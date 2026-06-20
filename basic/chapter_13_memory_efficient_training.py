#!/usr/bin/env python3
"""
Chapter 13 · Memory-Efficient Training — Complete Code Reference

Self-contained demos for document/chapter_13_memory_efficient_training.md.
Mirrors the pipeline in openAI/Problem_4_openAI_memory_efficient_training.py.

Sections:
  13.2   Training memory breakdown — params, grads, Adam states
  13.4   Mixed precision — dtype storage, AMP training step
  13.5   Gradient checkpointing — checkpoint() wrapper
  13.6   Four-scenario peak memory comparison (CUDA)
  13.7   DDP / TP / PP overview
  13.8   ZeRO / FSDP per-GPU memory estimate
  13.9   Technique selection + gradient accumulation
  13.10  MFU and tokens/s
  13.12  Transformer attention activation estimate
  13.13  LoRA parameter count
  13.15  FSDP / DeepSpeed config keys, PP bubble (GPipe vs 1F1B)
  13.20  Thought questions
  13.21  Interview quick reference

Run concept demos (CPU, no GPU required):
    python3 basic/chapter_13_memory_efficient_training.py

Run GPU memory comparison (needs CUDA):
    python3 basic/chapter_13_memory_efficient_training.py --gpu

Run full-scale benchmark (Problem_4 defaults, slower):
    python3 basic/chapter_13_memory_efficient_training.py --gpu --full
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    from torch.utils.checkpoint import checkpoint

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from torch.amp import GradScaler as AmpGradScaler
    from torch.amp import autocast as amp_autocast

    def make_scaler(enabled: bool):
        return AmpGradScaler("cuda") if enabled else None

    def autocast_ctx(enabled: bool):
        return amp_autocast("cuda", enabled=enabled)

except ImportError:
    from torch.cuda.amp import GradScaler as AmpGradScaler
    from torch.cuda.amp import autocast as amp_autocast

    def make_scaler(enabled: bool):
        return AmpGradScaler() if enabled else None

    def autocast_ctx(enabled: bool):
        return amp_autocast(enabled=enabled)


# =============================================================================
# 13.2  Training memory estimates (pure math)
# =============================================================================

DTYPE_BYTES = {"FP32": 4, "FP16": 2, "BF16": 2}


def estimate_adam_state_gb(num_params: int, bytes_per_param: float = 4.0) -> dict:
    """
    Params + gradients + Adam (m, v) without activations (chapter 13.2).

    Adam FP32: 4× parameter count in storage (W, grad, m, v).
    """
    total_bytes = num_params * bytes_per_param * 4
    return {
        "num_params": num_params,
        "params_gb": num_params * bytes_per_param / (1024 ** 3),
        "grads_gb": num_params * bytes_per_param / (1024 ** 3),
        "optimizer_gb": num_params * bytes_per_param * 2 / (1024 ** 3),
        "total_gb": total_bytes / (1024 ** 3),
    }


def activation_bytes_per_layer(
    batch: int,
    seq_len: int,
    hidden_dim: int,
    expansion: int = 4,
    bytes_per_elem: float = 4.0,
) -> float:
    """Rough FFN-style activation footprint per layer (chapter 13.3.1)."""
    # input + expanded intermediate + output (order-of-magnitude)
    elems = batch * seq_len * hidden_dim * (2 + expansion)
    return elems * bytes_per_elem


# =============================================================================
# 13.7  Technique metadata
# =============================================================================

@dataclass(frozen=True)
class TrainingTechnique:
    name: str
    memory_saving: str
    compute_overhead: str
    difficulty: str
    when_to_use: str


TECHNIQUES = (
    TrainingTechnique("Mixed precision (AMP)", "~30–50%", "none / faster", "low", "default on"),
    TrainingTechnique("Gradient checkpointing", "~50–80%", "+20–30%", "medium", "deep models, OOM"),
    TrainingTechnique("Gradient accumulation", "micro-batch peak", "+steps", "low", "large effective batch"),
    TrainingTechnique("ZeRO-3 / FSDP", "~4P/N per GPU", "AllGather", "high", "multi-GPU 7B+"),
    TrainingTechnique("Tensor Parallel (TP)", "shard layers", "AllReduce", "high", "layer too large"),
    TrainingTechnique("LoRA / QLoRA", "train adapters only", "small fwd", "medium", "finetuning"),
    TrainingTechnique("Flash Attention", "O(T) not O(T²)", "kernel", "low", "long context train"),
)


def estimate_zero_per_gpu_gb(
    num_params: int,
    num_gpus: int,
    bytes_per_param: float = 2.0,
    zero_stage: int = 3,
) -> dict:
    """
    Per-GPU training state under DDP vs ZeRO (chapter 13.8).

    Simplified Adam storage: param + grad + 2 optimizer moments = 4P bytes.
    """
    p = num_params * bytes_per_param
    ddp_per_gpu = 4 * p

    if zero_stage >= 3:
        zero_per_gpu = 4 * p / num_gpus
    elif zero_stage == 2:
        zero_per_gpu = p + 3 * p / num_gpus  # params local; grad + opt sharded
    elif zero_stage == 1:
        zero_per_gpu = 2 * p + 2 * p / num_gpus  # params + grad local; opt sharded
    else:
        zero_per_gpu = ddp_per_gpu

    return {
        "zero_stage": zero_stage,
        "num_gpus": num_gpus,
        "ddp_per_gpu_gb": ddp_per_gpu / (1024 ** 3),
        "zero_per_gpu_gb": zero_per_gpu / (1024 ** 3),
        "savings_vs_ddp": 1 - zero_per_gpu / ddp_per_gpu,
    }


def estimate_attention_activation_gb(
    batch: int,
    seq_len: int,
    n_heads: int,
    bytes_per_elem: float = 2.0,
) -> float:
    """Standard attention score matrix S: B × heads × T × T (chapter 13.12)."""
    elems = batch * n_heads * seq_len * seq_len
    return elems * bytes_per_elem / (1024 ** 3)


def compute_training_stats(
    *,
    num_params: int,
    global_batch: int,
    seq_len: int,
    step_time_s: float,
    peak_tflops_per_gpu: float = 312.0,
    num_gpus: int = 1,
) -> dict:
    """tokens/s and rough MFU (chapter 13.10)."""
    tokens_per_step = global_batch * seq_len
    tokens_per_s = tokens_per_step / step_time_s
    flops_per_token = 6 * num_params
    actual_flops_s = flops_per_token * tokens_per_s
    peak_flops_s = peak_tflops_per_gpu * 1e12 * num_gpus
    mfu = actual_flops_s / peak_flops_s if peak_flops_s else 0.0
    return {
        "tokens_per_s": tokens_per_s,
        "actual_tflops": actual_flops_s / 1e12,
        "peak_tflops": peak_tflops_per_gpu * num_gpus,
        "mfu": mfu,
    }


def lora_params_per_linear(hidden: int, rank: int) -> int:
    """LoRA A + B for one Linear(d, d) (chapter 13.13)."""
    return 2 * hidden * rank


def pipeline_bubble_fraction(stages: int, microbatches: int, schedule: str = "1f1b") -> float:
    """
    Pipeline bubble ratio (chapter 13.15.3).

    schedule: 'gpipe' or '1f1b'
    """
    p, m = stages, microbatches
    if p <= 1:
        return 0.0
    if schedule == "gpipe":
        return (p - 1) / (p + m - 1)
    return (p - 1) / (m + 2 * (p - 1))


FSDP_KEY_PARAMS = (
    ("FULL_SHARD", "ZeRO-3 equivalent, lowest memory"),
    ("SHARD_GRAD_OP", "ZeRO-2-like, often faster"),
    ("MixedPrecision(bfloat16)", "default on A100/H100"),
    ("auto_wrap_policy", "shard per TransformerBlock"),
    ("use_orig_params=True", "needed for LoRA / param groups"),
)

DEEPSPEED_KEY_PARAMS = (
    ("zero_optimization.stage", "1/2/3 = ZeRO stage"),
    ("overlap_comm", "overlap comm with compute"),
    ("offload_optimizer.device=cpu", "PCIe for GPU memory"),
    ("train_batch_size", "global = micro × accum × world"),
    ("activation_checkpointing", "DeepSpeed grad checkpointing"),
)


# =============================================================================
# 13.4 / 13.5  Model + training step (PyTorch)
# =============================================================================

if TORCH_AVAILABLE:

    class MemoryIntensiveBlock(nn.Module):
        """FFN-style block: expand 4× hidden_dim (chapter 13.3.1)."""

        def __init__(self, hidden_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.ReLU(),
                nn.Linear(4 * hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class LargeModel(nn.Module):
        """Stack of MemoryIntensiveBlock with optional checkpointing (chapter 13.3.2)."""

        def __init__(
            self,
            num_layers: int = 16,
            hidden_dim: int = 512,
            use_checkpointing: bool = False,
        ):
            super().__init__()
            self.use_checkpointing = use_checkpointing
            self.input_projection = nn.Linear(hidden_dim, hidden_dim)
            self.layers = nn.ModuleList(
                MemoryIntensiveBlock(hidden_dim) for _ in range(num_layers)
            )
            self.output_projection = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.input_projection(x)
            for layer in self.layers:
                if self.use_checkpointing and self.training:
                    x = checkpoint(layer, x, use_reentrant=False)
                else:
                    x = layer(x)
            return self.output_projection(x)

    def run_training_step(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        use_amp: bool = False,
        device: str = "cuda",
    ) -> float:
        """Single forward + backward + step; return peak GPU memory in GB (13.2.1, 13.4)."""
        scaler = make_scaler(use_amp)
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()

        with autocast_ctx(use_amp):
            loss = model(x).mean()

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        return torch.cuda.max_memory_allocated() / (1024 ** 3)

    def compare_four_scenarios(
        *,
        num_layers: int,
        hidden_dim: int,
        batch_size: int,
        seq_len: int,
    ) -> Dict[str, Union[float, str]]:
        """Four-scenario peak memory table (chapter 13.6)."""
        device = "cuda"
        configs = [
            ("Standard FP32", False, False),
            ("Mixed Precision", True, False),
            ("Gradient Checkpointing", False, True),
            ("Combined", True, True),
        ]
        results: Dict[str, Union[float, str]] = {}

        for name, use_amp, use_cp in configs:
            model = LargeModel(num_layers, hidden_dim, use_checkpointing=use_cp).to(device)
            optimizer = torch.optim.Adam(model.parameters())
            try:
                peak = run_training_step(
                    model,
                    optimizer,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    hidden_dim=hidden_dim,
                    use_amp=use_amp,
                    device=device,
                )
                results[name] = peak
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    results[name] = "OOM"
                else:
                    raise
            finally:
                del model, optimizer
                torch.cuda.empty_cache()

        return results

    def benchmark_step_times(
        *,
        num_layers: int,
        hidden_dim: int,
        batch_size: int,
        seq_len: int,
        iterations: int = 5,
        warmup: int = 2,
    ) -> Dict[str, float]:
        """Average step time per configuration (chapter 13.6.2)."""
        device = "cuda"
        configs = [
            ("Standard FP32", False, False),
            ("Mixed Precision", True, False),
            ("Gradient Checkpointing", False, True),
            ("Combined", True, True),
        ]
        timings: Dict[str, float] = {}

        for name, use_amp, use_cp in configs:
            model = LargeModel(num_layers, hidden_dim, use_checkpointing=use_cp).to(device)
            optimizer = torch.optim.Adam(model.parameters())

            for _ in range(warmup):
                run_training_step(
                    model,
                    optimizer,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    hidden_dim=hidden_dim,
                    use_amp=use_amp,
                    device=device,
                )

            start = time.perf_counter()
            for _ in range(iterations):
                run_training_step(
                    model,
                    optimizer,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    hidden_dim=hidden_dim,
                    use_amp=use_amp,
                    device=device,
                )
            timings[name] = (time.perf_counter() - start) / iterations

            del model, optimizer
            torch.cuda.empty_cache()

        return timings


# =============================================================================
# Demos
# =============================================================================

def demo_training_memory_breakdown() -> None:
    print("\n" + "=" * 60)
    print("[13.2] Training memory breakdown (Adam FP32, no activations)")
    print("=" * 60)
    for label, n in [("DistilBERT-scale", 67_000_000), ("LLaMA-7B", 7_000_000_000)]:
        est = estimate_adam_state_gb(n)
        print(f"\n  {label} ({n / 1e9:.2f}B params):")
        print(f"    params:     {est['params_gb']:.2f} GB")
        print(f"    grads:      {est['grads_gb']:.2f} GB")
        print(f"    Adam m,v:   {est['optimizer_gb']:.2f} GB")
        print(f"    subtotal:   {est['total_gb']:.2f} GB  (+ activations on top)")


def demo_dtype_and_activations() -> None:
    """Chapter 13.4: FP32 vs FP16 activation storage."""
    batch, seq, hidden = 16, 1024, 1024
    layers = 32
    fp32 = activation_bytes_per_layer(batch, seq, hidden, bytes_per_elem=4) * layers
    fp16 = activation_bytes_per_layer(batch, seq, hidden, bytes_per_elem=2) * layers

    print("\n" + "=" * 60)
    print("[13.4] Activation storage sketch (32 layers, 16×1024×1024)")
    print("=" * 60)
    print(f"  FP32 activations (rough): {fp32 / (1024 ** 3):.2f} GB")
    print(f"  FP16 activations (rough): {fp16 / (1024 ** 3):.2f} GB")
    print(f"  AMP often cuts activation footprint ~30–50% on GPU")


def demo_technique_table() -> None:
    print("\n" + "=" * 60)
    print("[13.9] Technique selection")
    print("=" * 60)
    print(f"  {'Technique':<28} | {'Memory':<12} | {'Compute':<12} | Use when")
    print("  " + "-" * 72)
    for t in TECHNIQUES:
        print(f"  {t.name:<28} | {t.memory_saving:<12} | {t.compute_overhead:<12} | {t.when_to_use}")


def demo_gradient_accumulation_pattern() -> None:
    """Chapter 13.7.1 — pseudocode pattern."""
    print("\n" + "=" * 60)
    print("[13.7.1] Gradient accumulation pattern")
    print("=" * 60)
    print("""
  accum_steps = 4
  for i, batch in enumerate(dataloader):
      loss = model(batch) / accum_steps
      loss.backward()
      if (i + 1) % accum_steps == 0:
          optimizer.step()
          optimizer.zero_grad(set_to_none=True)
  # Peak activation memory ≈ one micro-batch, not global batch
""")


def demo_ddp_vs_zero() -> None:
    """Chapter 13.7–13.8: DDP vs ZeRO-3 per-GPU memory."""
    n = 7_000_000_000
    gpus = 8
    print("\n" + "=" * 60)
    print(f"[13.8] DDP vs ZeRO per-GPU state (7B params, BF16, {gpus} GPUs)")
    print("=" * 60)
    for stage in (1, 2, 3):
        est = estimate_zero_per_gpu_gb(n, gpus, bytes_per_param=2.0, zero_stage=stage)
        print(
            f"  ZeRO-{stage}: {est['zero_per_gpu_gb']:.1f} GB/GPU  "
            f"(DDP baseline: {est['ddp_per_gpu_gb']:.1f} GB/GPU, "
            f"save {est['savings_vs_ddp']:.0%})"
        )


def demo_attention_activation() -> None:
    """Chapter 13.12: seq² effect on attention activations."""
    batch, heads = 4, 32
    print("\n" + "=" * 60)
    print("[13.12] Standard attention activation (B × heads × T²)")
    print("=" * 60)
    for seq in (2048, 4096, 8192):
        gb = estimate_attention_activation_gb(batch, seq, heads, bytes_per_elem=2.0)
        print(f"  seq={seq:5d}: ~{gb:.2f} GB  (FlashAttn → O(T), not O(T²))")


def demo_mfu() -> None:
    """Chapter 13.10: tokens/s and MFU."""
    stats = compute_training_stats(
        num_params=7_000_000_000,
        global_batch=512,
        seq_len=2048,
        step_time_s=50.0,  # illustrative; real step time depends on hardware
        peak_tflops_per_gpu=312.0,
        num_gpus=8,
    )
    print("\n" + "=" * 60)
    print("[13.10] Training throughput (7B, 512×2048, step=50s, 8×A100)")
    print("=" * 60)
    print(f"  tokens/s:        {stats['tokens_per_s']:,.0f}")
    print(f"  actual TFLOPs/s: {stats['actual_tflops']:.1f}")
    print(f"  peak TFLOPs/s:   {stats['peak_tflops']:.1f}")
    print(f"  MFU (cluster):   {stats['mfu']:.1%}  (30–55% typical in production)")


def demo_lora_params() -> None:
    """Chapter 13.13: LoRA param count."""
    hidden, rank, n_layers = 4096, 16, 32
    per_layer = lora_params_per_linear(hidden, rank)
    # two projections per transformer block (attn q/v or q/k/v/o simplified)
    per_block = per_layer * 4
    total = per_block * n_layers
    full = hidden * hidden * 4 * n_layers * 2  # rough 7B-scale order
    print("\n" + "=" * 60)
    print("[13.13] LoRA vs full finetune (illustrative)")
    print("=" * 60)
    print(f"  LoRA rank={rank}, hidden={hidden}, {n_layers} layers (4 matrices/layer)")
    print(f"  LoRA trainable params: ~{total / 1e6:.1f}M")
    print(f"  vs full block params (order): ~{full / 1e9:.1f}B+")


def demo_parallelism_overview() -> None:
    print("\n" + "=" * 60)
    print("[13.7] Parallelism cheat sheet")
    print("=" * 60)
    rows = [
        ("DDP", "shard data", "AllReduce grad", "throughput", "each GPU holds full 4P state"),
        ("TP", "shard weights in layer", "AllReduce/AllGather", "layer too big", "Megatron"),
        ("PP", "shard depth", "P2P activations", "very deep model", "1F1B schedule"),
        ("FSDP", "shard params+grad+opt", "AllGather", "7B+ multi-GPU", "ZeRO-3-like"),
    ]
    for name, cuts, comm, when, note in rows:
        print(f"  {name:5s} cuts {cuts:22s} comm={comm:18s} when={when}")


def demo_interview_quick_ref() -> None:
    print("\n" + "=" * 60)
    print("[13.20] AI Infra interview quick hits (training)")
    print("=" * 60)
    qa = [
        ("OOM on multi-GPU", "AMP → checkpoint → FSDP → TP → LoRA"),
        ("Low GPU util", "DataLoader, tokenize, storage IO first"),
        ("Low MFU", "comm, PP bubble, small batch, checkpoint"),
        ("Long context", "FlashAttention + checkpoint"),
        ("Finetune 7B on 1×24G", "QLoRA / LoRA + AMP"),
        ("FSDP vs DeepSpeed?", "FSDP=PyTorch native; DS=3D/offload"),
        ("PP bubble too high?", "1F1B schedule + increase microbatch M"),
    ]
    for q, a in qa:
        print(f"  Q: {q}")
        print(f"  A: {a}")


def demo_thought_question_5() -> None:
    stats = compute_training_stats(
        num_params=7_000_000_000,
        global_batch=1024,
        seq_len=4096,
        step_time_s=4.0,
        num_gpus=1,
    )
    print("\n" + "=" * 60)
    print("[13.19] Thought Q5: tokens/s (single-GPU rough MFU)")
    print("=" * 60)
    print(f"  tokens/s = 1024×4096/4 = {stats['tokens_per_s']:,.0f}")
    print(f"  MFU (1 GPU) = {stats['mfu']:.0%}  — divide by num_gpus in real clusters")


def demo_thought_question_6() -> None:
    n = lora_params_per_linear(4096, 16)
    print("\n" + "=" * 60)
    print("[13.19] Thought Q6: LoRA params one Linear(d=4096, r=16)")
    print("=" * 60)
    print(f"  2 × d × r = 2 × 4096 × 16 = {n:,} params")


def demo_pp_bubble() -> None:
    """Chapter 13.15.3: GPipe vs 1F1B bubble fractions."""
    p, m = 4, 8
    gpipe = pipeline_bubble_fraction(p, m, "gpipe")
    f1b = pipeline_bubble_fraction(p, m, "1f1b")
    print("\n" + "=" * 60)
    print(f"[13.15.3] Pipeline bubble (P={p} stages, M={m} microbatches)")
    print("=" * 60)
    print(f"  GPipe bubble: {(gpipe):.1%}  →  effective ~{(1-gpipe):.1%}")
    print(f"  1F1B bubble:  {(f1b):.1%}  →  effective ~{(1-f1b):.1%}")
    print("  Larger M → smaller bubble, but activation memory ∝ M")


def demo_fsdp_keys() -> None:
    print("\n" + "=" * 60)
    print("[13.15.1] FSDP key parameters")
    print("=" * 60)
    for key, desc in FSDP_KEY_PARAMS:
        print(f"  {key:28s} — {desc}")


def demo_deepspeed_keys() -> None:
    print("\n" + "=" * 60)
    print("[13.15.2] DeepSpeed ds_config.json keys")
    print("=" * 60)
    for key, desc in DEEPSPEED_KEY_PARAMS:
        print(f"  {key:32s} — {desc}")


def demo_framework_selection() -> None:
    print("\n" + "=" * 60)
    print("[13.15.4] Framework selection (illustrative)")
    print("=" * 60)
    rows = [
        ("7B finetune, 8×A100", "FSDP + BF16 + FlashAttn"),
        ("70B pretrain, 512+ GPUs", "Megatron TP+PP+DP or DeepSpeed 3D"),
        ("7B SFT, 1×24G", "QLoRA + AMP"),
        ("13B full, 32×A100", "FSDP + TP2 or ZeRO-3 + TP"),
    ]
    for scenario, stack in rows:
        print(f"  {scenario:28s} → {stack}")


def demo_thought_question_7() -> None:
    p, m = 4, 8
    print("\n" + "=" * 60)
    print("[13.20] Thought Q7: PP bubble GPipe vs 1F1B")
    print("=" * 60)
    g = pipeline_bubble_fraction(p, m, "gpipe")
    f = pipeline_bubble_fraction(p, m, "1f1b")
    print(f"  P={p}, M={m}: GPipe={g:.1%}, 1F1B={f:.1%}")


def demo_thought_question_1() -> None:
    """Chapter 13.11 Q1: 7B Adam FP32 minimum state."""
    n = 7_000_000_000
    est = estimate_adam_state_gb(n)
    print("\n" + "=" * 60)
    print("[13.11] Thought Q1: 7B params, Adam FP32 (params+grad+m+v)")
    print("=" * 60)
    print(f"  7B × 4 bytes × 4 copies ≈ {est['total_gb']:.0f} GB")
    print("  Real training needs activations + buffers → single-GPU FP32 Adam is impractical")


def demo_thought_question_3() -> None:
    print("\n" + "=" * 60)
    print("[13.11] Thought Q3: checkpointing vs INT8 for training")
    print("=" * 60)
    print("  Training:  prefer AMP + gradient checkpointing (+ ZeRO/FSDP at scale)")
    print("  Inference: prefer INT8/FP16 quantization (chapter 7) — no backward pass")


def _print_memory_table(results: Dict[str, Union[float, str]]) -> None:
    baseline: Optional[float] = None
    print(f"\n  {'Technique':<25} | {'Peak GB':<12} | {'Saved vs baseline'}")
    print("  " + "-" * 55)
    for name, mem in results.items():
        if mem == "OOM":
            print(f"  {name:<25} | {'OOM':<12} | N/A")
            continue
        assert isinstance(mem, float)
        if baseline is None:
            baseline = mem
            print(f"  {name:<25} | {mem:>8.2f} GB  | baseline")
        elif baseline > 0:
            saved = (baseline - mem) / baseline * 100
            print(f"  {name:<25} | {mem:>8.2f} GB  | {saved:+.1f}%")
        else:
            print(f"  {name:<25} | {mem:>8.2f} GB  | N/A")


def _print_timing_table(timings: Dict[str, float]) -> None:
    baseline = timings.get("Standard FP32", 1.0)
    print(f"\n  {'Technique':<25} | {'sec/step':<10} | {'vs FP32'}")
    print("  " + "-" * 50)
    for name, t in timings.items():
        ratio = baseline / t if t > 0 else 0.0
        print(f"  {name:<25} | {t:>8.4f}   | {ratio:.2f}×")


def demo_gpu_comparison(*, full: bool = False) -> bool:
    """Chapter 13.6: four-scenario CUDA comparison."""
    if not TORCH_AVAILABLE:
        print("\n[Skip 13.6] PyTorch required.")
        return False
    if not torch.cuda.is_available():
        print("\n[Skip 13.6] CUDA GPU required for memory comparison.")
        print("  Tip: python3 basic/chapter_13_memory_efficient_training.py --gpu")
        return False

    if full:
        num_layers, hidden_dim, batch_size, seq_len = 32, 1024, 16, 1024
        label = "full (Problem_4 scale)"
    else:
        num_layers, hidden_dim, batch_size, seq_len = 16, 512, 8, 512
        label = "default (lighter)"

    print("\n" + "=" * 60)
    print(f"[13.6] Four-scenario peak memory — {label}")
    print("=" * 60)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Model: layers={num_layers}, hidden={hidden_dim}")
    print(f"  Input: batch={batch_size}, seq={seq_len}, hidden={hidden_dim}")

    results = compare_four_scenarios(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    _print_memory_table(results)

    if full:
        print("\n[13.6.2] Step time benchmark (5 iterations)...")
        timings = benchmark_step_times(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            seq_len=seq_len,
            iterations=5,
            warmup=2,
        )
        _print_timing_table(timings)

    return True


# =============================================================================
# Main
# =============================================================================

def main(*, gpu: bool = False, full: bool = False) -> None:
    print("Chapter 13 · Memory-Efficient Training")
    print("Doc: document/chapter_13_memory_efficient_training.md")
    print("Full pipeline: openAI/Problem_4_openAI_memory_efficient_training.py\n")

    demo_training_memory_breakdown()
    demo_dtype_and_activations()
    demo_parallelism_overview()
    demo_ddp_vs_zero()
    demo_attention_activation()
    demo_mfu()
    demo_lora_params()
    demo_pp_bubble()
    demo_fsdp_keys()
    demo_deepspeed_keys()
    demo_framework_selection()
    demo_technique_table()
    demo_gradient_accumulation_pattern()
    demo_thought_question_1()
    demo_thought_question_3()
    demo_thought_question_5()
    demo_thought_question_6()
    demo_thought_question_7()
    demo_interview_quick_ref()

    if gpu:
        demo_gpu_comparison(full=full)
    elif TORCH_AVAILABLE and torch.cuda.is_available():
        print("\n" + "-" * 60)
        print("CUDA detected. Run with --gpu for four-scenario memory comparison.")
    else:
        print("\n" + "-" * 60)
        print("Tip: run with --gpu on a CUDA machine for live memory benchmarks")

    print("\n" + "=" * 60)
    print("Demos completed.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chapter 13 memory-efficient training demos")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Run four-scenario CUDA memory comparison",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use Problem_4 model size + step timing benchmark (requires --gpu)",
    )
    args = parser.parse_args()
    try:
        main(gpu=args.gpu, full=args.full)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise
