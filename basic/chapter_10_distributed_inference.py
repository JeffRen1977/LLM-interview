#!/usr/bin/env python3
"""
Chapter 10 · Distributed Inference & Model Parallelism — Complete Code Reference

Self-contained demos for document/chapter_10_distributed_inference.md.

Sections:
  10.1  GPU memory budget (weights + KV cache)
  10.3  Tensor Parallelism — column / row split of linear layers
  10.4  Pipeline Parallelism — stage assignment & bubble ratio
  10.5  TP + PP configuration helper
  10.9  Data parallel vs tensor parallel comparison

Run:
    python3 basic/chapter_10_distributed_inference.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# =============================================================================
# 10.1  Memory budget — when do we need multi-GPU?
# =============================================================================

@dataclass
class ModelSpec:
    """Simplified LLM spec for memory estimation."""

    name: str
    num_params_b: float  # billions
    n_layers: int
    d_model: int
    n_heads: int


def estimate_gpu_memory_gb(
    model: ModelSpec,
    seq_len: int,
    num_requests: int = 1,
    bytes_per_param: int = 2,
    kv_bytes: int = 2,
) -> dict:
    """
    Rough GPU memory (chapter 10.1 + chapter 8 KV formula).

    Weight_mem = params × bytes
    KV_mem     = 2 × N_layers × L × d_model × bytes × num_requests
    """
    weight_gb = model.num_params_b * 1e9 * bytes_per_param / (1024 ** 3)
    kv_per_req_gb = (
        2 * model.n_layers * seq_len * model.d_model * kv_bytes / (1024 ** 3)
    )
    kv_total_gb = kv_per_req_gb * num_requests
    total_gb = weight_gb + kv_total_gb
    return {
        "model": model.name,
        "weight_gb": weight_gb,
        "kv_per_request_gb": kv_per_req_gb,
        "kv_total_gb": kv_total_gb,
        "total_gb": total_gb,
        "seq_len": seq_len,
        "num_requests": num_requests,
    }


def memory_per_tp_rank(total_gb: float, tp_size: int) -> float:
    """Weight + KV both shrink ~linearly with TP (head / column split)."""
    return total_gb / tp_size


# =============================================================================
# 10.3  Tensor Parallelism — split linear Y = X @ W
# =============================================================================

def column_parallel_linear(x: np.ndarray, w_parts: List[np.ndarray]) -> np.ndarray:
    """
    Column parallel: W split along output dim (chapter 10.3.1).

    x:     [batch, in_dim]
    w_parts[i]: [in_dim, out_dim / tp]
    Returns concat of partial outputs — simulates AllGather-free case
    where next layer consumes split activations.
    """
    outputs = [x @ w for w in w_parts]
    return np.concatenate(outputs, axis=-1)


def row_parallel_linear(x_parts: List[np.ndarray], w_parts: List[np.ndarray]) -> np.ndarray:
    """
    Row parallel: W and X split along input dim; AllReduce = sum (chapter 10.3.1).

    x_parts[i]: [batch, in_dim / tp]
    w_parts[i]: [in_dim / tp, out_dim]
    """
    partial = sum(x @ w for x, w in zip(x_parts, w_parts))
    return partial  # AllReduce(sum) in distributed setting


def verify_tensor_parallel_linear(
    in_dim: int = 64,
    out_dim: int = 32,
    tp_size: int = 4,
    seed: int = 0,
) -> None:
    """TP linear must match single-GPU matmul."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((8, in_dim))
    w = rng.standard_normal((in_dim, out_dim))

    y_ref = x @ w

    # Column parallel
    w_cols = np.split(w, tp_size, axis=1)
    y_col = column_parallel_linear(x, list(w_cols))

    # Row parallel
    x_rows = np.split(x, tp_size, axis=1)
    w_rows = np.split(w, tp_size, axis=0)
    y_row = row_parallel_linear(list(x_rows), list(w_rows))

    print("\n" + "=" * 60)
    print("[10.3] Tensor Parallel Linear (TP={})".format(tp_size))
    print("=" * 60)
    print(f"  W shape: {w.shape}, X shape: {x.shape}")
    print(f"  Column parallel max diff: {np.abs(y_ref - y_col).max():.2e}")
    print(f"  Row parallel max diff:    {np.abs(y_ref - y_row).max():.2e}")


def tp_attention_heads(n_heads: int, tp_size: int) -> List[Tuple[int, int]]:
    """Assign head ranges to each TP rank (chapter 10.3.2)."""
    assert n_heads % tp_size == 0
    heads_per_rank = n_heads // tp_size
    return [
        (i * heads_per_rank, (i + 1) * heads_per_rank)
        for i in range(tp_size)
    ]


# =============================================================================
# 10.4  Pipeline Parallelism — stages & bubble
# =============================================================================

def pipeline_bubble_ratio(num_micro_batches: int, num_stages: int) -> float:
    """
    GPipe-style bubble fraction (chapter 10.4.1).

    bubble = (num_stages - 1) / (num_micro_batches + num_stages - 1)
    """
    if num_micro_batches < 1:
        return 1.0
    return (num_stages - 1) / (num_micro_batches + num_stages - 1)


def assign_layers_to_stages(n_layers: int, pp_size: int) -> List[List[int]]:
    """Split layer indices across PP stages (chapter 10.4)."""
    layers = list(range(n_layers))
    chunk = n_layers // pp_size
    stages = []
    for i in range(pp_size):
        start = i * chunk
        end = n_layers if i == pp_size - 1 else (i + 1) * chunk
        stages.append(layers[start:end])
    return stages


# =============================================================================
# 10.5  TP × PP configuration
# =============================================================================

def validate_parallel_config(
    num_gpus: int,
    tp_size: int,
    pp_size: int,
) -> dict:
    """Check TP × PP = num_gpus (chapter 10.5)."""
    valid = (tp_size * pp_size == num_gpus)
    return {
        "num_gpus": num_gpus,
        "tp_size": tp_size,
        "pp_size": pp_size,
        "valid": valid,
        "gpus_per_stage": tp_size,
        "num_stages": pp_size,
    }


# =============================================================================
# Demos
# =============================================================================

def demo_memory_budget() -> None:
    models = [
        ModelSpec("LLaMA-7B", 7.0, 32, 4096, 32),
        ModelSpec("LLaMA-70B", 70.0, 80, 8192, 64),
    ]
    print("\n" + "=" * 60)
    print("[10.1] GPU Memory Budget (FP16, L=2048, 1 request)")
    print("=" * 60)
    for m in models:
        est = estimate_gpu_memory_gb(m, seq_len=2048, num_requests=1, bytes_per_param=2)
        print(f"  {est['model']}: weights {est['weight_gb']:.1f} GB + "
              f"KV {est['kv_total_gb']:.2f} GB ≈ {est['total_gb']:.1f} GB total")
        for tp in (1, 2, 4, 8):
            if tp <= 8:
                per_card = memory_per_tp_rank(est["total_gb"], tp)
                print(f"    TP={tp}: ~{per_card:.1f} GB per GPU (idealized split)")


def demo_tp_heads() -> None:
    assignment = tp_attention_heads(32, tp_size=4)
    print("\n" + "=" * 60)
    print("[10.3] TP Head Assignment (32 heads, TP=4)")
    print("=" * 60)
    for rank, (lo, hi) in enumerate(assignment):
        print(f"  GPU {rank}: heads {lo}–{hi - 1}  ({hi - lo} heads)")


def demo_pipeline() -> None:
    stages = assign_layers_to_stages(32, pp_size=4)
    print("\n" + "=" * 60)
    print("[10.4] Pipeline Stages (32 layers, PP=4)")
    print("=" * 60)
    for i, layers in enumerate(stages):
        print(f"  Stage {i} (GPU group {i}): layers {layers[0]}–{layers[-1]}")

    print("\n  Pipeline bubble ratio (PP=4):")
    for mb in (1, 4, 16, 64):
        b = pipeline_bubble_ratio(mb, 4)
        print(f"    micro_batches={mb:2d} → bubble {b:.1%}")


def demo_tp_pp_config() -> None:
    print("\n" + "=" * 60)
    print("[10.5] TP × PP on 8 GPUs")
    print("=" * 60)
    for tp, pp in [(2, 4), (4, 2), (8, 1), (1, 8)]:
        cfg = validate_parallel_config(8, tp, pp)
        status = "✓" if cfg["valid"] else "✗"
        print(f"  TP={tp}, PP={pp}: {status}  "
              f"({cfg['num_stages']} stages × {cfg['gpus_per_stage']} TP ranks)")


def demo_dp_vs_tp() -> None:
    print("\n" + "=" * 60)
    print("[10.9] Data Parallel vs Tensor Parallel (conceptual)")
    print("=" * 60)
    print("  3× GPU Data Parallel:")
    print("    Each GPU: 100% model, different requests → 3× throughput")
    print("  3× GPU Tensor Parallel:")
    print("    Each GPU: ~33% weights, same request → 1 model on 3 cards")
    print("  Use DP when model fits one card; use TP when it does not.")


def main() -> None:
    print("Chapter 10 · Distributed Inference & Model Parallelism")
    print("Doc: document/chapter_10_distributed_inference.md\n")

    demo_memory_budget()
    verify_tensor_parallel_linear()
    demo_tp_heads()
    demo_pipeline()
    demo_tp_pp_config()
    demo_dp_vs_tp()

    print("\n" + "=" * 60)
    print("All demos completed.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise
