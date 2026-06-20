#!/usr/bin/env python3
"""
Chapter 7 · Model Quantization & Inference Optimization — Complete Code Reference

Self-contained demos for document/chapter_07_model_quantization.md.
Mirrors the pipeline in openAI/Problem_3_openAI_optimize_inference_model.py.

Sections:
  7.1  Inference goals — latency, memory, throughput
  7.3  Benchmark helpers — model size, latency, four ratios
  7.4  FP32 / FP16 / INT8 storage & dynamic quantization
  7.6  Compression ratio, speedup ratio, evaluation thresholds
  7.10 Thought questions (param count → MB)

Run concept demos (no model download):
    python3 basic/chapter_07_model_quantization.py

Run full DistilBERT benchmark (needs transformers + network):
    python3 basic/chapter_07_model_quantization.py --full
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# 7.1 / 7.4  Storage estimates (pure math)
# =============================================================================

DTYPE_BYTES = {
    "FP32": 4,
    "FP16": 2,
    "BF16": 2,
    "INT8": 1,
    "INT4": 0.5,
}


def estimate_weight_size_mb(num_params: int, bytes_per_param: float) -> float:
    """Theoretical weight size (chapter 7.3.2, 7.4.1)."""
    return num_params * bytes_per_param / (1024 ** 2)


def compression_vs_fp32(dtype: str) -> float:
    """Relative storage vs FP32."""
    return DTYPE_BYTES["FP32"] / DTYPE_BYTES[dtype]


# =============================================================================
# 7.6  Four key ratios
# =============================================================================

@dataclass
class OptimizationMetrics:
    fp32_size_mb: float
    optimized_size_mb: float
    fp32_latency_ms: float
    optimized_latency_ms: float

    @property
    def compression_ratio(self) -> float:
        if self.optimized_size_mb <= 0:
            return 0.0
        return self.fp32_size_mb / self.optimized_size_mb

    @property
    def speedup_ratio(self) -> float:
        if self.optimized_latency_ms <= 0:
            return 0.0
        return self.fp32_latency_ms / self.optimized_latency_ms

    @property
    def memory_saved_percent(self) -> float:
        if self.fp32_size_mb <= 0:
            return 0.0
        return (1.0 - self.optimized_size_mb / self.fp32_size_mb) * 100.0

    @property
    def time_saved_percent(self) -> float:
        if self.fp32_latency_ms <= 0:
            return 0.0
        return (1.0 - self.optimized_latency_ms / self.fp32_latency_ms) * 100.0


def evaluate_thresholds(metrics: OptimizationMetrics) -> dict:
    """Chapter 7.6 experience thresholds."""
    cr = metrics.compression_ratio
    sr = metrics.speedup_ratio
    return {
        "compression_ratio": cr,
        "compression_grade": "excellent" if cr > 2.0 else ("good" if cr > 1.5 else "needs_work"),
        "speedup_ratio": sr,
        "speedup_grade": "excellent" if sr > 1.5 else ("good" if sr > 1.2 else "needs_work"),
    }


# =============================================================================
# 7.3  Benchmark helpers (PyTorch)
# =============================================================================

if TORCH_AVAILABLE:

    def measure_state_dict_size_mb(model: nn.Module) -> float:
        """Serialize state_dict to temp file (chapter 7.3.2)."""
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            path = tmp.name
        try:
            torch.save(model.state_dict(), path)
            return os.path.getsize(path) / (1024 ** 2)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def measure_latency_ms(
        model: nn.Module,
        inputs: tuple,
        n_runs: int = 100,
        warmup: int = 10,
    ) -> float:
        """Average forward latency in ms (chapter 7.3.2)."""
        model.eval()
        with torch.no_grad():
            for _ in range(warmup):
                model(*inputs)
            start = time.perf_counter()
            for _ in range(n_runs):
                model(*inputs)
            elapsed = time.perf_counter() - start
        return (elapsed / n_runs) * 1000.0

    def apply_dynamic_int8_quant(model: nn.Module) -> nn.Module:
        """Dynamic PTQ on Linear layers (chapter 7.4.2)."""
        q = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        q.eval()
        return q

    def apply_fp16(model: nn.Module) -> nn.Module:
        """FP16 fallback (chapter 7.4.3)."""
        m = model.half()
        m.eval()
        return m


# =============================================================================
# 7.4  Minimal quantization demo (no HuggingFace)
# =============================================================================

if TORCH_AVAILABLE:

    class TinyClassifier(nn.Module):
        """Small MLP stand-in for DistilBERT Linear-heavy inference."""

        def __init__(self, dim: int = 768, hidden: int = 256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 2),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


def demo_storage_table() -> None:
    """FP32 / FP16 / INT8 size comparison (chapter 7.4.1)."""
    params = 67_000_000  # DistilBERT-scale
    print("\n" + "=" * 60)
    print("[7.4] Weight storage by dtype (67M params)")
    print("=" * 60)
    for dtype in ("FP32", "FP16", "INT8", "INT4"):
        mb = estimate_weight_size_mb(params, DTYPE_BYTES[dtype])
        ratio = compression_vs_fp32(dtype)
        print(f"  {dtype:5s}: {mb:7.1f} MB  (~{ratio:.1f}× vs FP32)")


def demo_four_ratios() -> None:
    """Synthetic before/after metrics (chapter 7.6)."""
    m = OptimizationMetrics(
        fp32_size_mb=256.0,
        optimized_size_mb=64.0,
        fp32_latency_ms=40.0,
        optimized_latency_ms=18.0,
    )
    ev = evaluate_thresholds(m)
    print("\n" + "=" * 60)
    print("[7.6] Four optimization ratios (example)")
    print("=" * 60)
    print(f"  Compression ratio:  {m.compression_ratio:.2f}×  ({ev['compression_grade']})")
    print(f"  Speedup ratio:      {m.speedup_ratio:.2f}×  ({ev['speedup_grade']})")
    print(f"  Memory saved:       {m.memory_saved_percent:.1f}%")
    print(f"  Time saved:         {m.time_saved_percent:.1f}%")


def demo_thought_question_1() -> None:
    """Chapter 7.10 Q1: 67M params FP32 → INT8."""
    n = 67_000_000
    fp32 = estimate_weight_size_mb(n, 4)
    int8 = estimate_weight_size_mb(n, 1)
    print("\n" + "=" * 60)
    print("[7.10] Thought Q1: 67M params FP32 → INT8")
    print("=" * 60)
    print(f"  FP32: ~{fp32:.0f} MB")
    print(f"  INT8: ~{int8:.0f} MB  (≈ 4× compression)")


def demo_tiny_model_quantization() -> None:
    """Dynamic INT8 on tiny MLP — no download (chapter 7.4.2)."""
    if not TORCH_AVAILABLE:
        print("\n[Skip 7.4] Tiny model quant demo requires PyTorch.")
        return

    torch.manual_seed(0)
    model = TinyClassifier()
    x = torch.randn(1, 768)

    fp32_size = measure_state_dict_size_mb(model)
    fp32_lat = measure_latency_ms(model, (x,))

    try:
        q_model = apply_dynamic_int8_quant(model)
        q_size = measure_state_dict_size_mb(q_model)
        q_lat = measure_latency_ms(q_model, (x,))
        method = "INT8 dynamic"
    except Exception as exc:
        print(f"\n[7.4] INT8 failed ({exc}), using FP16 fallback")
        q_model = apply_fp16(model)
        q_size = measure_state_dict_size_mb(q_model)
        q_lat = measure_latency_ms(q_model, (x.half(),))
        method = "FP16"

    metrics = OptimizationMetrics(fp32_size, q_size, fp32_lat, q_lat)
    ev = evaluate_thresholds(metrics)

    print("\n" + "=" * 60)
    print(f"[7.4] Tiny MLP quantization ({method})")
    print("=" * 60)
    print(f"  FP32: {fp32_size:.2f} MB, {fp32_lat:.3f} ms/forward")
    print(f"  Opt:  {q_size:.2f} MB, {q_lat:.3f} ms/forward")
    print(f"  Compression: {metrics.compression_ratio:.2f}×")
    print(f"  Speedup:     {metrics.speedup_ratio:.2f}×")
    print("  Note: INT8 speedup is strongest on CPU (oneDNN); GPU may differ.")


# =============================================================================
# Full DistilBERT pipeline (chapter 7.2–7.8, mirrors Problem_3)
# =============================================================================

def run_distilbert_benchmark() -> bool:
    """
    Full benchmark: FP32 baseline → INT8/FP16 → four ratios.
    Same flow as openAI/Problem_3_openAI_optimize_inference_model.py
    """
    if not TORCH_AVAILABLE:
        print("Full benchmark requires PyTorch.")
        return False

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        print("Full benchmark requires: pip install transformers")
        return False

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    print("\n" + "=" * 60)
    print("[7.2–7.8] Full DistilBERT benchmark")
    print("=" * 60)
    print(f"  Model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    fp32_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    fp32_model.eval()

    text = "This is a great library and I love using it!"
    inputs = tokenizer(text, return_tensors="pt")

    fp32_size = measure_state_dict_size_mb(fp32_model)
    fp32_lat = measure_latency_ms(fp32_model, (inputs["input_ids"], inputs["attention_mask"]))

    print(f"  FP32 baseline: {fp32_size:.2f} MB, {fp32_lat:.2f} ms/inference")

    try:
        opt_model = apply_dynamic_int8_quant(fp32_model)
        opt_lat = measure_latency_ms(opt_model, (inputs["input_ids"], inputs["attention_mask"]))
        label = "INT8"
    except Exception:
        opt_model = apply_fp16(fp32_model)
        fp16_inputs = inputs["input_ids"]  # embedding handles; attention mask stays long
        opt_lat = measure_latency_ms(opt_model, (fp16_inputs, inputs["attention_mask"]))
        label = "FP16"

    opt_size = measure_state_dict_size_mb(opt_model)
    metrics = OptimizationMetrics(fp32_size, opt_size, fp32_lat, opt_lat)
    ev = evaluate_thresholds(metrics)

    print(f"  {label} optimized: {opt_size:.2f} MB, {opt_lat:.2f} ms/inference")
    print(f"  Compression: {metrics.compression_ratio:.2f}× ({ev['compression_grade']})")
    print(f"  Speedup:     {metrics.speedup_ratio:.2f}× ({ev['speedup_grade']})")
    print(f"  Memory saved: {metrics.memory_saved_percent:.1f}%")
    return True


# =============================================================================
# Main
# =============================================================================

def main(full: bool = False) -> None:
    print("Chapter 7 · Model Quantization & Inference Optimization")
    print("Doc: document/chapter_07_model_quantization.md")
    print("Full pipeline: openAI/Problem_3_openAI_optimize_inference_model.py\n")

    demo_storage_table()
    demo_four_ratios()
    demo_thought_question_1()
    demo_tiny_model_quantization()

    if full:
        run_distilbert_benchmark()
    else:
        print("\n" + "-" * 60)
        print("Tip: run with --full for DistilBERT download benchmark")

    print("\n" + "=" * 60)
    print("Demos completed.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chapter 7 quantization demos")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full DistilBERT benchmark (downloads model)",
    )
    args = parser.parse_args()
    try:
        main(full=args.full)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise
