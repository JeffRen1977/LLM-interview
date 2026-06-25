#!/usr/bin/env python3
"""
Lab 06 · LLM Prefill / Decode simulation + KV Cache + latency metrics

Skills: TTFT, TPOT, compute vs memory bound phases
Maps to prep §3.2 — complements basic/chapter_08_inference_pipeline.py

Run:
    python3 qualcomm/lab_06_llm_prefill_decode.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.utils import print_header


@dataclass
class ModelShape:
    n_layers: int = 32
    d_model: int = 4096
    n_heads: int = 32


def estimate_prefill_flops(prompt_len: int, cfg: ModelShape) -> float:
    """Rough GEMM-dominated FLOPs for prefill (O(L^2) attention included)."""
    l, d, nl = prompt_len, cfg.d_model, cfg.n_layers
    attn = 4 * nl * l * l * d  # QK^T + softmaxV rough
    ffn = 8 * nl * l * d * d  # two large linear per layer
    return float(attn + ffn)


def estimate_decode_flops_per_step(seq_len: int, cfg: ModelShape) -> float:
    """One decode step: small GEMMs + read KV length L."""
    d, nl = cfg.d_model, cfg.n_layers
    attn = 4 * nl * seq_len * d  # read full KV
    ffn = 8 * nl * d * d
    return float(attn + ffn)


def estimate_kv_bytes(seq_len: int, cfg: ModelShape, bytes_per: int = 2) -> int:
    return 2 * cfg.n_layers * seq_len * cfg.d_model * bytes_per


@dataclass
class LatencyProfile:
    prefill_ms: float
    decode_ms_per_token: float

    def ttft_ms(self) -> float:
        return self.prefill_ms

    def tpot_ms(self) -> float:
        return self.decode_ms_per_token

    def e2e_ms(self, n_output: int) -> float:
        return self.prefill_ms + max(0, n_output - 1) * self.decode_ms_per_token


def simulate_timed_run(prompt_len: int, n_output: int, cfg: ModelShape) -> LatencyProfile:
    """
    Toy timing: prefill scales with L^2; decode scales with growing KV (memory-bound factor).
    Constants chosen for relative comparison, not real hardware.
    """
    prefill_cost = estimate_prefill_flops(prompt_len, cfg) / 1e11  # normalized units
    decode_costs = []
    for t in range(n_output):
        seq = prompt_len + t
        flops = estimate_decode_flops_per_step(seq, cfg)
        kv_gb = estimate_kv_bytes(seq, cfg) / (1024 ** 3)
        decode_costs.append(flops / 1e11 + kv_gb * 0.3)
    return LatencyProfile(
        prefill_ms=prefill_cost * 10,
        decode_ms_per_token=float(np.mean(decode_costs)) * 10 if decode_costs else 0.0,
    )


def demo_prefill_vs_decode() -> None:
    print_header("Prefill vs Decode bottleneck", "Lab 06")
    cfg = ModelShape()
    prompt, n_out = 512, 128
    prof = simulate_timed_run(prompt, n_out, cfg)

    print(f"  prompt_len={prompt}, output_tokens={n_out}")
    print(f"  TTFT (prefill):     {prof.ttft_ms():.1f} ms  [compute-bound]")
    print(f"  TPOT (decode avg):  {prof.tpot_ms():.1f} ms  [memory-bound KV read]")
    print(f"  E2E latency:        {prof.e2e_ms(n_out):.1f} ms")
    print(f"  KV at end:          {estimate_kv_bytes(prompt + n_out, cfg) / 1e9:.3f} GB (FP16)")


def demo_naive_vs_kv_cache_steps(n_output: int) -> None:
    print_header("Forward passes: naive vs KV cache", "Lab 06")
    naive = sum(range(1, n_output + 1))  # recompute all history each step
    cached = n_output  # one forward per new token after prefill
    print(f"  Generate {n_output} tokens:")
    print(f"  Naive attention passes:  O(T^2) ~ {naive} layer-steps")
    print(f"  With KV cache:           O(T)  ~ {cached} layer-steps (+ 1 prefill)")


def demo_edge_budget() -> None:
    print_header("Snapdragon memory budget (Genie constraints)", "Lab 06")
    cfg = ModelShape(n_layers=32, d_model=3072)  # 3B-class
    weight_int4_gb = 3.0 * 0.5  # ~1.5 GB
    for ctx in (2048, 4096):
        kv = estimate_kv_bytes(ctx, cfg, bytes_per=2) / (1024 ** 3)
        total = weight_int4_gb + kv
        print(f"  ctx={ctx}: weights~{weight_int4_gb:.1f}GB + KV~{kv:.2f}GB ≈ {total:.2f}GB")


def main() -> None:
    print_header("LLM Prefill / Decode & KV Metrics", "Lab 06")
    demo_prefill_vs_decode()
    demo_naive_vs_kv_cache_steps(100)
    demo_edge_budget()
    print("\nLab 06 completed.")


if __name__ == "__main__":
    main()
