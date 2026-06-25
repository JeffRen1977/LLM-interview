#!/usr/bin/env python3
"""
MoE inference concepts — demo for qualcomm_staff_ai_software_tools_prep.md §11

Sections:
  11.2  Dense vs MoE active params per token
  11.3  Top-k routing simulation
  11.4  Load imbalance across experts
  11.5  Memory: all experts vs active experts

Run:
    python3 basic/chapter_16_moe_inference.py
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass


@dataclass
class MoEConfig:
    num_experts: int = 8
    top_k: int = 2
    hidden_dim: int = 4096
    expert_ffn_dim: int = 14336  # e.g. Mixtral-scale FFN
    num_tokens: int = 1
    bytes_per_param: int = 2  # FP16


def dense_ffn_params(hidden: int, ffn: int) -> int:
    """One transformer FFN block: gate + up + down (SwiGLU-style ~3 matrices)."""
    return 3 * hidden * ffn


def moe_total_expert_params(num_experts: int, hidden: int, ffn: int) -> int:
    return num_experts * dense_ffn_params(hidden, ffn)


def moe_active_params(num_experts: int, top_k: int, hidden: int, ffn: int) -> int:
    """Per token: router + top_k expert FFNs."""
    router = hidden * num_experts
    active_ffn = top_k * dense_ffn_params(hidden, ffn)
    return router + active_ffn


def simulate_routing(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    seed: int = 0,
) -> Counter:
    """Toy gate logits -> top-k expert ids per token."""
    import random

    rng = random.Random(seed)
    counts: Counter = Counter()
    for _ in range(num_tokens):
        logits = [rng.random() for _ in range(num_experts)]
        top = sorted(range(num_experts), key=lambda i: logits[i], reverse=True)[:top_k]
        for e in top:
            counts[e] += 1
    return counts


def gb(params: int, bpp: int) -> float:
    return params * bpp / (1024**3)


def demo_dense_vs_moe() -> None:
    cfg = MoEConfig()
    dense = dense_ffn_params(cfg.hidden_dim, cfg.expert_ffn_dim)
    total = moe_total_expert_params(cfg.num_experts, cfg.hidden_dim, cfg.expert_ffn_dim)
    active = moe_active_params(cfg.num_experts, cfg.top_k, cfg.hidden_dim, cfg.expert_ffn_dim)

    print("\n" + "=" * 60)
    print("[11.2] Dense FFN vs MoE (Mixtral-scale dims, FP16)")
    print("=" * 60)
    print(f"  Dense one FFN block:     {gb(dense, 2):.2f} GB")
    print(f"  MoE all {cfg.num_experts} experts stored: {gb(total, 2):.2f} GB")
    print(f"  MoE active per token (top-{cfg.top_k}): {gb(active, 2):.3f} GB compute footprint")
    print(f"  Compute ratio active/total: {active/total:.1%}")


def demo_routing_load() -> None:
    cfg = MoEConfig()
    counts = simulate_routing(10_000, cfg.num_experts, cfg.top_k)
    print("\n" + "=" * 60)
    print("[11.4] Simulated expert load (10k tokens, top-2)")
    print("=" * 60)
    ideal = 10_000 * cfg.top_k / cfg.num_experts
    for e in range(cfg.num_experts):
        c = counts[e]
        bar = "#" * int(40 * c / max(counts.values()))
        print(f"  expert {e}: {c:5d}  (ideal~{ideal:.0f})  {bar}")
    print("  Load imbalance -> some experts hot, others idle (training uses aux loss)")


def demo_interview_answers() -> None:
    print("\n" + "=" * 60)
    print("[11.8] MoE interview quick hits")
    print("=" * 60)
    qa = [
        ("Why MoE?", "More capacity (params) without proportional compute per token"),
        ("Top-k?", "Each token uses k experts; k=2 common (Mixtral)"),
        ("vs TP for MoE?", "TP splits layers; MoE needs expert parallel + routing"),
        ("Edge challenge?", "Must store ALL expert weights; routing is dynamic"),
        ("QAIRT angle?", "Graph may need custom ops for sparse expert dispatch"),
    ]
    for q, a in qa:
        print(f"  Q: {q}")
        print(f"  A: {a}")


def main() -> None:
    print("Chapter 16 · MoE Inference Concepts (Qualcomm prep)")
    print("Doc: document/qualcomm_staff_ai_software_tools_prep.md §11\n")
    demo_dense_vs_moe()
    demo_routing_load()
    demo_interview_answers()
    print("\n" + "=" * 60)
    print("Demos completed.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise
