#!/usr/bin/env python3
"""
Lab 07 · MoE routing & expert dispatch

Skills: top-k gate, load balance, active vs total params
Maps to prep §11 — see also basic/chapter_16_moe_inference.py

Run:
    python3 qualcomm/lab_07_moe_routing.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.utils import print_header


def route_topk(logits: np.ndarray, k: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    idx = np.argpartition(probs, -k)[-k:]
    idx = idx[np.argsort(probs[idx])[::-1]]
    weights = probs[idx]
    weights /= weights.sum()
    return idx, weights


def expert_ffn(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    h = np.maximum(x @ w1, 0)
    return h @ w2


def moe_forward(
    x: np.ndarray,
    gate_w: np.ndarray,
    experts: List[Tuple[np.ndarray, np.ndarray]],
    k: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    logits = x @ gate_w
    ids, weights = route_topk(logits, k)
    out = np.zeros_like(x)
    for w, eid in zip(weights, ids):
        out += w * expert_ffn(x, experts[eid][0], experts[eid][1])
    return out, ids


def dispatch_by_expert(
    tokens: np.ndarray,
    gate_w: np.ndarray,
    k: int,
    num_experts: int,
) -> dict:
    """Group tokens per expert for batched expert parallel (training/serving pattern)."""
    buckets: dict = {i: [] for i in range(num_experts)}
    for i, x in enumerate(tokens):
        logits = x @ gate_w
        ids, _ = route_topk(logits, k)
        for eid in ids:
            buckets[int(eid)].append(i)
    return buckets


def demo_single_token() -> None:
    print_header("Single token MoE forward", "Lab 07")
    dim, num_experts, k = 16, 8, 2
    rng = np.random.default_rng(0)
    x = rng.standard_normal(dim).astype(np.float32)
    gate_w = rng.standard_normal((dim, num_experts)).astype(np.float32) * 0.1
    experts = [
        (rng.standard_normal((dim, dim * 2)).astype(np.float32) * 0.05,
         rng.standard_normal((dim * 2, dim)).astype(np.float32) * 0.05)
        for _ in range(num_experts)
    ]
    out, ids = moe_forward(x, gate_w, experts, k)
    print(f"  input dim={dim}, experts={num_experts}, top-k={k}")
    print(f"  selected experts: {ids.tolist()}")
    print(f"  output norm: {np.linalg.norm(out):.4f}")


def demo_load_balance() -> None:
    print_header("Expert load distribution (10k tokens)", "Lab 07")
    dim, num_experts, k = 32, 8, 2
    rng = np.random.default_rng(42)
    gate_w = rng.standard_normal((dim, num_experts)).astype(np.float32) * 0.1
    tokens = rng.standard_normal((10_000, dim)).astype(np.float32)
    counts: Counter = Counter()
    for x in tokens:
        ids, _ = route_topk(x @ gate_w, k)
        counts.update(ids.tolist())
    ideal = 10_000 * k / num_experts
    for e in range(num_experts):
        print(f"  expert {e}: {counts[e]:5d}  (ideal~{ideal:.0f})")


def demo_edge_challenge() -> None:
    print_header("Edge deployment challenge", "Lab 07")
    hidden, ffn, n_exp, k = 4096, 14336, 8, 2
    one_ffn = 3 * hidden * ffn
    total = n_exp * one_ffn * 2 / (1024 ** 3)
    active = (k * one_ffn + hidden * n_exp) * 2 / (1024 ** 3)
    print(f"  Store all experts (FP16):  {total:.2f} GB")
    print(f"  Compute per token (FP16):  {active:.3f} GB equiv")
    print("  Phone must STORE all experts; HTP dislikes dynamic routing")


def main() -> None:
    print_header("MoE Routing & Dispatch", "Lab 07")
    demo_single_token()
    demo_load_balance()
    demo_edge_challenge()
    print("\nLab 07 completed.")


if __name__ == "__main__":
    main()
