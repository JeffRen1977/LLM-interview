#!/usr/bin/env python3
"""
Chapter 11 · Speculative Decoding — Complete Code Reference

Self-contained demos for document/chapter_11_speculative_decoding.md.

Sections:
  11.3  Rejection sampling accept/reject (lossless property)
  11.4  Expected tokens per target forward vs acceptance rate α
  11.4  Speedup estimate with draft cost
  11.3  Simulated speculative decode loop

Run:
    python3 basic/chapter_11_speculative_decoding.py
"""

from __future__ import annotations

import sys
from typing import List, Tuple

import numpy as np


# =============================================================================
# 11.3  Rejection sampling — one token accept test
# =============================================================================

def acceptance_probability(p_target: float, q_draft: float) -> float:
    """
    Probability of accepting draft token x (chapter 11.3.2).

    accept with prob min(1, p_T(x) / q_D(x))
    """
    if q_draft <= 0:
        return 0.0
    return min(1.0, p_target / q_draft)


def resample_from_adjusted(
    p_target: np.ndarray,
    q_draft: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """
    On reject: sample from max(0, p_T - q_D), normalized (conceptual).
    Simplified demo uses residual distribution.
    """
    residual = np.maximum(p_target - q_draft, 0.0)
    total = residual.sum()
    if total <= 0:
        return int(rng.choice(len(p_target), p=p_target))
    return int(rng.choice(len(residual), p=residual / total))


def speculative_verify_step(
    draft_tokens: List[int],
    p_target_rows: np.ndarray,
    q_draft_rows: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[List[int], int]:
    """
    Verify γ draft tokens against target distributions (chapter 11.3.1).

    p_target_rows[i]: target distribution at position t+i+1, shape [vocab]
    q_draft_rows[i]:  draft distribution used to sample draft_tokens[i]
    Returns: (accepted_token_list, num_target_forwards=1)
    """
    accepted: List[int] = []
    for i, tok in enumerate(draft_tokens):
        p_t = p_target_rows[i, tok]
        q_d = q_draft_rows[i, tok]
        if rng.random() < acceptance_probability(p_t, q_d):
            accepted.append(tok)
        else:
            accepted.append(resample_from_adjusted(p_target_rows[i], q_draft_rows[i], rng))
            break
    else:
        # All γ accepted — sample one bonus token from last row (standard algorithm)
        bonus = int(rng.choice(len(p_target_rows[-1]), p=p_target_rows[-1]))
        accepted.append(bonus)
    return accepted, 1


# =============================================================================
# 11.4  Expected tokens & speedup formulas
# =============================================================================

def expected_tokens_per_round(alpha: float, gamma: int) -> float:
    """
    E[accepted] ≈ (1 - α^{γ+1}) / (1 - α)  (chapter 11.4)
    """
    if alpha >= 1.0:
        return float(gamma + 1)
    if alpha <= 0.0:
        return 1.0
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def speedup_estimate(
    alpha: float,
    gamma: int,
    draft_cost_ratio: float,
) -> float:
    """
    Speedup vs baseline 1 target forward → 1 token.

    draft_cost_ratio = (γ × T_draft) / T_target
    """
    baseline_tokens = 1.0
    spec_tokens = expected_tokens_per_round(alpha, gamma)
    baseline_cost = 1.0
    spec_cost = 1.0 + draft_cost_ratio
    return (spec_tokens / baseline_tokens) / (spec_cost / baseline_cost)


# =============================================================================
# 11.3  Full simulation with aligned / misaligned draft
# =============================================================================

def simulate_decode(
    vocab_size: int,
    gamma: int,
    num_rounds: int,
    alignment: float,
    seed: int = 0,
) -> dict:
    """
    Monte Carlo speculative decode.

    alignment ∈ [0,1]: how often draft matches target argmax
    (higher → higher effective α)
    """
    rng = np.random.default_rng(seed)
    total_tokens = 0
    total_target_forwards = 0

    for _ in range(num_rounds):
        draft_tokens = []
        p_rows = []
        q_rows = []
        for _ in range(gamma):
            p = rng.dirichlet(np.ones(vocab_size) * 0.3)
            if rng.random() < alignment:
                q = p.copy()
            else:
                q = rng.dirichlet(np.ones(vocab_size) * 0.3)
            tok = int(rng.choice(vocab_size, p=q))
            draft_tokens.append(tok)
            p_rows.append(p)
            q_rows.append(q)

        accepted, n_fwd = speculative_verify_step(
            draft_tokens, np.array(p_rows), np.array(q_rows), rng
        )
        total_tokens += len(accepted)
        total_target_forwards += n_fwd

    tokens_per_forward = total_tokens / total_target_forwards
    return {
        "gamma": gamma,
        "alignment": alignment,
        "rounds": num_rounds,
        "total_tokens": total_tokens,
        "target_forwards": total_target_forwards,
        "tokens_per_forward": tokens_per_forward,
    }


# =============================================================================
# Demos
# =============================================================================

def demo_acceptance_formula() -> None:
    print("\n" + "=" * 60)
    print("[11.3] Acceptance probability min(1, p_T / q_D)")
    print("=" * 60)
    cases = [(0.5, 0.5), (0.5, 0.25), (0.1, 0.5), (0.8, 0.4)]
    for p, q in cases:
        print(f"  p_T={p:.2f}, q_D={q:.2f} → accept prob {acceptance_probability(p, q):.2f}")


def demo_expected_tokens() -> None:
    print("\n" + "=" * 60)
    print("[11.4] Expected tokens per target forward E[(1-α^{γ+1})/(1-α)]")
    print("=" * 60)
    for alpha in (0.5, 0.7, 0.9):
        for gamma in (2, 4, 8):
            e = expected_tokens_per_round(alpha, gamma)
            print(f"  α={alpha:.1f}, γ={gamma} → E ≈ {e:.2f} tokens/forward")


def demo_speedup_table() -> None:
    print("\n" + "=" * 60)
    print("[11.4] Speedup estimate (draft cost = 10% of target per draft token)")
    print("=" * 60)
    draft_ratio_per_token = 0.1
    for alpha in (0.6, 0.75, 0.9):
        for gamma in (4, 8):
            s = speedup_estimate(alpha, gamma, gamma * draft_ratio_per_token)
            print(f"  α={alpha:.2f}, γ={gamma} → speedup ≈ {s:.2f}×")


def demo_simulation() -> None:
    print("\n" + "=" * 60)
    print("[11.3] Monte Carlo simulation (γ=4, 500 rounds)")
    print("=" * 60)
    for align in (0.3, 0.6, 0.9):
        r = simulate_decode(64, gamma=4, num_rounds=500, alignment=align, seed=42)
        print(
            f"  alignment={align:.1f} → {r['tokens_per_forward']:.2f} tokens/target forward "
            f"(baseline=1.0)"
        )


def demo_standard_vs_speculative_timeline() -> None:
    print("\n" + "=" * 60)
    print("[11.1] Standard vs Speculative (conceptual timeline)")
    print("=" * 60)
    print("  Standard Decode (3 tokens):")
    print("    Target → Target → Target   (3 forwards)")
    print("  Speculative (accept 2, resample 1):")
    print("    Draft×3 → Target×1         (1 target forward, ~3 tokens out)")


def main() -> None:
    print("Chapter 11 · Speculative Decoding")
    print("Doc: document/chapter_11_speculative_decoding.md\n")

    demo_standard_vs_speculative_timeline()
    demo_acceptance_formula()
    demo_expected_tokens()
    demo_speedup_table()
    demo_simulation()

    print("\n" + "=" * 60)
    print("All demos completed.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise
