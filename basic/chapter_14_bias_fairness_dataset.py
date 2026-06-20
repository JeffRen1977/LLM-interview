#!/usr/bin/env python3
"""
Chapter 14 · Dataset Bias & Fairness — Complete Code Reference

Self-contained demos for document/chapter_14_bias_fairness_dataset.md.
Mirrors the pipeline in openAI/Problem_6_openAI_fix_bias_dataset.py.

Sections:
  14.2   Bias source taxonomy
  14.4   Accuracy trap, imbalance ratio, grouped metrics
  14.5   Demographic parity, equalized odds (synthetic)
  14.6   Pre/In/Post mitigation — random oversample, class_weight
  14.7   Before/after model comparison (sklearn)
  14.11  Thought questions
  14.12  Interview quick reference

Run concept demos (numpy + sklearn only):
    python3 basic/chapter_14_bias_fairness_dataset.py

Full SMOTE + plots:
    python3 openAI/Problem_6_openAI_fix_bias_dataset.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# =============================================================================
# 14.2 / 14.4  Dataset + detection
# =============================================================================

def make_biased_dataset(
    n: int = 1000,
    minority_frac: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Problem 6 style: 95% class 0, 5% class 1."""
    rng = np.random.default_rng(seed)
    n_minority = int(n * minority_frac)
    n_majority = n - n_minority
    X = rng.random((n, 2))
    X[:, 0] *= 10.0   # feature1: income-like
    X[:, 1] *= 5.0    # feature2: credit-score-like
    y = np.array([0] * n_majority + [1] * n_minority)
    perm = rng.permutation(n)
    return X[perm], y[perm]


def imbalance_ratio(y: np.ndarray) -> float:
    counts = np.bincount(y.astype(int))
    majority = counts.max()
    minority = counts.min()
    return majority / minority if minority else float("inf")


def severity_label(ratio: float) -> str:
    if ratio > 10:
        return "severe"
    if ratio > 5:
        return "moderate"
    return "mild"


# =============================================================================
# 14.5  Fairness metrics (toy binary + sensitive attribute)
# =============================================================================

def demographic_parity_rate(y_pred: np.ndarray, group: np.ndarray, value) -> float:
    """P(y_hat=1 | group=value)."""
    mask = group == value
    if mask.sum() == 0:
        return 0.0
    return y_pred[mask].mean()


def equalized_odds_tpr(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray, value) -> float:
    mask = (group == value) & (y_true == 1)
    if mask.sum() == 0:
        return 0.0
    return y_pred[mask].mean()


def disparate_impact(pos_rate_minority: float, pos_rate_majority: float) -> float:
    """Four-fifths rule: ratio should often be >= 0.8."""
    if pos_rate_majority == 0:
        return 0.0
    return pos_rate_minority / pos_rate_majority


def detect_bias_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> Dict:
    results = {}
    for val in np.unique(sensitive):
        mask = sensitive == val
        results[str(val)] = {
            "count": int(mask.sum()),
            "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
            "f1": float(f1_score(y_true[mask], y_pred[mask], zero_division=0)),
        }
    return results


# =============================================================================
# 14.6  Mitigation
# =============================================================================

def random_oversample(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Simple minority oversample (SMOTE substitute without imblearn)."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    X_parts, y_parts = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) < max_count:
            extra = rng.choice(idx, size=max_count - len(idx), replace=True)
            idx = np.concatenate([idx, extra])
        X_parts.append(X[idx])
        y_parts.append(y[idx])
    X_out = np.vstack(X_parts)
    y_out = np.concatenate(y_parts)
    perm = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm]


# =============================================================================
# Demos
# =============================================================================

def demo_accuracy_trap() -> None:
    y = np.array([0] * 950 + [1] * 50)
    y_pred = np.zeros_like(y)  # predict all negative
    acc = accuracy_score(y, y_pred)
    rec = recall_score(y, y_pred, zero_division=0)
    print("\n" + "=" * 60)
    print("[14.4] Accuracy trap — predict all majority class")
    print("=" * 60)
    print(f"  Accuracy: {acc:.1%}   Recall (minority): {rec:.1%}")
    print("  High accuracy does NOT mean useful model on imbalanced data.")


def demo_dataset_stats() -> None:
    X, y = make_biased_dataset()
    ratio = imbalance_ratio(y)
    print("\n" + "=" * 60)
    print("[14.2 / 14.4] Synthetic biased dataset (Problem 6 scale)")
    print("=" * 60)
    print(f"  Shape: {X.shape},  class 0: {(y==0).sum()},  class 1: {(y==1).sum()}")
    print(f"  Imbalance ratio: {ratio:.1f}:1  ({severity_label(ratio)})")


def demo_train_compare() -> None:
    """14.6 / 14.7 — baseline vs oversampled train (Problem 6 flow)."""
    X, y = make_biased_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Baseline
    m0 = LogisticRegression(random_state=42, max_iter=1000)
    m0.fit(X_train, y_train)
    pred0 = m0.predict(X_test)

    # Oversampled train (SMOTE analog)
    X_res, y_res = random_oversample(X_train, y_train)
    m1 = LogisticRegression(random_state=42, max_iter=1000)
    m1.fit(X_res, y_res)
    pred1 = m1.predict(X_test)

    # class_weight balanced (in-processing, no resampling)
    m2 = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
    m2.fit(X_train, y_train)
    pred2 = m2.predict(X_test)

    print("\n" + "=" * 60)
    print("[14.7] Model comparison on held-out test (real distribution)")
    print("=" * 60)
    for name, pred in [
        ("Baseline (biased train)", pred0),
        ("Random oversample train", pred1),
        ("class_weight=balanced", pred2),
    ]:
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        print(f"  {name:28s}  acc={acc:.3f}  F1={f1:.3f}  recall={rec:.3f}")

    print("\n  Confusion matrix (oversample train):")
    cm = confusion_matrix(y_test, pred1)
    print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:3d}")
    print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:3d}")


def demo_fairness_metrics() -> None:
    rng = np.random.default_rng(0)
    n = 400
    group = rng.integers(0, 2, size=n)  # sensitive attribute
    y_true = rng.integers(0, 2, size=n)
    # Biased predictor: higher positive rate for group 0
    scores = rng.random(n) + (group == 0) * 0.15
    y_pred = (scores > 0.55).astype(int)

    p0 = demographic_parity_rate(y_pred, group, 0)
    p1 = demographic_parity_rate(y_pred, group, 1)
    tpr0 = equalized_odds_tpr(y_true, y_pred, group, 0)
    tpr1 = equalized_odds_tpr(y_true, y_pred, group, 1)
    di = disparate_impact(min(p0, p1), max(p0, p1))

    print("\n" + "=" * 60)
    print("[14.5] Fairness metrics (synthetic)")
    print("=" * 60)
    print(f"  Demographic parity P(y_hat=1): group0={p0:.2f}, group1={p1:.2f}")
    print(f"  TPR by group: group0={tpr0:.2f}, group1={tpr1:.2f}")
    print(f"  Disparate impact ratio: {di:.2f}  (rule-of-thumb target ≥ 0.8)")


def demo_grouped_eval() -> None:
    rng = np.random.default_rng(1)
    n = 300
    sensitive = rng.integers(0, 2, size=n)
    X = rng.random((n, 2))
    y = (X[:, 0] + sensitive * 0.5 > 0.8).astype(int)
    X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=42
    )
    model = LogisticRegression(max_iter=500)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    groups = detect_bias_by_group(y_te, pred, s_te)
    print("\n" + "=" * 60)
    print("[14.4.3] Group-wise evaluation")
    print("=" * 60)
    for g, stats in groups.items():
        print(f"  group {g}: n={stats['count']}, acc={stats['accuracy']:.3f}, F1={stats['f1']:.3f}")


def demo_mitigation_layers() -> None:
    print("\n" + "=" * 60)
    print("[14.6] Three-layer mitigation")
    print("=" * 60)
    rows = [
        ("Pre", "SMOTE, oversample, collect more minority data"),
        ("In", "class_weight, fair loss, adversarial debiasing"),
        ("Post", "group-specific thresholds, calibration"),
        ("LLM", "curated corpus, RLHF constraints, moderation API"),
    ]
    for layer, examples in rows:
        print(f"  {layer:5s} — {examples}")


def demo_thought_q1() -> None:
    print("\n" + "=" * 60)
    print("[14.11] Q1: all-negative predictor")
    print("=" * 60)
    print("  acc=95%, recall(minority)=0% — accuracy trap")


def demo_thought_q2() -> None:
    print("\n" + "=" * 60)
    print("[14.11] Q2: why SMOTE only on train?")
    print("=" * 60)
    print("  Test must reflect production distribution; oversampling test inflates metrics.")


def demo_interview_quick_ref() -> None:
    print("\n" + "=" * 60)
    print("[14.12] Interview quick hits")
    print("=" * 60)
    qa = [
        ("Imbalance vs fairness?", "class count vs group performance gap"),
        ("Main metric?", "F1 / recall on minority, not accuracy"),
        ("SMOTE?", "synthetic minority samples in kNN convex hull"),
        ("Demographic parity?", "similar positive prediction rate per group"),
        ("LLM bias eval?", "BBQ, CrowS-Pairs, red teaming, locale metrics"),
    ]
    for q, a in qa:
        print(f"  Q: {q}")
        print(f"  A: {a}")


def main() -> None:
    print("Chapter 14 · Dataset Bias & Fairness")
    print("Doc: document/chapter_14_bias_fairness_dataset.md")
    print("Full pipeline: openAI/Problem_6_openAI_fix_bias_dataset.py\n")

    demo_dataset_stats()
    demo_accuracy_trap()
    demo_train_compare()
    demo_fairness_metrics()
    demo_grouped_eval()
    demo_mitigation_layers()
    demo_thought_q1()
    demo_thought_q2()
    demo_interview_quick_ref()

    print("\n" + "-" * 60)
    print("Tip: run Problem_6 for SMOTE + matplotlib visualizations")
    print("\n" + "=" * 60)
    print("Demos completed.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise
