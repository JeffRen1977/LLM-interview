#!/usr/bin/env python3
"""
Chapter 15 · LLM Evaluation Framework — Complete Code Reference

Self-contained demos for document/chapter_15_llm_evaluation_framework.md.
Mirrors the pipeline in openAI/Problem_7_openAI_evaluation_framework.py.

Sections:
  15.2   EvaluationResult + EvaluationMetric pattern
  15.3   BLEU/ROUGE/BERTScore-style metrics (word overlap fallbacks)
  15.4   Human eval — A/B win rates, Likert simulation
  15.5   Online eval — implicit signals
  15.6   Mini EvaluationFramework.comprehensive_evaluation()
  15.12  Thought questions
  15.13  Interview quick reference

Run concept demos (numpy only):
    python3 basic/chapter_15_llm_evaluation_framework.py

Full framework (optional evaluate, bert-score):
    python3 openAI/Problem_7_openAI_evaluation_framework.py
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


# =============================================================================
# 15.2  Core types (Problem 7 pattern)
# =============================================================================

@dataclass
class EvaluationResult:
    metric_name: str
    score: float
    confidence: Optional[float] = None
    metadata: Optional[Dict] = None


class EvaluationMetric(ABC):
    @abstractmethod
    def compute(self, predictions: List[str], references: List[str]) -> EvaluationResult:
        ...


# =============================================================================
# 15.3  Automated metrics (fallback implementations)
# =============================================================================

def _token_f1(pred: str, ref: str) -> float:
    pred_w = set(pred.lower().split())
    ref_w = set(ref.lower().split())
    if not pred_w or not ref_w:
        return 0.0
    p = len(pred_w & ref_w) / len(pred_w)
    r = len(pred_w & ref_w) / len(ref_w)
    return 2 * p * r / (p + r) if (p + r) else 0.0


class SimpleBLEUMetric(EvaluationMetric):
    """Word precision proxy (Problem 7 BLEUMetric fallback)."""

    def compute(self, predictions: List[str], references: List[str]) -> EvaluationResult:
        scores = []
        for pred, ref in zip(predictions, references):
            pw = pred.lower().split()
            rw = set(ref.lower().split())
            scores.append(len(set(pw) & rw) / len(pw) if pw else 0.0)
        return EvaluationResult("BLEU", float(np.mean(scores)), metadata={"method": "word_precision"})


class SimpleROUGEMetric(EvaluationMetric):
    """Word recall proxy (Problem 7 ROUGEMetric fallback)."""

    def compute(self, predictions: List[str], references: List[str]) -> EvaluationResult:
        scores = []
        for pred, ref in zip(predictions, references):
            pw = set(pred.lower().split())
            rw = ref.lower().split()
            scores.append(len(pw & set(rw)) / len(rw) if rw else 0.0)
        return EvaluationResult("ROUGE-L", float(np.mean(scores)), metadata={"method": "word_recall"})


class SimpleBERTScoreMetric(EvaluationMetric):
    """Token F1 proxy for semantic similarity."""

    def compute(self, predictions: List[str], references: List[str]) -> EvaluationResult:
        scores = [_token_f1(p, r) for p, r in zip(predictions, references)]
        return EvaluationResult("BERTScore-F1", float(np.mean(scores)), metadata={"method": "token_f1"})


class SimplePerplexityMetric(EvaluationMetric):
    """Illustrative PPL — Problem 7 uses simulated value from avg length."""

    def compute(self, predictions: List[str], references: List[str]) -> EvaluationResult:
        avg_len = np.mean([len(t.split()) for t in predictions])
        ppl = 50.0 + avg_len * 0.5  # deterministic stand-in
        return EvaluationResult("Perplexity", ppl, metadata={"avg_length": avg_len, "simulated": True})


# =============================================================================
# 15.4 / 15.5  Human & online simulators (simplified Problem 7)
# =============================================================================

def ab_test_win_rates(
    outputs_a: List[str],
    outputs_b: List[str],
) -> Dict[str, float]:
    wins_a = wins_b = ties = 0
    for a, b in zip(outputs_a, outputs_b):
        sa = len(a.split())
        sb = len(b.split())
        if abs(sa - sb) <= 1:
            ties += 1
        elif sa > sb:
            wins_a += 1
        else:
            wins_b += 1
    total = max(len(outputs_a), 1)
    return {
        "model_a_win_rate": wins_a / total,
        "model_b_win_rate": wins_b / total,
        "tie_rate": ties / total,
    }


def likert_simulate(outputs: List[str], scale: int = 5) -> Dict:
    ratings = [min(scale, max(1, 2 + len(o.split()) // 8)) for o in outputs]
    return {
        "average": float(np.mean(ratings)),
        "std": float(np.std(ratings)),
        "ratings": ratings,
    }


def implicit_signals_simulate(outputs: List[str], users_per_output: int = 20) -> Dict:
    thumbs_up = thumbs_down = copied = 0
    for o in outputs:
        quality = min(1.0, len(set(o.split())) / max(len(o.split()), 1))
        for _ in range(users_per_output):
            if quality > 0.5:
                thumbs_up += 1
            else:
                thumbs_down += 1
            if quality > 0.7:
                copied += 1
    total = thumbs_up + thumbs_down
    return {
        "satisfaction_rate": thumbs_up / total if total else 0.0,
        "copied_responses": copied,
    }


# =============================================================================
# 15.6  Mini framework
# =============================================================================

class MiniEvaluationFramework:
    def __init__(self):
        self.metrics = {
            "bleu": SimpleBLEUMetric(),
            "rouge": SimpleROUGEMetric(),
            "bertscore": SimpleBERTScoreMetric(),
            "ppl": SimplePerplexityMetric(),
        }

    def comprehensive_evaluation(
        self, predictions: List[str], references: List[str]
    ) -> Dict:
        auto = {k: m.compute(predictions, references) for k, m in self.metrics.items()}
        human_ab = ab_test_win_rates(predictions, references)
        human_likert = likert_simulate(predictions)
        online = implicit_signals_simulate(predictions)
        return {"automated": auto, "human_ab": human_ab, "human_likert": human_likert, "online": online}


# Problem 7 demo data
DEMO_PREDICTIONS = [
    "The cat is sitting on the mat.",
    "Machine learning is a subset of artificial intelligence.",
    "The weather today is sunny and warm.",
    "Python is a popular programming language for data science.",
    "The quick brown fox jumps over the lazy dog.",
]

DEMO_REFERENCES = [
    "There is a cat on the mat.",
    "ML is part of AI technology.",
    "Today's weather is pleasant and sunny.",
    "Python programming is widely used in data analysis.",
    "A brown fox quickly jumps over a sleeping dog.",
]


# =============================================================================
# Demos
# =============================================================================

def demo_metric_comparison() -> None:
    print("\n" + "=" * 60)
    print("[15.3] Metric properties (interview)")
    print("=" * 60)
    rows = [
        ("PPL", "no reference", "fluency / LM fit", "pretrain"),
        ("BLEU", "yes", "n-gram precision", "MT"),
        ("ROUGE", "yes", "n-gram recall", "summarization"),
        ("BERTScore", "yes", "semantic similarity", "tasks with refs"),
        ("LLM-as-Judge", "optional", "helpfulness/safety", "chat eval"),
        ("MMLU etc.", "fixed MCQ", "knowledge", "capability"),
    ]
    for name, ref, measures, use in rows:
        print(f"  {name:14s} ref={ref:12s} measures={measures:22s} use={use}")


def demo_automated_metrics() -> None:
    fw = MiniEvaluationFramework()
    results = fw.comprehensive_evaluation(DEMO_PREDICTIONS, DEMO_REFERENCES)
    print("\n" + "=" * 60)
    print("[15.3 / 15.6] Automated metrics (Problem 7 sample data)")
    print("=" * 60)
    for name, res in results["automated"].items():
        print(f"  {res.metric_name:16s}: {res.score:.4f}")


def demo_human_online() -> None:
    fw = MiniEvaluationFramework()
    results = fw.comprehensive_evaluation(DEMO_PREDICTIONS, DEMO_REFERENCES)
    ab = results["human_ab"]
    lik = results["human_likert"]
    on = results["online"]
    print("\n" + "=" * 60)
    print("[15.4 / 15.5] Human & online simulation")
    print("=" * 60)
    print(f"  A/B pred vs ref win rates: A={ab['model_a_win_rate']:.0%}, B={ab['model_b_win_rate']:.0%}")
    print(f"  Likert avg (simulated): {lik['average']:.2f} ± {lik['std']:.2f}")
    print(f"  Online satisfaction: {on['satisfaction_rate']:.0%}, copies={on['copied_responses']}")


def demo_three_layers() -> None:
    print("\n" + "=" * 60)
    print("[15.1 / 15.9] Three-layer evaluation strategy")
    print("=" * 60)
    print("  Dev     → automated metrics + benchmark regression")
    print("  Pre-ship→ human Likert / A/B + red team + safety sets")
    print("  Prod    → thumbs up/down, copy rate, online A/B, SLA (ch.12)")


def demo_thought_q1() -> None:
    print("\n" + "=" * 60)
    print("[15.12] Q1: high BLEU, low user satisfaction")
    print("=" * 60)
    print("  BLEU ignores helpfulness, safety, factuality; add human + online signals.")


def demo_interview_quick_ref() -> None:
    print("\n" + "=" * 60)
    print("[15.13] Interview quick hits")
    print("=" * 60)
    qa = [
        ("Why hard to eval generation?", "open answers, subjective, multi-dim"),
        ("BLEU weakness?", "literal match, punishes paraphrase"),
        ("PPL for chat?", "weak; use benchmarks + human"),
        ("A/B vs Likert?", "pairwise compare vs multi-dim scoring"),
        ("Red team?", "safety worst-case, not average quality"),
        ("LLM-as-Judge risk?", "position/verbosity bias; calibrate to humans"),
    ]
    for q, a in qa:
        print(f"  Q: {q}")
        print(f"  A: {a}")


def main() -> None:
    print("Chapter 15 · LLM Evaluation Framework")
    print("Doc: document/chapter_15_llm_evaluation_framework.md")
    print("Full pipeline: openAI/Problem_7_openAI_evaluation_framework.py\n")

    demo_metric_comparison()
    demo_automated_metrics()
    demo_human_online()
    demo_three_layers()
    demo_thought_q1()
    demo_interview_quick_ref()

    print("\n" + "-" * 60)
    print("Tip: run Problem_7 for evaluate library + report generation")
    print("\n" + "=" * 60)
    print("Demos completed.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise
