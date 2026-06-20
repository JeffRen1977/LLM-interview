#!/usr/bin/env python3
"""
Chapter 12 · Inference Monitoring & SLA — Complete Code Reference

Self-contained demos for document/chapter_12_inference_monitoring_sla.md.

Sections:
  12.2  TTFT, TPOT, E2E latency from timestamps
  12.4  Percentiles P50 / P95 / P99
  12.5  SLA / SLO pass-fail checks
  12.8  Capacity planning (replicas, tokens/s)
  12.9  Cost per million tokens

Run:
    python3 basic/chapter_12_inference_monitoring_sla.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# =============================================================================
# 12.2  Request metrics from timestamps
# =============================================================================

@dataclass
class RequestTrace:
    """Single request timing trace (chapter 12.6.1)."""

    request_id: str
    t_arrive: float
    t_first_token: float
    token_times: List[float] = field(default_factory=list)  # each output token arrival
    t_done: Optional[float] = None
    prompt_tokens: int = 0
    output_tokens: int = 0

    @property
    def ttft(self) -> float:
        return self.t_first_token - self.t_arrive

    @property
    def tpots(self) -> List[float]:
        if len(self.token_times) < 2:
            return []
        return [
            self.token_times[i + 1] - self.token_times[i]
            for i in range(len(self.token_times) - 1)
        ]

    @property
    def tpot_mean(self) -> float:
        t = self.tpots
        return float(np.mean(t)) if t else 0.0

    @property
    def e2e(self) -> float:
        end = self.t_done if self.t_done is not None else (
            self.token_times[-1] if self.token_times else self.t_first_token
        )
        return end - self.t_arrive

    @property
    def e2e_estimate(self) -> float:
        """E2E ≈ TTFT + TPOT × (output_tokens - 1)"""
        n = max(self.output_tokens, 1)
        if n == 1:
            return self.ttft
        return self.ttft + self.tpot_mean * (n - 1)


def percentiles(values: List[float], ps: List[float] = None) -> dict:
    """P50 / P95 / P99 (chapter 12.4)."""
    ps = ps or [50, 95, 99]
    if not values:
        return {f"p{int(p)}": float("nan") for p in ps}
    arr = np.array(values)
    return {f"p{int(p)}": float(np.percentile(arr, p)) for p in ps}


def aggregate_traces(traces: List[RequestTrace]) -> dict:
    """Aggregate TTFT, TPOT, E2E across requests."""
    ttfts = [t.ttft for t in traces]
    tpots = [t.tpot_mean for t in traces if t.tpot_mean > 0]
    e2es = [t.e2e for t in traces]
    return {
        "ttft": percentiles(ttfts),
        "tpot": percentiles(tpots),
        "e2e": percentiles(e2es),
        "count": len(traces),
    }


# =============================================================================
# 12.5  SLA / SLO checks
# =============================================================================

@dataclass
class SLOThresholds:
    ttft_p99_ms: float = 800.0
    tpot_p99_ms: float = 50.0
    e2e_p99_ms: float = 30_000.0
    error_rate_max: float = 0.001


def check_slo(traces: List[RequestTrace], slo: SLOThresholds) -> dict:
    """Return pass/fail per SLI (chapter 12.5)."""
    agg = aggregate_traces(traces)
    results = {}
    for name, key in [("ttft", "ttft"), ("tpot", "tpot"), ("e2e", "e2e")]:
        p99_ms = agg[key]["p99"] * 1000
        threshold = getattr(slo, f"{name}_p99_ms")
        results[name] = {
            "p99_ms": p99_ms,
            "threshold_ms": threshold,
            "pass": p99_ms <= threshold,
        }
    results["overall_pass"] = all(r["pass"] for r in results.values())
    return results


# =============================================================================
# 12.8  Capacity planning
# =============================================================================

def estimate_concurrent_requests(
    gpu_mem_gb: float,
    weight_gb: float,
    kv_per_request_gb: float,
    reserve_gb: float = 2.0,
) -> int:
    """Max concurrent requests from memory (chapter 12.8)."""
    available = gpu_mem_gb - weight_gb - reserve_gb
    if available <= 0:
        return 0
    return int(available // kv_per_request_gb)


def system_tokens_per_sec(
    concurrent_requests: int,
    tpot_sec: float,
) -> float:
    """Rough system throughput (chapter 12.8.1)."""
    if tpot_sec <= 0:
        return 0.0
    return concurrent_requests / tpot_sec


def replicas_needed(target_qps: float, capacity_per_replica: float, redundancy: float = 1.3) -> int:
    """Replicas with headroom (chapter 12.12 Q3)."""
    if capacity_per_replica <= 0:
        return 0
    return int(np.ceil(target_qps / capacity_per_replica * redundancy))


# =============================================================================
# 12.9  Cost
# =============================================================================

def cost_per_million_tokens(
    monthly_gpu_cost_usd: float,
    monthly_output_tokens: int,
) -> float:
    """$/1M output tokens (chapter 12.9)."""
    if monthly_output_tokens <= 0:
        return float("inf")
    return monthly_gpu_cost_usd / (monthly_output_tokens / 1e6)


# =============================================================================
# Synthetic trace generator for demos
# =============================================================================

def generate_synthetic_traces(n: int = 100, seed: int = 42) -> List[RequestTrace]:
    rng = np.random.default_rng(seed)
    traces = []
    for i in range(n):
        t0 = 0.0
        ttft = rng.lognormal(mean=-1.0, sigma=0.5)  # skewed, some slow
        out_len = int(rng.integers(20, 200))
        tpot_base = rng.lognormal(mean=-3.0, sigma=0.3)
        times = [t0 + ttft]
        for _ in range(out_len - 1):
            times.append(times[-1] + tpot_base * rng.lognormal(0, 0.2))
        traces.append(
            RequestTrace(
                request_id=f"req-{i}",
                t_arrive=t0,
                t_first_token=t0 + ttft,
                token_times=times,
                t_done=times[-1],
                prompt_tokens=int(rng.integers(50, 2000)),
                output_tokens=out_len,
            )
        )
    return traces


# =============================================================================
# Demos
# =============================================================================

def demo_single_request_metrics() -> None:
    trace = RequestTrace(
        request_id="demo",
        t_arrive=0.0,
        t_first_token=0.25,
        token_times=[0.25, 0.28, 0.31, 0.34, 0.37],
        t_done=0.37,
        output_tokens=5,
    )
    print("\n" + "=" * 60)
    print("[12.2] Single request metrics")
    print("=" * 60)
    print(f"  TTFT:     {trace.ttft * 1000:.0f} ms")
    print(f"  TPOT avg: {trace.tpot_mean * 1000:.0f} ms")
    print(f"  E2E:      {trace.e2e * 1000:.0f} ms")
    print(f"  E2E est:  {trace.e2e_estimate * 1000:.0f} ms  (TTFT + TPOT×(N-1))")


def demo_percentiles() -> None:
    traces = generate_synthetic_traces(200)
    agg = aggregate_traces(traces)
    print("\n" + "=" * 60)
    print("[12.4] Aggregated percentiles (200 synthetic requests)")
    print("=" * 60)
    for metric in ("ttft", "tpot", "e2e"):
        p = agg[metric]
        print(
            f"  {metric.upper():5s}  P50={p['p50']*1000:6.0f}ms  "
            f"P95={p['p95']*1000:6.0f}ms  P99={p['p99']*1000:6.0f}ms"
        )


def demo_slo_check() -> None:
    traces = generate_synthetic_traces(500)
    slo = SLOThresholds(ttft_p99_ms=800, tpot_p99_ms=50, e2e_p99_ms=30_000)
    result = check_slo(traces, slo)
    print("\n" + "=" * 60)
    print("[12.5] SLO check")
    print("=" * 60)
    for name in ("ttft", "tpot", "e2e"):
        r = result[name]
        status = "PASS ✓" if r["pass"] else "FAIL ✗"
        print(f"  {name.upper()} P99={r['p99_ms']:.0f}ms  (SLO {r['threshold_ms']:.0f}ms)  {status}")
    print(f"  Overall: {'PASS ✓' if result['overall_pass'] else 'FAIL ✗'}")


def demo_capacity() -> None:
    concurrent = estimate_concurrent_requests(
        gpu_mem_gb=24, weight_gb=14, kv_per_request_gb=0.5
    )
    tps = system_tokens_per_sec(concurrent, tpot_sec=0.04)
    reps = replicas_needed(target_qps=1000, capacity_per_replica=200, redundancy=1.3)
    print("\n" + "=" * 60)
    print("[12.8] Capacity planning (LLaMA-7B on 24GB)")
    print("=" * 60)
    print(f"  Est. max concurrent (memory): {concurrent}")
    print(f"  Est. system tokens/s (TPOT=40ms): {tps:.0f}")
    print(f"  Replicas for 1000 req/s @ 200 req/s/replica (+30%): {reps}")


def demo_cost() -> None:
    cost = cost_per_million_tokens(
        monthly_gpu_cost_usd=8 * 2000,  # 8× A100-ish
        monthly_output_tokens=500_000_000,
    )
    print("\n" + "=" * 60)
    print("[12.9] Cost per 1M output tokens")
    print("=" * 60)
    print(f"  8 GPUs × $2000/mo, 500M tokens/mo → ${cost:.2f} / 1M tokens")


def main() -> None:
    print("Chapter 12 · Inference Monitoring & SLA")
    print("Doc: document/chapter_12_inference_monitoring_sla.md\n")

    demo_single_request_metrics()
    demo_percentiles()
    demo_slo_check()
    demo_capacity()
    demo_cost()

    print("\n" + "=" * 60)
    print("All demos completed.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise
