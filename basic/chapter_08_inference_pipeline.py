#!/usr/bin/env python3
"""
Chapter 8 · Inference Pipeline — Complete Code Reference

Self-contained demos for document/chapter_08_inference_pipeline.md.

Sections:
  8.2  Static vs Continuous Batching — decode steps, padding waste
  8.3  KV Cache — naive vs cached forward count, memory estimate
  8.3.4 PagedAttention block allocator (conceptual)
  8.4  Prefill vs Decode timeline
  8.6  TTFT + TPOT → E2E latency
  8.7  Continuous batching scheduler simulation
  8.10 Thought questions

Run:
    python3 basic/chapter_08_inference_pipeline.py

Requires: numpy only.
"""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Set

import numpy as np


# =============================================================================
# 8.2  Static vs Continuous Batching
# =============================================================================

def static_batch_decode_steps(output_lengths: List[int]) -> int:
    """Static batch must wait for longest request (chapter 8.2.2, 8.10 Q1)."""
    return max(output_lengths)


def continuous_batch_active_counts(
    output_lengths: List[int],
) -> List[int]:
    """
    Per decode step: how many requests still active (chapter 8.2.3).

    Returns list of length max(output_lengths); each entry = active count.
    """
    max_len = max(output_lengths)
    remaining = list(output_lengths)
    counts: List[int] = []
    for _ in range(max_len):
        counts.append(sum(1 for r in remaining if r > 0))
        remaining = [max(0, r - 1) for r in remaining]
    return counts


def static_batch_padding_tokens(lengths: List[int]) -> dict:
    """Padding waste in static batch (chapter 8.2.2)."""
    max_len = max(lengths)
    actual = sum(lengths)
    padded = max_len * len(lengths)
    return {
        "lengths": lengths,
        "max_len": max_len,
        "actual_tokens": actual,
        "padded_tokens": padded,
        "waste_tokens": padded - actual,
        "waste_percent": (1 - actual / padded) * 100 if padded else 0.0,
    }


# =============================================================================
# 8.3  KV Cache
# =============================================================================

def estimate_kv_cache_gb(
    n_layers: int,
    seq_len: int,
    d_model: int,
    bytes_per_elem: int = 2,
    num_requests: int = 1,
) -> float:
    """Total_KV = 2 × N × L × d_model × bytes × num_requests (chapter 8.3.3)."""
    per_request = 2 * n_layers * seq_len * d_model * bytes_per_elem
    return per_request * num_requests / (1024 ** 3)


@dataclass
class LayerKVCache:
    """Per-layer K/V cache for one request (chapter 8.3)."""

    k: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    v: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    def append(self, k_new: np.ndarray, v_new: np.ndarray) -> None:
        self.k = np.vstack([self.k, k_new]) if self.k.size else k_new
        self.v = np.vstack([self.v, v_new]) if self.v.size else v_new

    @property
    def seq_len(self) -> int:
        return self.k.shape[0] if self.k.size else 0


def count_forward_passes_naive_vs_cache(
    prompt_len: int,
    num_decode_tokens: int,
) -> dict:
    """
    Count full-sequence forwards without vs with KV cache (chapter 8.3.1).

    Naive decode step t processes t-1 tokens → sum_{t=1}^{T} (prompt + t - 1)
    With cache: 1 prefill + T decode steps of 1 token each
    """
    total_naive = 0
    for t in range(1, num_decode_tokens + 1):
        total_naive += prompt_len + t - 1  # re-process all prior tokens

    with_cache = prompt_len + num_decode_tokens  # prefill + one token per step
    return {
        "prompt_len": prompt_len,
        "decode_tokens": num_decode_tokens,
        "naive_token_forwards": total_naive,
        "cached_token_forwards": with_cache,
        "reduction_factor": total_naive / with_cache if with_cache else 0,
    }


# =============================================================================
# 8.3.4  PagedAttention — block allocator (conceptual)
# =============================================================================

@dataclass
class PagedKVAllocator:
    """Fixed-size block pool for KV cache (chapter 8.3.4)."""

    block_size: int
    num_blocks: int
    free_blocks: Deque[int] = field(default_factory=deque)
    block_tables: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.free_blocks = deque(range(self.num_blocks))

    def allocate_request(self, request_id: str, num_tokens: int) -> List[int]:
        blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < blocks_needed:
            raise MemoryError("KV pool full")
        assigned = [self.free_blocks.popleft() for _ in range(blocks_needed)]
        self.block_tables[request_id] = assigned
        return assigned

    def release_request(self, request_id: str) -> None:
        for b in self.block_tables.pop(request_id, []):
            self.free_blocks.append(b)

    @property
    def utilization(self) -> float:
        used = self.num_blocks - len(self.free_blocks)
        return used / self.num_blocks if self.num_blocks else 0.0


# =============================================================================
# 8.4 / 8.6  Prefill, Decode, TTFT, TPOT, E2E
# =============================================================================

def estimate_e2e_latency_ms(
    ttft_ms: float,
    tpot_ms: float,
    output_tokens: int,
) -> float:
    """E2E ≈ TTFT + TPOT × (output_tokens - 1) (chapter 8.6)."""
    if output_tokens <= 1:
        return ttft_ms
    return ttft_ms + tpot_ms * (output_tokens - 1)


# =============================================================================
# 8.7  Continuous batching scheduler (simplified vLLM loop)
# =============================================================================

@dataclass
class Request:
    req_id: str
    prompt_len: int
    max_output: int
    generated: int = 0
    phase: str = "prefill"  # prefill | decode

    @property
    def is_done(self) -> bool:
        return self.generated >= self.max_output

    def step(self) -> None:
        if self.phase == "prefill":
            self.phase = "decode"
        else:
            self.generated += 1


class ContinuousBatchScheduler:
    """Minimal iteration-level scheduler (chapter 8.7 pseudocode)."""

    def __init__(self, max_batch: int = 8):
        self.max_batch = max_batch
        self.running: List[Request] = []
        self.wait_queue: Deque[Request] = deque()
        self.log: List[str] = []

    def add_request(self, req: Request) -> None:
        if len(self.running) < self.max_batch:
            self.running.append(req)
        else:
            self.wait_queue.append(req)

    def run_step(self) -> List[str]:
        """One decode iteration; returns active request ids."""
        batch = self.running[: self.max_batch]
        active_ids = []
        for req in batch:
            req.step()
            active_ids.append(req.req_id)

        finished = [r for r in self.running if r.is_done]
        for r in finished:
            self.running.remove(r)
            self.log.append(f"release {r.req_id}")

        while self.wait_queue and len(self.running) < self.max_batch:
            self.running.append(self.wait_queue.popleft())
            self.log.append(f"admit {self.running[-1].req_id}")

        return active_ids


# =============================================================================
# Demos
# =============================================================================

def demo_static_vs_continuous() -> None:
    lengths = [10, 50, 100, 20]
    static_steps = static_batch_decode_steps(lengths)
    counts = continuous_batch_active_counts(lengths)
    print("\n" + "=" * 60)
    print("[8.2 / 8.10] Static vs Continuous Batching")
    print("=" * 60)
    print(f"  Output lengths: {lengths}")
    print(f"  Static decode steps: {static_steps}  (wait for max=100)")
    print(f"  Continuous: {len(counts)} steps, active/request step:")
    print(f"    first 5: {counts[:5]}  ...  last 5: {counts[-5:]}")


def demo_padding_waste() -> None:
    pad = static_batch_padding_tokens([10, 50, 30])
    print("\n" + "=" * 60)
    print("[8.2] Static batch padding waste")
    print("=" * 60)
    print(f"  Lengths {pad['lengths']}, pad to {pad['max_len']}")
    print(f"  Actual tokens: {pad['actual_tokens']}, padded: {pad['padded_tokens']}")
    print(f"  Waste: {pad['waste_tokens']} tokens ({pad['waste_percent']:.1f}%)")


def demo_kv_cache_forwards() -> None:
    r = count_forward_passes_naive_vs_cache(prompt_len=100, num_decode_tokens=50)
    print("\n" + "=" * 60)
    print("[8.3] Naive vs KV Cache — token-forward count")
    print("=" * 60)
    print(f"  Prompt={r['prompt_len']}, decode tokens={r['decode_tokens']}")
    print(f"  Naive (no cache):  {r['naive_token_forwards']} token-forwards")
    print(f"  With KV cache:     {r['cached_token_forwards']} token-forwards")
    print(f"  Reduction:         ~{r['reduction_factor']:.1f}×")


def demo_kv_memory() -> None:
    gb = estimate_kv_cache_gb(32, 2048, 4096, bytes_per_elem=2, num_requests=1)
    print("\n" + "=" * 60)
    print("[8.3] KV Cache memory (LLaMA-7B, L=2048, FP16)")
    print("=" * 60)
    print(f"  Per request: ~{gb:.2f} GB")


def demo_thought_question_2() -> None:
    gpu_gb = 24
    weight_gb = 2
    kv_per_req_gb = 1
    available = gpu_gb - weight_gb
    max_req = int(available // kv_per_req_gb)
    print("\n" + "=" * 60)
    print("[8.10] Thought Q2: max concurrent requests")
    print("=" * 60)
    print(f"  GPU {gpu_gb} GB - weights {weight_gb} GB = {available} GB for KV")
    print(f"  ~{kv_per_req_gb} GB/request → ~{max_req} concurrent (theoretical)")


def demo_paged_attention() -> None:
    alloc = PagedKVAllocator(block_size=16, num_blocks=20)
    alloc.allocate_request("A", 50)
    alloc.allocate_request("B", 30)
    print("\n" + "=" * 60)
    print("[8.3.4] PagedAttention block allocator")
    print("=" * 60)
    print(f"  Request A (50 tok): blocks {alloc.block_tables['A']}")
    print(f"  Request B (30 tok): blocks {alloc.block_tables['B']}")
    print(f"  Pool utilization: {alloc.utilization:.0%}")
    alloc.release_request("A")
    alloc.allocate_request("C", 40)
    print(f"  After A done, C (40 tok): blocks {alloc.block_tables['C']}")
    print(f"  Pool utilization: {alloc.utilization:.0%}")


def demo_prefill_decode_e2e() -> None:
    ttft, tpot, n_out = 200.0, 30.0, 100
    e2e = estimate_e2e_latency_ms(ttft, tpot, n_out)
    print("\n" + "=" * 60)
    print("[8.4 / 8.6] Prefill + Decode → E2E")
    print("=" * 60)
    print(f"  TTFT={ttft}ms, TPOT={tpot}ms, output={n_out} tokens")
    print(f"  E2E ≈ {ttft} + {tpot}×{n_out - 1} = {e2e:.0f} ms")


def demo_continuous_scheduler() -> None:
    sched = ContinuousBatchScheduler(max_batch=3)
    for rid, out_len in [("A", 3), ("B", 5), ("C", 2), ("D", 4)]:
        sched.add_request(Request(rid, prompt_len=10, max_output=out_len))

    print("\n" + "=" * 60)
    print("[8.7] Continuous batching scheduler (max_batch=3)")
    print("=" * 60)
    for step in range(8):
        active = sched.run_step()
        print(f"  Step {step + 1}: batch={active}")
    print(f"  Events: {sched.log}")


def main() -> None:
    print("Chapter 8 · Inference Pipeline")
    print("Doc: document/chapter_08_inference_pipeline.md\n")

    demo_static_vs_continuous()
    demo_padding_waste()
    demo_kv_cache_forwards()
    demo_kv_memory()
    demo_paged_attention()
    demo_prefill_decode_e2e()
    demo_continuous_scheduler()
    demo_thought_question_2()

    print("\n" + "=" * 60)
    print("All demos completed.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise
