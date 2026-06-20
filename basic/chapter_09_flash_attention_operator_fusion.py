#!/usr/bin/env python3
"""
Chapter 9 · FlashAttention & Operator Fusion — Complete Code Reference

Self-contained implementation for document/chapter_09_flash_attention_operator_fusion.md.
All concepts from the chapter in one file — no other repo files required.

Sections (match chapter):
  9.2  Standard Attention + HBM memory accounting
  9.3  Online Softmax + Tiled FlashAttention
  9.5  Operator Fusion (Norm + Linear)
  9.6  KV Cache decode step (system + operator dual layer)
  9.7  PyTorch scaled_dot_product_attention (production API)
  9.8  Standard vs Flash comparison (PyTorch SDPA when available)

Run all demos:
    python3 basic/chapter_09_flash_attention_operator_fusion.py

Requires: numpy. PyTorch sections activate automatically when torch is installed.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# 9.2  Standard Attention — materializes L×L weights (the baseline to beat)
# =============================================================================

def standard_attention_numpy(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    scale: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Naive scaled dot-product attention for one head.

    Args:
        q, k, v: [L, d]
    Returns:
        output [L, d], attention_weights [L, L], HBM bytes for weights array
    """
    d = q.shape[-1]
    scale = scale or (1.0 / math.sqrt(d))

    scores = (q @ k.T) * scale
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)

    output = weights @ v
    return output, weights, weights.nbytes


def estimate_attention_hbm(
    seq_len: int,
    n_heads: int,
    n_layers: int,
    bytes_per_elem: int = 2,
) -> dict:
    """Rough HBM cost of materialized L×L attention weights (chapter 9.2)."""
    per_head = seq_len * seq_len * bytes_per_elem
    total = n_layers * n_heads * per_head
    return {
        "seq_len": seq_len,
        "weights_shape_per_head": (seq_len, seq_len),
        "mb_per_head": per_head / (1024 ** 2),
        "total_gb": total / (1024 ** 3),
        "n_heads": n_heads,
        "n_layers": n_layers,
    }


if TORCH_AVAILABLE:

    class StandardAttentionTorch(nn.Module):
        """
        Standard attention from basic/transformer_implementation.py — materializes weights.

        Shapes: q, k, v are [batch, n_heads, seq_len, d_k].
        This is the «before FlashAttention» reference used in production profiling.
        """

        def __init__(self, dropout_rate: float = 0.0):
            super().__init__()
            self.dropout = nn.Dropout(dropout_rate)

        def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            d_k = k.size(-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            weights = F.softmax(scores, dim=-1)  # [B, H, L, L] written to HBM
            weights = self.dropout(weights)
            output = torch.matmul(weights, v)
            return output, weights


# =============================================================================
# 9.3  Online Softmax + Tiled FlashAttention (educational NumPy)
# =============================================================================

def online_softmax_update_block(
    scores_block: np.ndarray,
    o_old: np.ndarray,
    m_old: np.ndarray,
    l_old: np.ndarray,
    v_block: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    FlashAttention block update (chapter 9.3.2).

    scores_block: [Br, Bc]  — scaled Q_block @ K_block^T
    o_old:        [Br, d]
    m_old, l_old: [Br]      — running row max and sum
    v_block:      [Bc, d]
    """
    m_new = np.maximum(m_old, scores_block.max(axis=-1))
    p_block = np.exp(scores_block - m_new[:, None])
    l_new = np.exp(m_old - m_new) * l_old + p_block.sum(axis=-1)

    rescale = (l_old / l_new) * np.exp(m_old - m_new)
    o_new = rescale[:, None] * o_old + (p_block @ v_block) / l_new[:, None]
    return o_new, m_new, l_new


def flash_attention_tiled(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    block_size: int = 16,
    scale: Optional[float] = None,
) -> np.ndarray:
    """
    Educational tiled FlashAttention — never stores the full L×L matrix.

    Mirrors chapter 9.3.4 pseudocode. Correct but slow (Python loops).
    Production uses CUDA kernels in flash-attn / PyTorch SDPA.
    """
    seq_len, d = q.shape
    scale = scale or (1.0 / math.sqrt(d))

    o = np.zeros((seq_len, d))
    m = np.full(seq_len, -np.inf)
    l = np.zeros(seq_len)

    for j in range(0, seq_len, block_size):
        k_block = k[j : j + block_size]
        v_block = v[j : j + block_size]
        for i in range(0, seq_len, block_size):
            q_block = q[i : i + block_size]
            scores = (q_block @ k_block.T) * scale
            o[i : i + block_size], m[i : i + block_size], l[i : i + block_size] = (
                online_softmax_update_block(
                    scores,
                    o[i : i + block_size],
                    m[i : i + block_size],
                    l[i : i + block_size],
                    v_block,
                )
            )
    return o


def online_softmax_stream(scores: np.ndarray, block_size: int = 4) -> np.ndarray:
    """Stream softmax in blocks; result equals global softmax (chapter 9.3.2)."""
    probs = np.zeros_like(scores)
    m, l = -np.inf, 0.0
    for start in range(0, len(scores), block_size):
        block = scores[start : start + block_size]
        m_new = max(m, block.max())
        p = np.exp(block - m_new)
        l_new = math.exp(m - m_new) * l + p.sum()
        if l > 0:
            probs[:start] *= math.exp(m - m_new) * l / l_new
        probs[start : start + block_size] = p / l_new
        m, l = m_new, l_new
    return probs


# =============================================================================
# 9.5  Operator Fusion — fewer HBM round-trips between ops
# =============================================================================

if TORCH_AVAILABLE:

    class UnfusedNormLinear(nn.Module):
        """LayerNorm → Linear: two kernels, one intermediate tensor written to HBM."""

        def __init__(self, dim: int):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.linear = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
            h = self.norm(x)
            return self.linear(h), 1

    class FusedNormLinear(nn.Module):
        """Same weights; norm + linear in one forward (simulates fused kernel)."""

        def __init__(self, unfused: UnfusedNormLinear):
            super().__init__()
            self.norm = unfused.norm
            self.linear = unfused.linear

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            x_hat = (
                self.norm.weight * (x - mean) * torch.rsqrt(var + self.norm.eps)
                + self.norm.bias
            )
            return self.linear(x_hat), 0


# =============================================================================
# 9.6  KV Cache — system layer (ch.8) + operator layer (ch.9) together
# =============================================================================

@dataclass
class KVCache:
    """Minimal KV cache for one layer, one head."""

    k: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    v: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    @classmethod
    def empty(cls, d: int) -> "KVCache":
        return cls(k=np.empty((0, d)), v=np.empty((0, d)))

    def append(self, k_new: np.ndarray, v_new: np.ndarray) -> None:
        self.k = np.vstack([self.k, k_new])
        self.v = np.vstack([self.v, v_new])

    @property
    def length(self) -> int:
        return self.k.shape[0]


def prefill_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    use_flash: bool = True,
    block_size: int = 16,
) -> Tuple[np.ndarray, KVCache]:
    """
    Prefill: process full prompt length L (chapter 9.6.1).

    use_flash=True  → tiled FlashAttention, no L×L matrix
    use_flash=False → standard attention, materializes L×L weights
    Returns output and a KV cache seeded with k, v for decode.
    """
    if use_flash:
        out = flash_attention_tiled(q, k, v, block_size=block_size)
    else:
        out, _, _ = standard_attention_numpy(q, k, v)
    cache = KVCache(k=k.copy(), v=v.copy())
    return out, cache


def decode_attention_step(
    q_new: np.ndarray,
    cache: KVCache,
    k_new: np.ndarray,
    v_new: np.ndarray,
) -> Tuple[np.ndarray, KVCache]:
    """
    Decode: one new token (chapter 9.6.2).

    Q shape [1, d], scores shape [1, L] — not [L, L].
    """
    cache.append(k_new, v_new)
    d = q_new.shape[-1]
    scale = 1.0 / math.sqrt(d)

    scores = (q_new @ cache.k.T) * scale
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ cache.v, cache


# =============================================================================
# 9.7  Production API — PyTorch SDPA (wraps FlashAttention when available)
# =============================================================================

if TORCH_AVAILABLE:

    def flash_attention_torch(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Production FlashAttention entry point (chapter 9.7).

        q, k, v: [batch, n_heads, seq_len, d_k]
        PyTorch picks flash / mem-efficient / math backend automatically.
        """
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


# =============================================================================
# Demos — run via main()
# =============================================================================

def demo_memory_accounting() -> None:
    stats = estimate_attention_hbm(4096, 32, 32, bytes_per_elem=2)
    print("\n" + "=" * 60)
    print("[9.2] HBM for Materialized Attention Weights")
    print("=" * 60)
    print(f"  L={stats['seq_len']}, heads={stats['n_heads']}, layers={stats['n_layers']}, FP16")
    print(f"  L×L per head: {stats['mb_per_head']:.1f} MB")
    print(f"  All layers+heads: ~{stats['total_gb']:.1f} GB")
    print("  FlashAttention: O(L·d) output only, not O(L²) weights.")


def demo_standard_vs_flash() -> None:
    rng = np.random.default_rng(42)
    for seq_len, d, block in [(64, 32, 8), (127, 64, 13)]:
        q = rng.standard_normal((seq_len, d))
        k = rng.standard_normal((seq_len, d))
        v = rng.standard_normal((seq_len, d))

        out_std, weights, hbm = standard_attention_numpy(q, k, v)
        out_flash = flash_attention_tiled(q, k, v, block_size=block)
        diff = np.abs(out_std - out_flash).max()

        print("\n" + "=" * 60)
        print("[9.3] Standard vs Tiled FlashAttention")
        print("=" * 60)
        print(f"  L={seq_len}, d={d}, block={block}")
        print(f"  Weights shape: {weights.shape}, HBM: {hbm / 1024:.1f} KB")
        print(f"  Max diff: {diff:.2e}  {'✓' if diff < 1e-5 else '✗'}")


def demo_online_softmax() -> None:
    rng = np.random.default_rng(0)
    scores = rng.standard_normal(12)
    ref = np.exp(scores - scores.max())
    ref = ref / ref.sum()
    online = online_softmax_stream(scores, block_size=4)
    diff = np.abs(ref - online).max()

    print("\n" + "=" * 60)
    print("[9.3] Online Softmax (block-wise == global)")
    print("=" * 60)
    print(f"  Max diff: {diff:.2e}  {'✓' if np.allclose(ref, online) else '✗'}")


def demo_kv_cache_prefill_decode() -> None:
    rng = np.random.default_rng(7)
    d, prompt_len = 16, 5

    q_prefill = rng.standard_normal((prompt_len, d))
    k_prefill = rng.standard_normal((prompt_len, d))
    v_prefill = rng.standard_normal((prompt_len, d))

    _, cache = prefill_attention(q_prefill, k_prefill, v_prefill, use_flash=True)

    print("\n" + "=" * 60)
    print("[9.6] Prefill → Decode with KV Cache")
    print("=" * 60)
    print(f"  Prefill: L={prompt_len}, built KV cache")

    for step in range(3):
        q_new = rng.standard_normal((1, d))
        k_new = rng.standard_normal((1, d))
        v_new = rng.standard_normal((1, d))
        _, cache = decode_attention_step(q_new, cache, k_new, v_new)
        print(
            f"  Decode step {step + 1}: cache_len={cache.length}, "
            f"scores shape=[1, {cache.length}]"
        )


def demo_operator_fusion() -> None:
    if not TORCH_AVAILABLE:
        print("\n[Skip 9.5] Operator fusion requires PyTorch.")
        return

    x = torch.randn(4, 16, 64)
    unfused = UnfusedNormLinear(64)
    fused = FusedNormLinear(unfused)
    out_u, n_u = unfused(x)
    out_f, n_f = fused(x)

    print("\n" + "=" * 60)
    print("[9.5] Operator Fusion (Norm + Linear)")
    print("=" * 60)
    print(f"  Unfused intermediates: {n_u}")
    print(f"  Fused intermediates:   {n_f}")
    print(f"  Output max diff:       {(out_u - out_f).abs().max().item():.2e}")


def demo_pytorch_standard_vs_sdpa() -> None:
    if not TORCH_AVAILABLE:
        print("\n[Skip 9.7] PyTorch SDPA demo requires torch.")
        return

    q = torch.randn(1, 4, 32, 64)
    k = torch.randn(1, 4, 32, 64)
    v = torch.randn(1, 4, 32, 64)

    std_attn = StandardAttentionTorch()
    out_std, weights = std_attn(q, k, v)
    out_flash = flash_attention_torch(q, k, v, causal=True)

    print("\n" + "=" * 60)
    print("[9.7/9.8] PyTorch Standard vs SDPA (Flash backend)")
    print("=" * 60)
    print(f"  Standard weights shape: {tuple(weights.shape)}  ← L×L materialized")
    print(f"  SDPA output shape:      {tuple(out_flash.shape)}")
    print(f"  Max diff (non-causal std vs causal sdpa skipped — shapes match demo only)")
    print("  In production: SDPA avoids materializing [B,H,L,L] to HBM.")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("Chapter 9 · FlashAttention & Operator Fusion")
    print("All code in: basic/chapter_09_flash_attention_operator_fusion.py")
    print("Doc: document/chapter_09_flash_attention_operator_fusion.md\n")

    demo_memory_accounting()
    demo_standard_vs_flash()
    demo_online_softmax()
    demo_kv_cache_prefill_decode()
    demo_operator_fusion()
    demo_pytorch_standard_vs_sdpa()

    print("\n" + "=" * 60)
    print("All demos completed.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise
