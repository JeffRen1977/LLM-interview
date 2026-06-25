#!/usr/bin/env python3
"""
Lab 08 · LoRA merge for deployment

Skills: W' = W + BA, merge vs sidecar adapter, inference path
Maps to prep §3.5, chapter_13 §13.13

Run:
    python3 qualcomm/lab_08_lora_merge_deploy.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.utils import max_abs_diff, print_header, timed


class LinearWithLoRA(nn.Module):
    def __init__(self, in_f: int, out_f: int, rank: int = 4, alpha: float = 8.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_f, out_f, bias=False)
        self.lora_a = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(out_f, rank))
        self.scaling = alpha / rank

    def forward_adapter(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        delta = (x @ self.lora_a.T @ self.lora_b.T) * self.scaling
        return base + delta

    def merge_weights(self) -> torch.Tensor:
        delta_w = (self.lora_b @ self.lora_a) * self.scaling
        return self.linear.weight.data + delta_w


def demo_merge_equivalence() -> None:
    print_header("LoRA merge numerical equivalence", "Lab 08")
    torch.manual_seed(0)
    layer = LinearWithLoRA(64, 128, rank=8).eval()
    x = torch.randn(4, 64)

    with torch.no_grad():
        out_adapter = layer.forward_adapter(x).numpy()
        merged_w = layer.merge_weights()
        out_merged = (x @ merged_w.T).numpy()

    diff = max_abs_diff(out_adapter, out_merged)
    print(f"  adapter path vs merged weight max diff: {diff:.2e}")
    print("  Deploy: merge into base weights for single MatMul on NPU")


def demo_size_tradeoff() -> None:
    print_header("Merge vs sidecar (deployment trade-off)", "Lab 08")
    in_f, out_f, rank = 4096, 4096, 16
    base = in_f * out_f
    lora = rank * (in_f + out_f)
    print(f"  Base Linear params: {base:,}")
    print(f"  LoRA params (r={rank}): {lora:,}  ({100*lora/base:.2f}% of base)")
    print("  Sidecar: ship base INT4 + small adapter FP16 (multi-LoRA apps)")
    print("  Merge:   one graph, best HTP performance, per-task binary")


def demo_export_note() -> None:
    print_header("ONNX export after merge", "Lab 08")
    print("  1. merge_weights() in PyTorch")
    print("  2. torch.onnx.export merged model")
    print("  3. QNN compile — graph sees single MatMul per layer")


def main() -> None:
    print_header("LoRA Merge for Edge Deployment", "Lab 08")
    demo_merge_equivalence()
    demo_size_tradeoff()
    demo_export_note()
    print("\nLab 08 completed.")


if __name__ == "__main__":
    main()
