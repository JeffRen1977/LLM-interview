#!/usr/bin/env python3
"""
Lab 01 · PyTorch → ONNX export & numerical validation

Skills: torch.onnx.export, opset, onnx checker, ORT vs PyTorch golden compare
Maps to prep §3.4, §3.6, §10.5

Run:
    python3 qualcomm/lab_01_onnx_export.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn

# Allow running as script from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.utils import max_abs_diff, print_header, timed

try:
    import onnx
    from onnx import checker
    import onnxruntime as ort

    ONNX_STACK = True
except ImportError:
    ONNX_STACK = False


class MiniTransformerBlock(nn.Module):
    """Single block: LayerNorm → Linear(QKV) → attention stub → FFN."""

    def __init__(self, dim: int = 64, n_heads: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(b, seq, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-2, -1) / (self.head_dim ** 0.5), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, seq, d)
        x = x + self.proj(out)
        x = x + self.ffn(self.norm2(x))
        return x


def export_model(path: str, opset: int = 17) -> tuple[nn.Module, torch.Tensor]:
    model = MiniTransformerBlock(dim=64, n_heads=4).eval()
    dummy = torch.randn(1, 8, 64)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamic_axes={"input": {0: "batch", 1: "seq"}, "output": {0: "batch", 1: "seq"}},
    )
    return model, dummy


def validate_onnx(path: str) -> None:
    model_proto = onnx.load(path)
    checker.check_model(model_proto)
    print(f"  ONNX checker: OK ({len(model_proto.graph.node)} nodes)")
    print(f"  IR version: {model_proto.ir_version}, opset: {model_proto.opset_import[0].version}")


def compare_pytorch_ort(pytorch_model: nn.Module, onnx_path: str, inputs: torch.Tensor) -> None:
    with torch.no_grad():
        pt_out = pytorch_model(inputs).numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": inputs.numpy()})[0]

    diff = max_abs_diff(pt_out, ort_out)
    print(f"  PyTorch vs ORT max abs diff: {diff:.2e}")
    if diff > 1e-4:
        print("  WARNING: diff high — check export opset or dynamic axes")
    else:
        print("  Numerical match: PASS")


def demo_export_pitfalls() -> None:
    print_header("Common export pitfalls (interview talking points)", "Lab 01")
    tips = [
        "Control flow (if/for on tensor data) → use torch.export or rewrite",
        "Unsupported ops → replace with decomposed ops or register custom op",
        "Dynamic axes → declare batch/seq in dynamic_axes for LLM",
        "dtype mismatch → ensure input/output types match runtime",
        "eval() + no_grad() before export",
    ]
    for t in tips:
        print(f"  · {t}")


def main() -> None:
    print_header("PyTorch → ONNX Export & Validate", "Lab 01")
    if not ONNX_STACK:
        print("  Install: pip install onnx onnxruntime")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = os.path.join(tmp, "block.onnx")
        with timed("torch.onnx.export"):
            pt_model, dummy = export_model(onnx_path)
        validate_onnx(onnx_path)
        compare_pytorch_ort(pt_model, onnx_path, dummy)

    demo_export_pitfalls()
    print("\nLab 01 completed.")


if __name__ == "__main__":
    main()
