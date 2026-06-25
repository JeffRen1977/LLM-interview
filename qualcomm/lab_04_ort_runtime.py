#!/usr/bin/env python3
"""
Lab 04 · ONNX Runtime execution & Execution Provider selection

Skills: InferenceSession, provider priority, QNN EP fallback pattern
Maps to prep §3.4, §10.5 — QNN EP requires Snapdragon SDK; we simulate provider chain

Run:
    python3 qualcomm/lab_04_ort_runtime.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.utils import max_abs_diff, print_header, timed

try:
    import onnxruntime as ort

    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def export_mlp(path: str) -> None:
    m = TinyMLP().eval()
    x = torch.randn(2, 32)
    torch.onnx.export(m, x, path, input_names=["x"], output_names=["y"], opset_version=17)


def list_providers() -> List[str]:
    return ort.get_available_providers()


def run_with_providers(onnx_path: str, providers: List[str], x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(onnx_path, providers=providers)
    return sess.run(None, {"x": x})[0]


class MockProviderChain:
    """
    Simulates ORT provider fallback when QNN EP unavailable on dev machine.

    Real deployment:
        providers=["QNNExecutionProvider", "CPUExecutionProvider"]
    """

    def __init__(self, available: List[str]) -> None:
        self.available = available

    def resolve(self, requested: List[str]) -> Optional[str]:
        for p in requested:
            if p in self.available:
                return p
        return None


def demo_provider_chain() -> None:
    print_header("Execution Provider chain (QNN → CPU fallback)", "Lab 04")

    real = list_providers()
    print(f"  Machine ORT providers: {real}")

    mock_available = ["CPUExecutionProvider"]  # dev laptop
    chain = MockProviderChain(mock_available)
    requested = ["QNNExecutionProvider", "CPUExecutionProvider"]
    chosen = chain.resolve(requested)
    print(f"  Requested: {requested}")
    print(f"  Selected:  {chosen}")
    print("  On Snapdragon device + QAIRT SDK, QNNExecutionProvider appears in list")


def demo_session_run(onnx_path: str) -> None:
    print_header("ORT InferenceSession run", "Lab 04")
    with timed("sess.run CPU"):
        x = np.random.randn(2, 32).astype(np.float32)
        out1 = run_with_providers(onnx_path, ["CPUExecutionProvider"], x)
    out2 = run_with_providers(onnx_path, ["CPUExecutionProvider"], x)
    diff = max_abs_diff(out1, out2)
    print(f"  Deterministic re-run max diff (should be 0): {diff:.2e}")


def demo_session_options() -> None:
    print_header("ORT SessionOptions (production knobs)", "Lab 04")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    print(f"  graph_optimization_level: ORT_ENABLE_ALL")
    print(f"  intra_op_num_threads: {opts.intra_op_num_threads}")
    print("  Interview: ORT graph opt runs BEFORE EP partition to QNN")


def main() -> None:
    print_header("ONNX Runtime & EP Selection", "Lab 04")
    if not ORT_AVAILABLE:
        print("  Install: pip install onnxruntime")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "mlp.onnx")
        export_mlp(path)
        demo_provider_chain()
        demo_session_run(path)
    demo_session_options()
    print("\nLab 04 completed.")


if __name__ == "__main__":
    main()
