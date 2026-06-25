#!/usr/bin/env python3
"""
Lab 09 · Cross-layer debug pipeline (PyTorch → ONNX → quant golden)

Skills: layer-wise max diff, isolate failing op, Staff war-story methodology
Maps to prep §3.9, §10.7

Run:
    python3 qualcomm/lab_09_debug_golden_pipeline.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.utils import max_abs_diff, print_header

try:
    import onnxruntime as ort

    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

from lab_03_quantization_ptq import symmetric_quantize_per_channel, dequantize


class TwoLayerMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def export(path: str, model: nn.Module) -> None:
    x = torch.randn(1, 32)
    torch.onnx.export(model, x, path, input_names=["x"], output_names=["y"], opset_version=17)


def layerwise_hooks(model: nn.Module, x: torch.Tensor) -> List[Tuple[str, np.ndarray]]:
    activations: List[Tuple[str, np.ndarray]] = []

    def make_hook(name: str) -> Callable:
        def hook(_mod, _inp, out):
            activations.append((name, out.detach().numpy()))

        return hook

    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return activations


def inject_quant_error(w: np.ndarray) -> np.ndarray:
    q, p = symmetric_quantize_per_channel(w.astype(np.float32))
    return dequantize(q, p)


def debug_pipeline() -> None:
    print_header("Golden compare workflow", "Lab 09")
    torch.manual_seed(0)
    model = TwoLayerMLP().eval()
    x = torch.randn(1, 32)

    # Step 1: PyTorch golden
    with torch.no_grad():
        pt_out = model(x).numpy()
    print("  [1] PyTorch golden captured")

    # Step 2: Simulate quant error on fc1 only
    w_orig = model.fc1.weight.detach().numpy().copy()
    model.fc1.weight.data = torch.from_numpy(inject_quant_error(w_orig))
    with torch.no_grad():
        pt_quant = model(x).numpy()
    diff_quant = max_abs_diff(pt_out, pt_quant)
    print(f"  [2] After INT8-ish fc1 weight quant: max diff = {diff_quant:.4f}")

    # Step 3: Layer hooks localize
    acts = layerwise_hooks(model, x)
    for name, arr in acts:
        print(f"  [3] layer {name} output mean={arr.mean():.4f} std={arr.std():.4f}")

    # Step 4: ONNX path
    if ORT_AVAILABLE:
        model.fc1.weight.data = torch.from_numpy(w_orig)  # reset for fair export
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mlp.onnx")
            export(path, model)
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            ort_out = sess.run(None, {"x": x.numpy()})[0]
            diff_onnx = max_abs_diff(pt_out, ort_out)
            print(f"  [4] PyTorch vs ONNX ORT: max diff = {diff_onnx:.2e}")

    print("\n  Debug order (prep §3.9):")
    print("    reproduce → shrink graph → layer golden → quant on/off → layout/scale")


def main() -> None:
    print_header("Cross-Layer Debug Pipeline", "Lab 09")
    debug_pipeline()
    print("\nLab 09 completed.")


if __name__ == "__main__":
    main()
