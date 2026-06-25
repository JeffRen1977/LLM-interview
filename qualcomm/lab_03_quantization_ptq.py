#!/usr/bin/env python3
"""
Lab 03 · PTQ quantization (symmetric, per-tensor vs per-channel, calibration)

Skills: dynamic vs static scale, QDQ format, precision recovery knobs
Maps to prep §3.1 Q1–Q2, §10

Run:
    python3 qualcomm/lab_03_quantization_ptq.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.utils import max_abs_diff, print_header


@dataclass
class QuantParams:
    scale: np.ndarray
    zero_point: int = 0
    symmetric: bool = True


def symmetric_quantize_per_tensor(x: np.ndarray, bits: int = 8) -> Tuple[np.ndarray, QuantParams]:
    qmax = 2 ** (bits - 1) - 1
    scale = np.array([np.max(np.abs(x)) / qmax + 1e-8], dtype=np.float32)
    q = np.clip(np.round(x / scale), -qmax - 1, qmax).astype(np.int8)
    return q, QuantParams(scale=scale)


def symmetric_quantize_per_channel(w: np.ndarray, bits: int = 8) -> Tuple[np.ndarray, QuantParams]:
    """Weight matrix [out_features, in_features] — scale per output channel."""
    qmax = 2 ** (bits - 1) - 1
    out_ch = w.shape[0]
    scale = np.max(np.abs(w), axis=1) / qmax + 1e-8
    q = np.clip(np.round(w / scale[:, None]), -qmax - 1, qmax).astype(np.int8)
    return q, QuantParams(scale=scale.astype(np.float32))


def dequantize(q: np.ndarray, params: QuantParams) -> np.ndarray:
    if params.scale.size == 1:
        return q.astype(np.float32) * params.scale[0]
    return q.astype(np.float32) * params.scale[:, None]


def calibrate_static_scale(activations: List[np.ndarray], bits: int = 8) -> QuantParams:
    """Static PTQ: collect max abs over calibration batches."""
    qmax = 2 ** (bits - 1) - 1
    global_max = max(float(np.max(np.abs(a))) for a in activations)
    return QuantParams(scale=np.array([global_max / qmax + 1e-8], dtype=np.float32))


def dynamic_quantize_activation(x: np.ndarray, bits: int = 8) -> Tuple[np.ndarray, QuantParams]:
    """Dynamic: scale computed at runtime per tensor (no calibration set)."""
    return symmetric_quantize_per_tensor(x, bits)


def matmul_int8_sim(x_fp: np.ndarray, w_fp: np.ndarray) -> np.ndarray:
    """W8A16-style: quantize weight per-channel, activation stays FP16 path."""
    w_q, w_p = symmetric_quantize_per_channel(w_fp)
    w_dq = dequantize(w_q, w_p)
    return x_fp @ w_dq.T


def demo_per_tensor_vs_per_channel() -> None:
    print_header("per-tensor vs per-channel weight quant", "Lab 03")
    rng = np.random.default_rng(0)
    w = rng.standard_normal((128, 64)).astype(np.float32)
    w[0] *= 10.0  # one outlier channel

    q_pt, p_pt = symmetric_quantize_per_tensor(w)
    q_pc, p_pc = symmetric_quantize_per_channel(w)

    err_pt = max_abs_diff(w, dequantize(q_pt, p_pt))
    err_pc = max_abs_diff(w, dequantize(q_pc, p_pc))
    print(f"  per-tensor max error:   {err_pt:.4f}")
    print(f"  per-channel max error:  {err_pc:.4f}")
    print("  → outlier channel: per-channel wins (NPU weight quant default)")


def demo_dynamic_vs_static_activation() -> None:
    print_header("dynamic vs static activation quant", "Lab 03")
    rng = np.random.default_rng(1)
    calib = [rng.standard_normal((4, 64)).astype(np.float32) * s for s in (0.5, 1.0, 2.0, 5.0)]
    static_p = calibrate_static_scale(calib)

    test = rng.standard_normal((4, 64)).astype(np.float32) * 8.0  # OOD scale
    _, dyn_p = dynamic_quantize_activation(test)
    q_static = np.clip(np.round(test / static_p.scale[0]), -127, 127)
    q_dynamic = np.clip(np.round(test / dyn_p.scale[0]), -127, 127)

    err_static = max_abs_diff(test, q_static.astype(np.float32) * static_p.scale[0])
    err_dynamic = max_abs_diff(test, q_dynamic.astype(np.float32) * dyn_p.scale[0])
    print(f"  OOD activation — static calib error:  {err_static:.4f}")
    print(f"  OOD activation — dynamic scale error: {err_dynamic:.4f}")
    print("  → static needs good calib; dynamic adapts but harder on NPU")


def demo_qdq_format() -> None:
    print_header("QDQ node format (ONNX Runtime / QNN)", "Lab 03")
    x = np.array([0.1, -0.3, 0.8], dtype=np.float32)
    q, p = symmetric_quantize_per_tensor(x)
    print("  FP32:", x)
    print("  QuantizeLinear → INT8:", q, "scale=", p.scale[0])
    print("  DequantizeLinear → FP32:", dequantize(q, p))
    print("  Graph: x → Q → INT8 MatMul → DQ → y  (scale embedded for HTP)")


def demo_llm_weight_only() -> None:
    print_header("LLM W4A16 footprint (GPTQ/AWQ style)", "Lab 03")
    params_b = 7.0
    for label, bpp in [("FP16", 2), ("INT8", 1), ("INT4", 0.5)]:
        gb = params_b * bpp
        print(f"  7B × {label}: ~{gb:.1f} GB weights")


def main() -> None:
    print_header("PTQ Quantization Hands-on", "Lab 03")
    demo_per_tensor_vs_per_channel()
    demo_dynamic_vs_static_activation()
    demo_qdq_format()
    demo_llm_weight_only()
    print("\nLab 03 completed.")


if __name__ == "__main__":
    main()
