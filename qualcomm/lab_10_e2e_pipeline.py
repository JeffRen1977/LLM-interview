#!/usr/bin/env python3
"""
Lab 10 · End-to-end mini pipeline (export → transform → quant → partition → budget)

Skills: tie Labs 01–09 into one story you can tell in interviews
Maps to full prep doc §1 pipeline diagram

Run:
    python3 qualcomm/lab_10_e2e_pipeline.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.utils import max_abs_diff, print_header, timed

try:
    import onnx
    from onnx import checker
    import onnxruntime as ort

    STACK_OK = True
except ImportError:
    STACK_OK = False

from lab_02_graph_transform import constant_fold_identity, count_node_types, fuse_matmul_add_relu
from lab_03_quantization_ptq import symmetric_quantize_per_channel, dequantize
from lab_05_graph_partition_delegate import partition_graph


class DeployModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


@dataclass
class PipelineReport:
    onnx_nodes_before: int
    onnx_nodes_after: int
    htp_subgraphs: int
    cpu_subgraphs: int
    weight_mb_int8: float
    max_diff_pt_ort: float


def run_pipeline() -> PipelineReport:
    torch.manual_seed(0)
    model = DeployModel().eval()
    x = torch.randn(2, 32)

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = os.path.join(tmp, "raw.onnx")
        torch.onnx.export(model, x, raw_path, input_names=["x"], output_names=["y"], opset_version=17)
        m = onnx.load(raw_path)
        checker.check_model(m)
        before = sum(count_node_types(m).values())

        # graph transform (conceptual passes)
        m = constant_fold_identity(m)
        m = fuse_matmul_add_relu(m)
        after = sum(count_node_types(m).values())

        opt_path = os.path.join(tmp, "opt.onnx")
        onnx.save(m, opt_path)

        # ORT run
        sess = ort.InferenceSession(opt_path, providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"x": x.numpy()})[0]
        with torch.no_grad():
            pt_out = model(x).numpy()
        diff = max_abs_diff(pt_out, ort_out)

        # partition
        subs = partition_graph(m)
        htp = sum(1 for s in subs if s.backend == "HTP")
        cpu = sum(1 for s in subs if s.backend == "CPU")

        # quant footprint
        w = model.fc1.weight.detach().numpy()
        q, _ = symmetric_quantize_per_channel(w)
        int8_mb = q.nbytes / (1024 ** 2)

        return PipelineReport(
            onnx_nodes_before=before,
            onnx_nodes_after=after,
            htp_subgraphs=htp,
            cpu_subgraphs=cpu,
            weight_mb_int8=int8_mb,
            max_diff_pt_ort=diff,
        )


def print_story(r: PipelineReport) -> None:
    print_header("Interview narrative (30s)", "Lab 10")
    print("""
  I took a PyTorch model through the Qualcomm-style stack:
    1. torch.onnx.export → ONNX IR
    2. graph passes (fold/fuse) to cut nodes for HTP
    3. ORT CPU golden vs PyTorch (max diff shown below)
    4. partition: HTP subgraphs + CPU fallback
    5. INT8 per-channel weight sizing for deployment budget
  On device: QNN compile → context binary → Genie/ORT QNN EP.
""")


def main() -> None:
    print_header("End-to-End Deploy Pipeline", "Lab 10")
    if not STACK_OK:
        print("  Install: pip install onnx onnxruntime torch")
        sys.exit(1)

    with timed("full pipeline"):
        report = run_pipeline()

    print(f"  ONNX nodes: {report.onnx_nodes_before} → {report.onnx_nodes_after}")
    print(f"  Partition: HTP subgraphs={report.htp_subgraphs}, CPU={report.cpu_subgraphs}")
    print(f"  fc1 INT8 weight size: {report.weight_mb_int8:.3f} MB")
    print(f"  PyTorch vs ORT max diff: {report.max_diff_pt_ort:.2e}")
    print_story(report)
    print("Lab 10 completed.")


if __name__ == "__main__":
    main()
