#!/usr/bin/env python3
"""
Lab 05 · Graph partition & delegate (HTP vs CPU)

Skills: op support matrix, subgraph offload, fallback — core of QNN EP / ExecuTorch delegate
Maps to prep §3.4, §10.5

Run:
    python3 qualcomm/lab_05_graph_partition_delegate.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.utils import print_header

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import numpy as np

    ONNX_STACK = True
except ImportError:
    ONNX_STACK = False


# Simulated HTP supported ops (subset of QNN HTP backend)
HTP_SUPPORTED: Set[str] = {
    "MatMul",
    "Gemm",
    "Conv",
    "Relu",
    "Add",
    "Mul",
    "Softmax",
    "FusedMatMulAddRelu",
    "QuantizeLinear",
    "DequantizeLinear",
}

CPU_ONLY: Set[str] = {"LayerNormalization", "Gather", "Cast", "Reshape", "Transpose"}


@dataclass
class Subgraph:
    name: str
    backend: str
    nodes: List[str] = field(default_factory=list)


def partition_graph(model: onnx.ModelProto) -> List[Subgraph]:
    """
    Greedy partition: consecutive HTP-supported ops → HTP subgraph;
    unsupported op → CPU subgraph (delegate fallback).
    """
    subgraphs: List[Subgraph] = []
    current: Subgraph | None = None

    for node in model.graph.node:
        backend = "HTP" if node.op_type in HTP_SUPPORTED else "CPU"
        if current is None or current.backend != backend:
            current = Subgraph(name=f"sg_{len(subgraphs)}", backend=backend)
            subgraphs.append(current)
        current.nodes.append(f"{node.op_type}({node.name or node.output[0]})")

    return subgraphs


def build_mixed_graph() -> onnx.ModelProto:
    """Graph with both HTP-friendly and CPU-only ops."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 64])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 64])
    w = numpy_helper.from_array(np.random.randn(64, 64).astype(np.float32), "W")

    nodes = [
        helper.make_node("LayerNormalization", ["x"], ["n1"], name="ln1"),
        helper.make_node("MatMul", ["n1", "W"], ["mm1"], name="mm1"),
        helper.make_node("Relu", ["mm1"], ["r1"], name="relu1"),
        helper.make_node("Gather", ["r1"], ["g1"], name="gather1", axis=1),
        helper.make_node("MatMul", ["g1", "W"], ["mm2"], name="mm2"),
        helper.make_node("Add", ["mm2", "n1"], ["y"], name="add1"),
    ]
    graph = helper.make_graph(nodes, "mixed", [x], [y], initializer=[w])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def print_partition_plan(subgraphs: List[Subgraph]) -> None:
    for sg in subgraphs:
        print(f"\n  Subgraph {sg.name} → backend={sg.backend}")
        for n in sg.nodes:
            print(f"    · {n}")


def demo_interview_answers() -> None:
    print_header("Delegate / EP interview answers", "Lab 05")
    qa = [
        ("Whole-graph offload?", "Rare; usually partition into supported subgraphs"),
        ("Unsupported op?", "CPU fallback or graph surgery / custom op"),
        ("Dynamic shape?", "May block offline HTP compile; fix max_seq + padding"),
        ("ORT vs ExecuTorch?", "Same idea: partitioner marks delegate regions → QNN"),
    ]
    for q, a in qa:
        print(f"  Q: {q}")
        print(f"  A: {a}")


def main() -> None:
    print_header("Graph Partition & Delegate Simulation", "Lab 05")
    if not ONNX_STACK:
        print("  Install: pip install onnx numpy")
        sys.exit(1)

    model = build_mixed_graph()
    print(f"  Graph nodes: {len(model.graph.node)}")
    subgraphs = partition_graph(model)
    print_partition_plan(subgraphs)

    htp_ops = sum(len(s.nodes) for s in subgraphs if s.backend == "HTP")
    cpu_ops = sum(len(s.nodes) for s in subgraphs if s.backend == "CPU")
    print(f"\n  HTP ops: {htp_ops}, CPU fallback ops: {cpu_ops}")
    print("  Performance tip: minimize CPU↔HTP tensor copies between subgraphs")

    demo_interview_answers()
    print("\nLab 05 completed.")


if __name__ == "__main__":
    main()
