#!/usr/bin/env python3
"""
Lab 02 · ONNX graph transforms (compiler pass mindset)

Skills: constant folding, MatMul+Add+Relu fusion, dead code elimination
Maps to prep §3.4, §10.5 — same passes QNN converter / ORT optimizer apply

Run:
    python3 qualcomm/lab_02_graph_transform.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Set

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.utils import print_header

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    ONNX_STACK = True
except ImportError:
    ONNX_STACK = False


def build_unfused_graph() -> onnx.ModelProto:
    """MatMul → Add(bias) → Relu with constant bias initializer."""
    bias = numpy_helper.from_array(np.array([0.1, -0.2, 0.3], dtype=np.float32), name="bias")
    w = numpy_helper.from_array(np.random.randn(3, 3).astype(np.float32), name="W")

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 3])

    nodes = [
        helper.make_node("MatMul", ["x", "W"], ["mm_out"]),
        helper.make_node("Add", ["mm_out", "bias"], ["add_out"]),
        helper.make_node("Relu", ["add_out"], ["y"]),
        helper.make_node("Identity", ["y"], ["unused"]),  # dead code
    ]
    graph = helper.make_graph(nodes, "unfused", [x], [y], initializer=[w, bias])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def constant_fold_identity(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove Identity nodes that only pass through to unused outputs."""
    graph = model.graph
    used: Set[str] = {o.name for o in graph.output}
    for n in graph.node:
        for inp in n.input:
            used.add(inp)

    kept = []
    for n in graph.node:
        if n.op_type == "Identity" and n.output[0] not in used:
            print(f"  [fold] remove dead Identity → {n.output[0]}")
            continue
        kept.append(n)
    del graph.node[:]
    graph.node.extend(kept)
    return model


def fuse_matmul_add_relu(model: onnx.ModelProto) -> onnx.ModelProto:
    """Pattern match MatMul → Add(const) → Relu → single FusedMatMulAddRelu."""
    graph = model.graph
    init_map: Dict[str, np.ndarray] = {
        init.name: numpy_helper.to_array(init) for init in graph.initializer
    }
    node_map = {n.output[0]: n for n in graph.node if n.output}

    new_nodes: List[onnx.NodeProto] = []
    consumed: Set[str] = set()

    for node in graph.node:
        if node.output[0] in consumed:
            continue
        if node.op_type != "MatMul":
            new_nodes.append(node)
            continue

        mm_out = node.output[0]
        add_node = next((n for n in graph.node if n.op_type == "Add" and mm_out in n.input), None)
        if not add_node:
            new_nodes.append(node)
            continue
        relu_node = next((n for n in graph.node if n.op_type == "Relu" and add_node.output[0] in n.input), None)
        if not relu_node:
            new_nodes.append(node)
            continue

        bias_name = [x for x in add_node.input if x != mm_out][0]
        if bias_name not in init_map:
            new_nodes.append(node)
            continue

        fused = helper.make_node(
            "FusedMatMulAddRelu",
            node.input,
            relu_node.output,
            name="fused_1",
            domain="qualcomm.lab",
            bias=bias_name,
        )
        print("  [fuse] MatMul + Add + Relu → FusedMatMulAddRelu (1 kernel launch)")
        new_nodes.append(fused)
        consumed.update([mm_out, add_node.output[0]])

    del graph.node[:]
    graph.node.extend(new_nodes)
    return model


def count_node_types(model: onnx.ModelProto) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for n in model.graph.node:
        counts[n.op_type] = counts.get(n.op_type, 0) + 1
    return counts


def main() -> None:
    print_header("ONNX Graph Transforms (fold + fuse)", "Lab 02")
    if not ONNX_STACK:
        print("  Install: pip install onnx")
        sys.exit(1)

    model = build_unfused_graph()
    print(f"  Before: {count_node_types(model)}")

    model = constant_fold_identity(model)
    model = fuse_matmul_add_relu(model)
    print(f"  After:  {count_node_types(model)}")

    print("\n  Interview point: QNN/ORT run similar passes before HTP compile")
    print("  · fewer nodes → fewer kernel launches + less HBM traffic")
    print("  · fusion must preserve numerics — verify with golden tensors")
    print("\nLab 02 completed.")


if __name__ == "__main__":
    main()
