#!/usr/bin/env python3
"""
Run all Qualcomm interview labs in order.

    python3 qualcomm/run_all.py
    python3 qualcomm/run_all.py --lab 03
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

LABS = [
    ("01", "lab_01_onnx_export.py", "PyTorch → ONNX export"),
    ("02", "lab_02_graph_transform.py", "Graph fold & fuse"),
    ("03", "lab_03_quantization_ptq.py", "PTQ quantization"),
    ("04", "lab_04_ort_runtime.py", "ONNX Runtime & EP"),
    ("05", "lab_05_graph_partition_delegate.py", "Graph partition / delegate"),
    ("06", "lab_06_llm_prefill_decode.py", "Prefill / Decode / KV"),
    ("07", "lab_07_moe_routing.py", "MoE routing"),
    ("08", "lab_08_lora_merge_deploy.py", "LoRA merge"),
    ("09", "lab_09_debug_golden_pipeline.py", "Debug golden pipeline"),
    ("10", "lab_10_e2e_pipeline.py", "End-to-end pipeline"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab", type=str, default="", help="Run single lab, e.g. 03")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    selected = LABS
    if args.lab:
        selected = [t for t in LABS if t[0] == args.lab.zfill(2)]
        if not selected:
            print(f"Unknown lab: {args.lab}")
            return 1

    print("Qualcomm Interview Labs — run_all")
    print("=" * 60)
    failures = 0
    for lab_id, script, title in selected:
        path = root / script
        print(f"\n>>> Lab {lab_id}: {title}")
        rc = subprocess.call([sys.executable, str(path)], cwd=str(root))
        if rc != 0:
            failures += 1
            print(f"FAILED: {script} (exit {rc})")

    print("\n" + "=" * 60)
    if failures:
        print(f"Done with {failures} failure(s).")
        return 1
    print("All labs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
