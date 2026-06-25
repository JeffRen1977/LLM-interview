"""Utilities for Qualcomm lab demos."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator, Tuple

import numpy as np


def print_header(title: str, lab_id: str = "") -> None:
    prefix = f"[{lab_id}] " if lab_id else ""
    print("\n" + "=" * 60)
    print(f"{prefix}{title}")
    print("=" * 60)


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


@contextmanager
def timed(label: str) -> Generator[None, None, None]:
    t0 = time.perf_counter()
    yield
    ms = (time.perf_counter() - t0) * 1000
    print(f"  {label}: {ms:.3f} ms")
