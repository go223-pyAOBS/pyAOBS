"""Limit OpenMP/BLAS threading for native libs used with Qt (BurnMan, NumPy)."""

from __future__ import annotations

import os

_THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def limit_native_parallelism(*, threads: int = 1) -> None:
    """Set BLAS/OpenMP env vars if unset (safer when mixing Qt + BurnMan)."""
    val = str(max(1, int(threads)))
    for key in _THREAD_KEYS:
        os.environ.setdefault(key, val)
