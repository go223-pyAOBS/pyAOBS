"""
Numerical stabilization for BASALT+langmuir STATE / TEMPRUN FC.

Langmuir Kd at high T can drive a single phase (often plagioclase) to FA≈1 in
one Newton step, then FC composition updates blow up.  These helpers cap NR
increments, optionally crystallize one phase at a time (Ol → Pl → Cpx), and
limit solid fraction per equilibrium solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 0-based: OL=1, PLAG=0, CPX=2 — typical MORB FC order
DEFAULT_PHASE_FC_ORDER: tuple[int, ...] = (1, 0, 2)


@dataclass(frozen=True)
class StateStabilize:
    """Controls optional FC stabilization in ``WLStateSystem``."""

    enabled: bool = True
    max_nr_fa_delta: float = 0.12
    max_solid_fraction: float = 0.10
    sequential_phases: bool = True
    phase_order: tuple[int, ...] = DEFAULT_PHASE_FC_ORDER


# One TEMPRUN / cooling-step solve
DEFAULT_FC_STABILIZE = StateStabilize()

# Equilibrium-only or regression against literal Fortran
NO_STABILIZE = StateStabilize(
    enabled=False,
    max_solid_fraction=1.0,
    sequential_phases=False,
)


def filter_phases_sequential(
    phase_list: list[int],
    phase_order: tuple[int, ...] = DEFAULT_PHASE_FC_ORDER,
) -> list[int]:
    """Keep only the highest-priority saturated phase (Ol before Pl before Cpx)."""
    if len(phase_list) <= 1:
        return phase_list
    for phase in phase_order:
        if phase in phase_list:
            return [phase]
    return [phase_list[0]]


def cap_nr_delta(delta: float, max_delta: float) -> float:
    if max_delta <= 0.0:
        return delta
    return float(np.sign(delta) * min(abs(delta), max_delta))


def cap_solid_fraction(fa: np.ndarray, max_solid: float) -> np.ndarray:
    """Scale FA so total solid ≤ ``max_solid`` (per solve / TEMPRUN step)."""
    fa = np.asarray(fa, dtype=float).copy()
    f_sum = float(np.sum(fa))
    if f_sum <= max_solid + 1e-12 or f_sum <= 1e-12:
        return fa
    return fa * (max_solid / f_sum)
