"""
Polybaric crystallization pressure schedule (BASALT+langmuir.FOR DRIVER).

Fortran MODES(6)=1: loop PHI from P_HIGH to P_LOW in steps of DP (kbar),
call TEMPRUN at each segment, then CSJ=CLJ before the next lower P.

``polybaric_pressure_kbar(f_solid)`` retains the legacy F→P mapping for
diagnostics; the DRIVER uses discrete ``polybaric_pressure_levels_*``.
"""

from __future__ import annotations

import math

from .wl_kd import kbar_to_mpa

# KKHS02 Fig.2 / Fortran PARMS(5,6,4) defaults (kbar).
DEFAULT_P_HIGH_KBAR = 8.0
DEFAULT_P_LOW_KBAR = 1.0
DEFAULT_DP_KBAR = 0.5


def polybaric_pressure_levels_kbar(
    *,
    p_high_kbar: float = DEFAULT_P_HIGH_KBAR,
    p_low_kbar: float = DEFAULT_P_LOW_KBAR,
    dp_kbar: float = DEFAULT_DP_KBAR,
) -> list[float]:
    """
    Discrete P ladder matching Fortran DRIVER loop 90 (PHI from HIGH to LOW).

    Stops when ``PHI < P_LOW - 0.01``; ensures ``p_low_kbar`` is included.
    """
    p_high = float(p_high_kbar)
    p_low = float(p_low_kbar)
    dp = float(dp_kbar)
    if dp <= 1e-9:
        return [p_high]

    levels: list[float] = []
    phi = p_high
    while phi >= p_low - 0.01:
        levels.append(round(phi, 6))
        phi -= dp

    if not levels or levels[-1] > p_low + 1e-6:
        levels.append(p_low)
    return levels


def polybaric_pressure_levels_mpa(
    *,
    p_high_mpa: float = 800.0,
    p_low_mpa: float = 100.0,
    dp_mpa: float = 50.0,
) -> list[float]:
    """Polybaric P ladder in MPa (100 MPa = 1 kbar)."""
    return [
        kbar_to_mpa(p)
        for p in polybaric_pressure_levels_kbar(
            p_high_kbar=p_high_mpa / 100.0,
            p_low_kbar=p_low_mpa / 100.0,
            dp_kbar=dp_mpa / 100.0,
        )
    ]


def polybaric_pressure_kbar(
    f_solid: float,
    *,
    p_high_kbar: float = DEFAULT_P_HIGH_KBAR,
    p_low_kbar: float = DEFAULT_P_LOW_KBAR,
    dp_kbar: float = DEFAULT_DP_KBAR,
) -> float:
    """
    Effective crystallization P (kbar) at solid fraction ``f_solid``.

    Mirrors discrete P segments: as FC progresses, effective P steps down
    from ``p_high_kbar`` toward ``p_low_kbar`` in increments of ``dp_kbar``.
    """
    f = float(max(0.0, min(1.0, f_solid)))
    span = max(float(p_high_kbar) - float(p_low_kbar), 0.0)
    if span <= 1e-9 or float(dp_kbar) <= 1e-9:
        return float(p_high_kbar)

    if f >= 1.0 - 1e-9:
        return float(p_low_kbar)

    n_seg = max(1, int(math.ceil(span / float(dp_kbar))))
    seg_idx = min(n_seg - 1, int(f * n_seg))
    p = float(p_high_kbar) - seg_idx * float(dp_kbar)
    return float(max(float(p_low_kbar), p))


def polybaric_pressure_mpa(
    f_solid: float,
    *,
    p_high_mpa: float = 800.0,
    p_low_mpa: float = 100.0,
    dp_mpa: float = 50.0,
) -> float:
    """Polybaric P in MPa (100 MPa = 1 kbar)."""
    return kbar_to_mpa(
        polybaric_pressure_kbar(
            f_solid,
            p_high_kbar=p_high_mpa / 100.0,
            p_low_kbar=p_low_mpa / 100.0,
            dp_kbar=dp_mpa / 100.0,
        )
    )
