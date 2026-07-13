"""
W&L (1990) + Langmuir (1992) saturation-driven phase modes for basaltic FC.

Path-specific pressure: isobaric, equilibrium, or polybaric (FORTRAN MODES(6)).
"""

from __future__ import annotations

from typing import Literal, Mapping

from .wl_high_al_morb import HighAlFcRegime
from .wl_kd import DEFAULT_T_K
from .wl_partition import incremental_modes_from_saturation, saturation_indices
from .wl_polybaric import polybaric_pressure_mpa

CrystallizationPathMode = Literal["fc_100", "eq_100", "polybaric_fc"]


def phase_onset_f(p_mpa: float) -> tuple[float, float]:
    """Approximate F at Pl/Cpx appearance (diagnostic only)."""
    from .wl_partition import _mgo_plagioclase_in

    f_pl = float(max(0.08, 0.12 + 0.00015 * max(0.0, p_mpa - 100.0)))
    f_cpx = float(max(0.22, 0.34 - 0.00025 * max(0.0, p_mpa - 100.0)))
    _ = _mgo_plagioclase_in
    return f_pl, f_cpx


def incremental_modes(
    melt: Mapping[str, float],
    *,
    p_mpa: float,
    f_solid: float,
    path: CrystallizationPathMode,
    t_k: float = DEFAULT_T_K,
    polybaric_p_high_mpa: float = 800.0,
    polybaric_p_low_mpa: float = 100.0,
    polybaric_dp_mpa: float = 50.0,
    high_al_regime: HighAlFcRegime | None = None,
) -> tuple[float, float, float]:
    """Ol/Pl/Cpx modes from W&L + Langmuir saturation at effective P, T."""
    if path == "polybaric_fc":
        p_eff = polybaric_pressure_mpa(
            f_solid,
            p_high_mpa=polybaric_p_high_mpa,
            p_low_mpa=polybaric_p_low_mpa,
            dp_mpa=polybaric_dp_mpa,
        )
    elif path == "eq_100":
        p_eff = float(p_mpa) * 0.95
    else:
        p_eff = float(p_mpa)

    return incremental_modes_from_saturation(
        melt, p_mpa=p_eff, t_k=t_k, f_solid=f_solid, high_al_regime=high_al_regime,
    )
