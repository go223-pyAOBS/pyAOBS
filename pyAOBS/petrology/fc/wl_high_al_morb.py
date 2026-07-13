"""
High-Al MORB plagioclase-priority calibration (KKHS02 Fig.2 / Kinzler 1997 primary).

At ~1 kbar, high-Al MORB (Al2O3 ≳ 15 wt%) follows Ol → Pl before Cpx.  Paper Fig.2c
shows plagioclase from F ≈ 0.08 and clinopyroxene only after F ≈ 0.35–0.40.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

# Kinzler (1997) Fig.2 primary and similar MORB liquids.
HIGH_AL_AL2O3_WT = 15.0

# Plagioclase-in: raise MgO threshold with excess Al2O3 above tholeiitic baseline.
_MGO_PLAG_IN_100 = 8.5
_MGO_PLAG_P_SLOPE = 0.0015  # wt% MgO per MPa above 100 MPa
_AL2O3_PLAG_BOOST = 0.95  # wt% MgO per wt% Al2O3 above 14%

# Clinopyroxene-in @ 100 MPa (MgO wt%); delays Cpx on the Ol–Pl cotectic.
_MGO_CPX_IN_100 = 8.15
_MGO_CPX_P_SLOPE = 0.0012
_F_CPX_MIN_HIGH_AL = 0.32
_F_CPX_BLEND_START = 0.28  # no Cpx increment before this F (paper Cpx cum ≈0 @ F=0.22)

# Ol share on Ol–Pl segment (paper Fig.2c: ~pure Pl increments after F≈0.17).
_OL_PL_OL_FLOOR = 0.02
_OL_PL_OL_MGO_SLOPE = 0.012  # per wt% (mgo_pl_in − MgO)
_OL_PL_OL_CAP = 0.10
# Ol share on Ol→Pl limb @ F≈0.07–0.17 (paper cum ~82% Ol @ F=0.08).
_OL_PL_OL_F0 = 0.71
_OL_PL_OL_F_SLOPE = 7.1

# Smooth transitions (avoid Ol↔Pl Vp spikes when MgO hovers at plagioclase-in).
_PLAG_IN_BLEND_WT = 0.28  # MgO wt% scale for Ol→Ol+Pl sigmoid
_CPX_ONSET_F_BLEND = 0.05  # solid-fraction scale for Cpx entry
_CPX_LATCH_BLEND = 0.08  # latch ternary once blend exceeds this


@dataclass
class HighAlFcRegime:
    """Forward-only phase-field flags along an FC path (prevents Ol/Pl oscillation)."""

    enabled: bool = True
    high_al_morb: bool = True  # latched from primary melt; do not re-check Al2O3 on evolved melt
    primary_al2o3: float = 0.0  # latch Al2O3 for plagioclase-in threshold along path
    pl_field: bool = False
    cpx_field: bool = False


def _use_high_al(melt: Mapping[str, float], regime: HighAlFcRegime | None) -> bool:
    if regime is not None and regime.enabled:
        return regime.high_al_morb
    return is_high_al_morb(melt)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _ol_pl_shares_f_guided(
    f_solid: float,
    mg: float,
    mgo_pl_in: float,
) -> tuple[float, float]:
    """Ol/Pl on Ol–Pl segment; F-guided to match KKHS02 Fig.2c incremental schedule."""
    if f_solid < 0.07:
        gap = max(0.0, mgo_pl_in - mg)
        ol = min(_OL_PL_OL_CAP, _OL_PL_OL_FLOOR + _OL_PL_OL_MGO_SLOPE * gap)
    elif f_solid <= 0.17:
        ol = max(0.0, _OL_PL_OL_F0 - (f_solid - 0.07) * _OL_PL_OL_F_SLOPE)
    elif f_solid < 0.23:
        ol = 0.0
    else:
        ol = _OL_PL_OL_FLOOR
    ol = float(np.clip(ol, 0.0, 0.72))
    pl = max(1.0 - ol, 0.05)
    s = ol + pl
    return ol / s, pl / s


def incremental_modes_high_al(
    melt: Mapping[str, float],
    *,
    p_mpa: float,
    f_solid: float,
    regime: HighAlFcRegime,
    sat_ol: float,
    sat_pl: float,
    sat_cpx: float,
) -> tuple[float, float, float]:
    """
    High-Al MORB incremental modes with smooth Ol→Pl and delayed Cpx entry.

    Uses ``regime`` hysteresis so FC mass-balance noise at plagioclase-in does not
    flip between 100% olivine and plagioclase-rich increments (Vp spikes).
    """
    mg = float(melt.get("MgO", 0.0))
    mgo_pl_in = mgo_plagioclase_in(melt, p_mpa, regime=regime)

    if not regime.pl_field:
        pl_blend = _sigmoid((mgo_pl_in - mg) / _PLAG_IN_BLEND_WT)
        if pl_blend >= 0.80 or f_solid >= 0.07:
            regime.pl_field = True
        ol_s, pl_s = _ol_pl_shares_f_guided(f_solid, mg, mgo_pl_in)
        ol = (1.0 - pl_blend) + pl_blend * ol_s
        pl = pl_blend * pl_s
        s = ol + pl
        return ol / s, pl / s, 0.0

    ol_s, pl_s = _ol_pl_shares_f_guided(f_solid, mg, mgo_pl_in)
    if not regime.cpx_field:
        cpx_blend = 0.0
        if f_solid >= _F_CPX_BLEND_START:
            cpx_blend = _sigmoid((f_solid - _F_CPX_MIN_HIGH_AL) / _CPX_ONSET_F_BLEND)
            cpx_blend *= _sigmoid((sat_cpx - 0.04) / 0.06)
        ol_t = float(np.clip(0.01 + 0.02 * mg / max(mgo_pl_in, 1.0), 0.01, 0.06))
        cpx_t = float(np.clip(0.24 + 0.04 * min(sat_cpx, 6.0), 0.20, 0.50))
        pl_t = max(0.38, 1.0 - ol_t - cpx_t)
        st = ol_t + pl_t + cpx_t
        ol_t, pl_t, cpx_t = ol_t / st, pl_t / st, cpx_t / st
        if f_solid >= 0.30 and cpx_blend >= 0.55:
            regime.cpx_field = True
        ol = (1.0 - cpx_blend) * ol_s + cpx_blend * ol_t
        pl = (1.0 - cpx_blend) * pl_s + cpx_blend * pl_t
        cpx = cpx_blend * cpx_t
        s = ol + pl + cpx
        return ol / s, pl / s, cpx / s

    ol_t = float(np.clip(0.01 + 0.03 * mg / max(mgo_pl_in, 1.0), 0.01, 0.08))
    cpx_t = float(np.clip(0.28 + 0.04 * min(sat_cpx, 6.0), 0.24, 0.52))
    pl_t = max(0.40, 1.0 - ol_t - cpx_t)
    st = ol_t + pl_t + cpx_t
    return ol_t / st, pl_t / st, cpx_t / st


def is_high_al_morb(melt: Mapping[str, float], *, al2o3_min: float = HIGH_AL_AL2O3_WT) -> bool:
    return float(melt.get("Al2O3", 0.0)) >= float(al2o3_min)


def mgo_plagioclase_in(
    melt: Mapping[str, float],
    p_mpa: float,
    *,
    high_al: bool | None = None,
    regime: HighAlFcRegime | None = None,
) -> float:
    """MgO (wt%) at plagioclase saturation; boosted for high-Al MORB."""
    mgo_in = _MGO_PLAG_IN_100 - _MGO_PLAG_P_SLOPE * max(0.0, float(p_mpa) - 100.0)
    use_ha = high_al if high_al is not None else _use_high_al(melt, regime)
    if use_ha:
        if regime is not None and regime.enabled and regime.primary_al2o3 > 0.0:
            al = regime.primary_al2o3
        else:
            al = float(melt.get("Al2O3", 0.0))
        mgo_in += _AL2O3_PLAG_BOOST * max(0.0, al - 14.0)
    return mgo_in


def mgo_clinopyroxene_in(
    melt: Mapping[str, float],
    p_mpa: float,
    *,
    high_al: bool | None = None,
    regime: HighAlFcRegime | None = None,
) -> float | None:
    """MgO (wt%) below which Cpx may join; ``None`` = use default cpx gate."""
    use_ha = high_al if high_al is not None else _use_high_al(melt, regime)
    if not use_ha:
        return None
    return _MGO_CPX_IN_100 + _MGO_CPX_P_SLOPE * max(0.0, float(p_mpa) - 100.0)


def cpx_active_high_al(
    melt: Mapping[str, float],
    *,
    p_mpa: float,
    f_solid: float,
    sat_cpx: float,
    regime: HighAlFcRegime | None = None,
) -> bool:
    """Gate Cpx for high-Al MORB: F onset + STOICH proxy (MgO stays high on Fig.2 path)."""
    mgo_in = mgo_clinopyroxene_in(melt, p_mpa, regime=regime)
    if mgo_in is None:
        return sat_cpx > 0.08
    return f_solid >= _F_CPX_MIN_HIGH_AL and sat_cpx > 0.05


def ol_pl_binary_shares(
    melt: Mapping[str, float],
    *,
    p_mpa: float,
    mgo_pl_in: float,
    regime: HighAlFcRegime | None = None,
) -> tuple[float, float]:
    """Ol/Pl mode fractions on the binary segment (no Cpx)."""
    mg = float(melt.get("MgO", 0.0))
    if _use_high_al(melt, regime):
        gap = max(0.0, mgo_pl_in - mg)
        ol_share = float(min(_OL_PL_OL_CAP, _OL_PL_OL_FLOOR + _OL_PL_OL_MGO_SLOPE * gap))
        pl_share = max(1.0 - ol_share, 0.05)
        s = ol_share + pl_share
        return ol_share / s, pl_share / s

    n_an, n_ab, n_or = _feldspar_moles(melt)
    mg_mol = float(melt.get("MgO", 0.0)) / 40.305
    fe_mol = float(melt.get("FeO", 0.0)) / 71.844
    mg_num = mg_mol / max(mg_mol + fe_mol, 1e-12)
    ol_share = float(min(0.72, max(0.30, 0.38 + 0.32 * mg_num)))
    pl_share = 1.0 - ol_share
    return ol_share, pl_share


def _feldspar_moles(melt: Mapping[str, float]) -> tuple[float, float, float]:
    n_an = float(melt.get("CaO", 0.0)) / 56.078
    n_ab = 2.0 * float(melt.get("Na2O", 0.0)) / 61.979
    n_or = 2.0 * float(melt.get("K2O", 0.0)) / 94.196
    return n_an, n_ab, n_or
