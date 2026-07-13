"""
Weaver & Langmuir (1990) + Langmuir (1992) partition and saturation for basaltic FC.

Oxide-based surrogates aligned with ``BASALT+langmuir.FOR`` (KDCALC KDMODE=4,
STOICH pressure-dependent Cpx).  Full ``STATE`` Newton–Raphson is not ported.

References
----------
- Weaver & Langmuir (1990), Computers & Geosciences 16(1), 1–19.
- Langmuir et al. (1992) — extended BASALT.FOR (P-dependent Kd, polybaric FC).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .wl_kd import (
    DEFAULT_T_K,
    cpx_stoich_index_from_melt,
    mpa_to_kbar,
    olivine_fo_pct_from_kd,
    plagioclase_saturation_boost,
    plag_an_pct_from_kd,
)

_OXIDE_MW = {"MgO": 40.305, "FeO": 71.844, "CaO": 56.078, "Na2O": 61.979, "K2O": 94.196}

# Low-P plagioclase liquidus (W&L eq. 2 style) @ ~1 kbar, 1273 K — scaled by Langmuir P-boost.
_K_AN_SAT = 2.45
_K_AB_SAT = 0.38
_K_OR_SAT = 0.15

# Fallback Roeder–Emslie if use_langmuir_kd=False.
_KD_FE_MG = 0.30

from .wl_high_al_morb import (
    HighAlFcRegime,
    _use_high_al,
    cpx_active_high_al,
    incremental_modes_high_al,
    is_high_al_morb,
    mgo_clinopyroxene_in,
    mgo_plagioclase_in,
    ol_pl_binary_shares,
)


@dataclass(frozen=True)
class WLSaturation:
    """Continuous saturation indices (0 = subsolidus, >=1 = saturated)."""

    olivine: float
    plagioclase: float
    clinopyroxene: float


def feldspar_moles(melt: Mapping[str, float]) -> tuple[float, float, float]:
    """Molar An, Ab, Or on the feldspar join (W&L liquid components)."""
    n_an = float(melt.get("CaO", 0.0)) / _OXIDE_MW["CaO"]
    n_ab = 2.0 * float(melt.get("Na2O", 0.0)) / _OXIDE_MW["Na2O"]
    n_or = 2.0 * float(melt.get("K2O", 0.0)) / _OXIDE_MW["K2O"]
    return n_an, n_ab, n_or


def melt_an_fraction(melt: Mapping[str, float]) -> float:
    n_an, n_ab, n_or = feldspar_moles(melt)
    return n_an / max(n_an + n_ab + n_or, 1e-12)


def _mgo_plagioclase_in(p_mpa: float, melt: Mapping[str, float] | None = None) -> float:
    """MgO (wt%) at plagioclase saturation; high-Al boost when ``melt`` given."""
    if melt is not None:
        return mgo_plagioclase_in(melt, p_mpa)
    return mgo_plagioclase_in({"Al2O3": 0.0}, p_mpa, high_al=False)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _sat_weight(index: float, *, onset: float = 0.12, sharpness: float = 6.0) -> float:
    return _sigmoid(sharpness * (float(index) - onset))


def saturation_indices(
    melt: Mapping[str, float],
    *,
    p_mpa: float = 100.0,
    t_k: float = DEFAULT_T_K,
) -> WLSaturation:
    """
    W&L + Langmuir saturation indices for Ol, Pl, Cpx.

    Plagioclase: liquidus proxy scaled by ``plagioclase_saturation_boost(P)``.
    Clinopyroxene: ``cpx_stoich_index_from_melt`` (STOICH JP=3, P>4 kbar threshold 1.1).
    """
    p_kbar = mpa_to_kbar(p_mpa)
    n_an, n_ab, n_or = feldspar_moles(melt)
    f_sum = max(n_an + n_ab + n_or, 1e-12)
    an_l = n_an / f_sum
    ab_l = n_ab / f_sum
    or_l = n_or / f_sum

    p_boost = plagioclase_saturation_boost(p_kbar)
    pl_liq = p_boost * (_K_AN_SAT * an_l + _K_AB_SAT * ab_l + _K_OR_SAT * or_l)

    mg = float(melt.get("MgO", 0.0))
    mgo_pl_in = _mgo_plagioclase_in(p_mpa, melt)

    i_ol = float(np.clip((mg - 3.0) / max(mgo_pl_in - 3.0, 1.0), 0.0, 1.5))
    if mg > mgo_pl_in + 1.5:
        i_ol = max(i_ol, 1.0)

    i_pl = max(0.0, pl_liq - 1.0)

    cpx_idx = cpx_stoich_index_from_melt(melt, t_k=t_k, p_kbar=p_kbar)
    i_cpx = max(0.0, cpx_idx - 1.0)

    return WLSaturation(olivine=i_ol, plagioclase=i_pl, clinopyroxene=i_cpx)


def olivine_fo_pct(
    melt: Mapping[str, float],
    *,
    t_k: float = DEFAULT_T_K,
    p_mpa: float = 100.0,
    use_langmuir_kd: bool = True,
) -> float:
    if use_langmuir_kd:
        return olivine_fo_pct_from_kd(melt, t_k=t_k, p_kbar=mpa_to_kbar(p_mpa))
    mg_mol = float(melt.get("MgO", 0.0)) / _OXIDE_MW["MgO"]
    fe_mol = float(melt.get("FeO", 0.0)) / _OXIDE_MW["FeO"]
    fe_mg = fe_mol / max(mg_mol, 1e-12)
    fo = 1.0 / (1.0 + _KD_FE_MG * fe_mg)
    return float(np.clip(100.0 * fo, 20.0, 95.0))


def plagioclase_an_pct(
    melt: Mapping[str, float],
    *,
    t_k: float = DEFAULT_T_K,
    p_mpa: float = 100.0,
    use_langmuir_kd: bool = True,
) -> float:
    if use_langmuir_kd:
        return plag_an_pct_from_kd(melt, t_k=t_k, p_kbar=mpa_to_kbar(p_mpa))
    n_an, n_ab, n_or = feldspar_moles(melt)
    w_an = 1.05 * n_an
    w_ab = 0.48 * n_ab
    w_or = 0.10 * n_or
    total = w_an + w_ab + w_or
    if total <= 1e-12:
        return 70.0
    return float(np.clip(100.0 * w_an / total, 20.0, 95.0))


def clinopyroxene_di_pct(
    melt: Mapping[str, float],
    *,
    t_k: float = DEFAULT_T_K,
    p_mpa: float = 100.0,
) -> float:
    mg_mol = float(melt.get("MgO", 0.0)) / _OXIDE_MW["MgO"]
    fe_mol = float(melt.get("FeO", 0.0)) / _OXIDE_MW["FeO"]
    mg_num = mg_mol / max(mg_mol + fe_mol, 1e-12)
    p_kbar = mpa_to_kbar(p_mpa)
    # High-P cpx Kd (+0.004/kbar on MgO) stabilizes more magnesian cpx.
    p_shift = min(0.08, 0.004 * max(0.0, p_kbar - 1.0))
    return float(np.clip(100.0 * (0.90 * mg_num + 0.05 + p_shift), 20.0, 95.0))


def incremental_modes_from_saturation(
    melt: Mapping[str, float],
    *,
    p_mpa: float = 100.0,
    t_k: float = DEFAULT_T_K,
    f_solid: float = 0.0,
    high_al_regime: HighAlFcRegime | None = None,
) -> tuple[float, float, float]:
    """Incremental Ol/Pl/Cpx modes (W&L sequential cotectic logic)."""
    sat = saturation_indices(melt, p_mpa=p_mpa, t_k=t_k)
    mg = float(melt.get("MgO", 0.0))
    mgo_pl_in = _mgo_plagioclase_in(p_mpa, melt)

    if high_al_regime is not None and high_al_regime.enabled:
        return incremental_modes_high_al(
            melt,
            p_mpa=p_mpa,
            f_solid=f_solid,
            regime=high_al_regime,
            sat_ol=sat.olivine,
            sat_pl=sat.plagioclase,
            sat_cpx=sat.clinopyroxene,
        )

    if mg > mgo_pl_in + 0.25:
        return 1.0, 0.0, 0.0

    n_an, n_ab, n_or = feldspar_moles(melt)
    mg_mol = float(melt.get("MgO", 0.0)) / _OXIDE_MW["MgO"]
    fe_mol = float(melt.get("FeO", 0.0)) / _OXIDE_MW["FeO"]
    mg_num = mg_mol / max(mg_mol + fe_mol, 1e-12)
    an_idx = n_an / max(n_an + n_ab + n_or, 1e-12)

    pl_active = mg <= mgo_pl_in and sat.plagioclase > 0.05
    if _use_high_al(melt, high_al_regime):
        cpx_active = pl_active and cpx_active_high_al(
            melt, p_mpa=p_mpa, f_solid=f_solid, sat_cpx=sat.clinopyroxene,
            regime=high_al_regime,
        )
    else:
        cpx_active = pl_active and sat.clinopyroxene > 0.08

    if not pl_active:
        return 1.0, 0.0, 0.0

    w_ol = _sat_weight(sat.olivine, onset=0.08)
    w_pl = _sat_weight(sat.plagioclase, onset=0.10)
    w_cpx = _sat_weight(sat.clinopyroxene, onset=0.12) if cpx_active else 0.0

    if not cpx_active or w_cpx < 0.05:
        ol_share, pl_share = ol_pl_binary_shares(
            melt, p_mpa=p_mpa, mgo_pl_in=mgo_pl_in, regime=high_al_regime,
        )
        ol, pl, cpx = ol_share, pl_share, 0.0
        if not _use_high_al(melt, high_al_regime):
            ol *= w_ol
            pl *= w_pl
            cpx *= w_cpx
            total = ol + pl + cpx
            if total <= 1e-12:
                return 1.0, 0.0, 0.0
            return ol / total, pl / total, cpx / total
        return ol, pl, cpx
    else:
        if _use_high_al(melt, high_al_regime):
            ol_share = float(np.clip(0.05 + 0.08 * mg_num, 0.04, 0.18))
            cpx_share = float(np.clip(0.14 + 0.30 * w_cpx, 0.12, 0.36))
            pl_share = max(0.40, 1.0 - ol_share - cpx_share)
            s = ol_share + pl_share + cpx_share
            ol, pl, cpx = ol_share / s, pl_share / s, cpx_share / s
        else:
            ol_share = float(np.clip(0.14 + 0.22 * mg_num, 0.08, 0.38))
            cpx_share = float(np.clip(0.22 + 0.28 * (1.0 - an_idx) + 0.18 * w_cpx, 0.12, 0.48))
            pl_share = max(0.0, 1.0 - ol_share - cpx_share)
            if pl_share < 0.15:
                scale = 0.85 / max(ol_share + cpx_share, 1e-9)
                ol_share *= scale
                cpx_share *= scale
                pl_share = 0.15
            s = ol_share + pl_share + cpx_share
            ol, pl, cpx = ol_share / s, pl_share / s, cpx_share / s

    if _use_high_al(melt, high_al_regime) and cpx_active:
        return ol, pl, cpx

    ol *= w_ol
    pl *= w_pl
    cpx *= w_cpx
    total = ol + pl + cpx
    if total <= 1e-12:
        return 1.0, 0.0, 0.0
    return ol / total, pl / total, cpx / total
