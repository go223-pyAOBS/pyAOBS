"""
Residual Mantle Column (RMC) mixing — Brown & Lesher (2016) REEBOX PRO eq. (8), major-element limit.

Modes:
  - ``passive_langmuir``: triangular melting zone; longest flow line F(P).
  - ``active_langmuir``: all columns melt to F_max at lithosphere base (LIP).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .isentropic import IsentropicPath, IsentropicStep
from .lithology import Lithology

_OXIDE_KEYS = (
    "SiO2", "TiO2", "Al2O3", "Cr2O3", "FeO", "MgO", "CaO", "Na2O", "K2O", "P2O5", "H2O",
)


def _normalize_ox(ox: Mapping[str, float]) -> dict[str, float]:
    vals = {k: float(max(0.0, ox.get(k, 0.0))) for k in _OXIDE_KEYS}
    s = sum(vals.values())
    if s <= 0:
        raise ValueError("empty melt composition")
    return {k: 100.0 * v / s for k, v in vals.items()}


@dataclass(frozen=True)
class RmcMeltResult:
    pooled_melt_wt: dict[str, float]
    f_max: float
    fbar: float
    melt_flux_by_lithology: dict[str, float]
    rmc_mode: str


def _instant_melt_mix(
    lithologies: list[Lithology],
    step: IsentropicStep,
    *,
    df_dp: dict[str, float],
    dp: float,
) -> dict[str, float]:
    """Mass-weighted instantaneous melt oxides at one decompression increment."""
    acc = {k: 0.0 for k in _OXIDE_KEYS}
    mass = 0.0
    for L in lithologies:
        dfi = abs(df_dp.get(L.name, 0.0) * dp)
        if dfi <= 0.0:
            continue
        fi = step.f_by_name.get(L.name, 0.0)
        ox = L.melt_oxides_wt(step.p_gpa, max(fi, 0.02))
        w = L.u0 * dfi
        mass += w
        for k in _OXIDE_KEYS:
            acc[k] += w * ox.get(k, 0.0)
    if mass <= 0.0:
        p = step.p_gpa
        ox = lithologies[0].melt_oxides_wt(p, 0.02)
        return dict(ox)
    return _normalize_ox({k: acc[k] / mass for k in _OXIDE_KEYS})


def accumulate_column_melt(
    lithologies: list[Lithology],
    path: IsentropicPath,
    *,
    rmc_mode: str = "active_langmuir",
) -> RmcMeltResult:
    """
    Column-accumulated pooled melt (eq. 8 major-element form).

    C_column = (1/F_max) ∫ C_inst dF  along the active (longest) flow line.
    """
    if len(path.steps) < 2:
        raise ValueError("isentropic path too short")

    acc = {k: 0.0 for k in _OXIDE_KEYS}
    f_max = 0.0
    flux = {L.name: 0.0 for L in lithologies}

    for i in range(len(path.steps) - 1):
        st0 = path.steps[i]
        st1 = path.steps[i + 1]
        dp = st1.p_gpa - st0.p_gpa
        df_dp = st0.df_dp_by_name
        d_f_bulk = 0.0
        for L in lithologies:
            dfi = abs(df_dp.get(L.name, 0.0) * dp)
            w = L.u0 * dfi
            flux[L.name] += w
            d_f_bulk += w
        if d_f_bulk <= 0.0:
            continue
        inst = _instant_melt_mix(lithologies, st0, df_dp=df_dp, dp=dp)
        for k in _OXIDE_KEYS:
            acc[k] += d_f_bulk * inst[k]
        f_max += d_f_bulk

    if f_max <= 1e-9:
        p = path.p0_gpa
        ox = lithologies[0].melt_oxides_wt(p, 0.02)
        return RmcMeltResult(
            pooled_melt_wt=dict(ox),
            f_max=0.02,
            fbar=0.02,
            melt_flux_by_lithology=flux,
            rmc_mode=rmc_mode,
        )

    pooled = _normalize_ox({k: acc[k] / f_max for k in _OXIDE_KEYS})
    fbar = path.fbar()

    if rmc_mode == "passive_langmuir":
        # Passive: mean F ~ half of max along triangular domain (Langmuir 1992).
        fbar = 0.5 * f_max

    return RmcMeltResult(
        pooled_melt_wt=pooled,
        f_max=float(f_max),
        fbar=float(fbar),
        melt_flux_by_lithology=flux,
        rmc_mode=rmc_mode,
    )
