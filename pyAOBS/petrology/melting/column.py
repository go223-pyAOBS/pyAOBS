"""
Modern-track mantle melting column (REEBOX-inspired core, scoped to pyAOBS).

Implements the pieces we need — not a full REEBOX PRO clone:

  1. Column geometry: KKHS02 / Langmuir **active upwelling** (Tp, b, chi).
  2. Chemistry: **incremental batch** major elements via Kinzler (1997) batch melt
     (``kinzler_linear``) or pMELTS KLB-1 grid (``reebox`` + ``lithology_backend=pymelt``).
  3. Optional **pyroxenite** fraction:
     - ``melting_engine="kinzler_linear"``: linear end-member mix (legacy MVP).
     - ``melting_engine="reebox"``: Brown & Lesher (2016) REEBOX PRO isentropic + dual lithology RMC.

Downstream: ``norm_velocity_from_bulk_wt`` → V_bulk; ``invert.bulk_vp_bounds`` → V_LC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from petrology.active_upwelling import (
    DFDP_DEFAULT_PER_GPA,
    solve_active_upwelling,
)
from petrology.vp_regression import predict_vp_km_s

from .kinzler1997_batch import HZ_DEP1_WT
from .melt_chemistry import ChemistryBackend, melt_oxides

# G2-like enriched melt end-member (wt%) for pyroxenite contribution (MVP).
_PYROXENITE_MELT_WT: dict[str, float] = {
    "SiO2": 48.5,
    "TiO2": 2.2,
    "Al2O3": 14.0,
    "Cr2O3": 0.05,
    "FeO": 11.0,
    "MgO": 11.5,
    "CaO": 10.0,
    "Na2O": 2.4,
    "K2O": 0.35,
    "P2O5": 0.05,
    "H2O": 0.0,
}

_OXIDE_KEYS = tuple(k for k in HZ_DEP1_WT if k != "H2O") + ("H2O",)


@dataclass(frozen=True)
class MeltingColumnResult:
    """Primary melt + column state for LIP / H–Vp workflows."""

    tp_c: float
    b_km: float
    chi: float
    pyroxenite_frac: float
    p0_gpa: float
    pf_gpa: float
    pbar_gpa: float
    fbar: float
    f_max: float
    h_km: float
    pooled_melt_wt: dict[str, float]
    vp_bulk_eq1_km_s: float
    vp_bulk_norm_km_s: float | None = None
    rmc_mode: str = "active_langmuir"
    melting_engine: str = "kinzler_linear"


def _normalize_oxides(ox: Mapping[str, float]) -> dict[str, float]:
    vals = {k: float(max(0.0, ox.get(k, 0.0))) for k in _OXIDE_KEYS}
    s = sum(vals.values())
    if s <= 0:
        raise ValueError("empty oxide composition")
    return {k: 100.0 * v / s for k, v in vals.items()}


def _integrate_peridotite_melt(
    *,
    p0_gpa: float,
    pf_gpa: float,
    f_max: float,
    dfdp_per_gpa: float,
    n_steps: int,
    source_wt: Mapping[str, float] | None,
    chemistry_backend: ChemistryBackend = "kinzler1997",
) -> dict[str, float]:
    """
    Active (Langmuir) RMC: column-accumulated melt along longest flow line.

    F(P) = (dF/dP) * (P0 - P);  C_column = (1/F_max) * integral C_batch(P, F) dF.
    """
    if f_max <= 1e-9:
        return melt_oxides(p0_gpa, 0.02, chemistry_backend=chemistry_backend, source=source_wt)

    p_path = np.linspace(float(p0_gpa), float(pf_gpa), int(n_steps) + 1)
    acc = {k: 0.0 for k in _OXIDE_KEYS}
    f_prev = 0.0
    for i in range(len(p_path) - 1):
        p_mid = 0.5 * (p_path[i] + p_path[i + 1])
        f_mid = float(np.clip(dfdp_per_gpa * (p0_gpa - p_mid), 0.0, 0.995))
        d_f = max(f_mid - f_prev, dfdp_per_gpa * abs(p_path[i + 1] - p_path[i]))
        f_prev = f_mid
        if d_f <= 0.0:
            continue
        ox = melt_oxides(
            p_mid,
            f_mid,
            chemistry_backend=chemistry_backend,
            source=source_wt,
        )
        for k in _OXIDE_KEYS:
            acc[k] += d_f * float(ox.get(k, 0.0))

    return _normalize_oxides({k: acc[k] / f_max for k in _OXIDE_KEYS})


def _blend_pyroxenite(
    peridotite_melt: Mapping[str, float],
    *,
    pyroxenite_frac: float,
    pyroxenite_melt: Mapping[str, float] | None = None,
) -> dict[str, float]:
    phi = float(np.clip(pyroxenite_frac, 0.0, 0.99))
    if phi <= 1e-9:
        return dict(peridotite_melt)
    end = _normalize_oxides(pyroxenite_melt or _PYROXENITE_MELT_WT)
    per = _normalize_oxides(peridotite_melt)
    mixed = {k: (1.0 - phi) * per[k] + phi * end[k] for k in _OXIDE_KEYS}
    return _normalize_oxides(mixed)


def forward_melting_column(
    *,
    tp_c: float,
    b_km: float = 0.0,
    chi: float = 1.0,
    pyroxenite_frac: float = 0.0,
    melting_engine: str = "kinzler_linear",
    rmc_mode: str = "active_langmuir",
    dfdp_per_gpa: float = DFDP_DEFAULT_PER_GPA,
    n_steps: int = 48,
    n_isentropic_steps: int = 32,
    source_wt: Mapping[str, float] | None = None,
    vp_bias_km_s: float = 0.0,
    compute_norm_vp: bool = False,
    cipw_backend: str | None = None,
    mineral_backend: str | None = None,
    lithology_backend: str = "native",
    peridotite_lith: str = "katz_lherzolite",
    pyroxenite_lith: str = "pertermann_g2",
    peridotite_h2o_wt: float = 0.0,
    pyroxenite_h2o_wt: float = 0.0,
    peridotite_chemistry: ChemistryBackend | None = None,
) -> MeltingColumnResult:
    """
    Forward melting column for Modern LIP track.

    Parameters
    ----------
    tp_c, b_km, chi
        As in KKHS02 §4 / REEBOX active (Langmuir) end-member.
    pyroxenite_frac
        Mass fraction of G2 pyroxenite in source (0–0.99).
    melting_engine
        ``kinzler_linear`` (default) or ``reebox`` (isentropic heterogeneous).
    lithology_backend
        ``native`` or ``pymelt`` (pyMelt lithology registry).
    compute_norm_vp
        If True, run Pyrolite CIPW + BurnMan on pooled melt (slower).
    """
    if melting_engine == "reebox":
        from .heterogeneous import forward_heterogeneous_column

        het = forward_heterogeneous_column(
            tp_c=tp_c,
            b_km=b_km,
            chi=chi,
            pyroxenite_frac=pyroxenite_frac,
            rmc_mode=rmc_mode,
            n_isentropic_steps=n_isentropic_steps,
            vp_bias_km_s=vp_bias_km_s,
            compute_norm_vp=compute_norm_vp,
            cipw_backend=cipw_backend,
            mineral_backend=mineral_backend,
            lithology_backend=lithology_backend,
            peridotite_lith=peridotite_lith,
            pyroxenite_lith=pyroxenite_lith,
            peridotite_h2o_wt=peridotite_h2o_wt,
            pyroxenite_h2o_wt=pyroxenite_h2o_wt,
            peridotite_chemistry=peridotite_chemistry,
        )
        return MeltingColumnResult(
            tp_c=het.tp_c,
            b_km=het.b_km,
            chi=het.chi,
            pyroxenite_frac=het.pyroxenite_frac,
            p0_gpa=het.p0_gpa,
            pf_gpa=het.pf_gpa,
            pbar_gpa=het.pbar_gpa,
            fbar=het.fbar,
            f_max=het.f_max,
            h_km=het.h_km,
            pooled_melt_wt=het.pooled_melt_wt,
            vp_bulk_eq1_km_s=het.vp_bulk_eq1_km_s,
            vp_bulk_norm_km_s=het.vp_bulk_norm_km_s,
            rmc_mode=het.rmc_mode,
            melting_engine="reebox",
        )

    if melting_engine != "kinzler_linear":
        raise ValueError(f"unknown melting_engine: {melting_engine}")
    col = solve_active_upwelling(
        tp_c=tp_c,
        b_km=b_km,
        chi=chi,
        dfdp_per_gpa=dfdp_per_gpa,
        vp_bias_km_s=0.0,
    )
    chem: ChemistryBackend = peridotite_chemistry or (
        "pmelts_klb1" if lithology_backend == "pymelt" else "kinzler1997"
    )
    f_max = float(min(2.0 * col.fbar, 0.995))
    per_melt = _integrate_peridotite_melt(
        p0_gpa=col.p0_gpa,
        pf_gpa=col.pf_gpa,
        f_max=f_max,
        dfdp_per_gpa=dfdp_per_gpa,
        n_steps=n_steps,
        source_wt=source_wt,
        chemistry_backend=chem,
    )
    pooled = _blend_pyroxenite(per_melt, pyroxenite_frac=pyroxenite_frac)

    vp_eq1 = predict_vp_km_s(col.pbar_gpa, col.fbar) + float(vp_bias_km_s)
    vp_norm = None
    if compute_norm_vp:
        from petrology.norm_velocity import norm_velocity_from_bulk_wt

        nv = norm_velocity_from_bulk_wt(
            pooled,
            cipw_backend=cipw_backend,
            mineral_backend=mineral_backend,
        )
        vp_norm = float(nv["vp_km_s"]) + float(vp_bias_km_s)

    return MeltingColumnResult(
        tp_c=float(tp_c),
        b_km=float(b_km),
        chi=float(chi),
        pyroxenite_frac=float(pyroxenite_frac),
        p0_gpa=col.p0_gpa,
        pf_gpa=col.pf_gpa,
        pbar_gpa=col.pbar_gpa,
        fbar=col.fbar,
        f_max=f_max,
        h_km=col.h_km,
        pooled_melt_wt=pooled,
        vp_bulk_eq1_km_s=float(vp_eq1),
        vp_bulk_norm_km_s=vp_norm,
        rmc_mode="active_langmuir",
        melting_engine="kinzler_linear",
    )
