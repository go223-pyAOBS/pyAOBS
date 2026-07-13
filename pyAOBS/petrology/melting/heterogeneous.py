"""
Heterogeneous (peridotite + pyroxenite) REEBOX forward model.

Wires: lithology → isentropic path → RMC column melt → optional Vp.
"""

from __future__ import annotations

from dataclasses import dataclass

from petrology.vp_regression import predict_vp_km_s

from .isentropic import IsentropicPath, integrate_isentropic_path
from .lithology import Lithology, heterogeneous_source
from .melt_chemistry import ChemistryBackend
from .reebox_geometry import build_reebox_column
from .rmc import RmcMeltResult, accumulate_column_melt


@dataclass(frozen=True)
class HeterogeneousColumnResult:
    """REEBOX-core forward: isentropic heterogeneous melting + RMC."""

    tp_c: float
    b_km: float
    chi: float
    pyroxenite_frac: float
    rmc_mode: str
    p0_gpa: float
    pf_gpa: float
    pbar_gpa: float
    fbar: float
    f_max: float
    h_km: float
    pooled_melt_wt: dict[str, float]
    melt_flux_by_lithology: dict[str, float]
    vp_bulk_eq1_km_s: float
    vp_bulk_norm_km_s: float | None
    isentropic_path: IsentropicPath
    lithologies: tuple[Lithology, ...]


def forward_heterogeneous_column(
    *,
    tp_c: float,
    b_km: float = 0.0,
    chi: float = 1.0,
    pyroxenite_frac: float = 0.0,
    rmc_mode: str = "active_langmuir",
    n_isentropic_steps: int = 120,
    vp_bias_km_s: float = 0.0,
    compute_norm_vp: bool = False,
    cipw_backend: str | None = None,
    mineral_backend: str | None = None,
    geometry: str = "reebox",
    lithology_backend: str = "native",
    peridotite_lith: str = "katz_lherzolite",
    pyroxenite_lith: str = "pertermann_g2",
    peridotite_h2o_wt: float = 0.0,
    pyroxenite_h2o_wt: float = 0.0,
    peridotite_chemistry: ChemistryBackend | None = None,
) -> HeterogeneousColumnResult:
    """
    REEBOX-style forward: Brown & Lesher (2016) isentropic melting + RMC pooling.

    ``geometry="reebox"`` (default): Katz solidus P0, path Pf, triangular H (χ).
    ``geometry="kkhs02"``: legacy KKHS02 linear F(P) column bounds (reproduction only).
    ``lithology_backend="pymelt"`` — pyMelt lithology curves (see ``list_pymelt_lithology_keys``).
    """
    liths = heterogeneous_source(
        pyroxenite_frac=pyroxenite_frac,
        backend=lithology_backend,
        peridotite_key=peridotite_lith,
        pyroxenite_key=pyroxenite_lith,
        peridotite_h2o_wt=peridotite_h2o_wt,
        pyroxenite_h2o_wt=pyroxenite_h2o_wt,
        peridotite_chemistry=peridotite_chemistry,
    )

    if geometry == "kkhs02":
        from petrology.active_upwelling import solve_active_upwelling

        col = solve_active_upwelling(tp_c=tp_c, b_km=b_km, chi=chi, vp_bias_km_s=0.0)
        path = integrate_isentropic_path(
            liths,
            tp_c=tp_c,
            p0_gpa=col.p0_gpa,
            pf_gpa=col.pf_gpa,
            n_steps=n_isentropic_steps,
        )
        p0, pf, h_km = col.p0_gpa, col.pf_gpa, col.h_km
        fbar_kkhs02 = col.fbar
    else:
        geom = build_reebox_column(
            liths,
            tp_c=tp_c,
            b_km=b_km,
            chi=chi,
            n_isentropic_steps=n_isentropic_steps,
        )
        path = geom.path
        p0, pf, h_km = geom.p0_gpa, geom.pf_gpa, geom.h_km
        fbar_kkhs02 = None

    rmc = accumulate_column_melt(liths, path, rmc_mode=rmc_mode)

    pbar = 0.5 * (p0 + pf)
    if geometry == "kkhs02" and rmc_mode != "passive_langmuir":
        fbar = float(fbar_kkhs02)
    else:
        fbar = float(rmc.fbar)
    vp_eq1 = predict_vp_km_s(pbar, fbar) + float(vp_bias_km_s)
    vp_norm = None
    if compute_norm_vp:
        from petrology.norm_velocity import norm_velocity_from_bulk_wt

        nv = norm_velocity_from_bulk_wt(
            rmc.pooled_melt_wt,
            cipw_backend=cipw_backend,
            mineral_backend=mineral_backend,
        )
        vp_norm = float(nv["vp_km_s"]) + float(vp_bias_km_s)

    return HeterogeneousColumnResult(
        tp_c=float(tp_c),
        b_km=float(b_km),
        chi=float(chi),
        pyroxenite_frac=float(pyroxenite_frac),
        rmc_mode=rmc_mode,
        p0_gpa=p0,
        pf_gpa=pf,
        pbar_gpa=float(pbar),
        fbar=float(fbar),
        f_max=float(rmc.f_max),
        h_km=float(h_km),
        pooled_melt_wt=dict(rmc.pooled_melt_wt),
        melt_flux_by_lithology=dict(rmc.melt_flux_by_lithology),
        vp_bulk_eq1_km_s=float(vp_eq1),
        vp_bulk_norm_km_s=vp_norm,
        isentropic_path=path,
        lithologies=tuple(liths),
    )
