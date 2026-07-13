"""
3D scan: (Tp, chi, pyroxenite_frac) vs H + V_LC bounding.

When no point is fully feasible, reports Pareto-best by |dH| and Vp excess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from petrology.fractionation import delta_vp_km_s
from petrology.invert import bulk_vp_bounds

from .column import forward_melting_column
from .hvp_scan import HvpScanPoint, HvpScanResult, ProgressCallback, _count_grid, scan_hvp_lip


@dataclass(frozen=True)
class PhiScanSlice:
    pyroxenite_frac: float
    result: HvpScanResult
    best_pareto: HvpScanPoint | None


@dataclass(frozen=True)
class PhiScanResult:
    v_lc_obs_km_s: float
    h_obs_km: float
    b_km: float
    melting_engine: str
    slices: list[PhiScanSlice]

    @property
    def n_feasible_total(self) -> int:
        return sum(s.result.n_feasible for s in self.slices)

    def best_overall(self) -> tuple[PhiScanSlice, HvpScanPoint] | None:
        candidates: list[tuple[PhiScanSlice, HvpScanPoint, float]] = []
        for sl in self.slices:
            pt = sl.best_pareto
            if pt is None:
                continue
            score = _pareto_score(pt, v_lc_obs=self.v_lc_obs_km_s, h_obs=self.h_obs_km)
            candidates.append((sl, pt, score))
        if not candidates:
            return None
        sl, pt, _ = min(candidates, key=lambda x: x[2])
        return sl, pt


def _pareto_score(
    pt: HvpScanPoint,
    *,
    v_lc_obs: float,
    h_obs: float,
    h_scale_km: float = 3.0,
    vp_scale_km_s: float = 0.10,
) -> float:
    vp_excess = max(0.0, pt.vp_bulk_km_s - v_lc_obs)
    return (pt.h_match_km / h_scale_km) ** 2 + (vp_excess / vp_scale_km_s) ** 2


def _best_pareto_from_result(
    result: HvpScanResult,
    *,
    v_lc_obs: float,
    h_obs: float,
    prefer_in_bounds: bool = True,
) -> HvpScanPoint | None:
    if result.feasible:
        return min(result.feasible, key=lambda p: _pareto_score(p, v_lc_obs=v_lc_obs, h_obs=h_obs))
    if not result.points:
        return None
    pool = result.points
    if prefer_in_bounds:
        in_b = [p for p in pool if p.bulk_in_bounds]
        if in_b:
            return min(in_b, key=lambda p: abs(p.h_match_km))
    return min(pool, key=lambda p: _pareto_score(p, v_lc_obs=v_lc_obs, h_obs=h_obs))


def scan_hvp_lip_phi(
    *,
    v_lc_obs_km_s: float,
    h_obs_km: float,
    b_km: float = 0.0,
    phi_values: list[float] | None = None,
    tp_range_c: tuple[float, float] = (1200.0, 1500.0),
    tp_step_c: float = 10.0,
    chi_values: list[float] | None = None,
    h_tolerance_km: float = 3.0,
    melting_engine: str = "reebox",
    use_norm_vp: bool = False,
    vp_mode: str = "auto",
    vp_bias_km_s: float = -0.10,
    n_isentropic_steps: int = 32,
    verbose: bool = False,
    refine_norm_vp: int = 0,
    delta_vp_engine: str = "auto",
    delta_vp_wl_kw: dict | None = None,
    require_bulk_in_bounds: bool = True,
    lithology_backend: str = "native",
    peridotite_lith: str = "katz_lherzolite",
    pyroxenite_lith: str = "pertermann_g2",
    peridotite_h2o_wt: float = 0.0,
    pyroxenite_h2o_wt: float = 0.0,
    lithology_preset: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PhiScanResult:
    if phi_values is None:
        phi_values = [0.0, 0.05, 0.10, 0.15, 0.20]
    if chi_values is None:
        chi_values = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]

    n_grid = _count_grid(tp_range_c, tp_step_c, chi_values)
    refine_steps = (
        refine_norm_vp if (refine_norm_vp > 0 and melting_engine == "reebox" and not use_norm_vp) else 0
    )
    per_phi = n_grid + refine_steps
    total_steps = len(phi_values) * per_phi

    slices: list[PhiScanSlice] = []
    for i, phi in enumerate(phi_values):
        if verbose:
            print(f"Phi slice {i + 1}/{len(phi_values)}: Φ={phi:.2f}", flush=True)

        base = i * per_phi

        def _phi_progress(done: int, _total: int, msg: str, *, base=base) -> None:
            if progress_callback:
                progress_callback(min(base + done, total_steps), total_steps, msg)

        res = scan_hvp_lip(
            v_lc_obs_km_s=v_lc_obs_km_s,
            h_obs_km=h_obs_km,
            b_km=b_km,
            pyroxenite_frac=float(phi),
            tp_range_c=tp_range_c,
            tp_step_c=tp_step_c,
            chi_values=chi_values,
            h_tolerance_km=h_tolerance_km,
            melting_engine=melting_engine,
            use_norm_vp=use_norm_vp,
            vp_mode=vp_mode,
            vp_bias_km_s=vp_bias_km_s,
            n_isentropic_steps=n_isentropic_steps,
            verbose=verbose,
            refine_norm_vp=refine_norm_vp,
            delta_vp_engine=delta_vp_engine,
            delta_vp_wl_kw=delta_vp_wl_kw,
            require_bulk_in_bounds=require_bulk_in_bounds,
            lithology_backend=lithology_backend,
            peridotite_lith=peridotite_lith,
            pyroxenite_lith=pyroxenite_lith,
            peridotite_h2o_wt=peridotite_h2o_wt,
            pyroxenite_h2o_wt=pyroxenite_h2o_wt,
            lithology_preset=lithology_preset,
            progress_callback=_phi_progress if progress_callback else None,
        )
        best = _best_pareto_from_result(
            res, v_lc_obs=v_lc_obs_km_s, h_obs=h_obs_km
        )
        slices.append(PhiScanSlice(pyroxenite_frac=float(phi), result=res, best_pareto=best))
    if progress_callback:
        progress_callback(total_steps, total_steps, "Φ 扫描完成")
    return PhiScanResult(
        v_lc_obs_km_s=float(v_lc_obs_km_s),
        h_obs_km=float(h_obs_km),
        b_km=float(b_km),
        melting_engine=melting_engine,
        slices=slices,
    )


def forward_point_diagnostics(
    *,
    v_lc_obs_km_s: float,
    tp_c: float,
    chi: float,
    pyroxenite_frac: float,
    b_km: float = 0.0,
    melting_engine: str = "reebox",
    vp_bias_km_s: float = -0.10,
    use_norm_vp: bool = False,
) -> dict:
    """Single (Tp, chi, phi) closure diagnostics."""
    col = forward_melting_column(
        tp_c=tp_c,
        b_km=b_km,
        chi=chi,
        pyroxenite_frac=pyroxenite_frac,
        melting_engine=melting_engine,
        vp_bias_km_s=vp_bias_km_s,
        compute_norm_vp=use_norm_vp,
    )
    vp = col.vp_bulk_norm_km_s if (use_norm_vp and col.vp_bulk_norm_km_s) else col.vp_bulk_eq1_km_s
    d_vp = delta_vp_km_s(0.75, bulk_vp_km_s=vp)
    bounds = bulk_vp_bounds(v_lc_obs_km_s, f_lower=0.5, delta_vp_fc_km_s=d_vp)
    in_b = bounds.v_bulk_lower_km_s <= vp <= bounds.v_bulk_upper_km_s
    return {
        "tp_c": tp_c,
        "chi": chi,
        "pyroxenite_frac": pyroxenite_frac,
        "h_km": col.h_km,
        "vp_bulk_km_s": vp,
        "v_lc_obs_km_s": v_lc_obs_km_s,
        "vp_upper_km_s": bounds.v_bulk_upper_km_s,
        "vp_excess_km_s": max(0.0, vp - bounds.v_bulk_upper_km_s),
        "bulk_in_bounds": in_b,
        "melt_sio2": col.pooled_melt_wt.get("SiO2"),
        "melt_mgo": col.pooled_melt_wt.get("MgO"),
        "melting_engine": melting_engine,
    }
