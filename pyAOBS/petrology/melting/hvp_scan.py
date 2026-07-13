"""
Scan (Tp, chi) for LIP observations: H + V_LC bounding feasibility.

Uses ``forward_melting_column`` (norm Vp or eq.1) + KKHS02 Step-2 bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from petrology.fractionation import delta_vp_km_s
from petrology.invert import bulk_vp_bounds
from petrology.fc.delta_vp import R3_DELTA_VP_WL_KW

from .column import forward_melting_column
from .vp_fast import vp_from_melt_proxy
from ..thread_env import limit_native_parallelism

ProgressCallback = Callable[[int, int, str], None]


def _norm_vp_col_extras(
    *,
    norm_mineral_backend: str | None,
    norm_cipw_backend: str | None,
    delta_vp_engine: str,
    wl_kw: dict,
) -> dict:
    """Mineral/CIPW backends for CIPW+BurnMan norm Vp (R3 track defaults)."""
    out: dict = {}
    mineral = norm_mineral_backend
    if mineral is None and delta_vp_engine in ("wl1990", "auto"):
        mineral = wl_kw.get("mineral_backend")
    if mineral:
        out["mineral_backend"] = mineral
    if norm_cipw_backend:
        out["cipw_backend"] = norm_cipw_backend
    return out


def _merge_col_kw(
    lithology_col_kw: dict,
    *,
    norm_mineral_backend: str | None,
    norm_cipw_backend: str | None,
    delta_vp_engine: str,
    wl_kw: dict,
) -> dict:
    merged = dict(lithology_col_kw)
    merged.update(
        _norm_vp_col_extras(
            norm_mineral_backend=norm_mineral_backend,
            norm_cipw_backend=norm_cipw_backend,
            delta_vp_engine=delta_vp_engine,
            wl_kw=wl_kw,
        )
    )
    return merged


def _count_grid(tp_range_c: tuple[float, float], tp_step_c: float, chi_values: list[float]) -> int:
    tp_vals = np.arange(tp_range_c[0], tp_range_c[1] + tp_step_c * 0.5, tp_step_c)
    return len(tp_vals) * sum(1 for c in chi_values if c >= 1.0)


def _refine_norm_vp(
    points: list[HvpScanPoint],
    *,
    v_lc_obs_km_s: float,
    h_obs_km: float,
    b_km: float,
    pyroxenite_frac: float,
    melting_engine: str,
    vp_bias_km_s: float,
    h_tolerance_km: float,
    require_bulk_in_bounds: bool,
    f_solid: float,
    p_fc_mpa: float,
    f_lower: float,
    top_k: int,
    delta_vp_engine: str = "auto",
    delta_vp_wl_kw: dict | None = None,
    norm_mineral_backend: str | None = None,
    norm_cipw_backend: str | None = None,
    lithology_col_kw: dict | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[HvpScanPoint]:
    """Re-run top Pareto candidates with CIPW+BurnMan norm Vp."""
    limit_native_parallelism()
    if not points or top_k <= 0:
        return points

    def score(p: HvpScanPoint) -> float:
        vp_ex = max(0.0, p.vp_bulk_km_s - v_lc_obs_km_s)
        return (p.h_match_km / max(h_tolerance_km, 1e-6)) ** 2 + (vp_ex / 0.1) ** 2

    ranked = sorted(points, key=score)[:top_k]
    keys = {(p.tp_c, p.chi) for p in ranked}
    out: list[HvpScanPoint] = []
    refine_total = len(keys)
    refined = 0
    col_extras = _merge_col_kw(
        lithology_col_kw or {},
        norm_mineral_backend=norm_mineral_backend,
        norm_cipw_backend=norm_cipw_backend,
        delta_vp_engine=delta_vp_engine,
        wl_kw=dict(delta_vp_wl_kw or R3_DELTA_VP_WL_KW),
    )

    for pt in points:
        if (pt.tp_c, pt.chi) not in keys:
            if require_bulk_in_bounds:
                out.append(replace(pt, feasible=False))
            else:
                out.append(pt)
            continue
        refined += 1
        if progress_callback:
            progress_callback(
                refined,
                refine_total,
                f"BurnMan 精修 {refined}/{refine_total} (Tp={pt.tp_c:.0f}, χ={pt.chi:g})",
            )
        col = forward_melting_column(
            tp_c=pt.tp_c,
            b_km=b_km,
            chi=pt.chi,
            pyroxenite_frac=pyroxenite_frac,
            melting_engine=melting_engine,
            compute_norm_vp=True,
            vp_bias_km_s=vp_bias_km_s,
            n_isentropic_steps=48,
            **col_extras,
        )
        vp_bulk = col.vp_bulk_norm_km_s if col.vp_bulk_norm_km_s is not None else col.vp_bulk_eq1_km_s
        wl_kw = dict(delta_vp_wl_kw or R3_DELTA_VP_WL_KW)
        d_vp = delta_vp_km_s(
            f_solid,
            bulk_vp_km_s=vp_bulk,
            p_fc_mpa=p_fc_mpa,
            engine=delta_vp_engine,  # type: ignore[arg-type]
            melt_oxides_wt=col.pooled_melt_wt,
            **wl_kw,
        )
        bounds = bulk_vp_bounds(v_lc_obs_km_s, f_lower=f_lower, delta_vp_fc_km_s=d_vp)
        in_b = bounds.v_bulk_lower_km_s <= vp_bulk <= bounds.v_bulk_upper_km_s
        h_match = float(col.h_km - h_obs_km)
        v_lc_th = float(vp_bulk + d_vp)
        ok = abs(h_match) <= h_tolerance_km and (not require_bulk_in_bounds or in_b)
        out.append(
            HvpScanPoint(
                tp_c=pt.tp_c,
                chi=pt.chi,
                pyroxenite_frac=pyroxenite_frac,
                h_km=float(col.h_km),
                fbar=float(col.fbar),
                pbar_gpa=float(col.pbar_gpa),
                vp_bulk_km_s=float(vp_bulk),
                v_lc_theory_km_s=v_lc_th,
                bulk_in_bounds=bool(in_b),
                h_match_km=h_match,
                vlc_match_km_s=float(v_lc_th - v_lc_obs_km_s),
                feasible=bool(ok),
            )
        )
    return out


@dataclass(frozen=True)
class HvpScanPoint:
    tp_c: float
    chi: float
    pyroxenite_frac: float
    h_km: float
    fbar: float
    pbar_gpa: float
    vp_bulk_km_s: float
    v_lc_theory_km_s: float
    bulk_in_bounds: bool
    h_match_km: float
    vlc_match_km_s: float
    feasible: bool


@dataclass(frozen=True)
class HvpScanResult:
    v_lc_obs_km_s: float
    h_obs_km: float
    b_km: float
    pyroxenite_frac: float
    points: list[HvpScanPoint]
    feasible: list[HvpScanPoint]

    @property
    def n_feasible(self) -> int:
        return len(self.feasible)

    def tp_chi_ranges(self) -> dict[str, tuple[float, float] | None]:
        if not self.feasible:
            return {"tp_c": None, "chi": None}
        tp = [p.tp_c for p in self.feasible]
        chi = [p.chi for p in self.feasible]
        return {
            "tp_c": (float(min(tp)), float(max(tp))),
            "chi": (float(min(chi)), float(max(chi))),
        }


def scan_hvp_lip(
    *,
    v_lc_obs_km_s: float,
    h_obs_km: float,
    b_km: float = 0.0,
    pyroxenite_frac: float = 0.0,
    tp_range_c: tuple[float, float] = (1200.0, 1500.0),
    tp_step_c: float = 10.0,
    chi_values: list[float] | None = None,
    f_lower: float = 0.5,
    f_solid: float = 0.75,
    p_fc_mpa: float = 400.0,
    delta_vp_engine: str = "auto",
    delta_vp_wl_kw: dict | None = None,
    h_tolerance_km: float = 3.0,
    vlc_tolerance_km_s: float | None = None,
    require_bulk_in_bounds: bool = True,
    require_vlc_match: bool = False,
    use_norm_vp: bool = False,
    vp_mode: str = "auto",
    melting_engine: str = "kinzler_linear",
    vp_bias_km_s: float = -0.10,
    dfdp_per_gpa: float | None = None,
    n_isentropic_steps: int = 32,
    verbose: bool = False,
    refine_norm_vp: int = 0,
    norm_mineral_backend: str | None = None,
    norm_cipw_backend: str | None = None,
    lithology_backend: str = "native",
    peridotite_lith: str = "katz_lherzolite",
    pyroxenite_lith: str = "pertermann_g2",
    peridotite_h2o_wt: float = 0.0,
    pyroxenite_h2o_wt: float = 0.0,
    lithology_preset: str | None = None,
    peridotite_chemistry: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> HvpScanResult:
    """
    Grid search over Tp and chi; mark feasible models matching H and V_LC.

    Feasible if:
      - |H - h_obs| <= h_tolerance_km
      - ``bulk_in_bounds`` when ``require_bulk_in_bounds``
      - |V_LC_theory - V_LC_obs| <= vlc_tolerance when ``require_vlc_match``
        (off by default: V_LC_theory = V_bulk + ΔVp usually exceeds V_LC_obs)
    """
    if chi_values is None:
        chi_values = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]

    wl_kw = dict(delta_vp_wl_kw or R3_DELTA_VP_WL_KW)

    # vp_mode: auto | eq1 | melt_proxy | norm
    if vp_mode == "auto":
        if use_norm_vp:
            vp_mode = "norm"
        elif melting_engine == "reebox":
            vp_mode = "melt_proxy"
        else:
            vp_mode = "eq1"

    from .pymelt_lithology_adapter import resolve_lithology_col_kwargs

    lithology_col_kw = resolve_lithology_col_kwargs(
        lithology_backend=lithology_backend,
        lithology_preset=lithology_preset,
        peridotite_lith=peridotite_lith,
        pyroxenite_lith=pyroxenite_lith,
        peridotite_h2o_wt=peridotite_h2o_wt,
        pyroxenite_h2o_wt=pyroxenite_h2o_wt,
        peridotite_chemistry=peridotite_chemistry,  # type: ignore[arg-type]
    )
    lithology_col_kw = _merge_col_kw(
        lithology_col_kw,
        norm_mineral_backend=norm_mineral_backend,
        norm_cipw_backend=norm_cipw_backend,
        delta_vp_engine=delta_vp_engine,
        wl_kw=wl_kw,
    )

    tp_vals = np.arange(tp_range_c[0], tp_range_c[1] + tp_step_c * 0.5, tp_step_c)
    n_grid = _count_grid(tp_range_c, tp_step_c, chi_values)
    refine_steps = refine_norm_vp if (refine_norm_vp > 0 and vp_mode != "norm") else 0
    total_steps = n_grid + refine_steps
    if progress_callback:
        progress_callback(0, total_steps, f"网格扫描 0/{n_grid}")
    if verbose:
        lith_note = lithology_col_kw["lithology_backend"]
        if lith_note == "pymelt":
            lith_note = (
                f"pymelt:{lithology_col_kw['peridotite_lith']}+"
                f"{lithology_col_kw['pyroxenite_lith']}"
            )
        print(
            f"Scan grid: {n_grid} points (engine={melting_engine}, lith={lith_note}, "
            f"vp={vp_mode}, isentropic_steps={n_isentropic_steps})",
            flush=True,
        )

    points: list[HvpScanPoint] = []
    feasible: list[HvpScanPoint] = []

    col_kw: dict = {
        "vp_bias_km_s": 0.0 if vp_mode == "melt_proxy" else vp_bias_km_s,
        "compute_norm_vp": vp_mode == "norm",
        "melting_engine": melting_engine,
        "n_isentropic_steps": n_isentropic_steps,
        **lithology_col_kw,
    }
    if dfdp_per_gpa is not None:
        col_kw["dfdp_per_gpa"] = dfdp_per_gpa

    done = 0
    for tp in tp_vals:
        for chi in chi_values:
            if chi < 1.0:
                continue
            done += 1
            if progress_callback:
                progress_callback(
                    done,
                    total_steps,
                    f"网格 {done}/{n_grid} (Tp={tp:.0f}, χ={chi:g})",
                )
            elif verbose and done % 10 == 0:
                print(f"  ... {done}/{n_grid} (Tp={tp:.0f}, chi={chi:g})", flush=True)
            try:
                col = forward_melting_column(
                    tp_c=float(tp),
                    b_km=float(b_km),
                    chi=float(chi),
                    pyroxenite_frac=float(pyroxenite_frac),
                    **col_kw,
                )
            except (ValueError, RuntimeError):
                continue

            if vp_mode == "norm":
                vp_bulk = col.vp_bulk_norm_km_s if col.vp_bulk_norm_km_s is not None else col.vp_bulk_eq1_km_s
            elif vp_mode == "melt_proxy":
                vp_bulk = vp_from_melt_proxy(
                    col.pooled_melt_wt,
                    pbar_gpa=col.pbar_gpa,
                    fbar=col.fbar,
                    vp_bias_km_s=vp_bias_km_s,
                )
            else:
                vp_bulk = col.vp_bulk_eq1_km_s
            d_vp = delta_vp_km_s(
                f_solid,
                bulk_vp_km_s=vp_bulk,
                p_fc_mpa=p_fc_mpa,
                engine=delta_vp_engine,  # type: ignore[arg-type]
                melt_oxides_wt=col.pooled_melt_wt,
                **wl_kw,
            )
            bounds = bulk_vp_bounds(v_lc_obs_km_s, f_lower=f_lower, delta_vp_fc_km_s=d_vp)
            in_bounds = bounds.v_bulk_lower_km_s <= vp_bulk <= bounds.v_bulk_upper_km_s
            v_lc_th = float(vp_bulk + d_vp)
            h_match = float(col.h_km - h_obs_km)
            vlc_match = float(v_lc_th - v_lc_obs_km_s)

            ok = abs(h_match) <= h_tolerance_km
            if require_bulk_in_bounds:
                ok = ok and in_bounds
            if require_vlc_match and vlc_tolerance_km_s is not None:
                ok = ok and abs(vlc_match) <= vlc_tolerance_km_s

            pt = HvpScanPoint(
                tp_c=float(tp),
                chi=float(chi),
                pyroxenite_frac=float(pyroxenite_frac),
                h_km=float(col.h_km),
                fbar=float(col.fbar),
                pbar_gpa=float(col.pbar_gpa),
                vp_bulk_km_s=float(vp_bulk),
                v_lc_theory_km_s=v_lc_th,
                bulk_in_bounds=bool(in_bounds),
                h_match_km=h_match,
                vlc_match_km_s=vlc_match,
                feasible=bool(ok),
            )
            points.append(pt)
            if ok:
                feasible.append(pt)

    if refine_steps > 0:
        if verbose:
            print(
                f"Refining top {refine_norm_vp} with norm Vp "
                f"({lithology_col_kw.get('mineral_backend', 'default')})...",
                flush=True,
            )
        if progress_callback:
            progress_callback(n_grid, total_steps, f"BurnMan 精修 (top {refine_norm_vp})…")

        def _refine_progress(r_done: int, r_total: int, msg: str) -> None:
            if progress_callback:
                progress_callback(n_grid + r_done, total_steps, msg)

        points = _refine_norm_vp(
            points,
            v_lc_obs_km_s=v_lc_obs_km_s,
            h_obs_km=h_obs_km,
            b_km=b_km,
            pyroxenite_frac=pyroxenite_frac,
            melting_engine=melting_engine,
            vp_bias_km_s=vp_bias_km_s,
            h_tolerance_km=h_tolerance_km,
            require_bulk_in_bounds=require_bulk_in_bounds,
            f_solid=f_solid,
            p_fc_mpa=p_fc_mpa,
            f_lower=f_lower,
            delta_vp_engine=delta_vp_engine,
            delta_vp_wl_kw=wl_kw,
            norm_mineral_backend=norm_mineral_backend,
            norm_cipw_backend=norm_cipw_backend,
            top_k=refine_norm_vp,
            lithology_col_kw=lithology_col_kw,
            progress_callback=_refine_progress if progress_callback else None,
        )
        feasible = [p for p in points if p.feasible]

    if progress_callback:
        progress_callback(total_steps, total_steps, "扫描完成")

    return HvpScanResult(
        v_lc_obs_km_s=float(v_lc_obs_km_s),
        h_obs_km=float(h_obs_km),
        b_km=float(b_km),
        pyroxenite_frac=float(pyroxenite_frac),
        points=points,
        feasible=feasible,
    )
