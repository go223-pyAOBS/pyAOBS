"""
Vp bias calibration for norm-Vp / eq.(1) forward models vs KKHS02 Step-2 bounds.

Maps raw ``Vp_bulk`` from CIPW+BurnMan or eq.(1) to the interval
``[V_LC - f_lower * dVp_fc, V_LC]`` (see ``invert.bulk_vp_bounds``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from petrology.fractionation import delta_vp_km_s
from petrology.invert import BulkVpBounds, bulk_vp_bounds

BiasTarget = Literal["bound_mid", "bound_upper", "bound_lower"]
AutoBiasStrategy = Literal["point", "sweep"]


@dataclass(frozen=True)
class VpBiasRecommendation:
    track: str
    vp_raw_km_s: float
    vp_bias_km_s: float
    vp_calibrated_km_s: float
    bounds: BulkVpBounds
    target: str
    tp_c: float
    chi: float
    note: str = ""


def bulk_bounds_for_obs(
    v_lc_obs_km_s: float,
    *,
    vp_guess_km_s: float,
    f_lower: float = 0.5,
    f_solid: float = 0.75,
    p_fc_mpa: float = 400.0,
    delta_vp_engine: str = "wl1990",
    melt_oxides_wt: dict | None = None,
    delta_vp_wl_kw: dict | None = None,
) -> BulkVpBounds:
    from petrology.fc.wl1990 import load_kinzler1997_morb_primary

    melt = melt_oxides_wt
    if melt is None and delta_vp_engine == "wl1990":
        melt = load_kinzler1997_morb_primary()["oxides_wt_percent"]
    d_vp = delta_vp_km_s(
        f_solid,
        bulk_vp_km_s=float(vp_guess_km_s),
        p_fc_mpa=float(p_fc_mpa),
        engine=delta_vp_engine,  # type: ignore[arg-type]
        melt_oxides_wt=melt,
        **dict(delta_vp_wl_kw or {}),
    )
    return bulk_vp_bounds(v_lc_obs_km_s, f_lower=f_lower, delta_vp_fc_km_s=d_vp)


def bias_for_target(
    vp_raw_km_s: float,
    bounds: BulkVpBounds,
    *,
    target: BiasTarget = "bound_mid",
    margin_km_s: float = 0.02,
) -> float:
    """Return additive bias so ``vp_raw + bias`` hits the chosen bound target."""
    raw = float(vp_raw_km_s)
    lo = float(bounds.v_bulk_lower_km_s)
    hi = float(bounds.v_bulk_upper_km_s)
    mid = 0.5 * (lo + hi)
    m = float(margin_km_s)
    if target == "bound_upper":
        goal = hi - m
    elif target == "bound_lower":
        goal = lo + m
    else:
        goal = mid
    return float(goal - raw)


def recommend_vp_bias(
    *,
    track: str,
    vp_raw_km_s: float,
    v_lc_obs_km_s: float,
    tp_c: float,
    chi: float,
    target: BiasTarget = "bound_mid",
    margin_km_s: float = 0.02,
    f_lower: float = 0.5,
    f_solid: float = 0.75,
    p_fc_mpa: float = 400.0,
) -> VpBiasRecommendation:
    bounds = bulk_bounds_for_obs(
        v_lc_obs_km_s,
        vp_guess_km_s=vp_raw_km_s,
        f_lower=f_lower,
        f_solid=f_solid,
        p_fc_mpa=p_fc_mpa,
    )
    bias = bias_for_target(
        vp_raw_km_s,
        bounds,
        target=target,
        margin_km_s=margin_km_s,
    )
    return VpBiasRecommendation(
        track=track,
        vp_raw_km_s=float(vp_raw_km_s),
        vp_bias_km_s=bias,
        vp_calibrated_km_s=float(vp_raw_km_s) + bias,
        bounds=bounds,
        target=target,
        tp_c=float(tp_c),
        chi=float(chi),
    )


def forward_raw_vp(
    *,
    tp_c: float,
    chi: float,
    b_km: float,
    melting_engine: str,
    lithology_backend: str,
    vp_mode: str,
    peridotite_chemistry: str | None,
    pyroxenite_frac: float = 0.0,
    n_isentropic_steps: int = 48,
    mineral_backend: str | None = None,
    lithology_preset: str | None = None,
) -> tuple[float, float, float]:
    """Return (vp_raw, pbar, fbar) from one forward column."""
    from .column import forward_melting_column
    from .pymelt_lithology_adapter import resolve_lithology_col_kwargs

    lith_kw = resolve_lithology_col_kwargs(
        lithology_backend=lithology_backend,
        lithology_preset=lithology_preset,
        peridotite_chemistry=peridotite_chemistry,  # type: ignore[arg-type]
    )
    if mineral_backend:
        lith_kw["mineral_backend"] = mineral_backend

    col = forward_melting_column(
        tp_c=float(tp_c),
        b_km=float(b_km),
        chi=float(chi),
        pyroxenite_frac=float(pyroxenite_frac),
        melting_engine=melting_engine,
        compute_norm_vp=vp_mode == "norm",
        vp_bias_km_s=0.0,
        n_isentropic_steps=int(n_isentropic_steps),
        **lith_kw,
    )
    if vp_mode == "norm":
        vp = col.vp_bulk_norm_km_s if col.vp_bulk_norm_km_s is not None else col.vp_bulk_eq1_km_s
    else:
        vp = col.vp_bulk_eq1_km_s
    return float(vp), float(col.pbar_gpa), float(col.fbar)


def calibrate_track_at_tp_chi(
    *,
    track: str,
    tp_c: float,
    chi: float,
    v_lc_obs_km_s: float,
    b_km: float = 0.0,
    melting_engine: str = "reebox",
    lithology_backend: str = "native",
    vp_mode: str = "norm",
    peridotite_chemistry: str | None = None,
    target: BiasTarget = "bound_mid",
    margin_km_s: float = 0.02,
    n_isentropic_steps: int = 48,
) -> VpBiasRecommendation:
    vp_raw, _, _ = forward_raw_vp(
        tp_c=tp_c,
        chi=chi,
        b_km=b_km,
        melting_engine=melting_engine,
        lithology_backend=lithology_backend,
        vp_mode=vp_mode,
        peridotite_chemistry=peridotite_chemistry,
        n_isentropic_steps=n_isentropic_steps,
    )
    return recommend_vp_bias(
        track=track,
        vp_raw_km_s=vp_raw,
        v_lc_obs_km_s=v_lc_obs_km_s,
        tp_c=tp_c,
        chi=chi,
        target=target,
        margin_km_s=margin_km_s,
    )


def default_bias_sweep_grid() -> list[float]:
    return [round(float(x), 2) for x in np.arange(-1.0, -0.04, 0.05)]


def sweep_feasible_vp_bias(
    *,
    melting_engine: str,
    lithology_backend: str,
    vp_mode: str,
    peridotite_chemistry: str | None,
    v_lc_obs_km_s: float,
    h_obs_km: float,
    b_km: float = 0.0,
    pyroxenite_frac: float = 0.0,
    tp_range_c: tuple[float, float] = (1200.0, 1500.0),
    tp_step_c: float = 25.0,
    chi_values: list[float] | None = None,
    bias_values: list[float] | None = None,
    h_tolerance_km: float = 3.0,
    n_isentropic_steps: int = 48,
    lithology_preset: str | None = None,
    delta_vp_engine: str = "wl1990",
    delta_vp_wl_kw: dict | None = None,
    refine_norm_vp: int = 0,
) -> tuple[float, int]:
    """Sweep additive Vp bias to maximize full H+Vp feasible grid points."""
    from petrology.fc.delta_vp import R3_DELTA_VP_WL_KW
    from .hvp_scan import scan_hvp_lip

    chi_values = chi_values or [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    bias_values = bias_values or default_bias_sweep_grid()
    wl_kw = dict(delta_vp_wl_kw or R3_DELTA_VP_WL_KW)
    best_bias = float(bias_values[0])
    best_n = -1
    for bias in bias_values:
        res = scan_hvp_lip(
            v_lc_obs_km_s=v_lc_obs_km_s,
            h_obs_km=h_obs_km,
            b_km=b_km,
            pyroxenite_frac=pyroxenite_frac,
            tp_range_c=tp_range_c,
            tp_step_c=tp_step_c,
            chi_values=chi_values,
            h_tolerance_km=h_tolerance_km,
            melting_engine=melting_engine,
            lithology_backend=lithology_backend,
            lithology_preset=lithology_preset,
            peridotite_chemistry=peridotite_chemistry,
            use_norm_vp=vp_mode == "norm",
            vp_mode=vp_mode,
            vp_bias_km_s=float(bias),
            n_isentropic_steps=n_isentropic_steps,
            verbose=False,
            require_bulk_in_bounds=True,
            delta_vp_engine=delta_vp_engine,
            delta_vp_wl_kw=wl_kw,
            refine_norm_vp=refine_norm_vp,
            norm_mineral_backend=wl_kw.get("mineral_backend"),
        )
        if res.n_feasible > best_n:
            best_n = int(res.n_feasible)
            best_bias = float(bias)
    return best_bias, best_n


@dataclass(frozen=True)
class AutoVpBiasResult:
    track: str
    strategy: AutoBiasStrategy
    vp_bias_km_s: float
    n_feasible: int | None = None
    point_rec: VpBiasRecommendation | None = None


def auto_calibrate_track_bias(
    *,
    track: str,
    strategy: AutoBiasStrategy = "sweep",
    tp_c: float = 1425.0,
    chi: float = 10.0,
    v_lc_obs_km_s: float = 7.0,
    h_obs_km: float = 30.0,
    b_km: float = 0.0,
    melting_engine: str = "reebox",
    lithology_backend: str = "native",
    vp_mode: str = "norm",
    peridotite_chemistry: str | None = None,
    target: BiasTarget = "bound_mid",
    margin_km_s: float = 0.02,
    tp_range_c: tuple[float, float] = (1200.0, 1500.0),
    tp_step_c: float = 25.0,
    chi_values: list[float] | None = None,
    h_tolerance_km: float = 3.0,
    n_isentropic_steps: int = 48,
) -> AutoVpBiasResult:
    """Point calibration at (Tp, chi) or sweep for max feasible H+Vp grid."""
    point_rec = calibrate_track_at_tp_chi(
        track=track,
        tp_c=tp_c,
        chi=chi,
        v_lc_obs_km_s=v_lc_obs_km_s,
        b_km=b_km,
        melting_engine=melting_engine,
        lithology_backend=lithology_backend,
        vp_mode=vp_mode,
        peridotite_chemistry=peridotite_chemistry,
        target=target,
        margin_km_s=margin_km_s,
        n_isentropic_steps=n_isentropic_steps,
    )
    if strategy == "point":
        return AutoVpBiasResult(
            track=track,
            strategy=strategy,
            vp_bias_km_s=point_rec.vp_bias_km_s,
            point_rec=point_rec,
        )

    best_bias, best_n = sweep_feasible_vp_bias(
        melting_engine=melting_engine,
        lithology_backend=lithology_backend,
        vp_mode=vp_mode,
        peridotite_chemistry=peridotite_chemistry,
        v_lc_obs_km_s=v_lc_obs_km_s,
        h_obs_km=h_obs_km,
        b_km=b_km,
        tp_range_c=tp_range_c,
        tp_step_c=tp_step_c,
        chi_values=chi_values,
        h_tolerance_km=h_tolerance_km,
        n_isentropic_steps=n_isentropic_steps,
    )
    return AutoVpBiasResult(
        track=track,
        strategy=strategy,
        vp_bias_km_s=best_bias,
        n_feasible=best_n,
        point_rec=point_rec,
    )
