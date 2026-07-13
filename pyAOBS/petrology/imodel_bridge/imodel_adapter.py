"""Build ``CrustObservation`` from imodel velocity grids and interface geometry."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from petrology.imodel_bridge.export_contract import CrustObservation
from petrology.imodel_bridge.crust_geometry import (
    crust_thickness_bm_at_x,
    f_lower_from_lc_top,
    f_lower_from_polygon_at_x,
    igneous_column_at_x,
    lower_crust_relative_bounds,
    polygon_effective_x_range,
    polygon_top_depth_at_x,
)
from petrology.imodel_bridge.crust_metrics import metrics_from_depth_samples, metrics_from_lower_crust_polygon
from petrology.seismic.reference_state import correct_vp_to_reference_km_s


def crust_thickness_at_x(
    x_km: float,
    *,
    moho_z_fn: Callable[[float], float],
    igneous_top_z_fn: Callable[[float], float],
    **_: object,
) -> float:
    """Igneous column thickness H = |M − B| at profile x."""
    return crust_thickness_bm_at_x(
        x_km,
        igneous_top_z_fn=igneous_top_z_fn,
        moho_z_fn=moho_z_fn,
    )


def sample_grid_vp(
    grid_vp_fn: Callable[[float, float], float],
    x_km: float,
    z_km: float,
) -> float:
    v = float(grid_vp_fn(x_km, z_km))
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"无效 Vp @ x={x_km:g}, z={z_km:g}")
    return v


def pt_correct_sample(
    vp_insitu_km_s: float,
    depth_km: float,
    *,
    p_local_mpa_fn: Callable[[float, float], float] | None = None,
    t_local_c_fn: Callable[[float, float], float] | None = None,
    gradient_c_per_km: float = 20.0,
    surface_t_c: float = 0.0,
) -> float:
    if p_local_mpa_fn is not None and t_local_c_fn is not None:
        p = float(p_local_mpa_fn(0.0, depth_km))
        t = float(t_local_c_fn(0.0, depth_km))
        return correct_vp_to_reference_km_s(vp_insitu_km_s, p_local_mpa=p, t_local_c=t)
    from petrology.seismic.reference_state import correct_vp_depth_to_reference_km_s

    return correct_vp_depth_to_reference_km_s(
        vp_insitu_km_s,
        depth_km,
        gradient_c_per_km=gradient_c_per_km,
        surface_t_c=surface_t_c,
    )


def crust_from_polygon_selection(
    polygon_xz: Sequence[tuple[float, float]],
    *,
    grid_vp_fn: Callable[[float, float], float],
    h_whole_km: float,
    z_basement_km: float,
    z_moho_km: float,
    x_km: float | None = None,
    x_min_km: float | None = None,
    x_max_km: float | None = None,
    pt_correct: bool = True,
    p_local_mpa_fn: Callable[[float, float], float] | None = None,
    t_local_c_fn: Callable[[float, float], float] | None = None,
) -> CrustObservation:
    """Manual lower-crust polygon → harmonic V_LC; f_lower from polygon top edge at x."""

    x_lo, x_hi = polygon_effective_x_range(
        polygon_xz, x_min_km=x_min_km, x_max_km=x_max_km
    )
    if x_km is not None:
        x_use = float(np.clip(float(x_km), x_lo, x_hi))
    else:
        x_use = 0.5 * (x_lo + x_hi)
    _h, _h_lc, f_lower = f_lower_from_polygon_at_x(
        polygon_xz,
        x_use,
        z_basement_km=z_basement_km,
        z_moho_km=z_moho_km,
    )
    z_lc_top = polygon_top_depth_at_x(polygon_xz, x_use)
    if z_lc_top is None:
        raise ValueError(f"x={x_use:g} km 无法确定多边形顶边深度")

    def _wrapped_grid(x: float, z: float) -> float:
        vp = sample_grid_vp(grid_vp_fn, x, z)
        if not pt_correct:
            return vp
        return pt_correct_sample(
            vp,
            z,
            p_local_mpa_fn=p_local_mpa_fn,
            t_local_c_fn=t_local_c_fn,
        )

    obs = metrics_from_lower_crust_polygon(
        polygon_xz,
        _wrapped_grid,
        h_whole_km=h_whole_km,
        f_lower=f_lower,
        z_lower_top=z_lc_top,
        z_moho=z_moho_km,
        z_basement_km=z_basement_km,
        pt_correct=False,
        x_km=x_use,
        x_min_km=x_min_km,
        x_max_km=x_max_km,
    )
    return CrustObservation(
        h_whole_km=obs.h_whole_km,
        v_lc_km_s=obs.v_lc_km_s,
        f_lower=obs.f_lower,
        source="imodel_polygon",
        x_km=x_use,
        n_samples=obs.n_samples,
    )


def crust_from_vertical_profile(
    depths_km: Sequence[float],
    vp_insitu_km_s: Sequence[float],
    *,
    h_whole_km: float,
    f_lower: float = 0.7,
    z_surface_km: float = 0.0,
    x_km: float | None = None,
    pt_correct: bool = True,
    p_local_mpa_fn: Callable[[float, float], float] | None = None,
    t_local_c_fn: Callable[[float, float], float] | None = None,
    gradient_c_per_km: float = 20.0,
    source: str = "imodel_profile",
) -> CrustObservation:
    d = np.asarray(depths_km, dtype=float)
    v = np.asarray(vp_insitu_km_s, dtype=float)
    if pt_correct and p_local_mpa_fn is not None and t_local_c_fn is not None:
        v_corr = np.array(
            [pt_correct_sample(vi, zi, p_local_mpa_fn=p_local_mpa_fn, t_local_c_fn=t_local_c_fn) for zi, vi in zip(d, v)],
            dtype=float,
        )
        obs = metrics_from_depth_samples(
            d,
            v_corr,
            h_whole_km=h_whole_km,
            f_lower=f_lower,
            z_surface_km=z_surface_km,
            pt_correct=False,
            source=source,
            x_km=x_km,
        )
    else:
        obs = metrics_from_depth_samples(
            d,
            v,
            h_whole_km=h_whole_km,
            f_lower=f_lower,
            z_surface_km=z_surface_km,
            pt_correct=pt_correct,
            gradient_c_per_km=gradient_c_per_km,
            source=source,
            x_km=x_km,
        )
    return CrustObservation(
        h_whole_km=obs.h_whole_km,
        v_lc_km_s=obs.v_lc_km_s,
        f_lower=obs.f_lower,
        source=obs.source,
        x_km=obs.x_km,
        n_samples=obs.n_samples,
    )


def crust_from_interfaces_at_x(
    x_km: float,
    *,
    moho_z_fn: Callable[[float], float],
    igneous_top_z_fn: Callable[[float], float],
    grid_vp_fn: Callable[[float, float], float],
    lower_crust_top_z_fn: Callable[[float], float],
    n_depth_samples: int = 12,
    pt_correct: bool = True,
    p_local_mpa_fn: Callable[[float, float], float] | None = None,
    t_local_c_fn: Callable[[float, float], float] | None = None,
    **_: object,
) -> CrustObservation:
    """H from B/M; f_lower = 下地壳厚度/H from LC top interface at x."""
    z_top, z_bot, h = igneous_column_at_x(
        x_km,
        igneous_top_z_fn=igneous_top_z_fn,
        moho_z_fn=moho_z_fn,
    )
    lc_top_z = float(lower_crust_top_z_fn(x_km))
    f_used = f_lower_from_lc_top(
        z_basement_km=z_top,
        z_moho_km=z_bot,
        z_lc_top_km=lc_top_z,
    )
    z_rel_lo, z_rel_hi = lower_crust_relative_bounds(
        h,
        z_basement_km=z_top,
        lower_crust_top_z_km=lc_top_z,
    )
    margin = 0.05 * h
    z_abs_lo = z_top + z_rel_lo + margin
    z_abs_hi = z_top + z_rel_hi - margin
    if z_abs_hi <= z_abs_lo:
        raise ValueError(f"x={x_km:g} km 下地壳采样区间无效")
    depths = np.linspace(z_abs_lo, z_abs_hi, max(int(n_depth_samples), 3))
    vp_list: list[float] = []
    depth_list: list[float] = []
    for z in depths:
        try:
            vp_list.append(sample_grid_vp(grid_vp_fn, x_km, float(z)))
            depth_list.append(float(z))
        except ValueError:
            continue
    if len(depth_list) < 2:
        raise ValueError(f"x={x_km:g} km 下地壳内有效 Vp 样点不足")
    return crust_from_vertical_profile(
        depth_list,
        vp_list,
        h_whole_km=h,
        f_lower=f_used,
        z_surface_km=z_top,
        x_km=x_km,
        pt_correct=pt_correct,
        p_local_mpa_fn=p_local_mpa_fn,
        t_local_c_fn=t_local_c_fn,
        source="imodel_interfaces",
    )


def format_crust_observation_summary(obs: CrustObservation) -> str:
    lines = [
        f"H = {obs.h_whole_km:.2f} km",
        f"V_LC = {obs.v_lc_km_s:.4f} km/s @ 600 MPa, 400°C",
        f"f_lower = {obs.f_lower:.2f}",
        f"source = {obs.source}",
    ]
    if obs.x_km is not None:
        lines.append(f"x = {obs.x_km:.2f} km")
    if obs.n_samples is not None:
        lines.append(f"n_samples = {obs.n_samples}")
    if obs.v_lc_sigma_km_s is not None:
        lines.append(f"V_LC σ = {obs.v_lc_sigma_km_s:.4f} km/s")
    return "\n".join(lines)
