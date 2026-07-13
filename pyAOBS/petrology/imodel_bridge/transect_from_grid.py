"""Build along-transect samples from imodel velocity grid + interfaces."""

from __future__ import annotations

from typing import Callable, Literal, Sequence

import numpy as np
from matplotlib.path import Path as MplPath

from petrology.imodel_bridge.crust_geometry import (
    f_lower_from_lc_top,
    f_lower_from_polygon_at_x,
    igneous_column_at_x,
    lower_crust_relative_bounds,
    polygon_effective_x_range,
    polygon_top_depth_at_x,
)
from petrology.imodel_bridge.imodel_adapter import sample_grid_vp
from petrology.seismic.transect import TransectSample, TransectWindow, aggregate_transect_windows, prepare_samples

LowerCrustMode = Literal["interface", "polygon"]


def build_transect_rows_from_grid(
    *,
    x_values_km: Sequence[float],
    grid_vp_fn: Callable[[float, float], float],
    moho_z_fn: Callable[[float], float],
    igneous_top_z_fn: Callable[[float], float],
    lower_crust_top_z_fn: Callable[[float], float],
    n_depth_samples: int = 8,
) -> list[dict[str, float]]:
    """
    Sample lower crust along profile x positions.

    ``depth_km`` relative to B; ``f_lower`` computed at each x from LC top interface.
    """
    rows: list[dict[str, float]] = []
    for x in x_values_km:
        x = float(x)
        try:
            z_top, z_bot, h = igneous_column_at_x(
                x,
                igneous_top_z_fn=igneous_top_z_fn,
                moho_z_fn=moho_z_fn,
            )
            lc_top_z = float(lower_crust_top_z_fn(x))
            f_at_x = f_lower_from_lc_top(
                z_basement_km=z_top,
                z_moho_km=z_bot,
                z_lc_top_km=lc_top_z,
            )
            z_rel_lo, z_rel_hi = lower_crust_relative_bounds(
                h,
                z_basement_km=z_top,
                lower_crust_top_z_km=lc_top_z,
            )
        except ValueError:
            continue
        margin = 0.04 * h
        z_rels = np.linspace(
            z_rel_lo + margin,
            z_rel_hi - margin,
            max(int(n_depth_samples), 3),
        )
        for z_rel in z_rels:
            z_abs = z_top + float(z_rel)
            try:
                vp = sample_grid_vp(grid_vp_fn, x, z_abs)
            except ValueError:
                continue
            rows.append(
                {
                    "distance_km": x,
                    "depth_km": float(z_rel),
                    "vp_insitu_km_s": vp,
                    "h_whole_km": h,
                    "f_lower": f_at_x,
                }
            )
    return rows


def build_transect_rows_from_polygon(
    *,
    polygon_xz: Sequence[tuple[float, float]],
    grid_vp_fn: Callable[[float, float], float],
    moho_z_fn: Callable[[float], float],
    igneous_top_z_fn: Callable[[float], float],
    x_min_km: float,
    x_max_km: float,
    n_depth_samples: int = 8,
) -> list[dict[str, float]]:
    """Hand-drawn polygon mask; f_lower from polygon top edge depth at each x."""

    if len(polygon_xz) < 3:
        raise ValueError("多边形至少需要 3 个顶点")
    poly = MplPath(np.asarray(polygon_xz, dtype=float))
    x_lo, x_hi = polygon_effective_x_range(
        polygon_xz, x_min_km=x_min_km, x_max_km=x_max_km
    )

    gx = np.linspace(x_lo, x_hi, max(16, int((x_hi - x_lo) / 2.0) + 1))

    rows: list[dict[str, float]] = []
    for x in gx:
        try:
            z_top, z_bot, h = igneous_column_at_x(
                float(x),
                igneous_top_z_fn=igneous_top_z_fn,
                moho_z_fn=moho_z_fn,
            )
            _h, _h_lc, f_at_x = f_lower_from_polygon_at_x(
                polygon_xz,
                float(x),
                z_basement_km=z_top,
                z_moho_km=z_bot,
            )
            z_lc_top = polygon_top_depth_at_x(polygon_xz, float(x))
            if z_lc_top is None:
                continue
            z_rel_lo, z_rel_hi = lower_crust_relative_bounds(
                h,
                z_basement_km=z_top,
                lower_crust_top_z_km=float(z_lc_top),
            )
        except ValueError:
            continue
        margin = 0.04 * h
        z_rels = np.linspace(
            z_rel_lo + margin,
            z_rel_hi - margin,
            max(int(n_depth_samples), 3),
        )
        for z_rel in z_rels:
            z_abs = z_top + float(z_rel)
            if not poly.contains_point((float(x), float(z_abs))):
                continue
            if z_abs < z_top or z_abs > z_top + h:
                continue
            try:
                vp = sample_grid_vp(grid_vp_fn, float(x), float(z_abs))
            except ValueError:
                continue
            rows.append(
                {
                    "distance_km": float(x),
                    "depth_km": float(z_rel),
                    "vp_insitu_km_s": vp,
                    "h_whole_km": h,
                    "f_lower": f_at_x,
                }
            )
    return rows


def imodel_grid_to_transect_windows(
    *,
    x_min_km: float,
    x_max_km: float,
    dx_km: float,
    grid_vp_fn: Callable[[float, float], float],
    moho_z_fn: Callable[[float], float],
    igneous_top_z_fn: Callable[[float], float],
    lower_crust_mode: LowerCrustMode,
    lower_crust_top_z_fn: Callable[[float], float] | None = None,
    polygon_xz: Sequence[tuple[float, float]] | None = None,
    n_depth_samples: int = 8,
    window_half_width_km: float = 10.0,
    distance_step_km: float = 10.0,
    h_min_km: float = 15.0,
    delta_vp_max_km_s: float = 0.15,
    n_mc: int = 100,
    mc_depth_draws_per_station: int = 1,
    vp_pick_sigma_km_s: float = 0.0,
    gradient_c_per_km: float = 20.0,
    rng: np.random.Generator | None = None,
) -> tuple[list[dict[str, float]], list[TransectSample], list[TransectWindow]]:
    xs = np.arange(float(x_min_km), float(x_max_km) + 0.5 * float(dx_km), float(dx_km))
    if xs.size < 1:
        raise ValueError("X 采样为空")

    if lower_crust_mode == "polygon":
        if not polygon_xz:
            raise ValueError("多边形模式需要先在主图完成 Polygon Selection")
        rows = build_transect_rows_from_polygon(
            polygon_xz=polygon_xz,
            grid_vp_fn=grid_vp_fn,
            moho_z_fn=moho_z_fn,
            igneous_top_z_fn=igneous_top_z_fn,
            x_min_km=float(x_min_km),
            x_max_km=float(x_max_km),
            n_depth_samples=n_depth_samples,
        )
    else:
        if lower_crust_top_z_fn is None:
            raise ValueError("请选择下地壳顶界 v.in 界面")
        rows = build_transect_rows_from_grid(
            x_values_km=xs,
            grid_vp_fn=grid_vp_fn,
            moho_z_fn=moho_z_fn,
            igneous_top_z_fn=igneous_top_z_fn,
            lower_crust_top_z_fn=lower_crust_top_z_fn,
            n_depth_samples=n_depth_samples,
        )

    if not rows:
        raise ValueError("未生成有效剖面样点（检查 B/M、下地壳顶界与速度网格）")

    f_vals = [float(r["f_lower"]) for r in rows if "f_lower" in r]
    f_mean = float(np.mean(f_vals)) if f_vals else 0.5

    dist_range: tuple[float, float] | None = None
    if lower_crust_mode == "polygon" and polygon_xz:
        dist_range = polygon_effective_x_range(
            polygon_xz, x_min_km=x_min_km, x_max_km=x_max_km
        )

    samples = prepare_samples(rows, f_lower=f_mean, gradient_c_per_km=gradient_c_per_km)
    windows = aggregate_transect_windows(
        samples,
        window_half_width_km=window_half_width_km,
        distance_step_km=distance_step_km,
        distance_range_km=dist_range,
        h_min_km=h_min_km,
        f_lower=f_mean,
        delta_vp_max_km_s=delta_vp_max_km_s,
        n_mc=n_mc,
        mc_depth_draws_per_station=mc_depth_draws_per_station,
        vp_pick_sigma_km_s=vp_pick_sigma_km_s,
        rng=rng,
    )
    return rows, samples, windows
