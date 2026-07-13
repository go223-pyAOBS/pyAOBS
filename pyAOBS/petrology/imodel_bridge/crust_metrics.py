"""Compute H, V_LC, f_lower from imodel profiles / manual selections."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from petrology.imodel_bridge.crust_geometry import (
    crust_thickness_bm_at_x,
    f_lower_from_lc_top,
    polygon_effective_x_range,
)
from petrology.imodel_bridge.export_contract import CrustObservation
from petrology.seismic.reference_state import correct_vp_depth_to_reference_km_s
from petrology.seismic.transect import harmonic_mean_km_s, lower_crust_depth_mask


def metrics_from_depth_samples(
    depths_km: Sequence[float],
    vp_insitu_km_s: Sequence[float],
    *,
    h_whole_km: float,
    f_lower: float = 0.7,
    z_surface_km: float = 0.0,
    use_f_lower_mask: bool = True,
    pt_correct: bool = True,
    gradient_c_per_km: float = 20.0,
    surface_t_c: float = 0.0,
    source: str = "manual_polygon",
    x_km: float | None = None,
) -> CrustObservation:
    """Harmonic mean V_LC over lower-crust samples with optional P–T correction."""
    d_abs = np.asarray(depths_km, dtype=float)
    v = np.asarray(vp_insitu_km_s, dtype=float)
    h = float(h_whole_km)
    if use_f_lower_mask:
        d_col = d_abs - float(z_surface_km)
        mask = lower_crust_depth_mask(d_col, np.full_like(d_col, h), f_lower=f_lower)
        if not np.any(mask):
            raise ValueError("无样点落在下地壳区间")
    else:
        mask = np.isfinite(d_abs) & np.isfinite(v) & (v > 0.0)
        if not np.any(mask):
            raise ValueError("无有效速度样点")

    if pt_correct:
        v_ref = np.array(
            [
                correct_vp_depth_to_reference_km_s(
                    vi,
                    zi,
                    gradient_c_per_km=gradient_c_per_km,
                    surface_t_c=surface_t_c,
                )
                for zi, vi in zip(d_abs[mask], v[mask])
            ],
            dtype=float,
        )
    else:
        v_ref = v[mask]

    return CrustObservation(
        h_whole_km=h,
        v_lc_km_s=harmonic_mean_km_s(v_ref),
        f_lower=float(f_lower),
        source=source,
        x_km=x_km,
        n_samples=int(np.sum(mask)),
    )


def thickness_from_interfaces(
    x_km: float,
    *,
    igneous_top_z_fn: Callable[[float], float],
    moho_z_fn: Callable[[float], float],
) -> float:
    """Whole igneous thickness H = |M − B| at profile x."""
    return crust_thickness_bm_at_x(
        x_km,
        igneous_top_z_fn=igneous_top_z_fn,
        moho_z_fn=moho_z_fn,
    )


def metrics_from_lower_crust_polygon(
    polygon_xz: Sequence[tuple[float, float]],
    grid_vp_fn: Callable[[float, float], float],
    *,
    h_whole_km: float,
    f_lower: float | None = None,
    z_lower_top: float | None = None,
    z_moho: float | None = None,
    z_basement_km: float | None = None,
    pt_correct: bool = True,
    gradient_c_per_km: float = 20.0,
    x_km: float | None = None,
    x_min_km: float | None = None,
    x_max_km: float | None = None,
) -> CrustObservation:
    """
    Sample Vp inside a user polygon; compute harmonic V_LC.

    X 向仅在 ``polygon_effective_x_range``（多边形 ∩ 剖面）内采样。
    """
    if len(polygon_xz) < 3:
        raise ValueError("多边形至少需要 3 个顶点")

    from matplotlib.path import Path as MplPath

    poly = MplPath(np.asarray(polygon_xz, dtype=float))
    x_lo, x_hi = polygon_effective_x_range(
        polygon_xz, x_min_km=x_min_km, x_max_km=x_max_km
    )
    zs = np.array([p[1] for p in polygon_xz])
    z_lo, z_hi = float(zs.min()), float(zs.max())

    nx, nz = 24, 32
    gx = np.linspace(x_lo, x_hi, nx)
    gz = np.linspace(z_lo, z_hi, nz)
    depths: list[float] = []
    vels: list[float] = []
    for x in gx:
        for z in gz:
            if poly.contains_point((float(x), float(z))):
                depths.append(float(z))
                vels.append(float(grid_vp_fn(float(x), float(z))))

    if not depths:
        raise ValueError("多边形内无有效速度样点")

    h = float(h_whole_km)
    if f_lower is None:
        if z_lower_top is not None and z_moho is not None and z_basement_km is not None:
            f_lower = f_lower_from_lc_top(
                z_basement_km=float(z_basement_km),
                z_moho_km=float(z_moho),
                z_lc_top_km=float(z_lower_top),
            )
        else:
            raise ValueError("无法计算 f_lower：需提供 B/M 与下地壳顶界深度")

    return metrics_from_depth_samples(
        depths,
        vels,
        h_whole_km=h,
        f_lower=float(f_lower),
        use_f_lower_mask=False,
        pt_correct=pt_correct,
        gradient_c_per_km=gradient_c_per_km,
        source="manual_polygon",
        x_km=x_km,
    )
