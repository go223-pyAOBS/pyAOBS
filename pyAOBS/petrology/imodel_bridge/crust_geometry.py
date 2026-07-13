"""Igneous column (B→M) geometry and lower-crust depth bounds."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def igneous_column_at_x(
    x_km: float,
    *,
    igneous_top_z_fn: Callable[[float], float],
    moho_z_fn: Callable[[float], float],
) -> tuple[float, float, float]:
    """
    Whole igneous crust column at profile x.

    Top = B (沉积基底 / 火成柱顶); bottom = M (Moho).
    Returns ``(z_top, z_bottom, h)`` with depths positive downward.
    """
    z_top = float(igneous_top_z_fn(float(x_km)))
    z_bot = float(moho_z_fn(float(x_km)))
    if not np.isfinite(z_top) or not np.isfinite(z_bot):
        raise ValueError(f"x={x_km:g} km 处 B/M 界面深度无效")
    h = abs(z_bot - z_top)
    if h <= 0.0:
        raise ValueError(
            f"x={x_km:g} km 处壳厚 H≤0 (B={z_top:g}, M={z_bot:g})"
        )
    return z_top, z_bot, h


def crust_thickness_bm_at_x(
    x_km: float,
    *,
    igneous_top_z_fn: Callable[[float], float],
    moho_z_fn: Callable[[float], float],
) -> float:
    """H = |M − B| at x."""
    _z_top, _z_bot, h = igneous_column_at_x(
        x_km, igneous_top_z_fn=igneous_top_z_fn, moho_z_fn=moho_z_fn
    )
    return h


def f_lower_from_lc_top(
    *,
    z_basement_km: float,
    z_moho_km: float,
    z_lc_top_km: float,
) -> float:
    """
    f_lower = 下地壳厚度 / 全火成地壳厚度 = (M − z_lc_top) / (M − B).
    """
    h = abs(float(z_moho_km) - float(z_basement_km))
    if h <= 0.0:
        raise ValueError("全地壳厚度 H 必须为正")
    z_lc = float(z_lc_top_km)
    z_b = float(z_basement_km)
    z_m = float(z_moho_km)
    if not (min(z_b, z_m) <= z_lc <= max(z_b, z_m)):
        raise ValueError(
            f"下地壳顶界 z={z_lc:g} 须落在 B={z_b:g} 与 M={z_m:g} 之间"
        )
    h_lc = abs(z_m - z_lc)
    if h_lc <= 0.0:
        raise ValueError("下地壳厚度为零")
    return float(np.clip(h_lc / h, 0.05, 1.0))


def f_lower_from_thicknesses(*, h_whole_km: float, h_lower_km: float) -> float:
    """f_lower = 下地壳厚度 / 全火成地壳厚度。"""
    h = float(h_whole_km)
    h_lc = float(h_lower_km)
    if h <= 0.0:
        raise ValueError("全地壳厚度 H 必须为正")
    if h_lc <= 0.0:
        raise ValueError("下地壳厚度必须为正")
    return float(np.clip(h_lc / h, 0.05, 1.0))


def polygon_effective_x_range(
    polygon_xz: Sequence[tuple[float, float]],
    *,
    x_min_km: float | None = None,
    x_max_km: float | None = None,
) -> tuple[float, float]:
    """
    多边形模式 X 计算范围：多边形与剖面/网格 X 范围的交集（取最窄）。

    若未提供有效剖面范围 (x_max > x_min)，则仅用多边形顶点 X 包络。
    """
    if len(polygon_xz) < 3:
        raise ValueError("多边形至少需要 3 个顶点")
    xs = [float(p[0]) for p in polygon_xz]
    x_lo, x_hi = min(xs), max(xs)
    if (
        x_min_km is not None
        and x_max_km is not None
        and float(x_max_km) > float(x_min_km)
    ):
        x_lo = max(x_lo, float(x_min_km))
        x_hi = min(x_hi, float(x_max_km))
    if x_hi <= x_lo:
        prof = (
            f"[{float(x_min_km):g}, {float(x_max_km):g}]"
            if x_min_km is not None and x_max_km is not None
            else "未设"
        )
        raise ValueError(
            f"多边形与剖面 X 范围无交集 "
            f"(多边形 [{min(xs):g}, {max(xs):g}], 剖面 {prof})"
        )
    return x_lo, x_hi


def polygon_top_depth_at_x(
    polygon_xz: Sequence[tuple[float, float]],
    x_km: float,
    *,
    n_z_samples: int = 256,
) -> float | None:
    """
    多边形在 x 处的顶边深度（上边界，深度向下为正）。

    取该 x 竖线与多边形交集的最浅 z，而非全多边形顶点中的全局最浅点。
    """
    if len(polygon_xz) < 3:
        return None
    x_km = float(x_km)
    pts = [(float(p[0]), float(p[1])) for p in polygon_xz]
    xs = [p[0] for p in pts]
    zs = [p[1] for p in pts]
    x_lo, x_hi = min(xs), max(xs)
    if x_km < x_lo - 1e-9 or x_km > x_hi + 1e-9:
        return None

    top_z: float | None = None

    # 与多边形各边的交点（竖线 x = x_km）
    n = len(pts)
    for i in range(n):
        x0, z0 = pts[i]
        x1, z1 = pts[(i + 1) % n]
        dx = x1 - x0
        if abs(dx) < 1e-12:
            if abs(x_km - x0) < 1e-9:
                for z in (z0, z1):
                    top_z = z if top_z is None else min(top_z, z)
            continue
        t = (x_km - x0) / dx
        if -1e-12 <= t <= 1.0 + 1e-12:
            t = float(np.clip(t, 0.0, 1.0))
            z_cross = z0 + t * (z1 - z0)
            top_z = z_cross if top_z is None else min(top_z, z_cross)

    from matplotlib.path import Path as MplPath

    poly = MplPath(np.asarray(pts, dtype=float))
    z_lo, z_hi = min(zs), max(zs)
    for z in np.linspace(z_lo, z_hi, max(int(n_z_samples), 32)):
        if poly.contains_point((x_km, float(z))):
            top_z = float(z) if top_z is None else min(top_z, float(z))
            break

    return top_z


def f_lower_from_polygon_at_x(
    polygon_xz: Sequence[tuple[float, float]],
    x_km: float,
    *,
    z_basement_km: float,
    z_moho_km: float,
) -> tuple[float, float, float]:
    """
    多边形模式 f_lower：

    - 全地壳 H = |B − M|
    - 下地壳 h_lc = |多边形顶边(x) − M|
    - f_lower = h_lc / H
    """
    z_top = polygon_top_depth_at_x(polygon_xz, x_km)
    if z_top is None:
        raise ValueError(f"x={x_km:g} km 不在多边形水平范围内")
    h_whole = abs(float(z_moho_km) - float(z_basement_km))
    h_lc = abs(float(z_moho_km) - float(z_top))
    f_lower = f_lower_from_thicknesses(h_whole_km=h_whole, h_lower_km=h_lc)
    return h_whole, h_lc, f_lower


def lower_crust_relative_bounds(
    h_whole_km: float,
    *,
    z_basement_km: float,
    lower_crust_top_z_km: float,
) -> tuple[float, float]:
    """
    Lower crust depth interval relative to igneous top (B).

    Returns ``(z_rel_lo, z_rel_hi)`` within ``[0, H]``.
    """
    h = float(h_whole_km)
    if h <= 0.0:
        raise ValueError("H 必须为正")
    z_lo = float(lower_crust_top_z_km) - float(z_basement_km)
    z_lo = float(np.clip(z_lo, 0.0, h))
    z_hi = h
    if z_hi - z_lo <= 0.0:
        raise ValueError("下地壳厚度为零（检查下地壳顶界界面）")
    return z_lo, z_hi


def effective_f_lower(h_whole_km: float, z_rel_lo: float) -> float:
    """f_lower = (H − z_lc_top_rel) / H from relative lower-crust top."""
    h = float(h_whole_km)
    if h <= 0.0:
        raise ValueError("H 必须为正")
    return float(np.clip((h - float(z_rel_lo)) / h, 0.05, 1.0))
