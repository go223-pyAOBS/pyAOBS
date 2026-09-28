"""共用地形/水深色标（与位置 Map 默认「地形」色带一致）。"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def terrain_colormap_rgb(norm: np.ndarray, palette: str = "terrain") -> np.ndarray:
    """归一化高程 [0,1] → RGB(0–255)。"""
    n = np.clip(np.asarray(norm, dtype=float), 0.0, 1.0)
    pal = str(palette or "terrain").lower()
    if pal == "gray":
        stops = np.array([0.0, 1.0], dtype=float)
        colors = np.array([[30.0, 30.0, 30.0], [240.0, 240.0, 240.0]], dtype=float)
    elif pal == "gmt":
        stops = np.array([0.0, 0.18, 0.35, 0.5, 0.68, 0.85, 1.0], dtype=float)
        colors = np.array(
            [
                [20.0, 46.0, 115.0],
                [48.0, 98.0, 168.0],
                [94.0, 159.0, 201.0],
                [121.0, 173.0, 98.0],
                [171.0, 146.0, 95.0],
                [205.0, 190.0, 160.0],
                [248.0, 248.0, 248.0],
            ],
            dtype=float,
        )
    else:
        # 深海蓝 -> 浅海青 -> 陆地绿 -> 棕色 -> 近白高地（位置 Map 默认）
        stops = np.array([0.0, 0.35, 0.5, 0.78, 1.0], dtype=float)
        colors = np.array(
            [
                [26.0, 58.0, 118.0],
                [55.0, 126.0, 184.0],
                [102.0, 166.0, 86.0],
                [158.0, 122.0, 76.0],
                [240.0, 240.0, 240.0],
            ],
            dtype=float,
        )
        if pal == "terrain_r":
            colors = colors[::-1].copy()
    rgb = np.empty((n.size, 3), dtype=float)
    flat = n.reshape(-1)
    for c in range(3):
        rgb[:, c] = np.interp(flat, stops, colors[:, c])
    return rgb.reshape(n.shape + (3,))


def terrain_rgba_from_grid(
    z_grid: np.ndarray,
    *,
    palette: str = "terrain",
    shade_strength: float = 0.75,
    coast_enhance: bool = True,
    light_alt_deg: float = 45.0,
    light_az_deg: float = 315.0,
    alpha: int = 185,
) -> np.ndarray:
    """网格高程 → RGBA uint8（含 hillshade / 可选海岸增强）。"""
    zf = np.asarray(z_grid, dtype=float)
    zmin = float(np.nanpercentile(zf, 2.0))
    zmax = float(np.nanpercentile(zf, 98.0))
    span = max(1e-12, zmax - zmin)
    norm = np.clip((zf - zmin) / span, 0.0, 1.0)
    pal = str(palette or "terrain").lower()

    if pal == "gmt":
        sea = zf < 0.0
        land = ~sea
        rgb = np.zeros(zf.shape + (3,), dtype=float)
        if np.any(sea):
            z_sea = zf[sea]
            smin = float(np.nanpercentile(z_sea, 2.0))
            smax = float(np.nanpercentile(z_sea, 98.0))
            sspan = max(1e-12, smax - smin)
            s_norm = np.clip((z_sea - smin) / sspan, 0.0, 1.0)
            rgb[sea] = terrain_colormap_rgb(s_norm, palette="gmt")
        if np.any(land):
            z_land = zf[land]
            lmin = float(np.nanpercentile(z_land, 2.0))
            lmax = float(np.nanpercentile(z_land, 98.0))
            lspan = max(1e-12, lmax - lmin)
            l_norm = np.clip((z_land - lmin) / lspan, 0.0, 1.0)
            rgb[land] = terrain_colormap_rgb(l_norm, palette="terrain")
    else:
        rgb = terrain_colormap_rgb(norm, palette=pal)

    gy, gx = np.gradient(zf)
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az = np.radians(float(light_az_deg))
    alt = np.radians(float(light_alt_deg))
    hill = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    hill = np.clip((hill + 1.0) * 0.5, 0.0, 1.0)
    strength = float(np.clip(shade_strength, 0.0, 1.0))
    shade = (1.0 - 0.4 * strength) + (0.8 * strength) * hill
    rgb = np.clip(rgb * shade[..., None], 0.0, 255.0)

    edge2 = None
    if bool(coast_enhance):
        land = zf >= 0.0
        edge = np.zeros_like(land, dtype=bool)
        edge[1:, :] |= land[1:, :] != land[:-1, :]
        edge[:, 1:] |= land[:, 1:] != land[:, :-1]
        edge2 = edge.copy()
        edge2[:-1, :] |= edge[1:, :]
        edge2[1:, :] |= edge[:-1, :]
        edge2[:, :-1] |= edge[:, 1:]
        edge2[:, 1:] |= edge[:, :-1]
        coast_color = np.array([248.0, 242.0, 150.0], dtype=float)
        blend = 0.78
        rgb[edge2] = (1.0 - blend) * rgb[edge2] + blend * coast_color

    rgba = np.zeros(zf.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = rgb.astype(np.uint8)
    rgba[..., 3] = int(np.clip(alpha, 0, 255))
    if edge2 is not None:
        rgba[..., 3] = np.where(edge2, 235, rgba[..., 3])
    return rgba


def points_to_rgba_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    nx: int = 560,
    ny: int = 420,
    palette: str = "terrain",
    shade_strength: float = 0.75,
    coast_enhance: bool = True,
) -> Optional[Tuple[np.ndarray, float, float, float, float]]:
    """散点 → 着色栅格 RGBA，并返回 (rgba, xmin, xmax, ymin, ymax)。"""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[valid], y[valid], z[valid]
    if x.size == 0:
        return None
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if abs(xmax - xmin) < 1e-12 or abs(ymax - ymin) < 1e-12:
        return None
    sum_z, _, _ = np.histogram2d(
        y, x, bins=[ny, nx], range=[[ymin, ymax], [xmin, xmax]], weights=z
    )
    cnt, _, _ = np.histogram2d(y, x, bins=[ny, nx], range=[[ymin, ymax], [xmin, xmax]])
    with np.errstate(invalid="ignore", divide="ignore"):
        z_grid = sum_z / cnt
    if not np.any(np.isfinite(z_grid)):
        return None
    fill = float(np.nanmedian(z))
    z_grid = np.where(np.isfinite(z_grid), z_grid, fill)
    rgba = terrain_rgba_from_grid(
        z_grid,
        palette=palette,
        shade_strength=shade_strength,
        coast_enhance=coast_enhance,
    )
    return rgba, xmin, xmax, ymin, ymax
