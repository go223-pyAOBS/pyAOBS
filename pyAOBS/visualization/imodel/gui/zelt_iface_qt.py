"""v.in / Zelt 界面深度插值与海底自动检测（与 model_surface 逻辑一致）。"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def interpolate_interface_depth_at_x(zelt_model: Any, iface_idx: int, x: float) -> float:
    """单层界面在给定 x 处的深度 (km)，失败返回 0."""
    try:
        x_coords, z_coords = zelt_model.get_layer_geometry(iface_idx)
        x_array = np.asarray(x_coords, dtype=float)
        z_array = np.asarray(z_coords, dtype=float)
        valid = ~(np.isnan(x_array) | np.isnan(z_array))
        if not np.any(valid):
            return 0.0
        xv = x_array[valid]
        zv = z_array[valid]
        if len(xv) > 1:
            return float(np.interp(float(x), xv, zv))
        return float(zv[0])
    except Exception:
        return 0.0


def interpolate_interface_depths_on_grid(zelt_model: Any, iface_idx: int, x_vals: np.ndarray) -> np.ndarray:
    """界面在所有网格 x 上的深度数组。"""
    try:
        x_coords, z_coords = zelt_model.get_layer_geometry(iface_idx)
        x_array = np.asarray(x_coords, dtype=float)
        z_array = np.asarray(z_coords, dtype=float)
        valid = ~(np.isnan(x_array) | np.isnan(z_array))
        if not np.any(valid):
            return np.zeros_like(x_vals, dtype=float)
        xv = x_array[valid]
        zv = z_array[valid]
        return np.interp(np.asarray(x_vals, dtype=float), xv, zv)
    except Exception:
        return np.zeros_like(x_vals, dtype=float)


def detect_seafloor_depths_from_velocity(
    x_vals: np.ndarray,
    z_vals: np.ndarray,
    velocity_data: np.ndarray,
    *,
    vp_water_value: float = 1.5,
    vp_water_tolerance: float = 0.1,
) -> np.ndarray:
    """与 Tk ModelSurfaceMixin._detect_seafloor_from_velocity 一致。"""
    nz, nx = velocity_data.shape
    seafloor_depths = np.zeros(nx, dtype=float)
    for j in range(nx):
        vp_col = velocity_data[:, j]
        water_mask = np.abs(vp_col - vp_water_value) <= vp_water_tolerance
        rock_mask = ~water_mask
        if np.any(rock_mask):
            first_rock_idx = int(np.where(rock_mask)[0][0])
            if first_rock_idx > 0:
                seafloor_depths[j] = float(z_vals[first_rock_idx - 1])
            else:
                seafloor_depths[j] = float(z_vals[0])
        else:
            seafloor_depths[j] = float(z_vals[0])
    return seafloor_depths


def compute_seafloor_depth_map(
    grid_data,
    profile_extractor,
    *,
    zelt_model: Optional[Any],
    seafloor_interface_idx: Optional[int],
) -> np.ndarray:
    """返回与网格 x 列对齐的海底深度数组。"""
    x_vals = np.asarray(grid_data.coords[profile_extractor.x_coord].values, dtype=float)
    z_vals = np.asarray(grid_data.coords[profile_extractor.z_coord].values, dtype=float)
    vel_var = profile_extractor.velocity_var
    velocity_data = np.asarray(grid_data[vel_var].values, dtype=float)

    if zelt_model is not None and seafloor_interface_idx is not None:
        return interpolate_interface_depths_on_grid(zelt_model, int(seafloor_interface_idx), x_vals)

    return detect_seafloor_depths_from_velocity(x_vals, z_vals, velocity_data)


def plot_zelt_interfaces_on_axes(
    ax,
    zelt_model: Any,
    *,
    basement_interface_idx: Optional[int] = None,
    seafloor_interface_idx: Optional[int] = None,
    moho_interface_idx: Optional[int] = None,
) -> list:
    """在剖面轴上绘制 v.in 层界面（与 Tk ``PropertyPanelMixin._plot_prop_interfaces`` 一致）。"""
    artists: list = []
    if zelt_model is None:
        return artists
    try:
        n = len(zelt_model.depth_nodes)
        for i in range(n):
            x_coords, z_coords = zelt_model.get_layer_geometry(i)
            vm = ~(np.isnan(x_coords) | np.isnan(z_coords))
            if not np.any(vm):
                continue
            xv = np.asarray(x_coords, dtype=float)[vm]
            zv = np.asarray(z_coords, dtype=float)[vm]
            is_basement = (
                basement_interface_idx is not None and i == int(basement_interface_idx)
            )
            is_seafloor = (
                seafloor_interface_idx is not None and i == int(seafloor_interface_idx)
            )
            is_moho = moho_interface_idx is not None and i == int(moho_interface_idx)
            if i == n - 1:
                line, = ax.plot(xv, zv, "k--", linewidth=2.0, alpha=0.8)
            elif is_moho:
                line, = ax.plot(xv, zv, "g-", linewidth=2.5, alpha=0.9)
            elif is_basement:
                line, = ax.plot(xv, zv, "r-", linewidth=2.5, alpha=0.9)
            elif is_seafloor:
                line, = ax.plot(xv, zv, "b-", linewidth=2.5, alpha=0.9)
            else:
                line, = ax.plot(xv, zv, "k-", linewidth=1.5, alpha=0.7)
            artists.append(line)
    except Exception:
        pass
    return artists
