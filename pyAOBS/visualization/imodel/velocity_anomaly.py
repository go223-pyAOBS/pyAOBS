"""速度异常计算（相对参考模型）。"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import xarray as xr

from pyAOBS.model_building.zeltform import EnhancedZeltModel
from pyAOBS.visualization.imodel import ProfileExtractor

# 无海底面时：横向平均排除海水/填充（Vp≈1.5、≤0）。
_DEFAULT_EXCLUDE_VP_MAX = 1.6
_DEFAULT_MIN_VELOCITY = 0.05


def _grid_velocity_2d(
    grid_data: Any,
    *,
    velocity_var: str,
    x_coord: str,
    z_coord: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_vals = np.asarray(grid_data.coords[x_coord].values, dtype=float)
    z_vals = np.asarray(grid_data.coords[z_coord].values, dtype=float)
    da = grid_data[velocity_var].transpose(z_coord, x_coord)
    v = np.asarray(da.values, dtype=float)
    if v.ndim != 2:
        raise ValueError(f"velocity grid must be 2D, got ndim={v.ndim}")
    return x_vals, z_vals, v


def _infer_velocity_var(ds: Any) -> str:
    cand = ("velocity", "vp", "vs", "vel")
    names = list(getattr(ds, "data_vars", {}).keys())
    for c in cand:
        if c in names:
            return c
    if not names:
        raise ValueError("reference grid has no data variables")
    return str(names[0])


def _solid_earth_mask(
    v: np.ndarray,
    *,
    z_vals: np.ndarray,
    seafloor_depths: Optional[np.ndarray],
    min_velocity: float,
    exclude_vp_max: float,
) -> np.ndarray:
    """参与参考平均 / 异常解释的固体地球像元。"""
    solid = np.isfinite(v) & (v > float(min_velocity))
    if seafloor_depths is not None:
        sf = np.asarray(seafloor_depths, dtype=float).reshape(1, -1)
        if sf.shape[1] != v.shape[1]:
            raise ValueError(
                f"seafloor_depths length {sf.shape[1]} != nx {v.shape[1]}"
            )
        z = np.asarray(z_vals, dtype=float).reshape(-1, 1)
        # 海面以下、海底以上：水柱，不参与平均
        solid = solid & np.isfinite(sf) & (z >= sf)
    else:
        # 无海底面时用速度阈值排除海水（及明显填充）
        solid = solid & (v > float(exclude_vp_max))
    return solid


def _interp_fill_1d(y: np.ndarray) -> np.ndarray:
    """用两侧有效点线性填补 1D 参考曲线中的空隙。"""
    out = np.asarray(y, dtype=float).copy()
    idx = np.arange(out.size)
    good = np.isfinite(out)
    if not np.any(good):
        return out
    out[~good] = np.interp(idx[~good], idx[good], out[good])
    return out


def _reference_by_absolute_depth(v: np.ndarray, solid: np.ndarray) -> np.ndarray:
    v_m = np.where(solid, v, np.nan)
    with np.errstate(all="ignore"):
        ref_1d = np.nanmean(v_m, axis=1)
    ref_1d = _interp_fill_1d(ref_1d)
    return np.broadcast_to(ref_1d.reshape(-1, 1), v.shape).copy()


def _reference_by_depth_below_seafloor(
    v: np.ndarray,
    z_vals: np.ndarray,
    seafloor_depths: np.ndarray,
    solid: np.ndarray,
) -> np.ndarray:
    """按「海底以下深度」分箱做横向平均，再映射回 (z, x)。"""
    nz, nx = v.shape
    z = np.asarray(z_vals, dtype=float).reshape(-1, 1)
    sf = np.asarray(seafloor_depths, dtype=float).reshape(1, -1)
    if sf.shape[1] != nx:
        raise ValueError(f"seafloor_depths length {sf.shape[1]} != nx {nx}")

    dz = float(np.nanmedian(np.diff(np.asarray(z_vals, dtype=float)))) if nz > 1 else 0.1
    dz = abs(dz) if np.isfinite(dz) and dz != 0 else 0.1

    bsf = z - sf  # (nz, nx)
    use = solid & np.isfinite(bsf) & (bsf >= -0.5 * dz)
    if not np.any(use):
        return np.full_like(v, np.nan)

    k = np.rint(bsf / dz).astype(np.int64)
    k_use = k[use]
    k_min = int(np.min(k_use))
    k_max = int(np.max(k_use))
    # 允许略小于 0 的箱（数值圆整）
    offset = -min(k_min, 0)
    nbin = k_max + offset + 1
    kk = k_use + offset

    sums = np.bincount(kk, weights=v[use], minlength=nbin).astype(float)
    counts = np.bincount(kk, minlength=nbin).astype(float)
    ref_bsf = np.full(nbin, np.nan, dtype=float)
    good = counts > 0
    ref_bsf[good] = sums[good] / counts[good]
    ref_bsf = _interp_fill_1d(ref_bsf)

    ref_v = np.full_like(v, np.nan)
    below = np.isfinite(sf) & np.isfinite(bsf) & (bsf >= -0.5 * dz)
    kk_all = np.clip(k + offset, 0, nbin - 1)
    ref_v[below] = ref_bsf[kk_all[below]]
    return ref_v


def depthwise_horizontal_mean_velocity_anomaly(
    grid_data: Any,
    *,
    velocity_var: str,
    x_coord: str,
    z_coord: str,
    seafloor_depths: Optional[np.ndarray] = None,
    min_velocity: float = _DEFAULT_MIN_VELOCITY,
    exclude_vp_max: float = _DEFAULT_EXCLUDE_VP_MAX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算 depth-wise horizontal mean 参考下的速度异常。

    参考场构造：
    - 排除海水/无效像元后再做横向平均（避免 Vref 被水速拉低导致 |dv/v|≫100%）。
    - 若提供 ``seafloor_depths``，按**海底以下深度**分箱平均，再映射回剖面
      （海洋测线更合理；绝对深度平均会在海底起伏处混叠水/沉积/地壳）。

    返回:
        x_vals, z_vals, v, delta_v, ref_v
        其中 delta_v = V - V_ref；水柱/无效处 ref 与 delta 为 NaN。
    """
    x_vals, z_vals, v = _grid_velocity_2d(
        grid_data,
        velocity_var=velocity_var,
        x_coord=x_coord,
        z_coord=z_coord,
    )
    solid = _solid_earth_mask(
        v,
        z_vals=z_vals,
        seafloor_depths=seafloor_depths,
        min_velocity=min_velocity,
        exclude_vp_max=exclude_vp_max,
    )

    sf = None
    if seafloor_depths is not None:
        sf_arr = np.asarray(seafloor_depths, dtype=float).reshape(-1)
        if sf_arr.size == v.shape[1] and np.any(np.isfinite(sf_arr)):
            sf = sf_arr

    if sf is not None:
        ref_v = _reference_by_depth_below_seafloor(v, z_vals, sf, solid)
    else:
        ref_v = _reference_by_absolute_depth(v, solid)

    # 水柱与无效像元不解释异常（避免相对近零 Vref 爆炸）
    ref_v = np.where(solid, ref_v, np.nan)
    delta_v = np.where(solid, v - ref_v, np.nan)
    return x_vals, z_vals, v, delta_v, ref_v


def layer_average_velocity_anomaly(
    grid_data: Any,
    *,
    velocity_var: str,
    x_coord: str,
    z_coord: str,
    zelt_model: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算 layer-average 参考下的 ΔV (km/s)。返回 x, z, v, delta_v, ref_v。"""
    if zelt_model is None:
        raise ValueError("layer-average reference requires v.in model")

    x_vals, z_vals, v = _grid_velocity_2d(
        grid_data,
        velocity_var=velocity_var,
        x_coord=x_coord,
        z_coord=z_coord,
    )
    dx = float(np.nanmedian(np.diff(x_vals))) if x_vals.size > 1 else 2.0
    dz = float(np.nanmedian(np.diff(z_vals))) if z_vals.size > 1 else 0.5
    dx = abs(dx) if np.isfinite(dx) and dx != 0 else 2.0
    dz = abs(dz) if np.isfinite(dz) and dz != 0 else 0.5

    enhanced = EnhancedZeltModel(zelt_model)
    enhanced.process_velocity_model("average_velocity")
    ref_ds = enhanced.to_xarray(dx=dx, dz=dz)
    ref_da = ref_ds["velocity"].interp(
        x=xr.DataArray(x_vals, dims=("x",)),
        z=xr.DataArray(z_vals, dims=("z",)),
        method="linear",
    )
    ref_v = np.asarray(ref_da.values, dtype=float)
    if np.isnan(ref_v).any():
        ref_nn = ref_ds["velocity"].interp(
            x=xr.DataArray(x_vals, dims=("x",)),
            z=xr.DataArray(z_vals, dims=("z",)),
            method="nearest",
        )
        ref_v = np.where(np.isnan(ref_v), np.asarray(ref_nn.values, dtype=float), ref_v)
    delta_v = v - ref_v
    return x_vals, z_vals, v, delta_v, ref_v


def external_reference_velocity_anomaly(
    grid_data: Any,
    *,
    velocity_var: str,
    x_coord: str,
    z_coord: str,
    reference_grid: Any,
    reference_velocity_var: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算外部参考网格下的 ΔV (km/s)。返回 x, z, v, delta_v, ref_v。"""
    x_vals, z_vals, v = _grid_velocity_2d(
        grid_data,
        velocity_var=velocity_var,
        x_coord=x_coord,
        z_coord=z_coord,
    )
    ref_var = reference_velocity_var or _infer_velocity_var(reference_grid)
    try:
        ref_ext = ProfileExtractor(reference_grid)
        ref_x = ref_ext.x_coord
        ref_z = ref_ext.z_coord
    except Exception:
        if x_coord in reference_grid.coords and z_coord in reference_grid.coords:
            ref_x, ref_z = x_coord, z_coord
        else:
            raise ValueError("cannot detect x/z coordinates in reference grid")

    ref_da_base = reference_grid[ref_var].transpose(ref_z, ref_x)
    ref_da = ref_da_base.interp(
        {
            ref_x: xr.DataArray(x_vals, dims=("x",)),
            ref_z: xr.DataArray(z_vals, dims=("z",)),
        },
        method="linear",
    )
    ref_v = np.asarray(ref_da.values, dtype=float)
    if np.isnan(ref_v).any():
        ref_nn = ref_da_base.interp(
            {
                ref_x: xr.DataArray(x_vals, dims=("x",)),
                ref_z: xr.DataArray(z_vals, dims=("z",)),
            },
            method="nearest",
        )
        ref_v = np.where(np.isnan(ref_v), np.asarray(ref_nn.values, dtype=float), ref_v)
    delta_v = v - ref_v
    return x_vals, z_vals, v, delta_v, ref_v
