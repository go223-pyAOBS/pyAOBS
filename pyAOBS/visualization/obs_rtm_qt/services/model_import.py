# -*- coding: utf-8 -*-
"""
速度模型导入：复用 imodel ``load_velocity_grid``（v.in / .grd / .nc）
与 tomo2d ``SlownessMesh2D``（smesh），并写出 Madagascar ``.rsf``。
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .rsf_io import parse_rsf_header
from .velocity import read_vel_rsf, write_vel_rsf
from ..project import GridParams

# 预览栅格下限（km）：细于此时 to_xarray 双重循环极易卡死 UI
_PREVIEW_MIN_DX_KM = 0.5
_PREVIEW_MIN_DZ_KM = 0.25
_PREVIEW_MAX_NX = 800
_PREVIEW_MAX_NZ = 450


def preview_raster_spacing(
    vin_dx_km: float, vin_dz_km: float
) -> Tuple[float, float]:
    """预览用粗网格；转换/成像仍用用户设定的 vin_dx/dz。"""
    return (
        max(float(vin_dx_km), _PREVIEW_MIN_DX_KM),
        max(float(vin_dz_km), _PREVIEW_MIN_DZ_KM),
    )


def downsample_vel_meta(
    vel: np.ndarray,
    meta: Dict[str, float],
    *,
    max_nx: int = _PREVIEW_MAX_NX,
    max_nz: int = _PREVIEW_MAX_NZ,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """显示用抽稀，避免超大 ImageItem / Contours。"""
    vel = np.asarray(vel, dtype=np.float32)
    nz, nx = vel.shape
    sx = max(1, int(math.ceil(nx / float(max_nx))))
    sz = max(1, int(math.ceil(nz / float(max_nz))))
    if sx == 1 and sz == 1:
        return vel, meta
    v2 = vel[::sz, ::sx].copy()
    m2 = dict(meta)
    m2["d1"] = float(meta["d1"]) * sz
    m2["d2"] = float(meta["d2"]) * sx
    m2["n1"] = float(v2.shape[0])
    m2["n2"] = float(v2.shape[1])
    return v2, m2


def _pick_vel_var(ds) -> str:
    for name in ("velocity", "vp", "Vs", "vs", "V", "v"):
        if name in ds:
            return name
    # 第一个二维数据变量
    for name, da in ds.data_vars.items():
        if getattr(da, "ndim", 0) == 2:
            return name
    raise ValueError("数据集中未找到二维速度变量")


def _coord_1d(ds, candidates) -> np.ndarray:
    for c in candidates:
        if c in ds.coords:
            return np.asarray(ds.coords[c].values, dtype=float)
        if c in ds.dims:
            return np.asarray(ds[c].values, dtype=float)
    # 回退：按维度顺序
    dims = list(ds.dims)
    if len(dims) >= 2:
        # 常见 (z,x)
        return np.asarray(ds.coords[dims[0]].values, dtype=float)
    raise ValueError("无法识别坐标轴")


def dataset_to_vel_meta(ds) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    xr.Dataset → vel(nz,nx) float32 + meta(o1,d1,n1,o2,d2,n2)
    约定 n1=z 向下, n2=x。
    """
    vname = _pick_vel_var(ds)
    da = ds[vname]
    # 规范为 (z, x)
    dims = list(da.dims)
    z_dim = x_dim = None
    for d in dims:
        dl = d.lower()
        if dl in ("z", "depth", "y") and z_dim is None:
            z_dim = d
        if dl in ("x", "distance", "offset") and x_dim is None:
            x_dim = d
    if z_dim is None or x_dim is None:
        if len(dims) >= 2:
            z_dim, x_dim = dims[0], dims[1]
        else:
            raise ValueError("速度维度不足 2")
    da = da.transpose(z_dim, x_dim)
    vel = np.asarray(da.values, dtype=np.float32)
    # NaN → 按列线性插值（向量化索引，避免纯 Python 过慢）
    if np.isnan(vel).any():
        vel = vel.copy()
        nz, nx = vel.shape
        z_idx = np.arange(nz, dtype=float)
        for j in range(nx):
            col = vel[:, j]
            m = np.isnan(col)
            if not m.any():
                continue
            if m.all():
                col[:] = 1.5
            else:
                good = ~m
                col[m] = np.interp(z_idx[m], z_idx[good], col[good])
            vel[:, j] = col

    z = np.asarray(da.coords[z_dim].values, dtype=float)
    x = np.asarray(da.coords[x_dim].values, dtype=float)
    if z.size < 2 or x.size < 2:
        raise ValueError("网格过稀")
    # 保证 z 向下增大
    if z[0] > z[-1]:
        z = z[::-1]
        vel = vel[::-1, :]
    d1 = float(z[1] - z[0])
    d2 = float(x[1] - x[0])
    meta = {
        "o1": float(z[0]),
        "d1": d1,
        "n1": float(vel.shape[0]),
        "o2": float(x[0]),
        "d2": d2,
        "n2": float(vel.shape[1]),
    }
    return vel, meta


def _looks_like_smesh(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            parts = f.readline().split()
        if len(parts) < 4:
            return False
        nx, nz = int(float(parts[0])), int(float(parts[1]))
        return nx > 1 and nz > 1
    except Exception:
        return False


def _vel_meta_to_dataset(vel: np.ndarray, meta: Dict[str, float]) -> "object":
    import xarray as xr

    nz, nx = int(meta["n1"]), int(meta["n2"])
    z = float(meta["o1"]) + np.arange(nz, dtype=float) * float(meta["d1"])
    x = float(meta["o2"]) + np.arange(nx, dtype=float) * float(meta["d2"])
    return xr.Dataset(
        {"velocity": (("z", "x"), np.asarray(vel, dtype=float))},
        coords={"z": z, "x": x},
    )


def load_any_velocity_dataset(
    path: str,
    *,
    vin_dx_km: float = 0.025,
    vin_dz_km: float = 0.025,
) -> Tuple[object, object, str]:
    """
    加载为 imodel 同款 xr.Dataset。
    返回 (ds, zelt_or_None, kind)；kind: rsf | vin | grid | smesh
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(path)
    suf = p.suffix.lower()
    name = p.name.lower()

    from pyAOBS.visualization.imodel.gui.model_load import (
        is_vin_file_by_content,
        is_vin_path,
        load_velocity_grid,
    )

    if suf == ".rsf":
        vel, meta = read_vel_rsf(str(p))
        return _vel_meta_to_dataset(vel, meta), None, "rsf"

    is_vin = is_vin_path(p) or is_vin_file_by_content(str(p))
    try_smesh = suf in (".smesh", ".mesh") or (
        (not is_vin) and suf == "" and name not in ("v.in",) and _looks_like_smesh(str(p))
    )
    if try_smesh:
        from pyAOBS.model_building.tomoform import SlownessMesh2D

        mesh = SlownessMesh2D.from_file(str(p))
        ds = mesh.to_xarray(dx=float(vin_dx_km), dz=float(vin_dz_km))
        return ds, None, "smesh"

    ds, zelt = load_velocity_grid(
        str(p), vin_dx_km=float(vin_dx_km), vin_dz_km=float(vin_dz_km)
    )
    kind = "vin" if is_vin else "grid"
    return ds, zelt, kind


def load_velocity_for_preview(
    path: str,
    *,
    vin_dx_km: float = 0.025,
    vin_dz_km: float = 0.025,
) -> Tuple[object, object, str, float, float]:
    """
    专供 GUI 预览：粗网格栅格化 + 抽稀。
    返回 (ds, zelt_or_None, kind, used_dx, used_dz)
    """
    pdx, pdz = preview_raster_spacing(vin_dx_km, vin_dz_km)
    ds, zelt, kind = load_any_velocity_dataset(
        path, vin_dx_km=pdx, vin_dz_km=pdz
    )
    vel, meta = dataset_to_vel_meta(ds)
    vel, meta = downsample_vel_meta(vel, meta)
    ds2 = _vel_meta_to_dataset(vel, meta)
    return ds2, zelt, kind, pdx, pdz


def load_zelt_model_optional(path: str):
    """
    若路径为 Zelt v.in，只解析界面节点（供 Interfaces 叠层），不做栅格化。
    非 v.in / 失败 → None。
    """
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.is_file():
        return None
    try:
        from pyAOBS.visualization.imodel.gui.model_load import (
            is_vin_file_by_content,
            is_vin_path,
        )
        from pyAOBS.model_building.zeltform import ZeltVelocityModel2d
    except Exception:
        return None
    if not (is_vin_path(p) or is_vin_file_by_content(str(p))):
        return None
    try:
        return ZeltVelocityModel2d(model_file=str(p))
    except Exception:
        return None


def meta_from_rsf_header(path: str) -> Dict[str, float]:
    """只读 RSF 头，不载入二进制（用于同步网格）。"""
    meta = parse_rsf_header(path)
    return {
        "o1": float(meta.get("o1", 0)),
        "d1": float(meta.get("d1", 1)),
        "n1": float(int(meta["n1"])),
        "o2": float(meta.get("o2", 0)),
        "d2": float(meta.get("d2", 1)),
        "n2": float(int(meta.get("n2", 1))),
    }


def suggest_grid_from_model_file(
    path: str,
    *,
    vin_dx_km: float = 0.5,
    vin_dz_km: float = 0.25,
) -> Tuple[GridParams, str]:
    """
    从模型推断工区网格。v.in 用节点范围+用户 dx/dz，避免完整细网格栅格化。
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(path)
    suf = p.suffix.lower()

    from pyAOBS.visualization.imodel.gui.model_load import (
        is_vin_file_by_content,
        is_vin_path,
    )

    if suf == ".rsf":
        meta = meta_from_rsf_header(str(p))
        return suggest_grid_from_meta(meta), "rsf"

    is_vin = is_vin_path(p) or is_vin_file_by_content(str(p))
    if is_vin:
        from pyAOBS.model_building.zeltform import ZeltVelocityModel2d

        zelt = ZeltVelocityModel2d(model_file=str(p))
        x0, x1, z0, z1 = zelt.get_model_bounds()
        dx = max(float(vin_dx_km), 1e-6)
        dz = max(float(vin_dz_km), 1e-6)
        nx = max(2, int(math.ceil((x1 - x0) / dx)) + 1)
        nz = max(2, int(math.ceil((z1 - z0) / dz)) + 1)
        return (
            GridParams(oz=float(z0), dz=dz, nz=nz, ox=float(x0), dx=dx, nx=nx),
            "vin",
        )

    # grd / smesh 等：用预览级加载取 meta（已在粗网格）
    _ds, _z, kind = load_any_velocity_dataset(
        str(p),
        vin_dx_km=float(vin_dx_km),
        vin_dz_km=float(vin_dz_km),
    )
    vel, meta = dataset_to_vel_meta(_ds)
    return suggest_grid_from_meta(meta), kind


def load_any_velocity(
    path: str,
    *,
    vin_dx_km: float = 0.025,
    vin_dz_km: float = 0.025,
) -> Tuple[np.ndarray, Dict[str, float], str]:
    """
    加载任意支持格式 → (vel, meta, kind)
    kind: rsf | vin | grid | smesh
    """
    ds, _zelt, kind = load_any_velocity_dataset(
        path, vin_dx_km=vin_dx_km, vin_dz_km=vin_dz_km
    )
    if kind == "rsf":
        # 已由 read_vel_rsf 构造；dataset_to_vel_meta 亦可
        vel, meta = dataset_to_vel_meta(ds)
        return vel, meta, kind
    vel, meta = dataset_to_vel_meta(ds)
    return vel, meta, kind


def convert_model_to_rsf(
    src_path: str,
    out_rsf: str,
    *,
    vin_dx_km: float = 0.025,
    vin_dz_km: float = 0.025,
    grid: Optional[GridParams] = None,
    resample_to_project_grid: bool = False,
) -> Tuple[str, Dict[str, float], str]:
    """
    任意模型 → out_rsf。
    若 resample_to_project_grid 且给了 grid，则按工区网格重采样后写出。
    返回 (out_rsf, meta_used, kind)
    """
    from .velocity import resample_to_grid

    vel, meta, kind = load_any_velocity(
        src_path, vin_dx_km=vin_dx_km, vin_dz_km=vin_dz_km
    )
    if resample_to_project_grid and grid is not None:
        vel = resample_to_grid(vel, meta, grid)
        meta = {
            "o1": float(grid.oz),
            "d1": float(grid.dz),
            "n1": float(grid.nz),
            "o2": float(grid.ox),
            "d2": float(grid.dx),
            "n2": float(grid.nx),
        }
        g = grid
    else:
        g = GridParams(
            oz=float(meta["o1"]),
            dz=float(meta["d1"]),
            nz=int(meta["n1"]),
            ox=float(meta["o2"]),
            dx=float(meta["d2"]),
            nx=int(meta["n2"]),
        )
    os.makedirs(os.path.dirname(os.path.abspath(out_rsf)) or ".", exist_ok=True)
    write_vel_rsf(out_rsf, vel, g)
    return out_rsf, meta, kind


def suggest_grid_from_meta(meta: Dict[str, float]) -> GridParams:
    return GridParams(
        oz=float(meta["o1"]),
        dz=float(meta["d1"]),
        nz=int(meta["n1"]),
        ox=float(meta["o2"]),
        dx=float(meta["d2"]),
        nx=int(meta["n2"]),
    )
