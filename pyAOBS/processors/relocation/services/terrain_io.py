"""水深/地形文件加载与 UTM 转换（从 zplotpy 精简移植，无 Qt）。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def guess_coord_kind(xv: np.ndarray, yv: np.ndarray, xname: str = "", yname: str = "") -> str:
    xn = str(xname).lower()
    yn = str(yname).lower()
    if ("lon" in xn and "lat" in yn) or ("lon" in yn and "lat" in xn):
        return "geo"
    if xv.size == 0 or yv.size == 0:
        return "unknown"
    xmin = float(np.nanmin(xv[np.isfinite(xv)])) if np.any(np.isfinite(xv)) else 0.0
    xmax = float(np.nanmax(xv[np.isfinite(xv)])) if np.any(np.isfinite(xv)) else 0.0
    ymin = float(np.nanmin(yv[np.isfinite(yv)])) if np.any(np.isfinite(yv)) else 0.0
    ymax = float(np.nanmax(yv[np.isfinite(yv)])) if np.any(np.isfinite(yv)) else 0.0
    x_abs = float(np.nanmax(np.abs(xv[np.isfinite(xv)]))) if np.any(np.isfinite(xv)) else 0.0
    y_abs = float(np.nanmax(np.abs(yv[np.isfinite(yv)]))) if np.any(np.isfinite(yv)) else 0.0
    if -180.5 <= xmin <= 180.5 and -180.5 <= xmax <= 180.5 and -90.5 <= ymin <= 90.5 and -90.5 <= ymax <= 90.5:
        return "geo"
    if x_abs > 1000.0 and y_abs > 1000.0:
        return "utm"
    return "unknown"


def normalize_geo_lonlat_order(x_geo: np.ndarray, y_geo: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_geo = np.asarray(x_geo, dtype=float)
    y_geo = np.asarray(y_geo, dtype=float)
    xv = x_geo[np.isfinite(x_geo)]
    yv = y_geo[np.isfinite(y_geo)]
    if xv.size == 0 or yv.size == 0:
        return x_geo, y_geo
    x_min, x_max = float(np.min(xv)), float(np.max(xv))
    y_min, y_max = float(np.min(yv)), float(np.max(yv))
    x_is_lon = (-180.5 <= x_min <= 360.5) and (-180.5 <= x_max <= 360.5)
    x_is_lat = (-90.5 <= x_min <= 90.5) and (-90.5 <= x_max <= 90.5)
    y_is_lon = (-180.5 <= y_min <= 360.5) and (-180.5 <= y_max <= 360.5)
    y_is_lat = (-90.5 <= y_min <= 90.5) and (-90.5 <= y_max <= 90.5)
    if x_is_lat and y_is_lon and (not x_is_lon or not y_is_lat):
        return y_geo, x_geo
    return x_geo, y_geo


def lonlat_to_utm(
    lon: np.ndarray,
    lat: np.ndarray,
    override_zone: Optional[int] = None,
    override_hemisphere: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    valid = np.isfinite(lon) & np.isfinite(lat)
    if not np.any(valid):
        raise ValueError("经纬度数据无有效点，无法转换为UTM")
    lon_valid = lon[valid]
    lat_valid = lat[valid]
    lon0 = float(np.nanmean(lon_valid))
    lat0 = float(np.nanmean(lat_valid))
    zone_auto = int(np.floor((lon0 + 180.0) / 6.0) + 1)
    zone_auto = max(1, min(60, zone_auto))
    zone = int(override_zone) if override_zone is not None else int(zone_auto)
    zone = max(1, min(60, zone))
    hemi_hint = str(override_hemisphere or "auto").lower()
    if hemi_hint == "north":
        lat_sign = 1.0
    elif hemi_hint == "south":
        lat_sign = -1.0
    else:
        lat_sign = 1.0 if lat0 >= 0.0 else -1.0
    epsg = (32600 + zone) if lat_sign >= 0.0 else (32700 + zone)
    try:
        from pyproj import Transformer  # type: ignore
    except Exception as exc:
        raise RuntimeError("经纬度→UTM 需要 pyproj，请安装：pip install pyproj") from exc
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x_out = np.full(lon.shape, np.nan, dtype=float)
    y_out = np.full(lat.shape, np.nan, dtype=float)
    xx, yy = transformer.transform(lon_valid, lat_valid)
    x_out[valid] = np.asarray(xx, dtype=float)
    y_out[valid] = np.asarray(yy, dtype=float)
    info: Dict[str, object] = {
        "zone": float(zone),
        "epsg": float(epsg),
        "lon0": float(lon0),
        "lat0": float(lat0),
    }
    return x_out, y_out, info


def xy_to_utm_guess(x: float, y: float) -> Tuple[float, float]:
    """若像经纬则转 UTM，否则原样返回。"""
    xx, yy = float(x), float(y)
    if abs(xx) <= 180.5 and abs(yy) <= 90.5:
        try:
            xu, yu, _ = lonlat_to_utm(np.asarray([xx]), np.asarray([yy]))
            if np.isfinite(xu[0]) and np.isfinite(yu[0]):
                return float(xu[0]), float(yu[0])
        except Exception:
            pass
    return xx, yy


def load_terrain_meta(path: str, force_geo: bool = False) -> Dict[str, object]:
    """加载 .xyz/.txt 点集或 .nc/.grd 网格。"""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".xyz", ".txt"):
        arr = np.loadtxt(str(p), comments="#", dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape((1, -1))
        if arr.shape[1] < 3:
            raise ValueError("xyz文本至少需要三列：x y z")
        x = np.asarray(arr[:, 0], dtype=float)
        y = np.asarray(arr[:, 1], dtype=float)
        z = np.asarray(arr[:, 2], dtype=float)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x, y, z = x[valid], y[valid], z[valid]
        if x.size == 0:
            raise ValueError("xyz文本没有有效数据点")
        coord_kind = "geo" if force_geo else guess_coord_kind(x, y)
        return {"mode": "points", "x": x, "y": y, "z": z, "path": str(p), "coord_kind": coord_kind}

    if suffix in (".nc", ".grd"):
        try:
            import xarray as xr  # type: ignore
        except Exception as exc:
            raise RuntimeError("读取 .nc/.grd 需要 xarray：pip install xarray netCDF4") from exc
        try:
            from pyAOBS.visualization.xarray_nc import open_netcdf_like_dataset
            ds = open_netcdf_like_dataset(p)
        except Exception:
            ds = xr.open_dataset(str(p))
        try:
            data_var = None
            for name, var in ds.data_vars.items():
                if getattr(var, "ndim", 0) >= 2:
                    data_var = name
                    break
            if data_var is None:
                raise ValueError("nc/grd中未找到二维地形变量")
            da = ds[data_var].squeeze()
            if da.ndim < 2:
                raise ValueError("地形变量维度不足（需要二维）")
            dims = list(da.dims)
            ydim, xdim = dims[-2], dims[-1]
            x = np.asarray(ds[xdim].values, dtype=float)
            y = np.asarray(ds[ydim].values, dtype=float)
            z = np.asarray(da.values, dtype=float)
            if z.ndim > 2:
                z = z.reshape(z.shape[-2], z.shape[-1])
            if z.shape[0] != y.size or z.shape[1] != x.size:
                zt = z.T
                if zt.shape[0] == y.size and zt.shape[1] == x.size:
                    z = zt
                else:
                    raise ValueError("nc/grd网格维度与坐标长度不一致")
            coord_kind = "geo" if force_geo else guess_coord_kind(x, y, xdim, ydim)
            return {
                "mode": "grid",
                "x": x,
                "y": y,
                "z": z,
                "path": str(p),
                "coord_kind": coord_kind,
                "x_name": str(xdim),
                "y_name": str(ydim),
            }
        finally:
            try:
                ds.close()
            except Exception:
                pass

    raise ValueError("不支持的地形格式，请使用 .grd/.nc/.xyz/.txt")


def convert_terrain_to_utm(terrain_meta: Dict[str, object]) -> Optional[Dict[str, object]]:
    """geo → UTM 点集；已是 utm 则原样返回。"""
    mode = str((terrain_meta or {}).get("mode", "")).lower()
    coord_kind = str((terrain_meta or {}).get("coord_kind", "unknown")).lower()
    if mode not in ("grid", "points"):
        return None
    if coord_kind == "utm":
        out = dict(terrain_meta)
        out["coord_kind"] = "utm"
        return out
    if coord_kind != "geo":
        # unknown：若数值像 UTM 则当 utm；像经纬则当 geo
        x0 = np.asarray(terrain_meta.get("x", []), dtype=float).reshape(-1)
        y0 = np.asarray(terrain_meta.get("y", []), dtype=float).reshape(-1)
        kind2 = guess_coord_kind(x0, y0)
        if kind2 == "utm":
            out = dict(terrain_meta)
            out["coord_kind"] = "utm"
            return out
        if kind2 != "geo":
            return None
        coord_kind = "geo"

    try:
        if mode == "grid":
            x = np.asarray(terrain_meta.get("x", []), dtype=float)
            y = np.asarray(terrain_meta.get("y", []), dtype=float)
            z = np.asarray(terrain_meta.get("z", []), dtype=float)
            if x.size < 2 or y.size < 2 or z.size == 0:
                return None
            xx, yy = np.meshgrid(x, y, indexing="xy")
            lon, lat = normalize_geo_lonlat_order(xx.reshape(-1), yy.reshape(-1))
            x_utm, y_utm, _ = lonlat_to_utm(lon, lat)
            z_flat = np.asarray(z, dtype=float).reshape(-1)
            valid = np.isfinite(x_utm) & np.isfinite(y_utm) & np.isfinite(z_flat)
            if not np.any(valid):
                return None
            return {
                "mode": "points",
                "coord_kind": "utm",
                "x": np.asarray(x_utm[valid], dtype=float),
                "y": np.asarray(y_utm[valid], dtype=float),
                "z": np.asarray(z_flat[valid], dtype=float),
                "path": str(terrain_meta.get("path", "")),
            }

        x = np.asarray(terrain_meta.get("x", []), dtype=float).reshape(-1)
        y = np.asarray(terrain_meta.get("y", []), dtype=float).reshape(-1)
        z = np.asarray(terrain_meta.get("z", []), dtype=float).reshape(-1)
        lon, lat = normalize_geo_lonlat_order(x, y)
        x_utm, y_utm, _ = lonlat_to_utm(lon, lat)
        valid = np.isfinite(x_utm) & np.isfinite(y_utm) & np.isfinite(z)
        if not np.any(valid):
            return None
        return {
            "mode": "points",
            "coord_kind": "utm",
            "x": np.asarray(x_utm[valid], dtype=float),
            "y": np.asarray(y_utm[valid], dtype=float),
            "z": np.asarray(z[valid], dtype=float),
            "path": str(terrain_meta.get("path", "")),
        }
    except Exception:
        return None


def load_terrain_as_utm(path: str) -> Dict[str, object]:
    """一步：加载并转为 UTM meta（供 build_bathymetry_sampler）。"""
    raw = load_terrain_meta(path)
    utm = convert_terrain_to_utm(raw)
    if utm is None:
        raise ValueError("该水深文件无法转换为UTM坐标（或坐标类型无法识别）")
    return utm
