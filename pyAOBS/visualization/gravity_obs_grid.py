"""从重力格网 (.grd / NetCDF 等) 在剖面观测点上插值，用于与 Full Model 对比。

默认数据目录：``pyAOBS/visualization/data``（放入 ``gravity_world.grd``，由用户自备）。
等价于常说的 ``pyAOBS.visualization.data`` / ``pyobs.visualization.data`` 资源路径。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None  # type: ignore[misc, assignment]

from .xarray_nc import open_netcdf_like_dataset


DEFAULT_GRAVITY_OBS_FILENAME = "gravity_world.grd"


def open_gravity_netcdf_dataset(fp: Path | str):
    """与 zplotpy 地形读取一致：经 ``open_netcdf_like_dataset``（含多引擎 / decode_cf 与最终裸打开）。"""
    if xr is None:
        raise ImportError("需要 xarray")
    return open_netcdf_like_dataset(Path(fp).expanduser())

# 视作「经纬度同名」的水平坐标维度名（不包含距离用的 x/dist）
_HORIZONTAL_LON_DIM = frozenset({"lon", "longitude", "long", "longitude_degrees", "east", "easting"})
_HORIZONTAL_LAT_DIM = frozenset({"lat", "latitude", "north", "northing"})


def visualization_package_data_dir() -> Path:
    """返回 `visualization/data` 目录路径。"""
    return Path(__file__).resolve().parent / "data"


def default_observed_gravity_path(
    *,
    directory: Optional[Path | str] = None,
    filename: str = DEFAULT_GRAVITY_OBS_FILENAME,
) -> Path:
    """拼出观测重力格网文件的完整路径。"""
    root = Path(directory).expanduser() if directory else visualization_package_data_dir()
    return Path(root).expanduser() / str(filename)


def parse_lon_lat_endpoint_line(raw: str) -> Optional[Tuple[float, float, float, float]]:
    """解析单行四个数字，分隔符为空白或逗号/分号。

    **标准语义**（沿用既有逻辑）： ``lon0 lat0 lon1 lat1``
    — 剖面线在地球表面上的两端点经纬度。

    **常见误填**（本函数自动识别）： ``lon_min lon_max lat_min lat_max``
    — 多见于把「东经 / 北纬范围」连在一起写，例如::

        ``110 120 21 23`` 表示 东经 110–120°、北纬 21–23°，
        等价为端点西南 ``(lon_min, lat_min)`` → 东北 ``(lon_max, lat_max)``。

    启发式：第二个数若为经度取值（｜第二位｜＞90 通常不可能是纬度）且第三、四维在纬度范围内，
    则判为四角范围写法；否则会按 ``lon0 lat0 lon1 lat1`` 解释。
    跨日期变更线或非单调范围请改用明确端点写法 ``lon0 lat0 lon1 lat1``。
    """
    s = str(raw or "").strip()
    if not s:
        return None
    parts = [p for p in re.split(r"[\s,;]+", s) if p]
    if len(parts) != 4:
        return None
    try:
        a, b, c, d = (float(x) for x in parts)
    except ValueError:
        return None

    def ok_lat(u: float) -> bool:
        return abs(u) <= 90.0 + 1e-9

    def ok_lon_deg(u: float) -> bool:
        return abs(u) <= 360.0 + 1e-9

    # 误填：lon_min lon_max lat_min lat_max（第二位不可能是纬度时）
    if ok_lon_deg(a) and ok_lon_deg(b) and ok_lat(c) and ok_lat(d) and abs(b) > 90.0:
        lon_min, lon_max = sorted((a, b))
        lat_min, lat_max = sorted((c, d))
        return (lon_min, lat_min, lon_max, lat_max)

    return (a, b, c, d)


def extract_lon_lat_along_profile(
    grid_data: Any,
    x_coord: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """抽取与横向维 ``x_coord`` 对齐的 1D 经度 / 纬度（coords 或非维度变量）。"""
    if xr is None or grid_data is None:
        return None, None
    ds = grid_data if isinstance(grid_data, xr.Dataset) else None
    if ds is None:
        return None, None
    try:
        xv = ds[x_coord]
    except KeyError:
        return None, None
    if xv.ndim != 1 or len(xv.dims) != 1:
        return None, None
    xd = xv.dims[0]
    xkey = str(x_coord).lower()

    def _as_1d(name: str) -> Optional[np.ndarray]:
        try:
            v = ds[name]
        except KeyError:
            return None
        if v.ndim != 1 or v.dims[0] != xd:
            return None
        a = np.asarray(v.values, dtype=float)
        if not np.all(np.isfinite(a)):
            return None
        return a

    lon = lat = None
    for cand in list(ds.coords) + list(ds.data_vars):
        if str(cand) == str(x_coord):
            continue
        cl = cand.lower()
        look_lon = (
            any(cl == k or cl.startswith(k + "_") for k in {"lon", "longitude", "long", "east", "eastings"})
            and cl != xkey
        )
        look_lat = any(
            cl == k or cl.startswith(k + "_") for k in {"lat", "latitude", "north", "northings"}
        )

        cand_lon = (
            look_lon or cl.endswith("_lon") or cl.endswith("longitude") or "_lon_" in cl
        ) and "latitud" not in cl
        cand_lat = look_lat or cl.endswith("_lat") or "_lat_" in cl

        if cand_lon and lon is None:
            lon = _as_1d(str(cand))
        elif cand_lat and lat is None:
            lat = _as_1d(str(cand))
        if lon is not None and lat is not None:
            break
    return lon, lat


def lon_lat_along_simple_geodesic(
    x_obs_km: np.ndarray,
    x_ref_km: np.ndarray,
    lon0_deg: float,
    lat0_deg: float,
    lon1_deg: float,
    lat1_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """在模型水平距离上对两端经纬度线性插值（短剖面通常可接受）。

    ``x_obs_km`` 与网格 ``x_ref_km``（km）共线性段基准。"""
    x_obs = np.asarray(x_obs_km, dtype=float)
    x_grid = np.asarray(x_ref_km, dtype=float)
    if x_grid.size < 2:
        x0 = float(np.nanmin(x_obs)) if x_obs.size else 0.0
        x1 = float(np.nanmax(x_obs)) if x_obs.size else 1.0
    else:
        x0 = float(x_grid.min())
        x1 = float(x_grid.max())
    span = max(x1 - x0, 1e-12)
    t = (x_obs - x0) / span
    t = np.clip(t, 0.0, 1.0)
    lon = lon0_deg + t * (lon1_deg - lon0_deg)
    lat = lat0_deg + t * (lat1_deg - lat0_deg)
    return lon, lat


def profile_lon_lat_for_x_obs_km(
    *,
    x_obs_km: np.ndarray,
    x_grid_km: np.ndarray,
    grid_data: Any,
    x_coord: str,
    endpoint_line: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """为观测站 ``x_obs_km`` 求经纬度序列，用于在世界格网上插值。

    返回 ``(lon, lat, note)``；不可用则 lon/lat 为 None。
    """
    x_orig = np.asarray(x_grid_km, dtype=float)
    xv = np.asarray(x_obs_km, dtype=float)
    xname = str(x_coord).lower()

    lon_track, lat_track = extract_lon_lat_along_profile(grid_data, x_coord)

    if (
        lon_track is not None
        and lat_track is not None
        and lon_track.shape == lat_track.shape == x_orig.shape
    ):
        lon_st = np.interp(xv, x_orig, lon_track, left=np.nan, right=np.nan)
        lat_st = np.interp(xv, x_orig, lat_track, left=np.nan, right=np.nan)
        return lon_st, lat_st, "longitude/latitude from model Dataset along profile axis"

    pll = parse_lon_lat_endpoint_line(endpoint_line or "") if endpoint_line else None
    if pll is not None:
        lon0, la0, lon1, la1 = pll
        lon_st, lat_st = lon_lat_along_simple_geodesic(xv, x_orig, lon0, la0, lon1, la1)
        return (
            lon_st,
            lat_st,
            "manual lon/lat: endpoints (lon0 lat0 lon1 lat1) or range (lon_min lon_max lat_min lat_max); linear along km fraction",
        )

    if xr is not None and grid_data is not None and hasattr(grid_data, "coords"):
        if xname in _HORIZONTAL_LON_DIM:
            lon_vals = np.asarray(grid_data.coords[x_coord].values, dtype=float)
            if lon_vals.shape == x_orig.shape:
                pll_lat = parse_lon_lat_endpoint_line(endpoint_line or "") if endpoint_line else None
                if pll_lat is None:
                    return (
                        None,
                        None,
                        "horizontal coord is longitude: provide Profile Lon/Lat line "
                        "(lon0 lat0 lon1 lat1) to interpolate latitude along profile",
                    )
                lon0, la0, lon1, la1 = pll_lat
                lon_st = np.interp(xv, x_orig, lon_vals, left=np.nan, right=np.nan)
                _, lat_st = lon_lat_along_simple_geodesic(xv, x_orig, lon0, la0, lon1, la1)
                return lon_st, lat_st, "longitude from coord + latitude from endpoints (linear)"

    return (
        None,
        None,
        "cannot build lon/lat: add LON/LAT on model grid OR fill Profile Lon/Lat endpoints",
    )


def _is_lon_array(a: np.ndarray) -> bool:
    if a.size < 2:
        return False
    return bool(np.nanmax(a) <= 420.0 and np.nanmin(a) >= -420.0 and (np.nanmax(a) - np.nanmin(a)) <= 372.0)


def _is_lat_array(a: np.ndarray) -> bool:
    if a.size < 2:
        return False
    return bool(np.nanmax(a) <= 90.5 and np.nanmin(a) >= -90.5)


def dataset_looks_like_geographic_lon_lat_surface(ds: xr.Dataset) -> bool:
    """判断是否像经纬度平面格网（如全球重力/地形），而非 imodel 用的距深剖面网格 (km)。

    Sandwell-style ``gravity_world.grd`` 等：维度为纬度×经度、数值为异常而非速度。
    """
    try:
        da = _pick_primary_2d_data_array(ds)
    except ValueError:
        return False
    if getattr(da, "ndim", 0) != 2:
        return False
    d0, d1 = str(da.dims[0]), str(da.dims[1])
    try:
        c0 = np.asarray(da.coords[d0].values, dtype=float)
        c1 = np.asarray(da.coords[d1].values, dtype=float)
    except Exception:
        return False
    if _is_lon_array(c0) and _is_lat_array(c1):
        return True
    if _is_lon_array(c1) and _is_lat_array(c0):
        return True
    return False


def _coord_vals_1d(ds: xr.Dataset, da: Any, dim: str) -> np.ndarray:
    """取一维坐标（与 ``zplotpy.qt_fast_viewer._load_terrain_meta``：优先 ``ds[xdim]``）。"""
    dim = str(dim)
    try:
        if dim in ds:
            return np.asarray(ds[dim].values, dtype=float)
    except Exception:
        pass
    try:
        if dim in ds.coords:
            return np.asarray(ds.coords[dim].values, dtype=float)
    except Exception:
        pass
    if dim in getattr(da, "coords", {}):
        return np.asarray(da.coords[dim].values, dtype=float)
    raise KeyError(f"coordinate not found for dim={dim!r}")


def _recover_nx_ny_gmt_flat(
    *,
    nz: int,
    xr_: np.ndarray,
    yr_: np.ndarray,
    sp_: np.ndarray,
    nx_hint: int,
    ny_hint: int,
    pixel_registered: bool,
) -> Tuple[int, int]:
    """在 ``dimension``×size 不匹配时，用 spacing 或因数分解推断 (nx,ny)。

    GMT ``node_offset=1`` 为像素配准：点数 ≈ span/dx（无 gridline 的 +1）。
    """
    xmin, xmax = float(xr_[0]), float(xr_[1])
    ymin, ymax = float(yr_[0]), float(yr_[1])
    dx = float(abs(sp_[0])) if sp_.size else 0.0
    dy = float(abs(sp_[1])) if sp_.size >= 2 else 0.0

    def _ok(a: int, b: int) -> bool:
        return a >= 2 and b >= 2 and a * b == nz

    if _ok(nx_hint, ny_hint):
        return nx_hint, ny_hint

    if dx > 1e-60 and dy > 1e-60:
        xspan = xmax - xmin
        yspan = abs(ymax - ymin)
        # 像素：nx*dx≈span；网格线： (nx-1)*dx≈span
        nx_pix = max(2, int(round(xspan / dx)))
        ny_pix = max(2, int(round(yspan / dy)))
        nx_grid = max(2, int(round(xspan / dx)) + 1)
        ny_grid = max(2, int(round(yspan / dy)) + 1)

        if pixel_registered:
            if _ok(nx_pix, ny_pix):
                return nx_pix, ny_pix
            if _ok(nx_grid, ny_grid):
                return nx_grid, ny_grid
        else:
            if _ok(nx_grid, ny_grid):
                return nx_grid, ny_grid
            if _ok(nx_pix, ny_pix):
                return nx_pix, ny_pix

    # 因式分解 nz，按与 nx_hint、ny_hint 接近程度挑选
    import math as _math

    best_nx = best_ny = 0
    best_pen = float("inf")
    lim = int(_math.sqrt(nz)) + 2
    for nxa in range(2, max(lim, 3)):
        if nz % nxa != 0:
            continue
        nyb = nz // nxa
        if nyb < 2:
            continue
        pen = abs(nxa - nx_hint) + abs(nyb - ny_hint)
        if pen < best_pen:
            best_pen = pen
            best_nx, best_ny = nxa, nyb
    if best_nx > 0 and _ok(best_nx, best_ny):
        return best_nx, best_ny
    raise ValueError(f"无法将一维 z 长度 {nz} 恢复为整数 nx×ny")


def _gmt_z_node_offset_pixel(z_attrs: dict[str, Any], ds_attrs: dict[str, Any]) -> bool:
    """``z:node_offset=1`` 表示像素（cell）配准（GMT COARDS）；缺省当作网格线配准。"""
    for src in (z_attrs, ds_attrs):
        raw = src.get("node_offset")
        if raw is None:
            continue
        try:
            return int(float(raw)) == 1
        except (TypeError, ValueError):
            continue
    return False



def _is_gmt_flat_dataset(ds: xr.Dataset) -> bool:
    req = ["z", "dimension", "x_range", "y_range", "spacing"]
    if not all(k in ds.variables for k in req):
        return False
    try:
        return int(getattr(ds["z"], "ndim", 0)) == 1
    except Exception:
        return False


def _gmt_flat_grid_layout(ds: xr.Dataset) -> dict[str, Any]:
    """GMT nf/COARDS 一维 z：只读元数据与坐标，不加载整幅 z。"""
    if xr is None:
        raise ImportError("需要 xarray")
    req = ["z", "dimension", "x_range", "y_range", "spacing"]
    if any(k not in ds.variables for k in req):
        raise ValueError("__NOT_GMTPAD__ missing vars")

    zv = ds["z"]
    if getattr(zv, "ndim", 0) != 1:
        raise ValueError("__NOT_GMTPAD__ z_ndim!=1")

    nz_size = int(zv.size)
    if nz_size < 4:
        raise ValueError("__NOT_GMTPAD__ nz too short")

    xr_ = np.asarray(ds["x_range"].values, dtype=float).ravel()
    yr_ = np.asarray(ds["y_range"].values, dtype=float).ravel()
    sp_ = np.asarray(ds["spacing"].values, dtype=float).ravel()
    dv = np.asarray(ds["dimension"].values, dtype=float).ravel()
    if min(xr_.size, yr_.size, sp_.size, dv.size) < 2:
        raise ValueError("__NOT_GMTPAD__ side<2")

    nx_hint = max(2, int(round(float(dv[0]))))
    ny_hint = max(2, int(round(float(dv[1]))))
    pixel = _gmt_z_node_offset_pixel(dict(zv.attrs or {}), dict(ds.attrs or {}))
    nx, ny = _recover_nx_ny_gmt_flat(
        nz=nz_size,
        xr_=xr_,
        yr_=yr_,
        sp_=sp_,
        nx_hint=nx_hint,
        ny_hint=ny_hint,
        pixel_registered=pixel,
    )

    xmin, xmax = float(xr_[0]), float(xr_[1])
    ymin, ymax = float(yr_[0]), float(yr_[1])
    if pixel:
        wx = xmax - xmin
        wy = ymax - ymin
        xcoords = xmin + (np.arange(nx, dtype=float) + 0.5) * (wx / max(nx, 1))
        ycoords = ymin + (np.arange(ny, dtype=float) + 0.5) * (wy / max(ny, 1))
    else:
        xcoords = np.linspace(xmin, xmax, nx, dtype=float)
        ycoords = np.linspace(ymin, ymax, ny, dtype=float)

    return {
        "lon_1d": xcoords,
        "lat_1d": ycoords,
        "nx": nx,
        "ny": ny,
        "z_var": zv,
        "name": str(zv.name or "z"),
        "attrs": dict(getattr(zv, "attrs", {}) or {}),
        "full_shape": (ny, nx),
    }


def gmt_nf_coards_flat_to_dataarray(ds: xr.Dataset) -> xr.DataArray:
    """GMT 经典 nf/COARDS：二维场存成一维 ``z(xysize)``，范围与步长在 ``*_range``/``spacing``/``dimension``。

    Sandwell/V32 等与 ``dimension('side'); z('xysize')`` 结构对齐；供重力插值及 Qt 地形读取。
    """
    if xr is None:
        raise ImportError("需要 xarray")

    layout = _gmt_flat_grid_layout(ds)
    nz = np.asarray(layout["z_var"].values, dtype=float).ravel()
    nx, ny = int(layout["nx"]), int(layout["ny"])
    Z = nz.reshape((ny, nx), order="C")
    merged = layout["attrs"]
    return xr.DataArray(
        Z.astype(float),
        dims=("lat", "lon"),
        coords={"lon": layout["lon_1d"], "lat": layout["lat_1d"]},
        name=layout["name"],
        attrs=merged,
    )


def _terrain_pick_gravity_da(ds: xr.Dataset) -> xr.DataArray:
    """对齐 ``zplotpy/qt_fast_viewer._load_terrain_meta``：首个 ``ndim>=2`` data_var、squeeze、取末二维、reshape(z[-2:],x)。"""
    if xr is None:
        raise ImportError("需要 xarray")
    data_var: Optional[str] = None
    for name, var in ds.data_vars.items():
        if getattr(var, "ndim", 0) >= 2:
            data_var = str(name)
            break
    if data_var is None:
        for name in ds.variables:
            try:
                var = ds[name]
            except Exception:
                continue
            if getattr(var, "ndim", 0) >= 2:
                data_var = str(name)
                break
    if data_var is None:
        raise ValueError("__NO_VAR__")
    da0 = ds[data_var].squeeze()
    if da0.ndim < 2:
        raise ValueError("__NDIM_LT2__")
    dims = list(da0.dims)
    ydim, xdim = dims[-2], dims[-1]
    x = _coord_vals_1d(ds, da0, str(xdim))
    y = _coord_vals_1d(ds, da0, str(ydim))
    z = np.asarray(da0.values, dtype=float)
    if z.ndim > 2:
        z = z.reshape(z.shape[-2], z.shape[-1])
    if z.shape[0] != y.size or z.shape[1] != x.size:
        zt = z.T
        if zt.shape[0] == y.size and zt.shape[1] == x.size:
            z = zt
        else:
            raise ValueError(
                f"网格与坐标维度不一致 z.shape={z.shape}, y.len={y.size}, x.len={x.size}"
            )
    arr = xr.DataArray(
        z,
        dims=(str(ydim), str(xdim)),
        coords={str(ydim): y, str(xdim): x},
        name=str(data_var),
    )
    attrs = getattr(da0, "attrs", {}) or {}
    arr.attrs.update(attrs)
    return arr.astype(float)


def _pick_primary_2d_data_array(ds: xr.Dataset) -> xr.DataArray:
    """先 zplotpy 地形同款；失败再宽松扫描。"""
    if xr is None:
        raise ImportError("需要 xarray")

    errs: list[str] = []

    try:
        return _terrain_pick_gravity_da(ds)
    except Exception as e:
        errs.append(str(e))

    try:
        return gmt_nf_coards_flat_to_dataarray(ds)
    except Exception as e:
        errs.append(str(e))

    seen: set[str] = set()
    ordered: list[str] = []
    for vn in ds.data_vars:
        s = str(vn)
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    for vn in ds.variables:
        s = str(vn)
        if s not in seen:
            seen.add(s)
            ordered.append(s)

    def _try_any(name: str) -> Optional[xr.DataArray]:
        try:
            v = ds[name]
        except Exception:
            return None
        if getattr(v, "ndim", 0) < 2:
            return None
        try:
            return _terrain_pick_gravity_da(ds[[name]])
        except Exception:
            return None

    for vn in ordered:
        got = _try_any(vn)
        if got is not None:
            return got

    want = sorted(set(list(ds.variables)))
    hint = "; ".join([f"{x!s}{tuple(ds[x].dims)!r}" for x in want[:24] if x in ds.variables])
    tail = " ..." if len(want) > 24 else ""
    detail = errs[0] if errs else "unknown error"
    raise ValueError(
        "重力格网中没有可用的二维数据变量。"
        f" ({detail}) 变量节选: [{hint}{tail}]"
    )


def interpolate_observed_gravity_mgal(
    file_path: Path | str,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
) -> Tuple[np.ndarray, str]:
    """在 (lon, lat)（度）上对格网点值做双线性插值；返回数值与简短说明字符串。"""
    if xr is None:
        raise ImportError("读取 .grd 重力格网需要 xarray 与底层 netCDF4/h5netcdf 引擎")

    fp = Path(file_path).expanduser()
    if not fp.is_file():
        raise FileNotFoundError(f"重力数据文件不存在: {fp}")

    lon_deg = np.asarray(lon_deg, dtype=float)
    lat_deg = np.asarray(lat_deg, dtype=float)
    if lon_deg.shape != lat_deg.shape:
        raise ValueError("lon_deg 与 lat_deg 形状须一致")

    ds = open_gravity_netcdf_dataset(fp)
    try:
        da = _pick_primary_2d_data_array(ds)
        lat_dim, lon_dim = infer_geographic_dims(da)

        plat = xr.DataArray(lat_deg.astype(float).ravel(), dims=("grav_obs_pt",))
        plon = xr.DataArray(lon_deg.astype(float).ravel(), dims=("grav_obs_pt",))

        samp = da.interp({lat_dim: plat, lon_dim: plon}, method="linear")
        vals_flat = np.asarray(samp.values, dtype=float).ravel()

        vals_out = vals_flat.reshape(lon_deg.shape)
        attrs = getattr(da, "attrs", {}) or {}
        units = str(attrs.get("units", "") or "").strip()
        descr = (
            f"interp from {fp.name} «{da.name}» dims=({lat_dim},{lon_dim}); "
            f"attrs units={units!r}"
        )
        return vals_out, descr
    finally:
        ds.close()


def infer_geographic_dims(
    da: Any,
) -> tuple[str, str]:
    """辨认二维场 ``da`` 的维度名：返回 (lat_dim, lon_dim)。"""
    if xr is None:
        raise ImportError("需要 xarray")
    if da.ndim != 2:
        raise ValueError(f"平面图需要二维场，当前 ndim={da.ndim}")
    d0, d1 = str(da.dims[0]), str(da.dims[1])
    c0 = np.asarray(da.coords[d0].values, dtype=float)
    c1 = np.asarray(da.coords[d1].values, dtype=float)
    if _is_lat_array(c0) and _is_lon_array(c1):
        return d0, d1
    if _is_lon_array(c0) and _is_lat_array(c1):
        return d1, d0
    if _is_lat_array(c1) and _is_lon_array(c0):
        return d0, d1
    return d0, d1


def _pick_lazy_2d_data_array(ds: xr.Dataset) -> xr.DataArray:
    """返回二维 DataArray（尽量 lazy，不立即读入 values）。"""
    if xr is None:
        raise ImportError("需要 xarray")
    for _name, var in ds.data_vars.items():
        if getattr(var, "ndim", 0) >= 2:
            return ds[str(_name)].squeeze()
    for name in ds.variables:
        try:
            var = ds[name]
        except Exception:
            continue
        if getattr(var, "ndim", 0) >= 2:
            return ds[name].squeeze()
    raise ValueError("重力格网中没有可用的二维数据变量。")


def _index_span_for_coord(coord_1d: np.ndarray, lo: float, hi: float) -> tuple[int, int]:
    c = np.asarray(coord_1d, dtype=float)
    n = int(c.size)
    if n == 0:
        return 0, -1
    lo_f, hi_f = float(lo), float(hi)
    if lo_f > hi_f:
        lo_f, hi_f = hi_f, lo_f
    if c[-1] >= c[0]:
        i0 = int(np.searchsorted(c, lo_f, side="left"))
        i1 = int(np.searchsorted(c, hi_f, side="right")) - 1
    else:
        c_rev = c[::-1]
        i0 = n - 1 - int(np.searchsorted(c_rev, hi_f, side="left"))
        i1 = n - 1 - int(np.searchsorted(c_rev, lo_f, side="right")) + 1
    i0 = max(0, min(i0, n - 1))
    i1 = max(0, min(i1, n - 1))
    if i1 < i0:
        return i1, i0
    return i0, i1


def _normalize_gravity_lon_lat_z(
    lon_1d: np.ndarray,
    lat_1d: np.ndarray,
    z_2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon_1d = np.asarray(lon_1d, dtype=float)
    lat_1d = np.asarray(lat_1d, dtype=float)
    z_2d = np.asarray(z_2d, dtype=float)
    if lon_1d.size >= 2 and lon_1d[-1] < lon_1d[0]:
        lon_1d = lon_1d[::-1]
        z_2d = z_2d[:, ::-1]
    if lat_1d.size >= 2 and lat_1d[-1] < lat_1d[0]:
        lat_1d = lat_1d[::-1]
        z_2d = z_2d[::-1, :]
    if z_2d.shape != (lat_1d.size, lon_1d.size):
        raise ValueError(
            f"重力场维度与坐标不一致 z.shape={z_2d.shape}, lat={lat_1d.size}, lon={lon_1d.size}"
        )
    return lon_1d, lat_1d, z_2d


def _load_gmt_flat_arrays_subset(
    layout: dict[str, Any],
    bbox: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    lon_lo, lon_hi, lat_lo, lat_hi = bbox
    lon_1d = np.asarray(layout["lon_1d"], dtype=float)
    lat_1d = np.asarray(layout["lat_1d"], dtype=float)
    nx, ny = int(layout["nx"]), int(layout["ny"])
    zv = layout["z_var"]
    ix0, ix1 = _index_span_for_coord(lon_1d, lon_lo, lon_hi)
    iy0, iy1 = _index_span_for_coord(lat_1d, lat_lo, lat_hi)
    if ix1 < ix0 or iy1 < iy0:
        raise ValueError("bbox 与 GMT 格网无交集")

    blocks: list[np.ndarray] = []
    for j in range(iy0, iy1 + 1):
        s = j * nx + ix0
        e = j * nx + ix1 + 1
        blocks.append(np.asarray(zv[s:e].values, dtype=float))
    z_2d = np.vstack(blocks) if blocks else np.zeros((0, 0), dtype=float)
    lon_sub = lon_1d[ix0 : ix1 + 1]
    lat_sub = lat_1d[iy0 : iy1 + 1]
    lon_sub, lat_sub, z_2d = _normalize_gravity_lon_lat_z(lon_sub, lat_sub, z_2d)
    name = str(layout["name"])
    units = str(layout["attrs"].get("units", "") or "").strip()
    return lon_sub, lat_sub, z_2d, name, units


def read_gravity_grid_domain(
    file_path: Path | str,
) -> tuple[float, float, float, float]:
    """轻量读取格网经纬度范围 (lon_min, lon_max, lat_min, lat_max)。"""
    if xr is None:
        raise ImportError("需要 xarray")
    fp = Path(file_path).expanduser()
    if not fp.is_file():
        raise FileNotFoundError(f"重力数据文件不存在: {fp}")
    ds = open_gravity_netcdf_dataset(fp)
    try:
        if _is_gmt_flat_dataset(ds):
            layout = _gmt_flat_grid_layout(ds)
            lon = np.asarray(layout["lon_1d"], dtype=float)
            lat = np.asarray(layout["lat_1d"], dtype=float)
        else:
            da = _pick_lazy_2d_data_array(ds)
            lat_dim, lon_dim = infer_geographic_dims(da)
            lon = np.asarray(da.coords[lon_dim].values, dtype=float)
            lat = np.asarray(da.coords[lat_dim].values, dtype=float)
        return (
            float(np.nanmin(lon)),
            float(np.nanmax(lon)),
            float(np.nanmin(lat)),
            float(np.nanmax(lat)),
        )
    finally:
        ds.close()


def clamp_bbox_to_domain(
    bbox: tuple[float, float, float, float],
    domain: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lon_lo, lon_hi, lat_lo, lat_hi = bbox
    g_lon0, g_lon1, g_la0, g_la1 = domain
    cx0 = max(float(lon_lo), g_lon0)
    cx1 = min(float(lon_hi), g_lon1)
    cy0 = max(float(lat_lo), g_la0)
    cy1 = min(float(lat_hi), g_la1)
    return cx0, cx1, cy0, cy1


def estimate_plan_map_load_bbox(
    *,
    lon_p: np.ndarray,
    lat_p: np.ndarray,
    lon_s: Optional[np.ndarray] = None,
    lat_s: Optional[np.ndarray] = None,
    manual_bbox: Optional[tuple[float, float, float, float]] = None,
    zoom_to_profile: bool = True,
    grid_domain: Optional[tuple[float, float, float, float]] = None,
    pad_deg_min: float = 5.0,
) -> Optional[tuple[float, float, float, float]]:
    """在读取格网前估计加载 bbox；None 表示需读全球。"""
    eps = np.finfo(np.float64).resolution * 4000.0
    if manual_bbox is not None:
        bbox = manual_bbox
    elif zoom_to_profile:
        track_lon = np.asarray(lon_p, dtype=float).ravel()
        track_lat = np.asarray(lat_p, dtype=float).ravel()
        track_lon = track_lon[np.isfinite(track_lon)]
        track_lat = track_lat[np.isfinite(track_lat)]
        all_lon: list[float] = []
        all_lat: list[float] = []
        if track_lon.size:
            all_lon.extend([float(track_lon.min()), float(track_lon.max())])
        if track_lat.size:
            all_lat.extend([float(track_lat.min()), float(track_lat.max())])
        if (
            lon_s is not None
            and lat_s is not None
            and np.any(np.isfinite(lon_s))
            and np.any(np.isfinite(lat_s))
        ):
            gl = np.asarray(lon_s, dtype=float)
            gb = np.asarray(lat_s, dtype=float)
            gl = gl[np.isfinite(gl)]
            gb = gb[np.isfinite(gb)]
            if gl.size:
                all_lon.extend([float(gl.min()), float(gl.max())])
            if gb.size:
                all_lat.extend([float(gb.min()), float(gb.max())])
        if not all_lon or not all_lat:
            return None
        lo_lon, hi_lon = min(all_lon), max(all_lon)
        lo_la, hi_la = min(all_lat), max(all_lat)
        span_lon = max(hi_lon - lo_lon, eps)
        span_lat = max(hi_la - lo_la, eps)
        pad_lon = max(float(pad_deg_min), 0.06 * span_lon)
        pad_la = max(float(pad_deg_min), 0.06 * span_lat)
        bbox = (lo_lon - pad_lon, hi_lon + pad_lon, lo_la - pad_la, hi_la + pad_la)
    else:
        return None

    if grid_domain is not None:
        cx0, cx1, cy0, cy1 = clamp_bbox_to_domain(bbox, grid_domain)
        if cx0 >= cx1 - eps or cy0 >= cy1 - eps:
            return None
        return cx0, cx1, cy0, cy1
    return bbox


def load_gravity_plan_map_arrays(
    file_path: Path | str,
    *,
    bbox: Optional[tuple[float, float, float, float]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    """读取重力格网用于 lon-lat 平面图。

    ``bbox=(lon_lo, lon_hi, lat_lo, lat_hi)`` 时尽量只读该区域（GMT 一维 z 按行块读取）。

    返回 ``(lon_1d, lat_1d, z_2d, field_name, units_str)``，其中 ``z_2d`` 形状
    ``(len(lat_1d), len(lon_1d))``，维序与纬度、经度递增方向一致（若不递增则翻面）。
    """
    if xr is None:
        raise ImportError("需要 xarray")

    fp = Path(file_path).expanduser()
    if not fp.is_file():
        raise FileNotFoundError(f"重力数据文件不存在: {fp}")

    ds = open_gravity_netcdf_dataset(fp)
    try:
        if _is_gmt_flat_dataset(ds):
            layout = _gmt_flat_grid_layout(ds)
            if bbox is not None:
                return _load_gmt_flat_arrays_subset(layout, bbox)
            nz = np.asarray(layout["z_var"].values, dtype=float).ravel()
            nx, ny = int(layout["nx"]), int(layout["ny"])
            z_2d = nz.reshape((ny, nx), order="C")
            lon_1d, lat_1d, z_2d = _normalize_gravity_lon_lat_z(
                layout["lon_1d"], layout["lat_1d"], z_2d
            )
            name = str(layout["name"])
            units = str(layout["attrs"].get("units", "") or "").strip()
            return lon_1d, lat_1d, z_2d, name, units

        da = _pick_lazy_2d_data_array(ds)
        lat_dim, lon_dim = infer_geographic_dims(da)
        if bbox is not None:
            lon_lo, lon_hi, lat_lo, lat_hi = bbox
            try:
                da = da.sel(
                    {lon_dim: slice(lon_lo, lon_hi), lat_dim: slice(lat_lo, lat_hi)}
                )
            except Exception:
                pass
        da = da.load()
        da2 = da.transpose(lat_dim, lon_dim)
        lon_1d = np.asarray(da2.coords[lon_dim].values, dtype=float)
        lat_1d = np.asarray(da2.coords[lat_dim].values, dtype=float)
        z_2d = np.asarray(da2.values, dtype=float)
        name = str(da2.name or "gravity")
        attrs = getattr(da2, "attrs", {}) or {}
        units = str(attrs.get("units", "") or "").strip()
        lon_1d, lat_1d, z_2d = _normalize_gravity_lon_lat_z(lon_1d, lat_1d, z_2d)
        return lon_1d, lat_1d, z_2d, name, units
    finally:
        ds.close()


def crop_gravity_plan_map_arrays(
    lon_1d: np.ndarray,
    lat_1d: np.ndarray,
    z_2d: np.ndarray,
    lon_lo: float,
    lon_hi: float,
    lat_lo: float,
    lat_hi: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按经纬度矩形裁剪重力场（用于平面图只绘制视窗范围）。"""
    lon_1d = np.asarray(lon_1d, dtype=float)
    lat_1d = np.asarray(lat_1d, dtype=float)
    z_2d = np.asarray(z_2d, dtype=float)
    if lon_1d.size == 0 or lat_1d.size == 0 or z_2d.size == 0:
        return lon_1d, lat_1d, z_2d
    ix = np.where((lon_1d >= lon_lo) & (lon_1d <= lon_hi))[0]
    iy = np.where((lat_1d >= lat_lo) & (lat_1d <= lat_hi))[0]
    if ix.size == 0 or iy.size == 0:
        return lon_1d, lat_1d, z_2d
    i0, i1 = int(ix[0]), int(ix[-1])
    j0, j1 = int(iy[0]), int(iy[-1])
    return lon_1d[i0 : i1 + 1], lat_1d[j0 : j1 + 1], z_2d[j0 : j1 + 1, i0 : i1 + 1]


def decimate_gravity_plan_map_arrays(
    lon_1d: np.ndarray,
    lat_1d: np.ndarray,
    z_2d: np.ndarray,
    *,
    max_dim: int = 1600,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """显示用降采样：全球格网裁剪后仍过大时，按步长抽稀。"""
    lon_1d = np.asarray(lon_1d, dtype=float)
    lat_1d = np.asarray(lat_1d, dtype=float)
    z_2d = np.asarray(z_2d, dtype=float)
    ny, nx = z_2d.shape
    cap = max(64, int(max_dim))
    if ny <= cap and nx <= cap:
        return lon_1d, lat_1d, z_2d
    step_y = max(1, int(np.ceil(ny / cap)))
    step_x = max(1, int(np.ceil(nx / cap)))
    return lon_1d[::step_x], lat_1d[::step_y], z_2d[::step_y, ::step_x]


def imshow_extent_for_1d_lon_lat(
    lon_1d: np.ndarray,
    lat_1d: np.ndarray,
) -> tuple[float, float, float, float]:
    """规则 1D 经纬坐标 → ``imshow(extent=...)`` 四元组 (left, right, bottom, top)。"""
    lon_1d = np.asarray(lon_1d, dtype=float)
    lat_1d = np.asarray(lat_1d, dtype=float)
    if lon_1d.size >= 2:
        dlon = (float(lon_1d[-1]) - float(lon_1d[0])) / max(lon_1d.size - 1, 1)
    else:
        dlon = 0.0
    if lat_1d.size >= 2:
        dlat = (float(lat_1d[-1]) - float(lat_1d[0])) / max(lat_1d.size - 1, 1)
    else:
        dlat = 0.0
    left = float(lon_1d[0]) - 0.5 * dlon
    right = float(lon_1d[-1]) + 0.5 * dlon
    bottom = float(lat_1d[0]) - 0.5 * dlat
    top = float(lat_1d[-1]) + 0.5 * dlat
    return left, right, bottom, top
