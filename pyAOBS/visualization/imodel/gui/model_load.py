"""无 GUI 的速度网格加载（Tk/Qt 共用）。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import xarray as xr


def is_vin_path(path: Path) -> bool:
    """与公用 ``vin_io`` 一致的文件名启发。"""
    from pyAOBS.modeling.rayinvr.vin_io import is_vin_path as _is

    return _is(path)


def is_vin_file_by_content(filename: str) -> bool:
    """轻量内容探测（委托 ``vin_io``）。"""
    from pyAOBS.modeling.rayinvr.vin_io import is_vin_content

    return is_vin_content(filename)


def is_smesh_path(path: Path) -> bool:
    """TOMO2D smesh 文件名启发（含 ``out.smesh.1.1`` 这类后缀）。"""
    p = Path(path)
    name = p.name.lower()
    suf = p.suffix.lower()
    return ".smesh" in name or suf in {".smesh", ".mesh"}


def looks_like_smesh_content(path: str | Path) -> bool:
    """轻量内容探测：首行 ``nx nz v_water v_air``（与 obs_rtm 一致）。"""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            parts = f.readline().split()
        if len(parts) < 4:
            return False
        nx, nz = int(float(parts[0])), int(float(parts[1]))
        return nx > 1 and nz > 1
    except Exception:
        return False


_GEIGR_HINT = (
    "此文件看起来像经纬度平面格网（如 Sandwell ``gravity_world.grd`` 全球重力 anomaly），不是成像剖面速度网格。\n"
    "imodel **Load Vp Model** 需要：横向距离 × 深度坐标 (km)，以及 velocity / vp / vs 等量。\n\n"
    "若要为剖面叠加这条观测重力数据，请先加载速度模型；打开「Gravity Toolbox」，在观测重力中选择该 "
    ".grd 路径（world .grd）。\n\n"
    "若 Python 报错无法打开 `.grd`，请先安装：`pip install netCDF4`，或用 "
    "`gmt grdconvert in.grd out.nc=nc4`。"
)


def load_velocity_grid(
    path: str,
    *,
    vin_dx_km: float = 2.0,
    vin_dz_km: float = 0.5,
    smesh_dx_km: Optional[float] = None,
    smesh_dz_km: Optional[float] = None,
) -> Tuple[xr.Dataset, Optional[object]]:
    """加载网格 / v.in / TOMO2D smesh，返回 (xr.Dataset, zelt_model_or_none)。

    smesh 默认按节点原始间距栅格化（``smesh_dx_km`` / ``smesh_dz_km`` 为 None）；
    与 ``SlownessMesh2D.to_xarray`` 一致。v.in 仍用 ``vin_dx_km`` / ``vin_dz_km``。
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(path)

    # smesh 优先于 v.in（文件名含 .smesh 时绝不会是 Zelt）
    try_smesh = is_smesh_path(p) or (
        p.suffix.lower() in {"", ".mesh"} and looks_like_smesh_content(p)
    )
    if try_smesh:
        try:
            from pyAOBS.model_building.tomoform import SlownessMesh2D
        except ImportError:
            import importlib.util

            _tf = Path(__file__).resolve().parents[3] / "model_building" / "tomoform.py"
            _spec = importlib.util.spec_from_file_location("pyaobs_tomoform_smesh", _tf)
            if _spec is None or _spec.loader is None:
                raise
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            SlownessMesh2D = _mod.SlownessMesh2D

        mesh = SlownessMesh2D.from_file(str(p))
        ds = mesh.to_xarray(dx=smesh_dx_km, dz=smesh_dz_km)
        return ds, None

    from pyAOBS.modeling.rayinvr.vin_io import is_vin_file, load_zelt_model

    if is_vin_file(p):
        zelt = load_zelt_model(p)
        ds = zelt.to_xarray(dx=float(vin_dx_km), dz=float(vin_dz_km))
        return ds, zelt

    # 网格路径才拉 show_model（依赖 pygmt），避免 smesh/v.in 被连带卡住
    try:
        from pyAOBS.visualization.imodel.gravity_obs_grid import (
            dataset_looks_like_geographic_lon_lat_surface,
        )
        from pyAOBS.visualization.show_model import GridModelProcessor
    except ImportError:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from pyAOBS.visualization.imodel.gravity_obs_grid import (
            dataset_looks_like_geographic_lon_lat_surface,
        )
        from pyAOBS.visualization.show_model import GridModelProcessor

    proc = GridModelProcessor(grid_file=str(p))
    ds = proc.velocity_grid
    if ds is None:
        raise IOError(f"未能打开网格: {p}")
    if dataset_looks_like_geographic_lon_lat_surface(ds):
        try:
            ds.close()
        except Exception:
            pass
        proc.velocity_grid = None
        raise ValueError(_GEIGR_HINT)
    return ds, None
