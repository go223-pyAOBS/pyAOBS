"""无 GUI 的速度网格加载（Tk/Qt 共用）。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import xarray as xr

try:
    from pyAOBS.model_building.zeltform import ZeltVelocityModel2d
    from pyAOBS.visualization.gravity_obs_grid import dataset_looks_like_geographic_lon_lat_surface
    from pyAOBS.visualization.show_model import GridModelProcessor
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from pyAOBS.model_building.zeltform import ZeltVelocityModel2d
    from pyAOBS.visualization.gravity_obs_grid import dataset_looks_like_geographic_lon_lat_surface
    from pyAOBS.visualization.show_model import GridModelProcessor


def is_vin_path(path: Path) -> bool:
    """与 imodel_gui 一致的 v.in 文件名启发。"""
    name = path.name.lower()
    return name == "v.in" or name.endswith(".vin")


def is_vin_file_by_content(filename: str) -> bool:
    """轻量内容探测（与 ModelSurfaceMixin 逻辑一致，仅读前几行）。"""
    try:
        with open(filename, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline().strip()
            if not first_line:
                return False
            parts = first_line.split()
            if parts and parts[0].isdigit():
                second_line = f.readline().strip()
                if second_line:
                    parts2 = second_line.split()
                    if parts2 and parts2[0].isdigit():
                        third_line = f.readline().strip()
                        return bool(third_line)
    except OSError:
        pass
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
    path: str, *, vin_dx_km: float = 2.0, vin_dz_km: float = 0.5
) -> Tuple[xr.Dataset, Optional[object]]:
    """加载网格或 v.in，返回 (xr.Dataset, zelt_model_or_none)。"""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(path)

    is_vin = is_vin_path(p) or is_vin_file_by_content(str(p))
    if is_vin:
        zelt = ZeltVelocityModel2d(model_file=str(p))
        ds = zelt.to_xarray(dx=float(vin_dx_km), dz=float(vin_dz_km))
        return ds, zelt

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
