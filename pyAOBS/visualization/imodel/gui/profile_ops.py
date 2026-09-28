"""垂直剖面提取 / 平均 —— 与 Tk profiles 核心语义对齐。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
import xarray as xr

from pyAOBS.visualization.imodel import ProfileExtractor

DepthDatum = Literal["sf", "bm", "z0"]

DATUM_LABELS: dict[DepthDatum, str] = {
    "sf": "seafloor",
    "bm": "basement",
    "z0": "model top",
}


def format_v1d_filename(
    *,
    datum: DepthDatum,
    x_km: Optional[float] = None,
    x0_km: Optional[float] = None,
    x1_km: Optional[float] = None,
    kind: Literal["single", "avg", "envelope"] = "single",
) -> str:
    """建议文件名：``V_1D_from_{sf|bm|z0}_x….txt``（深度相对该基准）。"""
    tag = str(datum)
    if kind in ("avg", "envelope") and x0_km is not None and x1_km is not None:
        base = f"V_1D_from_{tag}_x{float(x0_km):.2f}-{float(x1_km):.2f}km"
        return f"{base}_envelope.txt" if kind == "envelope" else f"{base}.txt"
    if x_km is None:
        raise ValueError("single 文件名需要 x_km")
    return f"V_1D_from_{tag}_x{float(x_km):.2f}km.txt"


def write_v1d_txt(path: str | bytes, depth: np.ndarray, vp: np.ndarray) -> None:
    """两列空白分隔：depth(km)  vp(km/s)，无表头。"""
    d = np.asarray(depth, dtype=float).reshape(-1)
    v = np.asarray(vp, dtype=float).reshape(-1)
    if d.size != v.size:
        raise ValueError("depth / vp length mismatch")
    m = np.isfinite(d) & np.isfinite(v)
    arr = np.column_stack((d[m], v[m]))
    np.savetxt(path, arr, fmt="%.6f")


def write_v1d_envelope_txt(
    path: str | bytes,
    depth: np.ndarray,
    vp_min: np.ndarray,
    vp_max: np.ndarray,
) -> None:
    """三列空白分隔：depth  vp_min  vp_max（速度包络），无表头。"""
    d = np.asarray(depth, dtype=float).reshape(-1)
    lo = np.asarray(vp_min, dtype=float).reshape(-1)
    hi = np.asarray(vp_max, dtype=float).reshape(-1)
    if not (d.size == lo.size == hi.size):
        raise ValueError("depth / vp_min / vp_max length mismatch")
    m = np.isfinite(d) & np.isfinite(lo) & np.isfinite(hi)
    arr = np.column_stack((d[m], lo[m], hi[m]))
    np.savetxt(path, arr, fmt="%.6f")


def _interp_vp_to_grid(
    profile_depths: np.ndarray,
    profile_vp: np.ndarray,
    reference_depths: np.ndarray,
) -> np.ndarray:
    """线性插值到 reference_depths；区间外为 nan。"""
    d = np.asarray(profile_depths, dtype=float)
    v = np.asarray(profile_vp, dtype=float)
    m = np.isfinite(d) & np.isfinite(v)
    if np.sum(m) < 2:
        return np.full(reference_depths.shape, np.nan, dtype=float)
    d = d[m]
    v = v[m]
    order = np.argsort(d)
    d = d[order]
    v = v[order]
    uniq = np.concatenate(([True], np.diff(d) > 1e-9))
    d = d[uniq]
    v = v[uniq]
    if len(d) < 2:
        return np.full(reference_depths.shape, np.nan, dtype=float)
    return np.interp(
        np.asarray(reference_depths, dtype=float),
        d,
        v,
        left=np.nan,
        right=np.nan,
    )


def _apply_datum(
    profile: pd.DataFrame,
    x: float,
    datum_depth_fn: Optional[Callable[[float], float]],
) -> pd.DataFrame:
    """相对基准面校正；保证相对深度从 0 起算（在基准面处插值 Vp）。"""
    if datum_depth_fn is None:
        return profile.reset_index(drop=True)
    z0 = float(datum_depth_fn(float(x)))
    if not np.isfinite(z0) or z0 <= 0:
        return profile.reset_index(drop=True)

    depths = np.asarray(profile["depth"].values, dtype=float)
    vp = np.asarray(profile["vp"].values, dtype=float)
    m = np.isfinite(depths) & np.isfinite(vp)
    depths = depths[m]
    vp = vp[m]
    if depths.size < 1:
        raise RuntimeError(f"X={x:g} 处无有效速度采样")

    order = np.argsort(depths)
    depths = depths[order]
    vp = vp[order]

    if z0 >= float(depths[-1]):
        raise RuntimeError(f"X={x:g} 处基准面深度 {z0:g} km 超出模型底界")
    if z0 <= float(depths[0]):
        vp_at_datum = float(vp[0])
    else:
        vp_at_datum = float(np.interp(z0, depths, vp))

    adjusted = depths - z0
    keep = adjusted >= -1e-6
    if not np.any(keep):
        raise RuntimeError(f"X={x:g} 处剖面均在基准面之上，无有效深度")
    d_rel = np.maximum(adjusted[keep], 0.0)
    v_rel = vp[keep]

    order2 = np.argsort(d_rel)
    d_rel = d_rel[order2]
    v_rel = v_rel[order2]

    # 首行必须是 depth=0（界面处插值速度）
    if d_rel.size == 0 or d_rel[0] > 1e-9:
        d_rel = np.concatenate(([0.0], d_rel))
        v_rel = np.concatenate(([vp_at_datum], v_rel))
    else:
        d_rel = d_rel.copy()
        v_rel = v_rel.copy()
        d_rel[0] = 0.0
        v_rel[0] = vp_at_datum

    if d_rel.size > 1:
        uniq = np.ones(d_rel.size, dtype=bool)
        uniq[1:] = np.diff(d_rel) > 1e-9
        d_rel = d_rel[uniq]
        v_rel = v_rel[uniq]

    return pd.DataFrame({"depth": d_rel, "vp": v_rel}).reset_index(drop=True)


def _raw_vertical_at_x(grid_data: xr.Dataset, extractor: ProfileExtractor, x: float) -> pd.DataFrame:
    x_coord = extractor.x_coord
    z_coord = extractor.z_coord
    velocity_var = extractor.velocity_var
    try:
        velocity_data = grid_data[velocity_var]
        z_coords = np.asarray(grid_data.coords[z_coord].values, dtype=float)
        v_slice = velocity_data.sel({x_coord: float(x)}, method="nearest")
        vp_values = np.asarray(v_slice.values, dtype=float).reshape(-1)
        if vp_values.size != z_coords.size:
            if vp_values.size == 1:
                vp_values = np.full(z_coords.size, float(vp_values[0]), dtype=float)
            elif vp_values.size > z_coords.size:
                vp_values = vp_values[: z_coords.size]
            else:
                vp_values = np.pad(
                    vp_values,
                    (0, z_coords.size - vp_values.size),
                    constant_values=np.nan,
                )
        return pd.DataFrame({"depth": z_coords.copy(), "vp": vp_values})
    except Exception:
        return extractor.extract_vertical_profile(float(x))


def single_vertical_profile(
    grid_data: xr.Dataset,
    x: float,
    datum_depth_fn: Optional[Callable[[float], float]] = None,
    *,
    basement_depth_fn: Optional[Callable[[float], float]] = None,
) -> pd.DataFrame:
    """固定 X 抽取垂直 Vp–depth；可选相对基准面校正。"""
    # basement_depth_fn 保留兼容旧调用
    fn = datum_depth_fn if datum_depth_fn is not None else basement_depth_fn
    extractor = ProfileExtractor(grid_data)
    profile = _raw_vertical_at_x(grid_data, extractor, float(x))
    return _apply_datum(profile, float(x), fn)


@dataclass
class VerticalProfileBundle:
    """平均剖面 + 包络 + 各采样点剖面。"""

    mean: pd.DataFrame
    individuals: list[tuple[float, pd.DataFrame]] = field(default_factory=list)
    datum: DepthDatum = "z0"
    x_min: float = 0.0
    x_max: float = 0.0
    dx: float = 0.0

    @property
    def has_envelope(self) -> bool:
        return (
            "vp_min" in self.mean.columns
            and "vp_max" in self.mean.columns
            and len(self.individuals) > 1
        )


def extract_averaged_vertical_profile(
    grid_data: xr.Dataset,
    x_min: float,
    x_max: float,
    dx: float,
    datum_depth_fn: Optional[Callable[[float], float]] = None,
    *,
    basement_depth_fn: Optional[Callable[[float], float]] = None,
    datum: DepthDatum = "z0",
) -> VerticalProfileBundle:
    """
    在 [x_min, x_max] 上按 dx 取样；相对基准面校正后插值对齐，
    返回平均剖面（含 vp_min/vp_max 包络）及各点剖面。
    """
    if x_min >= x_max:
        raise ValueError("x_min 必须小于 x_max")
    if dx <= 0:
        raise ValueError("dx 必须大于 0")

    fn = datum_depth_fn if datum_depth_fn is not None else basement_depth_fn
    extractor = ProfileExtractor(grid_data)
    xs = np.arange(float(x_min), float(x_max) + 0.5 * float(dx), float(dx))
    if xs.size < 1:
        raise ValueError("X 采样为空")

    individuals: list[tuple[float, pd.DataFrame]] = []
    all_profiles: list[pd.DataFrame] = []
    for x in xs:
        raw = _raw_vertical_at_x(grid_data, extractor, float(x))
        profile = _apply_datum(raw, float(x), fn)
        individuals.append((float(x), profile.copy()))
        all_profiles.append(profile)

    if not all_profiles:
        raise RuntimeError("未能提取任何剖面")

    all_depths: list[float] = []
    for profile in all_profiles:
        d = np.asarray(profile["depth"].values, dtype=float)
        d = d[np.isfinite(d)]
        if d.size > 0:
            all_depths.extend(d.tolist())
    if not all_depths:
        raise RuntimeError("剖面中无有效深度")

    all_depths_array = np.asarray(all_depths, dtype=float)
    depth_min = float(np.nanmin(all_depths_array))
    depth_max = float(np.nanmax(all_depths_array))
    # 相对基准时强制深度网格从 0 起
    if fn is not None:
        depth_min = 0.0

    first_d = np.asarray(all_profiles[0]["depth"].values, dtype=float)
    first_d = first_d[np.isfinite(first_d)]
    if first_d.size > 1:
        depth_interval = float(np.mean(np.diff(np.sort(first_d))))
    else:
        depth_interval = 0.1
    depth_interval = max(depth_interval, 1e-4)

    reference_depths = np.arange(depth_min, depth_max + 0.5 * depth_interval, depth_interval)
    if reference_depths.size == 0 or abs(float(reference_depths[0]) - depth_min) > 1e-9:
        reference_depths = np.unique(np.concatenate(([depth_min], reference_depths)))


    vp_rows: list[np.ndarray] = []
    for profile in all_profiles:
        d = np.asarray(profile["depth"].values, dtype=float)
        v = np.asarray(profile["vp"].values, dtype=float)
        vp_rows.append(_interp_vp_to_grid(d, v, reference_depths))

    vp_matrix = np.asarray(vp_rows, dtype=float)
    averaged_vp = np.nanmean(vp_matrix, axis=0)
    vp_lo = np.nanmin(vp_matrix, axis=0)
    vp_hi = np.nanmax(vp_matrix, axis=0)
    valid_mask = np.isfinite(averaged_vp)
    if not np.any(valid_mask):
        raise RuntimeError("平均后无有效值")

    mean = pd.DataFrame(
        {
            "depth": reference_depths[valid_mask],
            "vp": averaged_vp[valid_mask],
            "vp_min": vp_lo[valid_mask],
            "vp_max": vp_hi[valid_mask],
        }
    ).reset_index(drop=True)

    return VerticalProfileBundle(
        mean=mean,
        individuals=individuals,
        datum=datum,
        x_min=float(x_min),
        x_max=float(x_max),
        dx=float(dx),
    )


def averaged_vertical_profile(
    grid_data: xr.Dataset,
    x_min: float,
    x_max: float,
    dx: float,
    basement_depth_fn: Optional[Callable[[float], float]] = None,
    *,
    datum_depth_fn: Optional[Callable[[float], float]] = None,
    datum: DepthDatum = "z0",
) -> pd.DataFrame:
    """兼容旧接口：仅返回平均剖面（含可选 vp_min/vp_max 列）。"""
    bundle = extract_averaged_vertical_profile(
        grid_data,
        x_min,
        x_max,
        dx,
        datum_depth_fn=datum_depth_fn,
        basement_depth_fn=basement_depth_fn,
        datum=datum,
    )
    return bundle.mean
