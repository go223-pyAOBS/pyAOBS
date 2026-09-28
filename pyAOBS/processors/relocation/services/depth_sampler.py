"""从地形 meta 构造深度采样器（返回 km，对齐 zplotpy 姿态校正语义）。"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from ..bathymetry_sampler import BathymetrySampler, build_bathymetry_sampler
from ..orientation_correction import OrientationObservation
from .terrain_io import xy_to_utm_guess

DepthSampler = Callable[[float, float], Optional[float]]


def normalize_depth_to_km(depth: float) -> float:
    d = abs(float(depth))
    return d / 1000.0 if d > 20.0 else d


def make_depth_sampler(
    terrain_meta_utm: Optional[dict],
    observations: Optional[List[OrientationObservation]] = None,
) -> Optional[DepthSampler]:
    """
    包装 BathymetrySampler：
    - 输入坐标若为经纬则先猜转为 UTM
    - 输出水深统一为 km
    """
    terrain_sampler: Optional[BathymetrySampler] = build_bathymetry_sampler(terrain_meta_utm)
    if terrain_sampler is None:
        return None
    coord_kind = str((terrain_meta_utm or {}).get("coord_kind", "utm")).lower()

    fallback_candidates: List[Tuple[float, float]] = []
    if observations:
        utm_vals = [o.source_xy_utm for o in observations if o.source_xy_utm is not None]
        geo_vals = [o.source_xy_geo for o in observations if o.source_xy_geo is not None]
        if utm_vals:
            u = np.asarray(utm_vals, dtype=float)
            fallback_candidates.append((float(np.median(u[:, 0])), float(np.median(u[:, 1]))))
        if geo_vals:
            g = np.asarray(geo_vals, dtype=float)
            gx, gy = float(np.median(g[:, 0])), float(np.median(g[:, 1]))
            fallback_candidates.append(xy_to_utm_guess(gx, gy))

    def _try_sample(cx: float, cy: float) -> Optional[float]:
        v = terrain_sampler(float(cx), float(cy))
        if v is not None and np.isfinite(float(v)):
            dk = normalize_depth_to_km(float(v))
            return dk if dk > 0 else None
        return None

    def _depth_sampler(x: float, y: float) -> Optional[float]:
        xx, yy = float(x), float(y)
        if coord_kind == "utm":
            xx, yy = xy_to_utm_guess(xx, yy)
        tries: List[Tuple[float, float]] = [(xx, yy), (yy, xx)]
        for cx, cy in fallback_candidates:
            tries.append((cx, cy))
        for cx, cy in tries:
            out = _try_sample(cx, cy)
            if out is not None:
                return out
        return None

    return _depth_sampler


def sample_initial_depth_km(
    observations: List[OrientationObservation],
    depth_sampler: DepthSampler,
) -> Optional[float]:
    """优先 OBS（receiver）XY，炮点坐标仅作回退。"""
    if not observations:
        return None
    for o in observations:
        d = depth_sampler(float(o.receiver_xyz[0]), float(o.receiver_xyz[1]))
        if d is not None and np.isfinite(float(d)) and float(d) > 0.0:
            return float(d)
    for o in observations:
        for xy in (o.source_xy_utm, o.source_xy_geo):
            if xy is None:
                continue
            d = depth_sampler(float(xy[0]), float(xy[1]))
            if d is not None and np.isfinite(float(d)) and float(d) > 0.0:
                return float(d)
    for o in observations:
        d = depth_sampler(float(o.source_xyz[0]), float(o.source_xyz[1]))
        if d is not None and np.isfinite(float(d)) and float(d) > 0.0:
            return float(d)
    return None
