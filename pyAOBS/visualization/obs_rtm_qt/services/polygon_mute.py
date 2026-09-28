# -*- coding: utf-8 -*-
"""
多边形 mute 几何（对齐 zplotpy.qt_fast_viewer._build_mute_inside_mask）。

坐标系: 点为 (x, t)，x 可为道序号或 offset_km；t 为秒。
默认：多边形内部保留，外部置零；invert=True 则相反（与 zplot Shift+M 一致）。
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]

# 工区 offset 表缓存：避免每次圈选都重读 shots_xz
_OFFSET_CACHE: Dict[str, Tuple[float, Dict[int, float]]] = {}


def build_mute_inside_mask(
    x_trace: float,
    t_display: np.ndarray,
    polygon: np.ndarray,
) -> np.ndarray:
    """纵向采样是否在多边形内（扫描线，同 qt_fast_viewer）。"""
    n = int(polygon.shape[0])
    if n < 3 or t_display.size == 0:
        return np.zeros_like(t_display, dtype=bool)
    y_hits: List[float] = []
    for i in range(n):
        x1, y1 = float(polygon[i, 0]), float(polygon[i, 1])
        x2, y2 = float(polygon[(i + 1) % n, 0]), float(polygon[(i + 1) % n, 1])
        if abs(x2 - x1) < 1e-12:
            if abs(x_trace - x1) < 1e-9:
                y_hits.extend([y1, y2])
            continue
        xmin, xmax = (x1, x2) if x1 <= x2 else (x2, x1)
        if x_trace < xmin or x_trace >= xmax:
            continue
        ratio = (x_trace - x1) / (x2 - x1)
        if 0.0 <= ratio <= 1.0:
            y_hits.append(y1 + ratio * (y2 - y1))
    if len(y_hits) < 2:
        return np.zeros_like(t_display, dtype=bool)
    y_hits.sort()
    mask = np.zeros_like(t_display, dtype=bool)
    for k in range(0, len(y_hits) - 1, 2):
        y0 = float(y_hits[k])
        y1 = float(y_hits[k + 1])
        lo, hi = (y0, y1) if y0 <= y1 else (y1, y0)
        mask |= (t_display >= lo) & (t_display <= hi)
    return mask


def _keep_mask_to_weights(keep: np.ndarray, n_taper: int) -> np.ndarray:
    """keep 区→1、mute 区→0；边界 n_taper 样点余弦过渡，减轻硬切假震相。"""
    keep = np.asarray(keep, dtype=bool)
    n = int(keep.size)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if n_taper <= 0 or not np.any(keep) or np.all(keep):
        return keep.astype(np.float64)
    muted = ~keep
    dist = np.empty(n, dtype=np.float64)
    last = -n
    for i in range(n):
        if muted[i]:
            last = i
            dist[i] = 0.0
        else:
            dist[i] = float(i - last)
    last = 2 * n
    for i in range(n - 1, -1, -1):
        if muted[i]:
            last = i
            dist[i] = 0.0
        else:
            dist[i] = min(dist[i], float(last - i))
    w = np.ones(n, dtype=np.float64)
    w[muted] = 0.0
    edge = keep & (dist <= float(n_taper))
    if np.any(edge):
        x = np.clip(dist[edge] / float(n_taper), 0.0, 1.0)
        w[edge] = 0.5 * (1.0 - np.cos(np.pi * x))
    return w


def apply_polygon_mute(
    gather: np.ndarray,
    times: np.ndarray,
    x_coords: Sequence[float],
    polygon_points: Sequence[Point],
    *,
    enabled: bool = True,
    invert: bool = False,
    display_vred: float = 0.0,
    x_reduce_origin: float = 0.0,
    tp: float = 0.15,
) -> np.ndarray:
    """
    gather (nt, ntr)；对每道用 x_coords[j] 与 times 做多边形 mute。
    invert=False: 保留多边形内部（外部置零）——与 zplot 默认一致。
    display_vred>0 时按折合时间判定，须与画布一致：
      t' = t - |x - x_reduce_origin| / vred
    （勿用 |x|/vred，OBS 不在 0 时会把圈内误清零）。
    tp: 边界余弦过渡（秒）；0=硬切。
    """
    data = np.asarray(gather, dtype=np.float32)
    if not enabled or len(polygon_points) < 3:
        return data.copy()
    polygon = np.asarray(polygon_points, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        return data.copy()

    out = data.copy()
    nt, ntr = out.shape
    times = np.asarray(times, dtype=np.float64)
    ns = int(min(nt, times.size))
    if ns <= 0 or ntr <= 0:
        return out
    t0 = times[:ns]
    dt = float(t0[1] - t0[0]) if ns > 1 else 0.004
    n_taper = int(round(max(0.0, float(tp)) / dt)) if dt > 1e-12 else 0
    vred = float(display_vred)
    x0 = float(x_reduce_origin)
    xs = np.asarray(x_coords, dtype=np.float64)
    if xs.size < ntr:
        xs = np.pad(xs, (0, ntr - xs.size), mode="edge")

    xmin = float(np.min(polygon[:, 0]))
    xmax = float(np.max(polygon[:, 0]))
    # 包围盒外：整道直接置零/保留，跳过扫描线
    for j in range(ntr):
        x = float(xs[j])
        if x < xmin or x > xmax:
            if not invert:
                out[:ns, j] = 0.0
            continue
        if vred > 0.0:
            t_use = t0 - abs(x - x0) / vred
        else:
            t_use = t0
        inside = build_mute_inside_mask(x, t_use, polygon)
        keep = ~inside if invert else inside
        if n_taper <= 0:
            out[:ns, j][~keep] = 0.0
        else:
            w = _keep_mask_to_weights(keep, n_taper)
            out[:ns, j] *= w.astype(np.float32)
    return out


def points_to_list(points: Iterable[Point]) -> List[List[float]]:
    return [[float(x), float(t)] for x, t in points]


def list_to_points(raw) -> List[Point]:
    out: List[Point] = []
    for p in raw or []:
        if len(p) >= 2:
            out.append((float(p[0]), float(p[1])))
    return out


def polygon_x_bounds(polygon_points: Sequence[Point]) -> Optional[Tuple[float, float]]:
    """多边形横轴包围盒；不足 3 点返回 None。"""
    if len(polygon_points) < 3:
        return None
    xs = [float(p[0]) for p in polygon_points]
    return float(min(xs)), float(max(xs))


def invalidate_offset_cache(workdir: Optional[str] = None) -> None:
    if workdir is None:
        _OFFSET_CACHE.clear()
        return
    _OFFSET_CACHE.pop(os.path.normpath(str(workdir)), None)


def shot_offset_km_map(project) -> Dict[int, float]:
    """全炮模型测线坐标 (km) 表；按工区路径+mtime 缓存。"""
    from .montage import _shot_x_by_index, montage_model_x_km
    from ..project import list_shot_rsf

    wd = os.path.normpath(str(getattr(project, "workdir", "") or ""))
    sx = project.path(project.shots_xz) if wd else ""
    ox = project.path(project.obs_xz) if wd else ""
    try:
        mtime = max(
            os.path.getmtime(sx) if sx and os.path.isfile(sx) else 0.0,
            os.path.getmtime(ox) if ox and os.path.isfile(ox) else 0.0,
        )
    except OSError:
        mtime = 0.0
    hit = _OFFSET_CACHE.get(wd)
    if hit is not None and abs(hit[0] - mtime) < 1e-9:
        return hit[1]

    shot_x = _shot_x_by_index(project)
    if shot_x:
        mapping = {int(i): float(x) for i, x in shot_x.items()}
    else:
        mapping = {}
        for path in list_shot_rsf(project):
            base = os.path.basename(path)
            try:
                ishot = int(base.replace("shot_", "").replace(".rsf", ""))
            except ValueError:
                continue
            mapping[ishot] = float(montage_model_x_km(project, ishot))
    _OFFSET_CACHE[wd] = (mtime, mapping)
    return mapping


def shot_ids_from_polygon(
    project,
    polygon_points: Sequence[Point],
    *,
    invert: bool = False,
) -> List[int]:
    """
    由多边形独立圈选炮号（与手选无关）。

    规则：炮点拼图偏移距落在多边形 x 包围盒内则入选；
    invert=True 时取包围盒外的炮。
    """
    bounds = polygon_x_bounds(polygon_points)
    if bounds is None:
        return []
    xmin, xmax = bounds
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    pad = max(1e-6, 0.01 * max(xmax - xmin, 1e-3))
    lo, hi = xmin - pad, xmax + pad

    mapping = shot_offset_km_map(project)
    out: List[int] = []
    for ishot, off in mapping.items():
        inside = lo <= float(off) <= hi
        if invert:
            if not inside:
                out.append(int(ishot))
        elif inside:
            out.append(int(ishot))
    out.sort()
    return out
