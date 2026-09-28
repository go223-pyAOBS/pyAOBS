"""将姿态解应用到三分量波形（主图旋转预览）。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pyAOBS.geometry_roles import DEFAULT_GEOM, GeomMode, physical_shot_obs_xyz, resolve_geom

from ..orientation_correction import OrientationCorrectionResult, rotate_components
from .models import AttitudeSolution


def rotate_from_result(
    r: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    result: OrientationCorrectionResult,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return rotate_components(r, t, z, result.azimuth_deg, result.tilt_deg)


def _len_to_km(v: float) -> float:
    d = abs(float(v))
    return d / 1000.0 if d > 50.0 else d


def _as_1d(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=float).reshape(-1)


def _paste_1d(orig: Any, new: np.ndarray) -> np.ndarray:
    out = _as_1d(orig).copy()
    n = int(min(out.size, np.asarray(new).size))
    if n > 0:
        out[:n] = np.asarray(new, dtype=float).reshape(-1)[:n]
    return out


def _shot_obs_xy(th: Any, *, geom: GeomMode, use_utm: bool) -> Tuple[np.ndarray, np.ndarray]:
    """物理 (shot_xy, obs_xy)。"""
    shot, obs = physical_shot_obs_xyz(th, geom=geom, use_utm=use_utm)
    return np.asarray(shot[:2], dtype=float), np.asarray(obs[:2], dtype=float)


def solution_cache_key(
    loaded_id: int,
    n_traces: int,
    sol: AttitudeSolution | Dict[str, float],
) -> Tuple:
    if isinstance(sol, AttitudeSolution):
        d = sol.to_dict()
    else:
        d = dict(sol or {})
    return (
        int(loaded_id),
        int(n_traces),
        float(d.get("azimuth_deg", 0.0)),
        float(d.get("tilt_deg", 0.0)),
        float(d.get("dx", 0.0)),
        float(d.get("dy", 0.0)),
        float(d.get("dz", 0.0)),
        float(d.get("time_shift_sec", 0.0)),
    )


def apply_orientation_to_gather(
    traces: List[np.ndarray],
    offsets: np.ndarray,
    trace_headers: list,
    solution: AttitudeSolution | Dict[str, float],
    *,
    prefer_utm: bool = True,
    geom: GeomMode = DEFAULT_GEOM,
) -> Tuple[List[np.ndarray], np.ndarray, int]:
    """
    对全部 (ishoti, ireci) 三分量组做旋转，并按水平位移投影更新偏移距。

    Returns:
        (out_traces, out_offsets, n_groups_rotated)
    """
    if isinstance(solution, AttitudeSolution):
        az = float(solution.azimuth_deg)
        tilt = float(solution.tilt_deg)
        dx = float(solution.dx)
        dy = float(solution.dy)
    else:
        az = float(solution.get("azimuth_deg", 0.0))
        tilt = float(solution.get("tilt_deg", 0.0))
        dx = float(solution.get("dx", 0.0))
        dy = float(solution.get("dy", 0.0))

    n = min(len(traces), len(trace_headers), int(np.asarray(offsets).size))
    # 深拷贝 1D，避免与 loaded 共享内存被后续处理链踩坏
    out_traces: List[np.ndarray] = [_as_1d(traces[i]).copy() for i in range(len(traces))]
    out_offsets = np.asarray(offsets, dtype=float).copy()
    if n == 0:
        return out_traces, out_offsets, 0

    geom_mode = resolve_geom(geom, list(trace_headers[:n]))
    use_utm = bool(prefer_utm)

    groups: Dict[Tuple[int, int], Dict[int, int]] = {}
    for i in range(n):
        th = trace_headers[i]
        shot = int(getattr(th, "ishoti", 0) or 0)
        rec = int(getattr(th, "ireci", 0) or 0)
        comp = int(getattr(th, "itypei", 0) or 0)
        if shot <= 0 or rec <= 0 or comp not in (1, 2, 3):
            continue
        groups.setdefault((shot, rec), {})[comp] = int(i)

    n_rot = 0
    group_index_sets: List[Tuple[int, int, int]] = []
    for g in groups.values():
        if not (1 in g and 2 in g and 3 in g):
            continue
        iz, ir, it = int(g[1]), int(g[2]), int(g[3])
        z = _as_1d(traces[iz])
        r = _as_1d(traces[ir])
        t = _as_1d(traces[it])
        nuse = int(min(z.size, r.size, t.size))
        if nuse < 8:
            continue
        if iz == ir or iz == it or ir == it:
            # 道头 itypei 冲突导致同道当多分量，拒绝旋转以免污染 Z
            continue
        # 主图/预览：|tilt|<1e-6 时只旋水平分量，Z 样点字节级保持原道
        if abs(float(tilt)) < 1e-6:
            r2, t2, _z_unused = rotate_components(r[:nuse], t[:nuse], z[:nuse], az, 0.0)
            out_traces[iz] = _as_1d(traces[iz]).copy()
            out_traces[ir] = _paste_1d(traces[ir], r2)
            out_traces[it] = _paste_1d(traces[it], t2)
        else:
            r2, t2, z2 = rotate_components(r[:nuse], t[:nuse], z[:nuse], az, tilt)
            out_traces[iz] = _paste_1d(traces[iz], z2)
            out_traces[ir] = _paste_1d(traces[ir], r2)
            out_traces[it] = _paste_1d(traces[it], t2)
        group_index_sets.append((iz, ir, it))
        n_rot += 1

    # 偏移距：沿 炮→OBS 方向投影水平位移（物理角色，geom=obs）
    disp = np.asarray([dx, dy], dtype=float)
    conv_samples: List[float] = []
    geom0_list: List[Optional[np.ndarray]] = [None] * n
    for i in range(n):
        th = trace_headers[i]
        try:
            shot_xy, obs_xy = _shot_obs_xy(th, geom=geom_mode, use_utm=use_utm)
            v0 = obs_xy - shot_xy
            geom0 = float(np.linalg.norm(v0))
            if np.isfinite(geom0) and geom0 > 1e-9:
                geom0_list[i] = np.asarray(v0, dtype=float)
                if i < out_offsets.size and np.isfinite(out_offsets[i]) and abs(float(out_offsets[i])) > 1e-9:
                    conv_samples.append(abs(float(out_offsets[i])) / geom0)
        except Exception:
            continue

    if conv_samples:
        conv = float(np.median(np.asarray(conv_samples, dtype=float)))
    else:
        nd = float(np.linalg.norm(disp))
        conv = _len_to_km(nd) / max(nd, 1e-9) if nd > 0 else 0.0
    if not np.isfinite(conv) or conv <= 0:
        conv = 0.0

    disp_km = _len_to_km(float(np.linalg.norm(disp)))
    for i in range(n):
        if i >= out_offsets.size or not np.isfinite(out_offsets[i]):
            continue
        v0 = geom0_list[i]
        if v0 is None:
            continue
        n0 = float(np.linalg.norm(v0))
        if not np.isfinite(n0) or n0 <= 1e-9:
            continue
        u0 = np.asarray(v0, dtype=float) / n0
        delta_km = float(np.dot(disp, u0)) * float(conv)
        delta_km = float(np.clip(delta_km, -disp_km, disp_km))
        out_offsets[i] = float(out_offsets[i]) + delta_km

    # 同一三分量组强制共用 Z 的偏移，避免 R/T 道头坐标缺失导致偏移飞出视窗（看起来像“空道”）
    for iz, ir, it in group_index_sets:
        if 0 <= iz < out_offsets.size and np.isfinite(out_offsets[iz]):
            off_z = float(out_offsets[iz])
            if 0 <= ir < out_offsets.size:
                out_offsets[ir] = off_z
            if 0 <= it < out_offsets.size:
                out_offsets[it] = off_z

    return out_traces, out_offsets, n_rot


def _add_to_attr(th: Any, names: Tuple[str, ...], delta: float) -> bool:
    """若道头存在任一字段则加上 delta，返回是否写入。"""
    if not np.isfinite(delta) or abs(float(delta)) < 1e-15:
        return False
    for name in names:
        if not hasattr(th, name):
            continue
        try:
            cur = float(getattr(th, name) or 0.0)
            if not np.isfinite(cur):
                cur = 0.0
            setattr(th, name, float(cur + float(delta)))
            return True
        except Exception:
            continue
    return False


def update_obs_geometry_headers(
    trace_headers: list,
    *,
    dx: float,
    dy: float,
    dz: float,
    geom: GeomMode = DEFAULT_GEOM,
    prefer_utm: bool = True,
) -> int:
    """把 (dx,dy,dz) 加到 OBS 侧道头坐标，返回更新道数。"""
    geom_mode = resolve_geom(geom, list(trace_headers or []))
    n_upd = 0
    for th in trace_headers or []:
        changed = False
        if geom_mode == "obs":
            # 本工区：OBS = s*
            if prefer_utm:
                changed |= _add_to_attr(th, ("sxutm", "sx_utm", "sx"), dx)
                changed |= _add_to_attr(th, ("syutm", "sy_utm", "sy"), dy)
            else:
                changed |= _add_to_attr(th, ("slong", "sx_geo", "s_lon"), dx)
                changed |= _add_to_attr(th, ("slat", "sy_geo", "s_lat"), dy)
            changed |= _add_to_attr(th, ("sz", "selev", "sdepth", "swdepth", "swdep"), dz)
        else:
            # segy 字面：OBS = r*/g*
            if prefer_utm:
                changed |= _add_to_attr(th, ("rxutm", "rx_utm", "gx", "rx"), dx)
                changed |= _add_to_attr(th, ("ryutm", "ry_utm", "gy", "ry"), dy)
            else:
                changed |= _add_to_attr(th, ("rlong", "rx_geo", "g_lon"), dx)
                changed |= _add_to_attr(th, ("rlat", "ry_geo", "g_lat"), dy)
            changed |= _add_to_attr(th, ("rz", "relev", "gelev"), dz)
        if changed:
            n_upd += 1
    return n_upd


def sync_offset_headers(trace_headers: list, offsets: np.ndarray) -> int:
    """把 offsets 数组写回道头 offsti/offset。"""
    n = 0
    off = np.asarray(offsets, dtype=float)
    for i, th in enumerate(trace_headers or []):
        if i >= off.size or not np.isfinite(off[i]):
            continue
        v = float(off[i])
        wrote = False
        for name in ("offsti", "offset", "offs"):
            if hasattr(th, name):
                try:
                    setattr(th, name, v)
                    wrote = True
                except Exception:
                    pass
        if wrote:
            n += 1
    return n


def commit_orientation_to_loaded(
    loaded: Dict[str, Any],
    solution: AttitudeSolution | Dict[str, float],
    *,
    prefer_utm: bool = True,
    geom: GeomMode = DEFAULT_GEOM,
) -> Dict[str, Any]:
    """
    把姿态解永久写入内存中的 loaded：
      - 旋转全部完整三分量组波形
      - 更新 OBS 道头几何 (dx,dy,dz)
      - 更新偏移距数组与道头

    不写磁盘。返回统计 dict。
    """
    traces = loaded.get("traces", [])
    offsets = np.asarray(loaded.get("offsets", []), dtype=float)
    headers = loaded.get("trace_headers", []) or []
    if isinstance(solution, AttitudeSolution):
        dx = float(solution.dx)
        dy = float(solution.dy)
        dz = float(solution.dz)
    else:
        dx = float(solution.get("dx", 0.0))
        dy = float(solution.get("dy", 0.0))
        dz = float(solution.get("dz", 0.0))

    out_traces, out_offsets, n_rot = apply_orientation_to_gather(
        list(traces),
        offsets,
        headers,
        solution,
        prefer_utm=prefer_utm,
        geom=geom,
    )
    # 写回波形 / 偏移（替换列表内容，尽量保持引用）
    if isinstance(traces, list):
        traces[:] = out_traces
    else:
        loaded["traces"] = out_traces
    loaded["offsets"] = np.asarray(out_offsets, dtype=float)

    n_geom = update_obs_geometry_headers(
        headers,
        dx=dx,
        dy=dy,
        dz=dz,
        geom=geom,
        prefer_utm=prefer_utm,
    )
    n_off = sync_offset_headers(headers, loaded["offsets"])
    return {
        "n_groups_rotated": int(n_rot),
        "n_headers_geom": int(n_geom),
        "n_headers_offset": int(n_off),
    }
