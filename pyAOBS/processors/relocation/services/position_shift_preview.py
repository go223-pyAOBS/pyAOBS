"""校正前后 OBS 位置对比（无 Qt）。

业务约定（主动源 OBS，对齐 pyAOBS.geometry_roles / RTM geom=obs）：
  - 待校正对象通常是 **唯一一台 OBS**（OrientationObservation.receiver_xyz）
  - 各道对应 **不同炮点**（OrientationObservation.source_xyz）
  - (dx,dy,dz) 加在 OBS 上
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..orientation_correction import OrientationObservation


@dataclass
class ObsShiftPoint:
    """唯一 OBS 校正前后位置。"""

    key: str
    x0: float
    y0: float
    z0: float
    x1: float
    y1: float
    z1: float
    n_obs: int = 1  # 参与汇总的观测条数
    trace_indices: List[int] = field(default_factory=list)

    @property
    def dx(self) -> float:
        return float(self.x1 - self.x0)

    @property
    def dy(self) -> float:
        return float(self.y1 - self.y0)

    @property
    def dz(self) -> float:
        return float(self.z1 - self.z0)

    @property
    def horizontal(self) -> float:
        return float(math.hypot(self.dx, self.dy))


@dataclass
class ObsShiftSummary:
    """单 OBS + 多炮点 的对比摘要。"""

    points: List[ObsShiftPoint]  # 长度应为 0 或 1
    shot_xy: np.ndarray  # (N, 2) 炮点
    dx: float
    dy: float
    dz: float
    horizontal: float
    azimuth_from_north_deg: float
    math_azimuth_deg: float
    coord_unit: str
    obs_side: str = "receiver"  # 判定 OBS 来自 observation 的哪一端
    message: str = ""


def _infer_coord_unit(xs: Sequence[float], ys: Sequence[float], disp: float) -> str:
    vals = [abs(float(v)) for v in list(xs) + list(ys) if np.isfinite(v)]
    if not vals:
        return "unknown"
    scale = max(vals)
    if scale > 1000.0 or abs(float(disp)) > 20.0:
        return "m"
    if scale < 50.0:
        return "km"
    return "unknown"


def _azimuth_from_north_cw(dx: float, dy: float) -> float:
    if abs(dx) < 1e-15 and abs(dy) < 1e-15:
        return float("nan")
    return float(math.degrees(math.atan2(dx, dy)) % 360.0)


def _math_azimuth(dx: float, dy: float) -> float:
    if abs(dx) < 1e-15 and abs(dy) < 1e-15:
        return float("nan")
    return float(math.degrees(math.atan2(dy, dx)) % 360.0)


def _xy_var(pts: np.ndarray) -> float:
    if pts.size == 0 or pts.shape[0] < 2:
        return 0.0
    return float(np.nanvar(pts[:, 0]) + np.nanvar(pts[:, 1]))


def _unique_xy(pts: np.ndarray, decimals: int = 1) -> np.ndarray:
    if pts.size == 0:
        return np.empty((0, 2), dtype=float)
    arr = np.asarray(pts, dtype=float)
    rounded = np.round(arr, int(decimals))
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return arr[np.sort(idx)]


def split_obs_and_shots(
    observations: List[OrientationObservation],
) -> Tuple[np.ndarray, np.ndarray, str, List[int]]:
    """
    返回 (obs_xyz Nx3, shot_xy Mx2, obs_side, trace_indices)。

    主路径：receiver=OBS，source=炮（geom=obs 组装后）。
    """
    rec_list: List[np.ndarray] = []
    src_list: List[np.ndarray] = []
    traces: List[int] = []
    for o in observations or []:
        rec = np.asarray(o.receiver_xyz[:3], dtype=float)
        src = np.asarray(o.source_xyz[:3], dtype=float)
        if not (np.isfinite(rec[:2]).all() and np.isfinite(src[:2]).all()):
            continue
        if abs(float(rec[0])) + abs(float(rec[1])) < 1e-12 and abs(float(src[0])) + abs(float(src[1])) < 1e-12:
            continue
        rec_list.append(rec)
        src_list.append(src)
        traces.append(int(o.trace_idx))

    if not rec_list:
        return (
            np.empty((0, 3), dtype=float),
            np.empty((0, 2), dtype=float),
            "receiver",
            [],
        )

    rec_arr = np.asarray(rec_list, dtype=float)
    src_arr = np.asarray(src_list, dtype=float)
    var_rec = _xy_var(rec_arr[:, :2])
    var_src = _xy_var(src_arr[:, :2])

    # 主路径（geometry_roles / geom=obs）：receiver=OBS，source=炮。
    # 仅当 receiver 明显更分散时做安全对调（上游角色仍反）。
    if var_rec > var_src * 2.0 + 1e-12:
        obs_xyz = src_arr
        shot_xy = rec_arr[:, :2]
        side = "source(swapped-sanity)"
    else:
        obs_xyz = rec_arr
        shot_xy = src_arr[:, :2]
        side = "receiver"

    return obs_xyz, shot_xy, side, traces


def build_obs_shift_summary(
    observations: List[OrientationObservation],
    position_correction: Tuple[float, float, float],
    *,
    round_decimals: int = 2,
) -> ObsShiftSummary:
    """
    汇总 **唯一 OBS** 校正前后位置 + 全部炮点。

    OBS 初值取各观测中 OBS 端坐标的中位数（抗噪）；校正后 = 初值 + (dx,dy,dz)。
    """
    dx, dy, dz = (float(v) for v in position_correction)
    horiz = float(math.hypot(dx, dy))
    az_n = _azimuth_from_north_cw(dx, dy)
    az_m = _math_azimuth(dx, dy)

    obs_xyz, shot_xy_raw, side, traces = split_obs_and_shots(observations)
    shot_xy = _unique_xy(shot_xy_raw, decimals=max(0, int(round_decimals) - 1))

    points: List[ObsShiftPoint] = []
    if obs_xyz.shape[0] > 0:
        x0 = float(np.nanmedian(obs_xyz[:, 0]))
        y0 = float(np.nanmedian(obs_xyz[:, 1]))
        z0 = float(np.nanmedian(obs_xyz[:, 2]))
        points.append(
            ObsShiftPoint(
                key="OBS",
                x0=x0,
                y0=y0,
                z0=z0,
                x1=x0 + dx,
                y1=y0 + dy,
                z1=z0 + dz,
                n_obs=int(obs_xyz.shape[0]),
                trace_indices=list(traces),
            )
        )

    xs: List[float] = []
    ys: List[float] = []
    for p in points:
        xs.extend([p.x0, p.x1])
        ys.extend([p.y0, p.y1])
    if shot_xy.size:
        xs.extend(shot_xy[:, 0].tolist())
        ys.extend(shot_xy[:, 1].tolist())
    unit = _infer_coord_unit(xs, ys, horiz)

    if not points:
        msg = "无有效 OBS 坐标，无法绘制位置对比"
    else:
        az_txt = f"{az_n:.1f}°" if np.isfinite(az_n) else "—"
        side_txt = "receiver端" if side == "receiver" else "source端(角色已按方差纠正)"
        msg = (
            f"OBS×1（取自{side_txt}，{points[0].n_obs}条观测中位数）| "
            f"炮点×{int(shot_xy.shape[0])} | "
            f"水平位移={horiz:.3f} {unit} | 方位(北顺时针)={az_txt} | dz={dz:.3f} {unit}"
        )

    return ObsShiftSummary(
        points=points,
        shot_xy=np.asarray(shot_xy, dtype=float),
        dx=dx,
        dy=dy,
        dz=dz,
        horizontal=horiz,
        azimuth_from_north_deg=az_n,
        math_azimuth_deg=az_m,
        coord_unit=unit,
        obs_side=side,
        message=msg,
    )


def collect_source_xy(observations: List[OrientationObservation]) -> np.ndarray:
    """兼容旧接口：返回判定后的炮点 XY。"""
    _obs, shot_xy, _side, _tr = split_obs_and_shots(observations)
    return _unique_xy(shot_xy, decimals=1)
