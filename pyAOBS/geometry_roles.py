# -*- coding: utf-8 -*-
"""
pyAOBS 炮点 / OBS 几何角色约定（全项目共用）
============================================

权威依据（本工区 OBS 装填，与 RTM ``--geom obs`` 一致）:

  - ``visualization/obs_rtm_qt/madagascar_obs_rtm/diag/obs_segy_geometry.txt``
  - ``processors/raw2sac/segy_trace_header.py``（``obs_xy_from_header`` / ``shot_xy_from_header``）

-------------------------------------------------------------------------------
1. 两套「名字」不要混用
-------------------------------------------------------------------------------

A) **道头槽位名**（SEGY/SU/Z 字段名，字面拷贝，不改字节布局）

   ========  ====================  ==============================
   槽位      SEGY 字面含义           本工区 OBS 实际装填（默认）
   ========  ====================  ==============================
   sx, sy    Source XY               **OBS** UTM
   gx, gy    Group XY                **炮点 / Gun** UTM
   selev     source elev             OBS 高程（海底，常为负）
   gelev     group elev              炮点高程（近海面）
   sdepth    source depth            OBS 深度（正）
   swdep     water @ source 槽       ≈ OBS 处水深
   gwdep     water @ group 槽        ≈ 炮点处水深
   ========  ====================  ==============================

   口诀（本工区）：**s* = Station/OBS（固定）**；**g* = Gun/炮（移动）**。
   这与 SEGY 英文 Source/Group **字面相反**，以工区装填为准。

B) **物理语义名**（算法 / OrientationObservation / RTM shots_xz·obs_xz）

   - ``shot_*`` / ``source_phys`` = 气枪炮点 ← 道头 ``gx,gy``（Z: ``rxutm``）
   - ``obs_*``  / ``receiver_obs`` = OBS      ← 道头 ``sx,sy``（Z: ``sxutm``）

Z 格式由 SU 字面映射（``su2z_hhb.py``，不交换）:

   sx,sy → sxutm,syutm / slong,slat
   gx,gy → rxutm,ryutm / rlong,rlat

因此对本工区数据：Z 的 ``s*`` ≈ OBS，Z 的 ``r*`` ≈ 炮点。
UI 若仍写「sxutm=震源」仅为 SEGY 字面注释，物理角色以本模块为准。

-------------------------------------------------------------------------------
2. geom 模式
-------------------------------------------------------------------------------

- ``obs``  （默认）：本工区 / RTM 约定 —— OBS=s*，炮=g*/r*
- ``segy``：严格 SEGY 字面 —— 震源=s*，检波=g*/r*
- ``auto``：未知装填时用「唯一 XY 更少的一侧 = OBS」启发式（仅回退）

新代码请显式传 ``geom="obs"``；禁止再各写一套方差/深度启发式当主路径。

-------------------------------------------------------------------------------
3. RTM 互易注意
-------------------------------------------------------------------------------

Madagascar 互易正传里 ``sou=OBS, rec=炮`` 是**波场角色**，不是道头 s*/g* 字面。
物理导入仍遵循：``shots_xz``←炮(gx)，``obs_xz``←OBS(sx)。
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Tuple

import numpy as np

GeomMode = Literal["obs", "segy", "auto"]

DEFAULT_GEOM: GeomMode = "obs"


def _f(th: Any, *names: str, default: float = 0.0) -> float:
    for name in names:
        if hasattr(th, name):
            try:
                v = getattr(th, name)
                if v is None:
                    continue
                return float(v)
            except Exception:
                continue
        if isinstance(th, dict) and name in th:
            try:
                return float(th[name])
            except Exception:
                continue
    return float(default)


def _valid_xy(x: float, y: float) -> bool:
    return bool(np.isfinite(x) and np.isfinite(y) and (abs(x) > 1e-9 or abs(y) > 1e-9))


def header_slot_obs_xy(th: Any, *, use_utm: bool = True) -> Tuple[float, float]:
    """道头槽：OBS XY（本工区 = sx/sy 或 Z sxutm）。"""
    if use_utm:
        return _f(th, "sxutm", "sx_utm", "sx"), _f(th, "syutm", "sy_utm", "sy")
    return _f(th, "slong", "sx_geo", "s_lon"), _f(th, "slat", "sy_geo", "s_lat")


def header_slot_shot_xy(th: Any, *, use_utm: bool = True) -> Tuple[float, float]:
    """道头槽：炮点 XY（本工区 = gx/gy 或 Z rxutm）。"""
    if use_utm:
        return _f(th, "rxutm", "rx_utm", "gx", "rx"), _f(th, "ryutm", "ry_utm", "gy", "ry")
    return _f(th, "rlong", "rx_geo", "g_lon"), _f(th, "rlat", "ry_geo", "g_lat")


def header_slot_obs_z(th: Any) -> float:
    """OBS 垂向：优先 sz/selev/sdepth/swdepth（s* 槽）。"""
    sw = _f(th, "swdepth", "swdep", default=0.0)
    sz = _f(th, "sz", "selev", "sdepth", default=sw)
    return float(sz)


def header_slot_shot_z(th: Any) -> float:
    """炮点垂向：优先 rz/relev/gelev（r*/g* 槽）。"""
    return float(_f(th, "rz", "relev", "gelev", default=0.0))


def infer_use_utm(trace_headers: list) -> bool:
    for th in trace_headers or []:
        sx, sy = header_slot_obs_xy(th, use_utm=True)
        gx, gy = header_slot_shot_xy(th, use_utm=True)
        if _valid_xy(sx, sy) or _valid_xy(gx, gy):
            return True
    return False


def infer_geom_auto(trace_headers: list, *, use_utm: Optional[bool] = None) -> GeomMode:
    """
    启发式：唯一 XY 更少的一侧为 OBS。
    若 s* 更固定 → 与本工区一致 → 返回 ``obs``；否则返回 ``segy``。
    """
    if use_utm is None:
        use_utm = infer_use_utm(trace_headers)

    def _nuniq(getter) -> int:
        pts = set()
        dec = 2 if use_utm else 6
        for th in trace_headers or []:
            x, y = getter(th)
            if _valid_xy(x, y):
                pts.add((round(x, dec), round(y, dec)))
        return len(pts)

    n_s = _nuniq(lambda th: header_slot_obs_xy(th, use_utm=use_utm))
    n_r = _nuniq(lambda th: header_slot_shot_xy(th, use_utm=use_utm))
    # s* 更固定 → OBS 在 s* → obs 模式；r* 更固定 → OBS 在 r* → 按 segy 字面用
    if n_s == 0 and n_r == 0:
        return DEFAULT_GEOM
    if n_s <= n_r:
        return "obs"
    return "segy"


def resolve_geom(
    geom: GeomMode,
    trace_headers: Optional[list] = None,
) -> GeomMode:
    if geom == "auto":
        return infer_geom_auto(list(trace_headers or []))
    if geom in ("obs", "segy"):
        return geom
    return DEFAULT_GEOM


def physical_shot_obs_xyz(
    th: Any,
    *,
    geom: GeomMode = DEFAULT_GEOM,
    use_utm: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 (shot_xyz, obs_xyz) —— **物理角色**，供联合反演 / RTM。

    - geom=obs : shot←r*/gx，obs←s*/sx
    - geom=segy: shot←s*/sx，obs←r*/gx
    """
    s_xy = header_slot_obs_xy(th, use_utm=use_utm)  # 槽位名 s*
    r_xy = header_slot_shot_xy(th, use_utm=use_utm)  # 槽位名 r*/g*
    s_z = header_slot_obs_z(th)
    r_z = header_slot_shot_z(th)

    s_xyz = np.asarray([s_xy[0], s_xy[1], s_z], dtype=float)
    r_xyz = np.asarray([r_xy[0], r_xy[1], r_z], dtype=float)

    mode = geom if geom != "auto" else DEFAULT_GEOM
    if mode == "obs":
        # 本工区：s*=OBS，r*=炮
        return r_xyz.copy(), s_xyz.copy()
    # segy 字面：s*=震源/炮，r*=检波
    return s_xyz.copy(), r_xyz.copy()


def physical_shot_xy_optional(
    th: Any,
    *,
    geom: GeomMode = DEFAULT_GEOM,
    use_utm: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    返回 (shot_xy_geo, shot_xy_utm) 供水深/预览；按物理炮点。
    仅填充与 use_utm 对应的一侧，另一侧为 None。
    """
    shot, _obs = physical_shot_obs_xyz(th, geom=geom, use_utm=use_utm)
    xy = np.asarray(shot[:2], dtype=float)
    if not _valid_xy(float(xy[0]), float(xy[1])):
        return None, None
    if use_utm:
        return None, xy
    return xy, None


def orientation_rec_role_label(geom: GeomMode) -> str:
    """
    兼容旧 qt_fast_viewer 的 rec_role 字符串：
    ``sx`` = OBS 在道头 s* 槽；``rx`` = OBS 在道头 r* 槽。
    """
    mode = geom if geom != "auto" else DEFAULT_GEOM
    return "sx" if mode == "obs" else "rx"
