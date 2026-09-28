"""道头几何与三分量分组 —— 物理炮点/OBS 角色走 geometry_roles（RTM obs）。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pyAOBS.geometry_roles import (
    DEFAULT_GEOM,
    GeomMode,
    physical_shot_obs_xyz,
    physical_shot_xy_optional,
    resolve_geom,
)


# itypei: 1=Z, 2=R, 3=T, 4=水听器（与 zplotpy 约定一致）
ITYPE_Z = 1
ITYPE_R = 2
ITYPE_T = 3


class GeometryResolver:
    """从 loaded Z 数据解析炮/OBS 坐标与 3C 分组。"""

    def __init__(self, loaded: Dict[str, Any], geom: GeomMode = DEFAULT_GEOM):
        self.loaded = loaded
        self.trace_headers: List[Any] = list(loaded.get("trace_headers") or [])
        self.offsets = np.asarray(loaded.get("offsets", []), dtype=float)
        self.geom: GeomMode = resolve_geom(geom, self.trace_headers)

    def find_3c_group(self, trace_idx: int) -> Tuple[Optional[Dict[int, int]], str]:
        if trace_idx < 0 or trace_idx >= len(self.trace_headers):
            return None, f"道索引越界: {trace_idx}"
        th0 = self.trace_headers[trace_idx]
        ishoti = int(getattr(th0, "ishoti", -1))
        ireci = int(getattr(th0, "ireci", getattr(th0, "irec", -1)))
        group: Dict[int, int] = {}
        for i, th in enumerate(self.trace_headers):
            if int(getattr(th, "ishoti", -2)) != ishoti:
                continue
            irec = int(getattr(th, "ireci", getattr(th, "irec", -2)))
            if irec != ireci:
                continue
            itype = int(getattr(th, "itypei", 0) or 0)
            if itype in (ITYPE_Z, ITYPE_R, ITYPE_T):
                group[itype] = i
        # 回退：同炮 + 最近偏移距补齐缺失分量（与 zplotpy 一致）
        if len(group) < 3 and self.offsets.size == len(self.trace_headers):
            x0 = float(self.offsets[trace_idx]) if 0 <= trace_idx < self.offsets.size else 0.0
            for c in (ITYPE_Z, ITYPE_R, ITYPE_T):
                if c in group:
                    continue
                best_i, best_dx = -1, float("inf")
                for i, th in enumerate(self.trace_headers):
                    if int(getattr(th, "ishoti", -2)) != ishoti:
                        continue
                    if int(getattr(th, "itypei", 0) or 0) != c:
                        continue
                    dx = abs(float(self.offsets[i]) - x0) if i < self.offsets.size else float("inf")
                    if dx < best_dx:
                        best_dx = dx
                        best_i = i
                if best_i >= 0:
                    group[c] = best_i
        if ITYPE_Z not in group or ITYPE_R not in group or ITYPE_T not in group:
            missing = [k for k in (ITYPE_Z, ITYPE_R, ITYPE_T) if k not in group]
            return None, f"缺少分量 itypei={missing} (ishoti={ishoti}, ireci={ireci})"
        return group, ""

    def extract_xyz(self, trace_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (shot_xyz, obs_xyz) 物理角色。"""
        th = self.trace_headers[trace_idx]
        use_utm = True
        sx = float(getattr(th, "sxutm", 0.0) or 0.0)
        sy = float(getattr(th, "syutm", 0.0) or 0.0)
        rx = float(getattr(th, "rxutm", 0.0) or 0.0)
        ry = float(getattr(th, "ryutm", 0.0) or 0.0)
        if abs(sx) + abs(sy) + abs(rx) + abs(ry) < 1e-9:
            use_utm = False
        shot, obs = physical_shot_obs_xyz(th, geom=self.geom, use_utm=use_utm)
        return np.asarray(shot, dtype=float), np.asarray(obs, dtype=float)

    def extract_source_coords(
        self, trace_idx: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """物理炮点 (geo, utm)。"""
        th = self.trace_headers[trace_idx]
        geo, _ = physical_shot_xy_optional(th, geom=self.geom, use_utm=False)
        _, utm = physical_shot_xy_optional(th, geom=self.geom, use_utm=True)
        return geo, utm

    def offset_km(self, trace_idx: int) -> float:
        if 0 <= trace_idx < self.offsets.size:
            return float(self.offsets[trace_idx])
        th = self.trace_headers[trace_idx]
        return float(getattr(th, "offsti", getattr(th, "offset", 0.0)) or 0.0)
