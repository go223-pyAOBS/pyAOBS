# -*- coding: utf-8 -*-
"""Trace geometry / orientation helpers mixed into QtFastViewer."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class TraceGeomMixin:
    """道坐标提取与几何角色推断。"""

    def _orientation_geom_mode(self) -> str:
        """姿态/位置图几何模式：默认本工区 obs（与 RTM --geom obs 一致）。"""
        return str(getattr(self, "_orientation_geom", "obs") or "obs")


    def _extract_trace_xyz(self, trace_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (shot_xyz, obs_xyz) 物理角色，见 pyAOBS.geometry_roles。"""
        headers = self.loaded.get("trace_headers", []) if self.loaded is not None else []
        if trace_idx < 0 or trace_idx >= len(headers):
            return np.zeros(3, dtype=float), np.zeros(3, dtype=float)
        th = headers[trace_idx]
        try:
            from pyAOBS.geometry_roles import (
                infer_use_utm,
                physical_shot_obs_xyz,
                resolve_geom,
            )
            use_utm = infer_use_utm(headers)
            geom = resolve_geom(self._orientation_geom_mode(), headers)  # type: ignore[arg-type]
            shot, obs = physical_shot_obs_xyz(th, geom=geom, use_utm=use_utm)
            return np.asarray(shot, dtype=float), np.asarray(obs, dtype=float)
        except Exception:
            pass
        # 回退：本工区 obs —— s*=OBS，r*=炮
        use_utm, rec_role = self._infer_rec_role_for_orientation()
        sxutm = float(getattr(th, "sxutm", 0.0) or 0.0)
        syutm = float(getattr(th, "syutm", 0.0) or 0.0)
        rxutm = float(getattr(th, "rxutm", 0.0) or 0.0)
        ryutm = float(getattr(th, "ryutm", 0.0) or 0.0)
        swdepth = float(getattr(th, "swdepth", 0.0) or 0.0)
        sz = float(getattr(th, "sz", swdepth) or swdepth)
        rz = float(getattr(th, "rz", 0.0) or 0.0)
        if use_utm and any(abs(v) > 1e-6 for v in (sxutm, syutm, rxutm, ryutm)):
            if rec_role == "sx":
                src = np.array([rxutm, ryutm, rz], dtype=float)
                rec = np.array([sxutm, syutm, sz], dtype=float)
            else:
                src = np.array([sxutm, syutm, sz], dtype=float)
                rec = np.array([rxutm, ryutm, rz], dtype=float)
            return src, rec
        slon = float(getattr(th, "slong", 0.0) or 0.0)
        slat = float(getattr(th, "slat", 0.0) or 0.0)
        rlon = float(getattr(th, "rlong", 0.0) or 0.0)
        rlat = float(getattr(th, "rlat", 0.0) or 0.0)
        if rec_role == "sx":
            src = np.array([rlon, rlat, rz], dtype=float)
            rec = np.array([slon, slat, sz], dtype=float)
        else:
            src = np.array([slon, slat, sz], dtype=float)
            rec = np.array([rlon, rlat, rz], dtype=float)
        return src, rec


    def _extract_trace_source_coords(self, trace_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """物理炮点 XY（geo / utm）；供水深回退与预览。"""
        headers = self.loaded.get("trace_headers", []) if self.loaded is not None else []
        if trace_idx < 0 or trace_idx >= len(headers):
            return None, None
        th = headers[trace_idx]
        try:
            from pyAOBS.geometry_roles import (
                infer_use_utm,
                physical_shot_xy_optional,
                resolve_geom,
            )
            use_utm = infer_use_utm(headers)
            geom = resolve_geom(self._orientation_geom_mode(), headers)  # type: ignore[arg-type]
            return physical_shot_xy_optional(th, geom=geom, use_utm=use_utm)
        except Exception:
            pass
        use_utm, rec_role = self._infer_rec_role_for_orientation()
        sxutm = float(getattr(th, "sxutm", 0.0) or 0.0)
        syutm = float(getattr(th, "syutm", 0.0) or 0.0)
        slon = float(getattr(th, "slong", 0.0) or 0.0)
        slat = float(getattr(th, "slat", 0.0) or 0.0)
        rxutm = float(getattr(th, "rxutm", 0.0) or 0.0)
        ryutm = float(getattr(th, "ryutm", 0.0) or 0.0)
        rlon = float(getattr(th, "rlong", 0.0) or 0.0)
        rlat = float(getattr(th, "rlat", 0.0) or 0.0)
        src_utm = None
        src_geo = None
        if use_utm:
            if rec_role == "sx":
                if abs(rxutm) > 1e-9 or abs(ryutm) > 1e-9:
                    src_utm = np.array([rxutm, ryutm], dtype=float)
            else:
                if abs(sxutm) > 1e-9 or abs(syutm) > 1e-9:
                    src_utm = np.array([sxutm, syutm], dtype=float)
        else:
            if rec_role == "sx":
                if abs(rlon) > 1e-9 or abs(rlat) > 1e-9:
                    src_geo = np.array([rlon, rlat], dtype=float)
            else:
                if abs(slon) > 1e-9 or abs(slat) > 1e-9:
                    src_geo = np.array([slon, slat], dtype=float)
        return src_geo, src_utm


    def _infer_rec_role_for_orientation(self) -> Tuple[bool, str]:
        """
        返回 (use_utm, rec_role)。
        rec_role: 道头哪一侧槽位是 OBS —— ``sx``=s*槽（本工区默认），``rx``=r*槽。
        主路径走 geometry_roles（默认 geom=obs，与 RTM 一致）。
        """
        trace_headers = self.loaded.get("trace_headers", []) if self.loaded is not None else []
        if not trace_headers:
            return False, "sx"
        try:
            from pyAOBS.geometry_roles import (
                infer_use_utm,
                orientation_rec_role_label,
                resolve_geom,
            )
            use_utm = infer_use_utm(trace_headers)
            geom = resolve_geom(self._orientation_geom_mode(), trace_headers)  # type: ignore[arg-type]
            return use_utm, orientation_rec_role_label(geom)
        except Exception:
            pass
        # 回退：本工区默认 OBS 在 s*
        use_utm = False
        for th in trace_headers:
            sxutm = float(getattr(th, "sxutm", 0.0) or 0.0)
            syutm = float(getattr(th, "syutm", 0.0) or 0.0)
            rxutm = float(getattr(th, "rxutm", 0.0) or 0.0)
            ryutm = float(getattr(th, "ryutm", 0.0) or 0.0)
            if (abs(sxutm) > 1e-9 or abs(syutm) > 1e-9) or (abs(rxutm) > 1e-9 or abs(ryutm) > 1e-9):
                use_utm = True
                break
        return use_utm, "sx"
