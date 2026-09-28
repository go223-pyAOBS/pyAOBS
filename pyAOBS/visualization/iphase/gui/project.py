# -*- coding: utf-8 -*-
"""iphase 工区工程（内存 + meta/iphase_project.json）。"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AnalysisParams:
    """分析参数快照（与主窗控件对齐）。"""

    theory_mode: str = "1D"
    psp_export_mode: str = "picked"
    share_y: bool = True
    window_points: int = 11
    smooth_half_win: str = "5"
    psp_phase_id: int = 40
    picked_policy: str = "插值"
    strict_diff_pair: bool = False
    force_recompute: bool = False
    theory2d_auto_fallback: bool = True
    use_rin_enabled_filter: bool = True
    h_cr: float = 2.0
    vp_cr: float = 3.5
    vs_cr: float = 1.5
    obs_mark_y: float = 0.2
    pois_left: str = ""
    pois_right: str = ""
    pps_pss_ratio: float = 0.6
    equi_write_equiv_psp: bool = False
    section_field_mode: str = "point"


@dataclass
class WorkflowParams:
    """当前数据路径（相对工区或绝对）。"""

    tx_files: List[str] = field(default_factory=list)
    seafloor_path: str = ""
    shot_depth_path: str = ""
    rin_path: str = ""


@dataclass
class IphaseProject:
    """一个 iphase 工区目录下的状态。"""

    name: str = "untitled"
    workdir: str = ""
    version: int = 1
    analysis: AnalysisParams = field(default_factory=AnalysisParams)
    workflow: WorkflowParams = field(default_factory=WorkflowParams)
    notes: str = ""
    dirty: bool = False

    def ensure_workdir(self) -> str:
        if not self.workdir:
            raise ValueError("未设置工区目录 workdir")
        from .services.workdir_layout import prepare_workdir

        prepare_workdir(self)
        return self.workdir

    def path(self, *parts: str) -> str:
        return os.path.join(self.workdir, *parts)

    def abs_or_join(self, p: str) -> str:
        s = str(p or "").strip()
        if not s:
            return ""
        if os.path.isabs(s):
            return s
        if not self.workdir:
            return s
        return os.path.normpath(os.path.join(self.workdir, s))

    def to_rel_or_abs(self, p: str) -> str:
        """尽量存相对工区路径，否则绝对路径。"""
        s = str(p or "").strip()
        if not s or not self.workdir:
            return s
        try:
            abs_p = os.path.abspath(s)
            work = os.path.abspath(self.workdir)
            common = os.path.commonpath([abs_p, work])
            if common == work:
                return os.path.relpath(abs_p, work)
        except Exception:
            pass
        return s

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("dirty", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IphaseProject":
        def _filter(dc_cls, raw: dict):
            names = set(dc_cls.__dataclass_fields__.keys())
            return {k: v for k, v in (raw or {}).items() if k in names}

        analysis = AnalysisParams(**_filter(AnalysisParams, d.get("analysis") or {}))
        wf_raw = dict(d.get("workflow") or {})
        if "tx_files" in wf_raw and not isinstance(wf_raw["tx_files"], list):
            wf_raw["tx_files"] = list(wf_raw["tx_files"] or [])
        workflow = WorkflowParams(**_filter(WorkflowParams, wf_raw))
        skip = {"analysis", "workflow", "dirty"}
        kw = {k: v for k, v in d.items() if k not in skip and k in cls.__dataclass_fields__}
        return cls(analysis=analysis, workflow=workflow, **kw)

    def save(self, path: Optional[str] = None) -> str:
        self.ensure_workdir()
        from .services.workdir_layout import project_json_path

        path = path or project_json_path(self.workdir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        self.dirty = False
        return path

    @classmethod
    def load(cls, path: str) -> "IphaseProject":
        with open(path, "r", encoding="utf-8") as f:
            proj = cls.from_dict(json.load(f))
        from .services.workdir_layout import infer_workdir_from_json

        if not proj.workdir:
            proj.workdir = infer_workdir_from_json(path)
        proj.dirty = False
        return proj

    @classmethod
    def create_new(cls, workdir: str, name: str = "") -> "IphaseProject":
        workdir = os.path.abspath(workdir)
        nm = name.strip() or os.path.basename(workdir.rstrip(os.sep)) or "untitled"
        proj = cls(name=nm, workdir=workdir)
        proj.ensure_workdir()
        proj.dirty = True
        return proj

    @classmethod
    def resolve_open_path(cls, path: str) -> Optional[str]:
        """目录或 iphase_project.json → 可 load 的 json；目录无 json 则 None。"""
        from .services.workdir_layout import project_json_path

        p = os.path.abspath(str(path or "").strip())
        if not p:
            return None
        if os.path.isfile(p):
            return p
        if os.path.isdir(p):
            cand = project_json_path(p)
            return cand if os.path.isfile(cand) else None
        return None
