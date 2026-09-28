# -*- coding: utf-8 -*-
"""OBS 姿态校正工区工程（内存 + meta/relocation_project.json）。"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .services.models import AttitudeSolution, AttitudeUiParams


@dataclass
class InputParams:
    """输入数据路径与几何约定。"""

    dfile: str = ""
    hfile: str = ""
    rfile: str = ""
    terrain_path: str = ""
    geom: str = "obs"  # obs | segy | auto（与 geometry_roles / RTM 一致）
    notes: str = ""


@dataclass
class WorkflowParams:
    """工作流状态路径（相对工区或绝对路径）。"""

    waveop_path: str = "outputs/waveop.json"
    picks_path: str = "outputs/picks.out"
    solution_path: str = "outputs/attitude_solution.json"
    viewer_params_path: str = "outputs/viewer_params.json"
    current_apick: int = 1


@dataclass
class RelocationProject:
    """一个姿态校正工区目录下的状态。"""

    name: str = "untitled"
    workdir: str = ""
    version: int = 1
    inputs: InputParams = field(default_factory=InputParams)
    workflow: WorkflowParams = field(default_factory=WorkflowParams)
    attitude_ui: AttitudeUiParams = field(default_factory=AttitudeUiParams)
    attitude_solution: AttitudeSolution = field(default_factory=AttitudeSolution)
    # 内嵌快照（可选；打开时优先，其次读 workflow 路径）
    waveform_selections: List[Dict[str, Any]] = field(default_factory=list)
    waveop_corrected_ttrue: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    def ensure_workdir(self) -> str:
        if not self.workdir:
            raise ValueError("未设置工区目录 workdir")
        from .services.workdir_layout import prepare_workdir

        prepare_workdir(self)
        return self.workdir

    def path(self, *parts: str) -> str:
        return os.path.join(self.workdir, *parts)

    def abs_or_join(self, p: str) -> str:
        """相对工区路径 → 绝对路径；已是绝对则原样返回。"""
        s = str(p or "").strip()
        if not s:
            return ""
        if os.path.isabs(s):
            return s
        if not self.workdir:
            return s
        return os.path.normpath(os.path.join(self.workdir, s))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # AttitudeUiParams / AttitudeSolution 已是普通 dict
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RelocationProject":
        def _filter(dc_cls, raw: dict):
            names = set(dc_cls.__dataclass_fields__.keys())
            return {k: v for k, v in (raw or {}).items() if k in names}

        inp = InputParams(**_filter(InputParams, d.get("inputs") or {}))
        wf = WorkflowParams(**_filter(WorkflowParams, d.get("workflow") or {}))
        ui = AttitudeUiParams.from_dict(d.get("attitude_ui") or {})
        sol = AttitudeSolution.from_dict(d.get("attitude_solution") or {})
        skip = {"inputs", "workflow", "attitude_ui", "attitude_solution"}
        kw = {k: v for k, v in d.items() if k not in skip and k in cls.__dataclass_fields__}
        return cls(
            inputs=inp,
            workflow=wf,
            attitude_ui=ui,
            attitude_solution=sol,
            **kw,
        )

    def save(self, path: Optional[str] = None) -> str:
        self.ensure_workdir()
        from .services.workdir_layout import project_json_path

        path = path or project_json_path(self.workdir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path: str) -> "RelocationProject":
        with open(path, "r", encoding="utf-8") as f:
            proj = cls.from_dict(json.load(f))
        from .services.workdir_layout import infer_workdir_from_json

        if not proj.workdir:
            proj.workdir = infer_workdir_from_json(path)
        return proj
