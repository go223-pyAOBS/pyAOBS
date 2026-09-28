# -*- coding: utf-8 -*-
"""imodel 工区工程（内存 + meta/imodel_project.json）。

与 Workbench 会话键 ``imodel_gui`` 字段对齐，便于双向同步。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AnalysisParams:
    """分析/界面显示参数快照。"""

    show_interfaces: bool = True
    basement_selection: str = ""
    seafloor_selection: str = ""
    moho_selection: str = ""


@dataclass
class WorkflowParams:
    """数据路径（相对工区或绝对）+ 重力/petrology 桥接。"""

    vp_model: str = ""
    vs_model: str = ""
    interface_files: List[str] = field(default_factory=list)
    gravity_obs_data_dir: str = ""
    gravity_obs_filename: str = ""
    gravity_obs_overlay: bool = False
    gravity_profile_lon_lat_csv: str = ""
    petrology_obs_json: str = ""
    petrology_f_lower: Optional[float] = None
    petrology_export_x_km: Optional[float] = None
    petrology_transect_windows: str = ""
    petrology_observation: Optional[Dict[str, Any]] = None
    petrology_observations: Optional[List[Dict[str, Any]]] = None


@dataclass
class ImodelProject:
    """一个 imodel 工区目录下的状态。"""

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

    @property
    def is_open(self) -> bool:
        return bool(self.workdir and os.path.isdir(self.workdir))

    def project_json(self) -> str:
        from .services.workdir_layout import project_json_path

        if not self.workdir:
            return ""
        return project_json_path(self.workdir)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("dirty", None)
        return d

    def to_imodel_gui_section(self) -> Dict[str, Any]:
        """导出为 Workbench ``imodel_gui`` 段（路径尽量绝对，便于 run 追踪）。"""
        wf = self.workflow
        an = self.analysis
        sec: Dict[str, Any] = {
            "model_file": self.abs_or_join(wf.vp_model) if wf.vp_model else "",
            "vs_model_file": self.abs_or_join(wf.vs_model) if wf.vs_model else "",
            "show_interfaces": bool(an.show_interfaces),
            "basement_selection": str(an.basement_selection or "").strip(),
            "seafloor_selection": str(an.seafloor_selection or "").strip(),
            "moho_selection": str(an.moho_selection or "").strip(),
            "interface_files": [
                self.abs_or_join(p) for p in (wf.interface_files or []) if str(p).strip()
            ],
            "gravity_obs_data_dir": self.abs_or_join(wf.gravity_obs_data_dir)
            if wf.gravity_obs_data_dir
            else "",
            "gravity_obs_filename": str(wf.gravity_obs_filename or "").strip(),
            "gravity_obs_overlay": bool(wf.gravity_obs_overlay),
            "gravity_profile_lon_lat_csv": self.abs_or_join(wf.gravity_profile_lon_lat_csv)
            if wf.gravity_profile_lon_lat_csv
            else "",
            "petrology_obs_json": self.abs_or_join(wf.petrology_obs_json)
            if wf.petrology_obs_json
            else "",
            "petrology_transect_windows": self.abs_or_join(wf.petrology_transect_windows)
            if wf.petrology_transect_windows
            else "",
            "imodel_project": self.project_json() or self.workdir,
        }
        if wf.petrology_f_lower is not None:
            sec["petrology_f_lower"] = float(wf.petrology_f_lower)
        if wf.petrology_export_x_km is not None:
            sec["petrology_export_x_km"] = float(wf.petrology_export_x_km)
        if isinstance(wf.petrology_observation, dict):
            sec["petrology_observation"] = wf.petrology_observation
        if isinstance(wf.petrology_observations, list):
            sec["petrology_observations"] = list(wf.petrology_observations)
        return sec

    @classmethod
    def from_imodel_gui_section(
        cls,
        sec: Dict[str, Any],
        *,
        workdir: str = "",
        name: str = "",
    ) -> "ImodelProject":
        """从 Workbench ``imodel_gui`` 段构造工区（路径原样写入 workflow）。"""
        sec = sec if isinstance(sec, dict) else {}
        ifaces = sec.get("interface_files")
        if not isinstance(ifaces, list):
            ifaces = []
        wf = WorkflowParams(
            vp_model=str(sec.get("model_file", "") or "").strip(),
            vs_model=str(sec.get("vs_model_file", "") or "").strip(),
            interface_files=[str(x).strip() for x in ifaces if str(x).strip()],
            gravity_obs_data_dir=str(sec.get("gravity_obs_data_dir", "") or "").strip(),
            gravity_obs_filename=str(sec.get("gravity_obs_filename", "") or "").strip(),
            gravity_obs_overlay=bool(sec.get("gravity_obs_overlay", False)),
            gravity_profile_lon_lat_csv=str(
                sec.get("gravity_profile_lon_lat_csv", "") or ""
            ).strip(),
            petrology_obs_json=str(sec.get("petrology_obs_json", "") or "").strip(),
            petrology_transect_windows=str(
                sec.get("petrology_transect_windows", "") or ""
            ).strip(),
            petrology_observation=(
                sec.get("petrology_observation")
                if isinstance(sec.get("petrology_observation"), dict)
                else None
            ),
            petrology_observations=(
                list(sec["petrology_observations"])
                if isinstance(sec.get("petrology_observations"), list)
                else None
            ),
        )
        if "petrology_f_lower" in sec and sec["petrology_f_lower"] is not None:
            try:
                wf.petrology_f_lower = float(sec["petrology_f_lower"])
            except (TypeError, ValueError):
                pass
        if "petrology_export_x_km" in sec and sec["petrology_export_x_km"] is not None:
            try:
                wf.petrology_export_x_km = float(sec["petrology_export_x_km"])
            except (TypeError, ValueError):
                pass
        an = AnalysisParams(
            show_interfaces=bool(sec.get("show_interfaces", True)),
            basement_selection=str(sec.get("basement_selection", "") or "").strip(),
            seafloor_selection=str(sec.get("seafloor_selection", "") or "").strip(),
            moho_selection=str(sec.get("moho_selection", "") or "").strip(),
        )
        wd = str(workdir or "").strip()
        nm = name.strip() or (os.path.basename(wd.rstrip(os.sep)) if wd else "untitled")
        return cls(name=nm, workdir=wd, analysis=an, workflow=wf)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ImodelProject":
        def _filter(dc_cls, raw: dict):
            names = set(dc_cls.__dataclass_fields__.keys())
            return {k: v for k, v in (raw or {}).items() if k in names}

        analysis = AnalysisParams(**_filter(AnalysisParams, d.get("analysis") or {}))
        wf_raw = dict(d.get("workflow") or {})
        if "interface_files" in wf_raw and not isinstance(wf_raw["interface_files"], list):
            wf_raw["interface_files"] = list(wf_raw["interface_files"] or [])
        # 兼容早期扁平字段
        if not wf_raw.get("vp_model") and d.get("model_file"):
            wf_raw["vp_model"] = d.get("model_file")
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
    def load(cls, path: str) -> "ImodelProject":
        with open(path, "r", encoding="utf-8") as f:
            proj = cls.from_dict(json.load(f))
        from .services.workdir_layout import infer_workdir_from_json

        inferred = infer_workdir_from_json(path)
        if not proj.workdir or not os.path.isdir(proj.workdir):
            proj.workdir = inferred
        proj.dirty = False
        return proj

    @classmethod
    def create_new(cls, workdir: str, name: str = "") -> "ImodelProject":
        workdir = os.path.abspath(workdir)
        nm = name.strip() or os.path.basename(workdir.rstrip(os.sep)) or "untitled"
        proj = cls(name=nm, workdir=workdir)
        proj.ensure_workdir()
        proj.dirty = True
        return proj

    @classmethod
    def resolve_open_path(cls, path: str) -> Optional[str]:
        """目录或 imodel_project.json → 返回可 load 的 json 路径。"""
        from .services.workdir_layout import PROJECT_JSON, project_json_path

        p = os.path.abspath(str(path or "").strip())
        if not p:
            return None
        if os.path.isfile(p):
            return p
        if os.path.isdir(p):
            cand = project_json_path(p)
            if os.path.isfile(cand):
                return cand
            # 空目录也可当作“新建后保存”的目标；打开时若无 json 则返回 None
            return None
        return None
