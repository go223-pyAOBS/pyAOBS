# -*- coding: utf-8 -*-
"""idata 工区工程（内存 + meta/idata_project.json）。"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ConvertParams:
    """转换页表单快照（与 ConvertPanel.collect_state_fields 对齐）。"""

    fields: Dict[str, str] = field(default_factory=dict)
    convert_tab_index: int = 0


# 解释模式默认：约定（炮=sx/sy，OBS=gx/gy）
DEFAULT_GEOM = "segy"


@dataclass
class WorkflowParams:
    """当前数据与最近产出路径（相对工区或绝对）。"""

    current_data: str = ""  # 当前打开的 SEGY/SU
    last_segy: str = ""
    last_su: str = ""
    geom: str = DEFAULT_GEOM  # segy=约定；obs=旧对调兼容
    stage_index: int = 0


@dataclass
class IdataProject:
    """一个 idata 工区目录下的状态。"""

    name: str = "untitled"
    workdir: str = ""
    version: int = 1
    convert: ConvertParams = field(default_factory=ConvertParams)
    workflow: WorkflowParams = field(default_factory=WorkflowParams)
    notes: str = ""
    dirty: bool = False  # 工程元数据脏（非道头脏）

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

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("dirty", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IdataProject":
        def _filter(dc_cls, raw: dict):
            names = set(dc_cls.__dataclass_fields__.keys())
            return {k: v for k, v in (raw or {}).items() if k in names}

        conv_raw = d.get("convert") or {}
        # 兼容旧：fields 直接在 convert 下
        if "fields" in conv_raw or "convert_tab_index" in conv_raw:
            conv = ConvertParams(
                fields=dict(conv_raw.get("fields") or {}),
                convert_tab_index=int(conv_raw.get("convert_tab_index", 0) or 0),
            )
        else:
            conv = ConvertParams(**_filter(ConvertParams, conv_raw))
        wf = WorkflowParams(**_filter(WorkflowParams, d.get("workflow") or {}))
        skip = {"convert", "workflow", "dirty"}
        kw = {k: v for k, v in d.items() if k not in skip and k in cls.__dataclass_fields__}
        return cls(convert=conv, workflow=wf, **kw)

    @classmethod
    def resolve_open_path(cls, path: str) -> str | None:
        """目录或 idata_project.json → 可 load 的 json；目录无 json 则 None。"""
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
    def load(cls, path: str) -> "IdataProject":
        with open(path, "r", encoding="utf-8") as f:
            proj = cls.from_dict(json.load(f))
        from .services.workdir_layout import infer_workdir_from_json

        if not proj.workdir:
            proj.workdir = infer_workdir_from_json(path)
        proj.dirty = False
        return proj

    @classmethod
    def create_new(cls, workdir: str, name: str = "") -> "IdataProject":
        workdir = os.path.abspath(workdir)
        nm = name.strip() or os.path.basename(workdir.rstrip(os.sep)) or "untitled"
        proj = cls(name=nm, workdir=workdir)
        proj.workflow.geom = DEFAULT_GEOM
        proj.ensure_workdir()
        proj.dirty = True
        return proj
