# -*- coding: utf-8 -*-
"""tomo2d 工区工程（内存 + meta/tomo2d_project.json）。"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Tomo2dProject:
    """一个 tomo2d 工区目录下的状态。

    ``profile`` 与 GUI ``FormState.to_profile_dict()`` 对齐（表单键）；
    ``workdir`` / ``bin_path`` 亦单独存放，加载时写回表单。
    """

    name: str = "untitled"
    workdir: str = ""
    version: int = 1
    bin_path: str = ""
    profile: Dict[str, Any] = field(default_factory=dict)
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
        s = str(p or "").strip()
        if not s or not self.workdir:
            return s
        try:
            abs_p = os.path.abspath(s)
            work = os.path.abspath(self.workdir)
            common = os.path.commonpath([abs_p, work])
            if common == work:
                return os.path.relpath(abs_p, work).replace("\\", "/")
        except Exception:
            pass
        return s

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("dirty", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Tomo2dProject":
        names = set(cls.__dataclass_fields__.keys()) - {"dirty"}
        kw = {k: v for k, v in (d or {}).items() if k in names}
        prof = kw.get("profile")
        if not isinstance(prof, dict):
            kw["profile"] = {}
        else:
            kw["profile"] = dict(prof)
        return cls(**kw)

    def save(self, path: Optional[str] = None) -> str:
        self.ensure_workdir()
        from .services.workdir_layout import project_json_path

        path = path or project_json_path(self.workdir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            f.write("\n")
        self.dirty = False
        return path

    @classmethod
    def load(cls, path: str) -> "Tomo2dProject":
        from .services.workdir_layout import (
            infer_workdir_from_json,
            resolve_project_json,
        )

        json_path = resolve_project_json(path)
        with open(json_path, "r", encoding="utf-8") as f:
            proj = cls.from_dict(json.load(f))
        inferred = infer_workdir_from_json(json_path)
        if not proj.workdir or not os.path.isdir(proj.workdir):
            proj.workdir = inferred
        proj.dirty = False
        return proj

    @classmethod
    def create_new(cls, workdir: str, name: str = "") -> "Tomo2dProject":
        workdir = os.path.abspath(workdir)
        nm = name.strip() or os.path.basename(workdir.rstrip(os.sep)) or "untitled"
        proj = cls(name=nm, workdir=workdir)
        proj.ensure_workdir()
        proj.dirty = True
        return proj

    @property
    def is_open(self) -> bool:
        return bool(self.workdir and os.path.isdir(self.workdir))

    @classmethod
    def resolve_open_path(cls, path: str) -> str | None:
        """目录或 tomo2d_project.json → 可 load 的 json；目录无 json 则 None。"""
        from .services.workdir_layout import project_json_path

        raw = str(path or "").strip()
        if not raw:
            return None
        p = os.path.abspath(raw)
        if os.path.isfile(p):
            return p
        if os.path.isdir(p):
            cand = project_json_path(p)
            return cand if os.path.isfile(cand) else None
        return None

    def apply_to_form_state(self, state) -> None:
        """将工区写回 FormState（含 profile）。"""
        mapping = dict(self.profile)
        mapping["work_dir"] = self.workdir
        if self.bin_path:
            mapping["bin_path"] = self.bin_path
        elif "bin_path" not in mapping:
            mapping.setdefault("bin_path", "")
        state.apply_mapping(mapping)

    def capture_from_form_state(self, state) -> None:
        """从表单快照写入工区（相对化路径由调用方先 normalize）。"""
        profile = dict(state.to_profile_dict())
        wd = str(profile.pop("work_dir", "") or self.workdir).strip()
        if wd:
            self.workdir = os.path.abspath(wd)
        bp = str(profile.get("bin_path", self.bin_path) or "").strip()
        self.bin_path = bp
        # work_dir 以工程字段为准，profile 内保留相对语义字段
        profile["work_dir"] = self.workdir
        if bp:
            profile["bin_path"] = bp
        self.profile = profile
        self.dirty = True
