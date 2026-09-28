# -*- coding: utf-8 -*-
"""idata 工区目录约定。

布局::

    meta/idata_project.json
    inputs/     原始 RAW/SAC/UKOOA/config（可选拷贝）
    outputs/    转换产出 SEGY/SU
    convert/    中间产物（可选）
    cache/
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from ..project import IdataProject

META_DIR = "meta"
PROJECT_JSON = os.path.join(META_DIR, "idata_project.json")
INPUTS_DIR = "inputs"
OUTPUTS_DIR = "outputs"
CONVERT_DIR = "convert"
CACHE_DIR = "cache"

LAYOUT_DIRS: Tuple[str, ...] = (META_DIR, INPUTS_DIR, OUTPUTS_DIR, CONVERT_DIR, CACHE_DIR)


def project_json_path(workdir: str) -> str:
    return os.path.join(workdir, PROJECT_JSON)


def infer_workdir_from_json(json_path: str) -> str:
    abs_p = os.path.abspath(json_path)
    parent = os.path.dirname(abs_p)
    if os.path.basename(parent).lower() == META_DIR:
        return os.path.dirname(parent)
    return parent


def ensure_layout_dirs(project: "IdataProject") -> None:
    if not project.workdir:
        raise ValueError("未设置工区目录 workdir")
    os.makedirs(project.workdir, exist_ok=True)
    for rel in LAYOUT_DIRS:
        os.makedirs(os.path.join(project.workdir, rel), exist_ok=True)


def apply_layout_defaults(project: "IdataProject") -> None:
    wf = project.workflow
    if not (wf.current_data or "").strip():
        wf.current_data = ""
    if not (wf.last_su or "").strip():
        wf.last_su = ""
    if not (wf.last_segy or "").strip():
        wf.last_segy = ""


def prepare_workdir(project: "IdataProject") -> str:
    ensure_layout_dirs(project)
    apply_layout_defaults(project)
    return project.workdir
