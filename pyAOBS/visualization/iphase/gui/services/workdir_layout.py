# -*- coding: utf-8 -*-
"""iphase 工区目录约定。

布局::

    meta/iphase_project.json
    inputs/     走时/地形/炮深等输入（可引用或拷贝）
    outputs/    PSP 导出、图像等
    cache/      反演/正演缓存
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from ..project import IphaseProject

META_DIR = "meta"
PROJECT_JSON = os.path.join(META_DIR, "iphase_project.json")
INPUTS_DIR = "inputs"
OUTPUTS_DIR = "outputs"
CACHE_DIR = "cache"

LAYOUT_DIRS: Tuple[str, ...] = (META_DIR, INPUTS_DIR, OUTPUTS_DIR, CACHE_DIR)


def project_json_path(workdir: str) -> str:
    return os.path.join(workdir, PROJECT_JSON)


def infer_workdir_from_json(json_path: str) -> str:
    abs_p = os.path.abspath(json_path)
    parent = os.path.dirname(abs_p)
    if os.path.basename(parent).lower() == META_DIR:
        return os.path.dirname(parent)
    return parent


def ensure_layout_dirs(project: "IphaseProject") -> None:
    if not project.workdir:
        raise ValueError("未设置工区目录 workdir")
    os.makedirs(project.workdir, exist_ok=True)
    for rel in LAYOUT_DIRS:
        os.makedirs(os.path.join(project.workdir, rel), exist_ok=True)


def prepare_workdir(project: "IphaseProject") -> str:
    ensure_layout_dirs(project)
    return project.workdir
