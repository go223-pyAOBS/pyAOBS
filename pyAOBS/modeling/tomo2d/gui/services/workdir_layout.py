# -*- coding: utf-8 -*-
"""tomo2d 工区目录约定。

布局::

    meta/tomo2d_project.json
    inputs/     smesh / geom / data / 边界等输入（可引用或拷贝）
    outputs/    gen_smesh / 反演产出等
    runs/       tt_inverse 可复现运行包
    cache/      临时文件
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from ..project import Tomo2dProject

META_DIR = "meta"
PROJECT_JSON = os.path.join(META_DIR, "tomo2d_project.json")
INPUTS_DIR = "inputs"
OUTPUTS_DIR = "outputs"
RUNS_DIR = "runs"
CACHE_DIR = "cache"

LAYOUT_DIRS: Tuple[str, ...] = (
    META_DIR,
    INPUTS_DIR,
    OUTPUTS_DIR,
    RUNS_DIR,
    CACHE_DIR,
)


def project_json_path(workdir: str) -> str:
    return os.path.join(workdir, PROJECT_JSON)


def infer_workdir_from_json(json_path: str) -> str:
    abs_p = os.path.abspath(json_path)
    parent = os.path.dirname(abs_p)
    if os.path.basename(parent).lower() == META_DIR:
        return os.path.dirname(parent)
    return parent


def resolve_project_json(path: str) -> str:
    """目录 → meta/tomo2d_project.json；文件则原样返回。"""
    p = os.path.abspath(path)
    if os.path.isdir(p):
        return project_json_path(p)
    return p


def ensure_layout_dirs(project: "Tomo2dProject") -> None:
    if not project.workdir:
        raise ValueError("未设置工区目录 workdir")
    os.makedirs(project.workdir, exist_ok=True)
    for rel in LAYOUT_DIRS:
        os.makedirs(os.path.join(project.workdir, rel), exist_ok=True)


def prepare_workdir(project: "Tomo2dProject") -> str:
    ensure_layout_dirs(project)
    return project.workdir
