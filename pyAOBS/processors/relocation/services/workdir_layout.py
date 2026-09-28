# -*- coding: utf-8 -*-
"""姿态校正工区目录约定。

布局::

    meta/relocation_project.json
    inputs/          原始 Z / hdr / rec / 水深（可选拷贝或仅记路径）
    outputs/         waveop / picks / solution / viewer_params
    cache/
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from ..project import RelocationProject

META_DIR = "meta"
PROJECT_JSON = os.path.join(META_DIR, "relocation_project.json")
INPUTS_DIR = "inputs"
OUTPUTS_DIR = "outputs"
CACHE_DIR = "cache"

LAYOUT_DIRS: Tuple[str, ...] = (META_DIR, INPUTS_DIR, OUTPUTS_DIR, CACHE_DIR)


def project_json_path(workdir: str) -> str:
    return os.path.join(workdir, PROJECT_JSON)


def infer_workdir_from_json(json_path: str) -> str:
    """meta/xxx.json → 工区根；否则取 JSON 所在目录。"""
    abs_p = os.path.abspath(json_path)
    parent = os.path.dirname(abs_p)
    if os.path.basename(parent).lower() == META_DIR:
        return os.path.dirname(parent)
    return parent


def ensure_layout_dirs(project: "RelocationProject") -> None:
    if not project.workdir:
        raise ValueError("未设置工区目录 workdir")
    os.makedirs(project.workdir, exist_ok=True)
    for rel in LAYOUT_DIRS:
        os.makedirs(os.path.join(project.workdir, rel), exist_ok=True)


def apply_layout_defaults(project: "RelocationProject") -> None:
    wf = project.workflow
    if not (wf.waveop_path or "").strip():
        wf.waveop_path = os.path.join(OUTPUTS_DIR, "waveop.json")
    if not (wf.picks_path or "").strip():
        wf.picks_path = os.path.join(OUTPUTS_DIR, "picks.out")
    if not (wf.solution_path or "").strip():
        wf.solution_path = os.path.join(OUTPUTS_DIR, "attitude_solution.json")
    if not (wf.viewer_params_path or "").strip():
        wf.viewer_params_path = os.path.join(OUTPUTS_DIR, "viewer_params.json")


def prepare_workdir(project: "RelocationProject") -> str:
    ensure_layout_dirs(project)
    apply_layout_defaults(project)
    return project.workdir
