"""imodel GUI 服务层（工区目录约定等）。"""

from __future__ import annotations

from .workdir_layout import (
    CACHE_DIR,
    INPUTS_DIR,
    META_DIR,
    OUTPUTS_DIR,
    PROJECT_JSON,
    infer_workdir_from_json,
    prepare_workdir,
    project_json_path,
)

__all__ = [
    "CACHE_DIR",
    "INPUTS_DIR",
    "META_DIR",
    "OUTPUTS_DIR",
    "PROJECT_JSON",
    "infer_workdir_from_json",
    "prepare_workdir",
    "project_json_path",
]
