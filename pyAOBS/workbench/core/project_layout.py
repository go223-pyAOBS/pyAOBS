"""Workbench 工程目录约定。

工作台只分辨「工区」；OBS / 炮 / 测线 / 不同参数的计算结果由各 GUI 在工区内分辨。

v2 布局::

    data/raw, data/derived
    tools/<idata|zplotpy|tomo2d|imodel|iphase|vedit>/
    _wb/runs, _wb/state, _wb/reports

旧工程（``datasets/`` + 根下 ``runs/`` ``state/``）仍可打开。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


LAYOUT_V2 = "workbench_v2"
LAYOUT_V1 = "workbench_v1"

DEFAULT_LAYOUT_DIRS = (
    "data/raw",
    "data/derived",
    "tools",
    "_wb/runs",
    "_wb/state",
    "_wb/reports",
)

TOOL_SLOTS = (
    "idata",
    "zplotpy",
    "tomo2d",
    "imodel",
    "iphase",
    "vedit",
)

LEGACY_LAYOUT_DIRS = (
    "datasets/raw",
    "datasets/processed",
    "picks",
    "models",
    "interpretation",
    "workflows",
    "runs",
    "state",
    "reports",
)

DEFAULT_WORKSPACES: dict[str, str] = {
    "data.gui": "tools/idata",
    "zplotpy.gui": "tools/zplotpy",
    "tomo2d.gui": "tools/tomo2d",
    "imodel.gui": "tools/imodel",
    "iphase.gui": "tools/iphase",
    "vedit.gui": "tools/vedit",
}

PLUGIN_WORKDIR_FORM_KEY: dict[str, str] = {
    "data.gui": "data_workdir",
    "zplotpy.gui": "zplot_workdir",
    "tomo2d.gui": "tomo_workdir",
    "imodel.gui": "imodel_workdir",
    "iphase.gui": "iphase_workdir",
    "vedit.gui": "vedit_workdir",
}

PLUGIN_NODE_FORM_KEY: dict[str, str] = {
    "data.gui": "data_node",
    "zplotpy.gui": "zplot_node",
    "tomo2d.gui": "tomo_node",
    "imodel.gui": "imodel_node",
    "iphase.gui": "iphase_node",
    "vedit.gui": "vedit_node",
}

_PLACEHOLDER_NODE_IDS = frozenset(
    {
        "",
        "OBS_node",
        "obs_node",
        "tomo2d_shell",
        "tomo2d_inverse",
        "tomo2d_forward",
        "workspace",
    }
)


def node_id_from_work_dir(work_dir: str | Path | None) -> str:
    """工区 ID 默认取用户工区目录名（不是工具名）。"""
    raw = str(work_dir or "").strip().replace("\\", "/")
    if not raw:
        return ""
    name = Path(raw).name.strip()
    if not name or name in {".", "..", "tools", "data"}:
        return ""
    return name


def default_node_id(plugin_id: str = "", work_dir: str = "") -> str:
    """工区 ID 取用户工区目录名；与插件/工具名无关。``plugin_id`` 仅保留兼容旧调用。"""
    del plugin_id
    return resolve_node_id(work_dir=work_dir)


def resolve_node_id(*candidates: str, work_dir: str = "") -> str:
    """User-typed 工区名 first, then the work_dir folder name."""
    for raw in candidates:
        text = str(raw or "").strip()
        if text and text not in _PLACEHOLDER_NODE_IDS:
            return text
    name = node_id_from_work_dir(work_dir)
    if name:
        return name
    return "workspace"


def detect_layout(root: str | Path) -> str:
    root_path = Path(root)
    if (
        (root_path / "_wb").is_dir()
        or (root_path / "tools").is_dir()
        or (root_path / "data").is_dir()
    ):
        return LAYOUT_V2
    return LAYOUT_V1


def required_layout_dirs(root: str | Path) -> tuple[str, ...]:
    if detect_layout(root) == LAYOUT_V2:
        return DEFAULT_LAYOUT_DIRS
    return LEGACY_LAYOUT_DIRS


def primary_runs_dir(root: str | Path) -> Path:
    root_path = Path(root)
    if detect_layout(root_path) == LAYOUT_V2:
        path = root_path / "_wb" / "runs"
        path.mkdir(parents=True, exist_ok=True)
        return path
    path = root_path / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_runs_dirs(root: str | Path) -> list[Path]:
    root_path = Path(root)
    out: list[Path] = []
    for rel in ("_wb/runs", "runs"):
        path = root_path / rel
        if path.is_dir() and path not in out:
            out.append(path)
    return out


def primary_state_dir(root: str | Path) -> Path:
    root_path = Path(root)
    if detect_layout(root_path) == LAYOUT_V2:
        path = root_path / "_wb" / "state"
        path.mkdir(parents=True, exist_ok=True)
        return path
    path = root_path / "state"
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_state_dirs(root: str | Path) -> list[Path]:
    root_path = Path(root)
    out: list[Path] = []
    for rel in ("_wb/state", "state"):
        path = root_path / rel
        if path.is_dir() and path not in out:
            out.append(path)
    return out


def state_rel_from_root(root: str | Path) -> Path:
    """Relative state dir used in manifests (posix)."""
    root_path = Path(root)
    if detect_layout(root_path) == LAYOUT_V2:
        return Path("_wb") / "state"
    return Path("state")


def workspaces_from_metadata(metadata: Mapping[str, Any] | None) -> dict[str, str]:
    merged = dict(DEFAULT_WORKSPACES)
    raw = (metadata or {}).get("workspaces")
    if isinstance(raw, dict):
        for key, value in raw.items():
            text = str(value or "").strip()
            if text:
                merged[str(key)] = text.replace("\\", "/")
    return merged


def workspace_rel_for_plugin(
    metadata: Mapping[str, Any] | None, plugin_id: str
) -> str:
    return workspaces_from_metadata(metadata).get(plugin_id, "").strip()


def resolve_workspace_path(
    root: str | Path,
    metadata: Mapping[str, Any] | None,
    plugin_id: str,
) -> Path | None:
    rel = workspace_rel_for_plugin(metadata, plugin_id)
    if not rel:
        return None
    path = Path(rel).expanduser()
    if not path.is_absolute():
        path = Path(root) / path
    return path


def path_for_registry(root: str | Path, path: str | Path) -> str:
    """Store a workspace path relative to the workbench root when possible."""
    root_path = Path(root).expanduser().resolve()
    target = Path(path).expanduser()
    if not target.is_absolute():
        return str(target).replace("\\", "/")
    try:
        resolved = target.resolve()
        rel = resolved.relative_to(root_path)
        return str(rel).replace("\\", "/")
    except ValueError:
        return str(target)


def fill_gui_form_workspaces(
    form: dict[str, str],
    workspaces: Mapping[str, str],
    *,
    overwrite: bool = False,
) -> dict[str, str]:
    """Fill empty GUI work_dir / node_id fields from the workspace registry."""
    out = dict(form)
    for plugin_id, work_key in PLUGIN_WORKDIR_FORM_KEY.items():
        current = str(out.get(work_key, "") or "").strip()
        if overwrite or not current:
            rel = str(workspaces.get(plugin_id, "") or "").strip()
            if rel:
                out[work_key] = rel
        node_key = PLUGIN_NODE_FORM_KEY.get(plugin_id)
        if not node_key:
            continue
        node_cur = str(out.get(node_key, "") or "").strip()
        work_for_name = str(out.get(work_key, "") or "").strip()
        if overwrite or node_cur in _PLACEHOLDER_NODE_IDS:
            out[node_key] = node_id_from_work_dir(work_for_name)
    return out


def workdir_from_form(
    plugin_id: str,
    form: Mapping[str, str],
    workspaces: Mapping[str, str] | None = None,
) -> str:
    key = PLUGIN_WORKDIR_FORM_KEY.get(plugin_id, "")
    if key:
        text = str(form.get(key, "") or "").strip()
        if text:
            return text
    if workspaces:
        return str(workspaces.get(plugin_id, "") or "").strip()
    return ""
