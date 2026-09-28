"""Workbench 会话状态：``PYAOBS_GUI_STATE_FILE`` 中的 tomo2d_gui。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def project_path_from_env() -> Path | None:
    raw = os.environ.get("PYAOBS_TOMO2D_PROJECT", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def state_file_path() -> Path | None:
    raw = os.environ.get("PYAOBS_GUI_STATE_FILE", "").strip()
    if not raw:
        return None
    return Path(raw)


def _read_raw() -> dict[str, Any]:
    path = state_file_path()
    if path is None or not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _write_raw(raw: dict[str, Any]) -> None:
    path = state_file_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass


def load_workbench_profile() -> dict[str, Any]:
    raw = _read_raw()
    state = raw.get("tomo2d_gui", {})
    if not isinstance(state, dict):
        return {}
    profile = state.get("profile", {})
    return dict(profile) if isinstance(profile, dict) else {}


def save_workbench_profile(
    profile: dict[str, Any],
    *,
    last_project: str | None = None,
) -> None:
    path = state_file_path()
    if path is None:
        return
    try:
        raw = _read_raw()
        block = raw.get("tomo2d_gui")
        if not isinstance(block, dict):
            block = {}
        block["profile"] = dict(profile)
        if last_project is not None:
            block["last_project"] = str(last_project)
        raw["tomo2d_gui"] = block
        _write_raw(raw)
    except Exception:
        pass


def load_last_project_path() -> str:
    raw = _read_raw()
    state = raw.get("tomo2d_gui", {})
    if not isinstance(state, dict):
        return ""
    p = state.get("last_project", "")
    return str(p).strip() if p else ""
