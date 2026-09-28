"""zplotpy Qt 会话：记住上次工区（``PYAOBS_GUI_STATE_FILE`` / ``PYAOBS_ZPLOTPY_PROJECT``）。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional


def gui_state_file_from_env() -> Optional[Path]:
    raw = os.environ.get("PYAOBS_GUI_STATE_FILE", "").strip()
    return Path(raw).expanduser() if raw else None


def project_path_from_env() -> Optional[Path]:
    raw = os.environ.get("PYAOBS_ZPLOTPY_PROJECT", "").strip()
    return Path(raw).expanduser() if raw else None


def load_zplotpy_section(state_file: Path) -> dict[str, Any]:
    if not state_file.exists():
        return {}
    try:
        raw = json.loads(state_file.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        obj = raw.get("zplotpy_gui", {})
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def save_zplotpy_section(
    state_file: Path,
    *,
    workdir: str = "",
    project_json: str = "",
) -> None:
    raw: dict[str, Any] = {}
    if state_file.exists():
        try:
            loaded = json.loads(state_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                raw = loaded
        except Exception:
            raw = {}
    sec = raw.get("zplotpy_gui")
    if not isinstance(sec, dict):
        sec = {}
    if workdir:
        sec["workdir"] = str(workdir)
    if project_json:
        sec["project_json"] = str(project_json)
    raw["zplotpy_gui"] = sec
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
