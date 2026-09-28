"""
State store for pyAOBS Workbench.

Phase-3 minimal scope:
- persist/load project UI state at _wb/state/ui_state.json (legacy: state/)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .project_layout import iter_state_dirs, primary_state_dir
from .project_manager import ProjectContext, ProjectError


UI_STATE_FILE = "ui_state.json"


@dataclass(frozen=True)
class UIStateRef:
    """Reference to persisted UI state file for a project."""

    path: Path


class StateStore:
    """Read/write UI state under project state/ directory."""

    def get_ui_state_ref(self, project: ProjectContext) -> UIStateRef:
        return UIStateRef(path=primary_state_dir(project.root) / UI_STATE_FILE)

    def save_ui_state(self, project: ProjectContext, state: dict[str, Any]) -> UIStateRef:
        ref = self.get_ui_state_ref(project)
        payload = {
            "schema_version": 1,
            "ui_state": state,
        }
        ref.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return ref

    def load_ui_state(self, project: ProjectContext) -> dict[str, Any]:
        candidates = [primary_state_dir(project.root) / UI_STATE_FILE]
        for state_dir in iter_state_dirs(project.root):
            path = state_dir / UI_STATE_FILE
            if path not in candidates:
                candidates.append(path)
        path = next((p for p in candidates if p.exists()), candidates[0])
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ProjectError(f"Invalid UI state file '{path}': {exc}") from exc

        if not isinstance(payload, dict):
            raise ProjectError(f"Invalid UI state payload type in '{path}'.")
        ui_state = payload.get("ui_state", {})
        if not isinstance(ui_state, dict):
            raise ProjectError(f"Invalid ui_state field type in '{path}'.")
        return ui_state

