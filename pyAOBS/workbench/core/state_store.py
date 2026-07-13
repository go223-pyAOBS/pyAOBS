"""
State store for pyAOBS Workbench.

Phase-3 minimal scope:
- persist/load project UI state at state/ui_state.json
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .project_manager import ProjectContext, ProjectError


UI_STATE_FILE = "ui_state.json"


@dataclass(frozen=True)
class UIStateRef:
    """Reference to persisted UI state file for a project."""

    path: Path


class StateStore:
    """Read/write UI state under project state/ directory."""

    def get_ui_state_ref(self, project: ProjectContext) -> UIStateRef:
        state_dir = project.resolve("state")
        state_dir.mkdir(parents=True, exist_ok=True)
        return UIStateRef(path=state_dir / UI_STATE_FILE)

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
        ref = self.get_ui_state_ref(project)
        if not ref.path.exists():
            return {}
        try:
            payload = json.loads(ref.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ProjectError(f"Invalid UI state file '{ref.path}': {exc}") from exc

        if not isinstance(payload, dict):
            raise ProjectError(f"Invalid UI state payload type in '{ref.path}'.")
        ui_state = payload.get("ui_state", {})
        if not isinstance(ui_state, dict):
            raise ProjectError(f"Invalid ui_state field type in '{ref.path}'.")
        return ui_state

