"""
Project manager for pyAOBS Workbench.

Phase-1 scope:
- create/open project
- validate canonical project layout
- persist/load minimal project metadata
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


PROJECT_META_FILE = "project.yaml"

# Note: JSON text is a valid YAML 1.2 subset. We keep .yaml extension to align
# with workbench design while avoiding a hard dependency on pyyaml.
DEFAULT_LAYOUT_DIRS = (
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


class ProjectError(RuntimeError):
    """Raised when project creation/open/validation fails."""


@dataclass(frozen=True)
class ProjectContext:
    """In-memory handle for a project workspace."""

    root: Path
    metadata: dict[str, Any]

    @property
    def project_file(self) -> Path:
        return self.root / PROJECT_META_FILE

    def resolve(self, relative_path: str | Path) -> Path:
        return self.root / Path(relative_path)


class ProjectManager:
    """Create, open, and validate workbench projects."""

    def __init__(self, layout_dirs: tuple[str, ...] = DEFAULT_LAYOUT_DIRS) -> None:
        self.layout_dirs = layout_dirs

    def create_project(
        self,
        root: str | Path,
        name: str | None = None,
        *,
        overwrite: bool = False,
    ) -> ProjectContext:
        root_path = Path(root).expanduser().resolve()
        project_file = root_path / PROJECT_META_FILE

        if project_file.exists() and not overwrite:
            raise ProjectError(
                f"Project already exists at '{root_path}'. "
                "Use overwrite=True to reinitialize."
            )

        root_path.mkdir(parents=True, exist_ok=True)
        self._ensure_layout(root_path)

        now = _utc_now_iso()
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "name": name or root_path.name,
            "created_at": now,
            "updated_at": now,
            "tool": "pyAOBS-workbench",
        }
        self._write_metadata(project_file, metadata)
        return ProjectContext(root=root_path, metadata=metadata)

    def open_project(self, root: str | Path) -> ProjectContext:
        root_path = Path(root).expanduser().resolve()
        project_file = root_path / PROJECT_META_FILE
        if not project_file.exists():
            raise ProjectError(
                f"Not a pyAOBS workbench project: missing '{PROJECT_META_FILE}' under '{root_path}'."
            )

        metadata = self._read_metadata(project_file)
        self.validate_project(root_path)
        return ProjectContext(root=root_path, metadata=metadata)

    def validate_project(self, root: str | Path) -> None:
        root_path = Path(root).expanduser().resolve()
        project_file = root_path / PROJECT_META_FILE
        if not root_path.exists() or not root_path.is_dir():
            raise ProjectError(f"Project directory does not exist: '{root_path}'.")
        if not project_file.exists():
            raise ProjectError(f"Missing project metadata file: '{project_file}'.")

        missing_dirs = [
            str(root_path / rel)
            for rel in self.layout_dirs
            if not (root_path / rel).exists()
        ]
        if missing_dirs:
            raise ProjectError(
                "Project layout is incomplete. Missing directories:\n- "
                + "\n- ".join(missing_dirs)
            )

    def touch_updated_at(self, context: ProjectContext) -> ProjectContext:
        metadata = dict(context.metadata)
        metadata["updated_at"] = _utc_now_iso()
        self._write_metadata(context.project_file, metadata)
        return ProjectContext(root=context.root, metadata=metadata)

    def _ensure_layout(self, root_path: Path) -> None:
        for rel in self.layout_dirs:
            (root_path / rel).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _read_metadata(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ProjectError(f"Invalid project metadata format in '{path}': {exc}") from exc

    @staticmethod
    def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

