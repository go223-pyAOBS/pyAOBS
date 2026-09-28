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

from .project_layout import (
    DEFAULT_LAYOUT_DIRS,
    DEFAULT_WORKSPACES,
    LAYOUT_V2,
    TOOL_SLOTS,
    detect_layout,
    path_for_registry,
    required_layout_dirs,
    workspaces_from_metadata,
)


PROJECT_META_FILE = "project.yaml"

# Re-exported for tests / callers that imported from this module.
__all__ = [
    "PROJECT_META_FILE",
    "DEFAULT_LAYOUT_DIRS",
    "ProjectError",
    "ProjectContext",
    "ProjectManager",
]


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

    @property
    def workspaces(self) -> dict[str, str]:
        return workspaces_from_metadata(self.metadata)


class ProjectManager:
    """Create, open, and validate workbench projects."""

    def __init__(self, layout_dirs: tuple[str, ...] | None = None) -> None:
        self.layout_dirs = layout_dirs if layout_dirs is not None else DEFAULT_LAYOUT_DIRS

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
            "schema_version": 2,
            "layout": LAYOUT_V2,
            "name": name or root_path.name,
            "created_at": now,
            "updated_at": now,
            "tool": "pyAOBS-workbench",
            "workspaces": dict(DEFAULT_WORKSPACES),
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
            for rel in required_layout_dirs(root_path)
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

    def set_workspace(
        self,
        context: ProjectContext,
        plugin_id: str,
        path: str | Path,
    ) -> ProjectContext:
        """Register a tool workspace path (relative to the workbench root when possible)."""
        pid = str(plugin_id or "").strip()
        if not pid:
            raise ProjectError("plugin_id is required to register a workspace.")
        rel = path_for_registry(context.root, path)
        metadata = dict(context.metadata)
        workspaces = dict(metadata.get("workspaces") or {})
        if not isinstance(workspaces, dict):
            workspaces = {}
        workspaces[pid] = rel
        metadata["workspaces"] = workspaces
        metadata["updated_at"] = _utc_now_iso()
        if "layout" not in metadata:
            metadata["layout"] = detect_layout(context.root)
        self._write_metadata(context.project_file, metadata)
        return ProjectContext(root=context.root, metadata=metadata)

    def _ensure_layout(self, root_path: Path) -> None:
        dirs = self.layout_dirs if self.layout_dirs is not None else DEFAULT_LAYOUT_DIRS
        for rel in dirs:
            (root_path / rel).mkdir(parents=True, exist_ok=True)
        if dirs == DEFAULT_LAYOUT_DIRS or "tools" in dirs:
            for slot in TOOL_SLOTS:
                (root_path / "tools" / slot).mkdir(parents=True, exist_ok=True)

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
