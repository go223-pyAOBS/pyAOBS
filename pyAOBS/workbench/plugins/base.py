"""Base abstractions for Workbench plugins."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class PluginValidationError(ValueError):
    """Raised when plugin input is invalid."""


@dataclass(frozen=True)
class PluginCommandSpec:
    """Executable command specification for RunManager."""

    node_id: str
    command: list[str]
    cwd: Path | None
    inputs: list[Path]
    env: dict[str, str]
    params: dict


class WorkbenchPlugin(Protocol):
    """Minimal plugin protocol for phase-4 runnable node panel."""

    id: str
    name: str
    description: str

    def build_spec(self, project_root: Path, payload: dict) -> PluginCommandSpec:
        """Build runnable command spec from raw UI payload."""
        ...

