"""
pyAOBS Workbench package.

This package hosts project-oriented infrastructure that can orchestrate
multiple GUI tools (e.g., zplotpy / tomo2d / imodel / iphase) under a
single project workspace.

User guide: see `pyAOBS/workbench/README.md`.
"""

from .core.project_manager import ProjectContext, ProjectManager, ProjectError
from .core.run_manager import RunContext, RunManager
from .core.state_store import StateStore, UIStateRef

__all__ = [
    "ProjectContext",
    "ProjectManager",
    "ProjectError",
    "RunContext",
    "RunManager",
    "StateStore",
    "UIStateRef",
    "WorkbenchMainWindow",
]


def __getattr__(name: str):
    if name == "WorkbenchMainWindow":
        from .shell.main_window import WorkbenchMainWindow

        return WorkbenchMainWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

