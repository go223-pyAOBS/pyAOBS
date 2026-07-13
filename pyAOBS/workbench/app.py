"""
Minimal Workbench app entry for phase-1 projectization.

Default UI: PySide6 (``shell_qt``). Set ``PYAOBS_WORKBENCH_UI=tk`` for legacy Tk shell.
"""

from __future__ import annotations

import os
from pathlib import Path

from .core.project_manager import ProjectContext, ProjectError, ProjectManager
from .core.run_manager import RunContext, RunManager
from .core.state_store import StateStore
from .shell.main_window import WorkbenchMainWindow


def bootstrap_project(project_root: str | Path, name: str | None = None) -> ProjectContext:
    """
    Create a project if needed, otherwise open and validate it.
    """
    pm = ProjectManager()
    root_path = Path(project_root).expanduser().resolve()
    project_file = root_path / "project.yaml"
    if project_file.exists():
        return pm.open_project(root_path)
    return pm.create_project(root_path, name=name)


def run_node_command(
    project: ProjectContext,
    node_id: str,
    command: list[str],
    *,
    params: dict | None = None,
    inputs: list[str | Path] | None = None,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
) -> RunContext:
    """Execute one node command under project/runs and persist manifest/logs."""
    rm = RunManager()
    return rm.run_command(
        project=project,
        node_id=node_id,
        command=command,
        params=params,
        inputs=inputs,
        env=env,
        cwd=cwd,
    )


def save_project_ui_state(project: ProjectContext, state: dict) -> Path:
    """Persist UI state to project/state/ui_state.json."""
    store = StateStore()
    ref = store.save_ui_state(project, state)
    return ref.path


def load_project_ui_state(project: ProjectContext) -> dict:
    """Load UI state from project/state/ui_state.json (or return empty dict)."""
    store = StateStore()
    return store.load_ui_state(project)


def launch_workbench() -> None:
    """Launch Workbench main window (PySide6 by default)."""
    ui = os.environ.get("PYAOBS_WORKBENCH_UI", "qt").strip().lower()
    if ui == "tk":
        from .shell.tk_main_window import WorkbenchMainWindowTk

        WorkbenchMainWindowTk().run()
        return

    try:
        from .shell_qt.app import main as qt_main
    except ImportError as exc:
        raise RuntimeError(
            "PySide6 未就绪，无法启动 Qt Workbench。请执行："
            " pip install 'pyAOBS[gui-qt]' 或 pip install PySide6\n"
            "若需旧版界面： set PYAOBS_WORKBENCH_UI=tk"
        ) from exc
    qt_main()


if __name__ == "__main__":
    launch_workbench()
