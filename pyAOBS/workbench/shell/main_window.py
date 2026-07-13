"""Workbench shell entry — Qt default, Tk legacy fallback."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tk_main_window import WorkbenchMainWindowTk
    from ..shell_qt.main_window import WorkbenchMainWindow as WorkbenchMainWindowQt


def _load_workbench_main_window_class():
    if os.environ.get("PYAOBS_WORKBENCH_UI", "qt").lower() == "tk":
        from .tk_main_window import WorkbenchMainWindowTk

        return WorkbenchMainWindowTk
    from ..shell_qt.main_window import WorkbenchMainWindow

    return WorkbenchMainWindow


def WorkbenchMainWindow(*args, **kwargs):
    """Lazy factory — avoids importing PySide6 until the window is created."""
    cls = _load_workbench_main_window_class()
    return cls(*args, **kwargs)


WorkbenchMainWindow.__doc__ = "Workbench main window (Qt default; Tk if PYAOBS_WORKBENCH_UI=tk)."

__all__ = ["WorkbenchMainWindow"]
