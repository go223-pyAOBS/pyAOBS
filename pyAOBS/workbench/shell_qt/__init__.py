"""PySide6 Workbench shell."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .main_window import WorkbenchMainWindow

__all__ = ["WorkbenchMainWindow"]


def __getattr__(name: str):
    if name == "WorkbenchMainWindow":
        from .main_window import WorkbenchMainWindow

        return WorkbenchMainWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
