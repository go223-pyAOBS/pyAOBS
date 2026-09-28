# -*- coding: utf-8 -*-
"""
zplotpy — OBS / 炮集波形显示、拾取与 V 选波工区

A modern Python GUI for interactive seismic phase picking, based on the original
ZPLOT Fortran code by Colin A. Zelt (1994), modified by Haibo Huang (2023).
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"
__author__ = "Haibo Huang"

from .project import ZplotProject

__all__ = [
    "__version__",
    "ZplotProject",
    "QtFastViewer",
    "ZPlotGUI",
]


def __getattr__(name: str) -> Any:
    """懒加载 Qt 查看器，避免 ``import …zplotpy.core`` 时强依赖 PySide6/pyqtgraph。"""
    if name in ("QtFastViewer", "ZPlotGUI"):
        from .gui.qt_fast_viewer import QtFastViewer, ZPlotGUI

        return QtFastViewer if name == "QtFastViewer" else ZPlotGUI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
