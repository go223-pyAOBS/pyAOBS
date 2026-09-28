"""zplotpy GUI（工程化工区 + 波形工作台）。

入口::

    python -m pyAOBS.visualization.zplotpy.gui

主窗为 ``ZplotProjectWindow``；波形底座 ``QtFastViewer``。
纯查看器：``python -m pyAOBS.visualization.zplotpy.gui.qt_fast_viewer``。
"""

from __future__ import annotations

from typing import Any

__all__ = ["ZplotProjectWindow", "QtFastViewer", "ZPlotGUI", "main"]


def __getattr__(name: str) -> Any:
    if name in ("ZplotProjectWindow", "main"):
        from .project_window import ZplotProjectWindow, main

        return ZplotProjectWindow if name == "ZplotProjectWindow" else main
    if name in ("QtFastViewer", "ZPlotGUI"):
        from .qt_fast_viewer import QtFastViewer, ZPlotGUI

        return QtFastViewer if name == "QtFastViewer" else ZPlotGUI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    from .project_window import main

    raise SystemExit(main())
