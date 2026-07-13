"""
zplotpy GUI 兼容入口。

保留原有模块路径：
    python -m pyAOBS.visualization.zplotpy.gui
并统一转发到 Qt Fast Viewer（当前最终版本）。
"""

from __future__ import annotations

from .qt_fast_viewer import QtFastViewer, ZPlotGUI, main

__all__ = ["QtFastViewer", "ZPlotGUI", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
