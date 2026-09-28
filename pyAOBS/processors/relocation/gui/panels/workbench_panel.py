# -*- coding: utf-8 -*-
"""阶段 2：嵌入 zplotpy 姿态工作台（RelocationViewer）。"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..main_window import RelocationViewer


class WorkbenchPanel(QWidget):
    """把 RelocationViewer 嵌为子控件（非独立顶层窗）。"""

    workbench_ready = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.viewer: Optional[RelocationViewer] = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._host = QWidget(self)
        self._host_lay = QVBoxLayout(self._host)
        self._host_lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._host)

    def ensure_viewer(self) -> RelocationViewer:
        if self.viewer is not None:
            return self.viewer
        # 直接以子控件构造，避免顶层窗再降级造成闪烁
        viewer = RelocationViewer(self._host)
        viewer.setWindowFlags(Qt.WindowType.Widget)
        # 退出只走工程主窗；禁用波形台 Q 键 close()
        if hasattr(viewer, "set_allow_shortcut_quit"):
            viewer.set_allow_shortcut_quit(False)
        else:
            viewer._allow_shortcut_quit = False
        try:
            viewer.menuBar().setVisible(False)
        except Exception:
            pass
        try:
            viewer.statusBar().setVisible(True)
        except Exception:
            pass
        self._host_lay.addWidget(viewer, stretch=1)
        self.viewer = viewer
        self.workbench_ready.emit()
        return viewer
