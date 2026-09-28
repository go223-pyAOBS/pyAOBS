# -*- coding: utf-8 -*-
"""阶段 2：嵌入 zplotpy QtFastViewer。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

if TYPE_CHECKING:
    from ..qt_fast_viewer import QtFastViewer


class WorkbenchPanel(QWidget):
    """把 QtFastViewer 嵌为子控件（非独立顶层窗）。"""

    workbench_ready = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.viewer: Optional["QtFastViewer"] = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._host = QWidget(self)
        self._host_lay = QVBoxLayout(self._host)
        self._host_lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._host)

    def ensure_viewer(self) -> "QtFastViewer":
        if self.viewer is not None:
            return self.viewer
        from ..qt_fast_viewer import QtFastViewer

        # 直接以子控件构造，避免「先顶层窗再 setParent」造成的闪白/闪缩
        self._host.setUpdatesEnabled(False)
        try:
            viewer = QtFastViewer(self._host)
            viewer.setWindowFlags(Qt.WindowType.Widget)
            if hasattr(viewer, "set_allow_shortcut_quit"):
                viewer.set_allow_shortcut_quit(False)
            else:
                viewer._allow_shortcut_quit = False
            try:
                viewer.menuBar().setVisible(False)
                viewer.menuBar().setMaximumHeight(0)
            except Exception:
                pass
            try:
                viewer.statusBar().setVisible(True)
            except Exception:
                pass
            self._host_lay.addWidget(viewer, stretch=1)
            self.viewer = viewer
        finally:
            self._host.setUpdatesEnabled(True)
        self.workbench_ready.emit()
        return viewer
