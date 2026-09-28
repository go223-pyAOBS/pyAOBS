"""Matplotlib 科学图的 PySide6 嵌入面板。

交互对齐 idata / zplotpy / relocation 的 **pyqtgraph ViewBox.PanMode**：

- 左键拖拽：平移
- 右键拖拽：连续缩放（沿拖动方向，非框选）
- 滚轮：以光标为中心缩放
- 双击 / Reset View：复位到首次完整显示的数据范围（home）
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
except ImportError:  # pragma: no cover
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from matplotlib.figure import Figure

from pyAOBS.utils.mpl_plot_nav import PyqtgraphStyleNav


class IPhasePlotCanvas(QWidget):
    """Full-width PySide6 plot panel: QtAgg + pyqtgraph-like mouse."""

    def __init__(self, parent=None, *, figsize=(13, 8), dpi=100) -> None:
        super().__init__(parent)
        self.setObjectName("IphasePlotPanel")
        self.fig = Figure(figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.toolbar = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)

        bar = QHBoxLayout()
        hint = QLabel("滚轮缩放 · 左拖平移 · 右拖缩放 · 双击复位")
        hint.setStyleSheet("color: #666; font-size: 11px;")
        bar.addWidget(hint)
        bar.addStretch(1)
        btn_reset = QPushButton("Reset View")
        btn_reset.setToolTip("复位到数据范围（等同双击空白）")
        bar.addWidget(btn_reset)
        lay.addLayout(bar)
        lay.addWidget(self.canvas, stretch=1)

        self._nav = PyqtgraphStyleNav(self.canvas)
        btn_reset.clicked.connect(self._nav.reset_view)

    def draw(self) -> None:
        self.canvas.draw_idle()

    # ---- view persistence across redraw（委托共享导航）----
    def capture_views(self):
        return self._nav.capture_views()

    def remember_current_views(self) -> None:
        self._nav.remember_current_views()

    def remember_home_views(self) -> None:
        self._nav.remember_home_views()

    def schedule_home_refresh(self) -> None:
        self._nav.schedule_home_refresh()

    def clear_saved_views(self) -> None:
        self._nav.clear_saved_views()

    def restore_saved_views(self) -> bool:
        return self._nav.restore_saved_views()

    def snapshot_before_clear(self) -> None:
        self._nav.snapshot_before_clear()

    def reset_view(self) -> None:
        self._nav.reset_view()
