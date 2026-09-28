"""Modeless matplotlib tool window (replaces tk.Toplevel + FigureCanvasTkAgg)."""

from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
except ImportError:  # pragma: no cover
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from ..styles import apply_tool_window_chrome, show_modeless_tool_window


class PlotToolWindow(QMainWindow):
    """Host a matplotlib Figure in a non-modal tool window."""

    def __init__(self, title: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title or "iphase")
        self.resize(1100, 780)
        self._central = QWidget()
        self.setCentralWidget(self._central)
        self._lay = QVBoxLayout(self._central)
        self._lay.setContentsMargins(4, 4, 4, 4)
        self._lay.setSpacing(2)
        self.canvas = None
        self.toolbar = None
        self.fig = None

    def set_figure(self, fig, *, with_toolbar: bool = False) -> None:
        self.fig = fig
        while self._lay.count():
            item = self._lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.canvas = FigureCanvasQTAgg(fig)
        self.toolbar = None
        self._lay.addWidget(self.canvas, stretch=1)
        self.canvas.draw_idle()

    def show_modeless(self, *, activate: bool = True) -> None:
        apply_tool_window_chrome(self)
        show_modeless_tool_window(self, activate=activate)
