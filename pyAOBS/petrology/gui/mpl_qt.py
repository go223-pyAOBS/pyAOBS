"""Matplotlib Qt backend + pyqtgraph 式绘图导航（与 imodel/iphase 同一套）。"""

from __future__ import annotations

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from pyAOBS.utils.mpl_plot_nav import (
    PyqtgraphStyleNav,
    install_plot_nav_bar,
    notify_plot_updated,
)

__all__ = [
    "FigureCanvasQTAgg",
    "PyqtgraphStyleNav",
    "install_plot_nav_bar",
    "notify_plot_updated",
]
