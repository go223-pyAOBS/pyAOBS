"""Matplotlib Qt backend imports (qtagg with qt5agg fallback)."""

from __future__ import annotations

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT

__all__ = ["FigureCanvasQTAgg", "NavigationToolbar2QT"]
