# -*- coding: utf-8 -*-
"""单道波形预览（pyqtgraph，失败则退回占位）。"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

try:
    import pyqtgraph as pg

    _HAS_PG = True
except Exception:
    pg = None  # type: ignore
    _HAS_PG = False


class TracePreviewWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._info = QLabel("未选择道")
        self._info.setStyleSheet("color:#64748b;")
        lay.addWidget(self._info)
        self._plot = None
        self._curve = None
        if _HAS_PG:
            self._plot = pg.PlotWidget()
            self._plot.setBackground("w")
            self._plot.showGrid(x=True, y=True, alpha=0.3)
            self._plot.setLabel("bottom", "time", units="s")
            self._curve = self._plot.plot(pen=pg.mkPen("#2563eb", width=1))
            lay.addWidget(self._plot, stretch=1)
        else:
            lay.addWidget(QLabel("pyqtgraph 未安装，无法预览波形"), stretch=1)

    def clear(self) -> None:
        self._info.setText("未选择道")
        if self._curve is not None:
            self._curve.setData([], [])

    def show_trace(
        self,
        samples: np.ndarray,
        *,
        dt_us: int = 0,
        title: str = "",
    ) -> None:
        y = np.asarray(samples, dtype=float).ravel()
        if y.size == 0:
            self.clear()
            return
        dt = float(dt_us) * 1e-6 if dt_us and dt_us > 0 else 1.0
        x = np.arange(y.size, dtype=float) * dt
        self._info.setText(title or f"npts={y.size}")
        if self._curve is not None:
            self._curve.setData(x, y)
