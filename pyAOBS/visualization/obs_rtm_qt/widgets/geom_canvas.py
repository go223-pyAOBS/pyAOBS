# -*- coding: utf-8 -*-
"""工区几何预览：炮 / OBS / 网格框（pyqtgraph）。"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

try:
    import pyqtgraph as pg
except ImportError as exc:  # pragma: no cover
    pg = None  # type: ignore
    _PG_ERR = exc
else:
    _PG_ERR = None


class GeomCanvas(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if pg is None:
            self.plot = None
            lay.addWidget(QLabel("需要 pyqtgraph\n%s" % _PG_ERR))
            return
        self.plot = pg.PlotWidget(background="w")
        self.plot.setLabel("bottom", "Distance", units="km")
        self.plot.setLabel("left", "Depth", units="km")
        self.plot.invertY(True)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self._grid_box = self.plot.plot(
            pen=pg.mkPen("#94a3b8", width=1.5, style=Qt.PenStyle.DashLine)
        )
        self._shots = pg.ScatterPlotItem(
            size=5, brush=pg.mkBrush("#ef4444"), pen=None, symbol="t"
        )
        self._obs = pg.ScatterPlotItem(
            size=16,
            brush=pg.mkBrush("#39ff14"),
            pen=pg.mkPen("#14532d", width=2),
            symbol="d",
        )
        self._obs_x_line = self.plot.plot(
            pen=pg.mkPen("#0ea5e9", width=1, style=Qt.PenStyle.DotLine)
        )
        self.plot.addItem(self._shots)
        self.plot.addItem(self._obs)
        lay.addWidget(self.plot)

    def clear(self, msg: str = "") -> None:
        if self.plot is None:
            return
        self._grid_box.setData([], [])
        self._shots.setData([], [])
        self._obs.setData([], [])
        self._obs_x_line.setData([], [])
        self.plot.setTitle(msg or "")

    def show_geometry(
        self,
        *,
        shots: Optional[List[Tuple[float, float]]] = None,
        obs: Optional[List[Tuple[float, float]]] = None,
        ox: float = 0.0,
        dx: float = 0.025,
        nx: int = 100,
        oz: float = 0.0,
        dz: float = 0.025,
        nz: int = 100,
        obs_x_mark: Optional[float] = None,
        title: str = "",
    ) -> None:
        if self.plot is None:
            return
        x0 = float(ox)
        x1 = float(ox) + max(int(nx) - 1, 0) * float(dx)
        z0 = float(oz)
        z1 = float(oz) + max(int(nz) - 1, 0) * float(dz)
        self._grid_box.setData([x0, x1, x1, x0, x0], [z0, z0, z1, z1, z0])
        # 海量炮点时抽稀，避免 WSL/pyqtgraph 崩溃
        max_scatter = 2500
        shots_draw = shots
        n_shots = len(shots or [])
        if shots and n_shots > max_scatter:
            step = max(1, n_shots // max_scatter)
            shots_draw = shots[::step]
        if shots_draw:
            self._shots.setData(
                x=[p[0] for p in shots_draw], y=[p[1] for p in shots_draw]
            )
        else:
            self._shots.setData([], [])
        if obs:
            self._obs.setData(x=[p[0] for p in obs], y=[p[1] for p in obs])
        else:
            self._obs.setData([], [])
        if obs_x_mark is not None:
            self._obs_x_line.setData([float(obs_x_mark), float(obs_x_mark)], [z0, z1])
        else:
            self._obs_x_line.setData([], [])

        xs = [x0, x1]
        zs = [z0, z1]
        if shots:
            xs += [p[0] for p in shots]
            zs += [p[1] for p in shots]
        if obs:
            xs += [p[0] for p in obs]
            zs += [p[1] for p in obs]
        pad_x = max(1.0, 0.05 * (max(xs) - min(xs) + 1e-6))
        pad_z = max(0.2, 0.05 * (max(zs) - min(zs) + 1e-6))
        self.plot.setXRange(min(xs) - pad_x, max(xs) + pad_x, padding=0)
        self.plot.setYRange(min(zs) - pad_z, max(zs) + pad_z, padding=0)
        self.plot.setTitle(
            title
            or "几何预览  shots=%d  OBS=%d  (灰框=网格)"
            % (len(shots or []), len(obs or []))
        )
