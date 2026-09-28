# -*- coding: utf-8 -*-
"""炮点 / OBS 平面分布画布（分色分形 + 图例）。"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

try:
    import pyqtgraph as pg

    _HAS_PG = True
except Exception:
    pg = None  # type: ignore
    _HAS_PG = False


class GeomCanvas(QWidget):
    """点击最近点发出 trace_clicked(row)。"""

    trace_clicked = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._info = QLabel("无几何数据")
        self._info.setWordWrap(True)
        self._info.setStyleSheet("color:#334155;")
        lay.addWidget(self._info)
        self._plot = None
        self._scatter_shot = None
        self._scatter_obs = None
        self._scatter_sel = None
        self._legend = None
        self._trace_xy: List[Tuple[float, float, int]] = []
        self._view_initialized = False
        self._sel_row: Optional[int] = None
        self._sel_xy: Optional[Tuple[float, float]] = None
        if _HAS_PG:
            self._plot = pg.PlotWidget()
            self._plot.setBackground("w")
            self._plot.showGrid(x=True, y=True, alpha=0.3)
            self._plot.setLabel("bottom", "X", units="m")
            self._plot.setLabel("left", "Y", units="m")
            self._plot.setAspectLocked(True)
            self._scatter_shot = pg.ScatterPlotItem(
                size=8,
                brush=pg.mkBrush("#ef4444"),
                pen=pg.mkPen("#7f1d1d", width=0.5),
                symbol="t",
                name="炮点",
            )
            self._scatter_obs = pg.ScatterPlotItem(
                size=12,
                brush=pg.mkBrush("#2563eb"),
                pen=pg.mkPen("#1e3a8a", width=1),
                symbol="d",
                name="OBS",
            )
            self._scatter_sel = pg.ScatterPlotItem(
                size=18,
                brush=pg.mkBrush("#16a34a"),
                pen=pg.mkPen("#14532d", width=2),
                symbol="o",
                name="选中炮",
            )
            self._plot.addItem(self._scatter_shot)
            self._plot.addItem(self._scatter_obs)
            self._plot.addItem(self._scatter_sel)
            self._legend = self._plot.addLegend(offset=(8, 8))
            self._legend.addItem(self._scatter_shot, "炮点")
            self._legend.addItem(self._scatter_obs, "OBS")
            self._legend.addItem(self._scatter_sel, "选中炮")
            self._plot.scene().sigMouseClicked.connect(self._on_click)
            lay.addWidget(self._plot, stretch=1)
        else:
            lay.addWidget(QLabel("pyqtgraph 未安装，无法显示几何 Map"), stretch=1)

    def clear(self) -> None:
        self._info.setText("无几何数据")
        self._trace_xy = []
        self._view_initialized = False
        self._sel_row = None
        self._sel_xy = None
        if self._scatter_shot is not None:
            self._scatter_shot.setData([], [])
            self._scatter_obs.setData([], [])
            self._scatter_sel.setData([], [])

    def set_legend_labels(self, shot_label: str, obs_label: str) -> None:
        if self._legend is None or self._scatter_shot is None:
            return
        self._legend.removeItem(self._scatter_shot)
        self._legend.removeItem(self._scatter_obs)
        if self._scatter_sel is not None:
            try:
                self._legend.removeItem(self._scatter_sel)
            except Exception:
                pass
        self._legend.addItem(self._scatter_shot, shot_label)
        self._legend.addItem(self._scatter_obs, obs_label)
        if self._scatter_sel is not None:
            self._legend.addItem(self._scatter_sel, "选中炮")

    def set_points(
        self,
        shots: Sequence[Tuple[float, float]],
        obs: Sequence[Tuple[float, float]],
        *,
        trace_anchors: Optional[Sequence[Tuple[float, float, int]]] = None,
        title: str = "",
        shot_label: str = "炮点",
        obs_label: str = "OBS",
        reset_view: bool = False,
    ) -> None:
        """
        shots/obs: 唯一物理点；
        trace_anchors: (x,y,trace_index) 用于点击选道。
        reset_view: True 时重新自适应范围；False 保留当前缩放/平移。
        """
        sx = [float(p[0]) for p in shots]
        sy = [float(p[1]) for p in shots]
        ox = [float(p[0]) for p in obs]
        oy = [float(p[1]) for p in obs]
        self._trace_xy = list(trace_anchors or [])
        self._info.setText(
            title
            or f"{shot_label}: {len(shots)}  |  {obs_label}: {len(obs)}"
        )
        self.set_legend_labels(shot_label, obs_label)
        if self._scatter_shot is not None:
            self._scatter_shot.setData(x=sx, y=sy)
            self._scatter_obs.setData(x=ox, y=oy)
            # 恢复选中高亮（刷新点集时不丢掉）
            if self._sel_row is not None:
                self.set_selected_trace(self._sel_row, self._sel_xy)
            else:
                self._scatter_sel.setData([], [])
            if reset_view or not self._view_initialized:
                self._auto_range(sx + ox, sy + oy)
                self._view_initialized = True

    def _auto_range(self, xs: Sequence[float], ys: Sequence[float]) -> None:
        if self._plot is None or not xs or not ys:
            return
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        if abs(xmax - xmin) < 1e-6:
            xmin -= 100.0
            xmax += 100.0
        if abs(ymax - ymin) < 1e-6:
            ymin -= 100.0
            ymax += 100.0
        pad_x = max(50.0, 0.05 * (xmax - xmin))
        pad_y = max(50.0, 0.05 * (ymax - ymin))
        self._plot.setXRange(xmin - pad_x, xmax + pad_x, padding=0)
        self._plot.setYRange(ymin - pad_y, ymax + pad_y, padding=0)

    def set_selected_trace(
        self, row: int, xy: Optional[Tuple[float, float]] = None
    ) -> None:
        if self._scatter_sel is None:
            return
        if xy is None:
            for x, y, idx in self._trace_xy:
                if idx == row:
                    xy = (x, y)
                    break
        self._sel_row = int(row) if row is not None else None
        self._sel_xy = (float(xy[0]), float(xy[1])) if xy is not None else None
        if xy is None:
            self._scatter_sel.setData([], [])
            return
        self._scatter_sel.setData(x=[xy[0]], y=[xy[1]])

    def _on_click(self, ev) -> None:
        if self._plot is None or not self._trace_xy:
            return
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        # 忽略画布外 / 图例区误点
        vb = self._plot.plotItem.vb
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        pos = vb.mapSceneToView(ev.scenePos())
        x, y = float(pos.x()), float(pos.y())
        best_i = None
        best_xy = None
        best_d = None
        for ax, ay, idx in self._trace_xy:
            d = (ax - x) ** 2 + (ay - y) ** 2
            if best_d is None or d < best_d:
                best_d = d
                best_i = idx
                best_xy = (ax, ay)
        if best_i is not None and best_xy is not None:
            # 先高亮再通知（切页后仍保留）
            self.set_selected_trace(best_i, best_xy)
            self.trace_clicked.emit(int(best_i))
