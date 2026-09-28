"""反演监视绘图：原生 pyqtgraph。

χ²/RMS 折线用本模块（pyqtgraph）。右侧速度场走 Matplotlib imshow，见 inv_monitor_model。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

pg.setConfigOptions(
    imageAxisOrder="row-major",
    antialias=False,
    foreground="k",
    background="w",
)

_PLOT_INK = "#111111"


def style_plot_ink(plot) -> None:
    """坐标轴刻度/标题用近黑。pyqtgraph 默认 foreground 是 ``'d'``（深灰）。"""
    item = getattr(plot, "getPlotItem", lambda: plot)()
    pen = pg.mkPen(_PLOT_INK)
    for name in ("left", "bottom", "right", "top"):
        try:
            ax = item.getAxis(name)
            ax.setPen(pen)
            ax.setTextPen(pen)
        except Exception:
            continue
    title = getattr(item, "titleLabel", None)
    if title is not None:
        try:
            title.setAttr("color", _PLOT_INK)
        except Exception:
            pass


def _make_glw(parent=None) -> pg.GraphicsLayoutWidget:
    """与 vedit ContourPlotWidget 相同：默认视口，不替换 QWidget。"""
    glw = pg.GraphicsLayoutWidget(parent)
    glw.setBackground("w")
    try:
        glw.setViewportUpdateMode(glw.ViewportUpdateMode.FullViewportUpdate)
        glw.setCacheMode(glw.CacheModeFlag.CacheNone)
    except Exception:
        pass
    return glw


def _hex_pen(color: str, width: float = 1.0, alpha: float = 1.0):
    c = str(color)
    if c.startswith("#") and len(c) == 7:
        r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
        return pg.mkPen((r, g, b, int(255 * alpha)), width=width)
    return pg.mkPen(c, width=width)


def _hex_brush(color: str, alpha: float = 0.75):
    c = str(color)
    if c.startswith("#") and len(c) == 7:
        r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
        return pg.mkBrush(r, g, b, int(255 * alpha))
    return pg.mkBrush(c)


class _PgHintBar(QWidget):
    def __init__(self, on_reset, on_save=None, parent=None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("滚轮缩放 · 左拖平移 · 右拖缩放 · 双击复位")
        hint.setStyleSheet("color:#666;font-size:11px;")
        self.hint = hint
        row.addWidget(hint)
        row.addStretch(1)
        if on_save is not None:
            btn_save = QPushButton("保存图像…")
            btn_save.setToolTip("保存为 PNG / JPEG / TIFF / PDF / PS / EPS / SVG")
            btn_save.clicked.connect(on_save)
            row.addWidget(btn_save)
        btn = QPushButton("Reset View")
        btn.setToolTip("复位到数据范围（等同双击空白）")
        btn.clicked.connect(on_reset)
        row.addWidget(btn)


def save_glw_png(parent: QWidget, glw: QWidget, *, start_dir: str = "", default_name: str = "plot.png") -> None:
    """兼容旧名：弹出保存框，支持 PNG/JPEG/PDF/PS/SVG 等。"""
    from .export_figure import save_graphics_widget

    save_graphics_widget(parent, glw, start_dir=start_dir, default_name=default_name)


class MonitorCurveWidget(QWidget):
    """左侧 χ² / RMS。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self._glw = _make_glw()
        self._save_dir = ""
        lay.addWidget(_PgHintBar(self.reset_view, self.save_png, self))
        lay.addWidget(self._glw, stretch=1)

        self.chi = self._glw.addPlot(row=0, col=0, title="等待 -L 数据行…")
        self.rms = self._glw.addPlot(row=1, col=0)
        self.chi.showGrid(x=True, y=True, alpha=0.3)
        self.rms.showGrid(x=True, y=True, alpha=0.3)
        self.chi.setLabel("left", "χ²")
        self.rms.setLabel("left", "RMS")
        self.rms.setLabel("bottom", "日志行序")
        self.rms.setXLink(self.chi)
        for p in (self.chi, self.rms):
            style_plot_ink(p)
            try:
                p.setMenuEnabled(False)
                p.getViewBox().setMenuEnabled(False)
            except Exception:
                pass
        self._glw.ci.layout.setRowStretchFactor(0, 1)
        self._glw.ci.layout.setRowStretchFactor(1, 1)
        self._home: dict | None = None

    def reset_view(self) -> None:
        if self._home:
            self.chi.setXRange(*self._home["x"], padding=0)
            self.chi.setYRange(*self._home["chi"], padding=0.06)
            self.rms.setYRange(*self._home["rms"], padding=0.06)
        else:
            self.chi.enableAutoRange()
            self.rms.enableAutoRange()

    def set_save_dir(self, path: str | Path | None) -> None:
        self._save_dir = str(path) if path else ""

    def save_png(self) -> None:
        save_glw_png(
            self, self._glw, start_dir=self._save_dir, default_name="inv_monitor_chi_rms.png"
        )

    def show_waiting(self) -> None:
        self.chi.clear()
        self.rms.clear()
        self.chi.setTitle("等待 -L 数据行…")
        self._home = None

    def update_snapshot(self, snap) -> None:
        self.chi.clear()
        self.rms.clear()
        x = list(snap.iters)
        if not x:
            self.chi.setTitle("等待 -L 数据行…")
            return
        self.chi.setTitle("χ² 随迭代（行序）")
        self.chi.plot(x, list(snap.chi), pen=pg.mkPen("#1f77b4", width=1.5), symbol="o", symbolSize=5)
        if snap.pred_chi:
            self.chi.plot(
                x, list(snap.pred_chi), pen=pg.mkPen("#ff7f0e", width=1.2, style=Qt.PenStyle.DashLine)
            )
        self.chi.addItem(pg.InfiniteLine(x[-1], angle=90, pen=pg.mkPen("#d62728", style=Qt.PenStyle.DotLine)))
        self.chi.plot([x[-1]], [snap.chi[-1]], pen=None, symbol="o", symbolSize=11, symbolBrush="#d62728")

        self.rms.plot(x, list(snap.rms), pen=pg.mkPen("#ff7f0e", width=1.5), symbol="o", symbolSize=5)
        if snap.rough_v:
            self.rms.plot(
                x, list(snap.rough_v), pen=pg.mkPen("#2ca02c", width=1.2, style=Qt.PenStyle.DashLine)
            )
        self.rms.addItem(pg.InfiniteLine(x[-1], angle=90, pen=pg.mkPen("#d62728", style=Qt.PenStyle.DotLine)))
        self.rms.plot([x[-1]], [snap.rms[-1]], pen=None, symbol="o", symbolSize=11, symbolBrush="#d62728")

        self.chi.enableAutoRange()
        self.rms.enableAutoRange()
        xr = self.chi.viewRange()[0]
        self._home = {
            "x": (float(xr[0]), float(xr[1])),
            "chi": tuple(self.chi.viewRange()[1]),
            "rms": tuple(self.rms.viewRange()[1]),
        }
