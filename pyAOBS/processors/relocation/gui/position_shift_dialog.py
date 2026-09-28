# -*- coding: utf-8 -*-
"""唯一 OBS 校正前后 + 多炮点分布（非模态）。"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
)

from ..orientation_correction import OrientationCorrectionResult, OrientationObservation
from ..services.position_shift_preview import ObsShiftSummary, build_obs_shift_summary
from .dialog_utils import show_modeless_dialog


def unit_label(unit: str) -> str:
    if unit == "m":
        return "m"
    if unit == "km":
        return "km"
    return "coord"


def draw_obs_shift_on_plot(plot: pg.PlotWidget, summary: ObsShiftSummary) -> None:
    """把 OBS 漂移/炮点画到已有 PlotWidget（可复用到结果 Tab）。"""
    pi = plot.getPlotItem()
    shot = np.asarray(summary.shot_xy, dtype=float)
    if shot.size > 0 and shot.ndim == 2 and shot.shape[1] >= 2:
        sp_src = pg.ScatterPlotItem(
            x=shot[:, 0],
            y=shot[:, 1],
            size=10,
            symbol="t",
            brush=pg.mkBrush(239, 68, 68, 150),
            pen=pg.mkPen("#7f1d1d", width=1.0),
            name=f"炮点×{shot.shape[0]}",
        )
        sp_src.setZValue(5)
        pi.addItem(sp_src)

    if not summary.points:
        return

    p = summary.points[0]
    before = pg.ScatterPlotItem(
        x=[p.x0],
        y=[p.y0],
        size=16,
        symbol="o",
        brush=pg.mkBrush(148, 163, 184, 230),
        pen=pg.mkPen("#334155", width=1.6),
        name="校正前 OBS",
    )
    after = pg.ScatterPlotItem(
        x=[p.x1],
        y=[p.y1],
        size=16,
        symbol="o",
        brush=pg.mkBrush(16, 185, 129, 230),
        pen=pg.mkPen("#065f46", width=1.6),
        name="校正后 OBS",
    )
    before.setZValue(20)
    after.setZValue(21)
    pi.addItem(before)
    pi.addItem(after)

    arrow_pen = pg.mkPen("#dc2626", width=2.4)
    if p.horizontal > 1e-12:
        line = pg.PlotDataItem(
            [p.x0, p.x1],
            [p.y0, p.y1],
            pen=arrow_pen,
            name=f"OBS 位移 (|Δxy|={summary.horizontal:.3f})",
        )
        line.setZValue(15)
        pi.addItem(line)
        ux, uy = p.dx / p.horizontal, p.dy / p.horizontal
        head = max(p.horizontal * 0.18, summary.horizontal * 0.12, 1e-6)
        c, s = np.cos(np.deg2rad(25.0)), np.sin(np.deg2rad(25.0))
        bx1 = p.x1 - head * (ux * c - uy * s)
        by1 = p.y1 - head * (ux * s + uy * c)
        bx2 = p.x1 - head * (ux * c + uy * s)
        by2 = p.y1 - head * (-ux * s + uy * c)
        wing = pg.PlotDataItem([bx1, p.x1, bx2], [by1, p.y1, by2], pen=arrow_pen)
        wing.setZValue(16)
        pi.addItem(wing)

    az = summary.azimuth_from_north_deg
    az_txt = f"{az:.1f}°" if np.isfinite(az) else "—"
    unit = unit_label(summary.coord_unit)
    text = pg.TextItem(
        html=(
            f"<span style='color:#0f172a; font-size:11pt;'>"
            f"OBS |Δxy|={summary.horizontal:.3f} {unit}<br/>"
            f"az(N→E cw)={az_txt}<br/>"
            f"dz={summary.dz:.3f} {unit}"
            f"</span>"
        ),
        anchor=(0, 1),
    )
    text.setPos(float(p.x1), float(p.y1))
    text.setZValue(40)
    pi.addItem(text)

    xs = [p.x0, p.x1]
    ys = [p.y0, p.y1]
    if shot.size > 0:
        xs.extend(shot[:, 0].tolist())
        ys.extend(shot[:, 1].tolist())
    xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
    ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
    pad = max(
        summary.horizontal * 2.0,
        0.05 * max(xmax - xmin, ymax - ymin, 1.0),
        1.0,
    )
    if summary.coord_unit == "km":
        pad = max(pad, 0.05)
    plot.setXRange(xmin - pad, xmax + pad, padding=0.0)
    plot.setYRange(ymin - pad, ymax + pad, padding=0.0)


class PositionShiftDialog(QDialog):
    def __init__(
        self,
        summary: ObsShiftSummary,
        observations: Optional[List[OrientationObservation]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("OBS 位置校正对比（单台 OBS）")
        self.setModal(False)
        self.resize(960, 720)

        unit = unit_label(summary.coord_unit)
        lay = QVBoxLayout(self)

        az = summary.azimuth_from_north_deg
        az_txt = f"{az:.1f}°" if np.isfinite(az) else "—"
        n_shot = int(summary.shot_xy.shape[0]) if summary.shot_xy is not None else 0
        head = QLabel(
            f"OBS×1　炮点×{n_shot}　"
            f"|Δxy|={summary.horizontal:.3f} {unit}　"
            f"方向(北顺时针)={az_txt}　"
            f"dx={summary.dx:.3f}  dy={summary.dy:.3f}  dz={summary.dz:.3f} {unit}",
            self,
        )
        head.setWordWrap(True)
        head.setStyleSheet("color:#0f172a; font-weight:600;")
        lay.addWidget(head)

        tip = QLabel(
            "图例：灰圆=校正前 OBS（唯一）；绿圆=校正后 OBS；红箭头=OBS 水平位移；"
            "浅红三角=各道对应炮点（多炮）。"
            "OBS 取道头中坐标更“固定”的一侧（XY 方差更小）；炮点为高散一侧。",
            self,
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#475569;")
        lay.addWidget(tip)

        self.plot = pg.PlotWidget(background="w")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setLabel("bottom", f"X ({unit})")
        self.plot.setLabel("left", f"Y ({unit})")
        self.plot.setAspectLocked(True)
        self.plot.addLegend(offset=(10, 10))
        lay.addWidget(self.plot, stretch=2)

        draw_obs_shift_on_plot(self.plot, summary)

        table = QPlainTextEdit(self)
        table.setReadOnly(True)
        table.setMaximumHeight(160)
        lines = [summary.message, ""]
        if summary.points:
            p = summary.points[0]
            lines.append(
                f"OBS 校正前: ({p.x0:.3f}, {p.y0:.3f}, {p.z0:.3f})\n"
                f"OBS 校正后: ({p.x1:.3f}, {p.y1:.3f}, {p.z1:.3f})\n"
                f"参与观测条数: {p.n_obs}　炮点数: {n_shot}　OBS端字段: {summary.obs_side}"
            )
        table.setPlainText("\n".join(lines))
        lay.addWidget(table, stretch=1)


def show_position_shift_preview(
    observations: List[OrientationObservation],
    result: OrientationCorrectionResult,
    parent=None,
) -> Optional[QDialog]:
    summary = build_obs_shift_summary(observations, result.position_correction)
    if not summary.points:
        return None
    dlg = PositionShiftDialog(summary, observations=observations, parent=parent)
    show_modeless_dialog(dlg, activate=True)
    return dlg
