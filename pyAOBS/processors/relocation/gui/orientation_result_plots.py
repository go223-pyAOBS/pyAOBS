# -*- coding: utf-8 -*-
"""姿态结果图：OBS 漂移、方位箭头对比（可嵌入 Tab，非独立模态窗）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pyAOBS.utils.qt_file_dialog import get_save_file_name

from ..orientation_correction import (
    OrientationCorrectionResult,
    OrientationObservation,
    PpolTraceSummary,
    compute_ppol_trace_results,
)
from ..services.position_shift_preview import build_obs_shift_summary
from ..services.terrain_style import (
    points_to_rgba_grid,
    terrain_colormap_rgb,
    terrain_rgba_from_grid,
)
from .dialog_utils import show_modeless_dialog
from .position_shift_dialog import draw_obs_shift_on_plot, unit_label


def _set_integer_iter_axis(plot: pg.PlotWidget, n_iters: int) -> None:
    """迭代轮次为离散整数，避免底轴出现 1.0 / 1.5 等浮点刻度。"""
    n = max(1, int(n_iters))
    ticks = [(float(i), str(i)) for i in range(1, n + 1)]
    try:
        plot.getAxis("bottom").setTicks([ticks])
    except Exception:
        pass
    try:
        plot.setXRange(0.5, float(n) + 0.5, padding=0.0)
    except Exception:
        pass


def save_widget_figure(
    capture_widget: QWidget,
    *,
    parent: Optional[QWidget] = None,
    default_stem: str = "orientation_figure",
    caption: str = "保存本组图",
) -> str:
    """弹出保存对话框，将 ``capture_widget`` 截图导出为 PNG/JPEG。成功返回路径，取消返回空串。"""
    host = parent or capture_widget
    path, _ = get_save_file_name(
        host,
        caption,
        str(Path.cwd() / f"{default_stem}.png"),
        "PNG (*.png);;JPEG (*.jpg *.jpeg);;All files (*)",
        default_suffix=".png",
    )
    if not path:
        return ""
    try:
        QApplication.processEvents()
        pix = capture_widget.grab()
        if pix.isNull():
            raise RuntimeError("截图为空，请稍后重试")
        if not pix.save(path):
            raise RuntimeError(f"无法写入文件：{path}")
    except Exception as exc:
        QMessageBox.warning(host, "保存失败", str(exc))
        return ""
    return path


def attach_save_figures_button(
    layout: Union[QVBoxLayout, QHBoxLayout],
    capture_widget: QWidget,
    *,
    default_stem: str = "orientation_figure",
    parent: Optional[QWidget] = None,
    button_text: str = "保存本组图…",
) -> QPushButton:
    """在布局底部加入「保存本组图」按钮（截取 ``capture_widget``）。"""
    row = QHBoxLayout()
    row.setContentsMargins(0, 4, 0, 0)
    btn = QPushButton(button_text)
    btn.setToolTip("将本页图组导出为 PNG/JPEG（不含底栏明细表时可只截图区）")
    row.addStretch(1)
    row.addWidget(btn)
    layout.addLayout(row)

    def _on_save() -> None:
        out = save_widget_figure(
            capture_widget,
            parent=parent or capture_widget.window(),
            default_stem=default_stem,
        )
        if out:
            btn.setToolTip(f"已保存：{out}")

    btn.clicked.connect(_on_save)
    return btn


def draw_terrain_utm_underlay(
    plot: pg.PlotWidget,
    terrain_meta_utm: Optional[Dict[str, Any]],
    *,
    palette: str = "terrain",
    shade_strength: float = 0.75,
    coast_enhance: bool = True,
    opacity: float = 0.55,  # 兼容旧参数；着色图已含 alpha
) -> bool:
    """在已有 PlotWidget 下叠加 UTM 地形底图（色标与位置 Map 默认「地形」一致）。"""
    del opacity  # RGBA 内已带透明度
    if not isinstance(terrain_meta_utm, dict):
        return False
    mode = str(terrain_meta_utm.get("mode", "")).lower()
    pi = plot.getPlotItem()
    pal = str(palette or "terrain")
    try:
        if mode == "grid":
            x = np.asarray(terrain_meta_utm.get("x", []), dtype=float)
            y = np.asarray(terrain_meta_utm.get("y", []), dtype=float)
            z = np.asarray(terrain_meta_utm.get("z", []), dtype=float)
            if x.size < 2 or y.size < 2 or z.size == 0:
                return False
            rgba = terrain_rgba_from_grid(
                z,
                palette=pal,
                shade_strength=float(shade_strength),
                coast_enhance=bool(coast_enhance),
            )
            img = pg.ImageItem(rgba, axisOrder="row-major")
            xmin, xmax = float(np.min(x)), float(np.max(x))
            ymin, ymax = float(np.min(y)), float(np.max(y))
            img.setRect(QRectF(xmin, ymin, max(1e-6, xmax - xmin), max(1e-6, ymax - ymin)))
            img.setZValue(-20)
            pi.addItem(img)
            return True

        x = np.asarray(terrain_meta_utm.get("x", []), dtype=float).reshape(-1)
        y = np.asarray(terrain_meta_utm.get("y", []), dtype=float).reshape(-1)
        z = np.asarray(terrain_meta_utm.get("z", []), dtype=float).reshape(-1)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x, y, z = x[valid], y[valid], z[valid]
        if x.size == 0:
            return False
        packed = points_to_rgba_grid(
            x,
            y,
            z,
            nx=700 if x.size > 80000 else 560,
            ny=520 if x.size > 80000 else 420,
            palette=pal,
            shade_strength=float(shade_strength),
            coast_enhance=bool(coast_enhance),
        )
        if packed is not None:
            rgba, xmin, xmax, ymin, ymax = packed
            img = pg.ImageItem(rgba, axisOrder="row-major")
            img.setRect(QRectF(xmin, ymin, max(1e-6, xmax - xmin), max(1e-6, ymax - ymin)))
            img.setZValue(-20)
            pi.addItem(img)
            return True
        zmin = float(np.nanpercentile(z, 2.0))
        zmax = float(np.nanpercentile(z, 98.0))
        span = max(1e-12, zmax - zmin)
        norm = np.clip((z - zmin) / span, 0.0, 1.0)
        rgb = terrain_colormap_rgb(norm, palette=pal)
        spots = [
            {
                "pos": (float(x[i]), float(y[i])),
                "size": 3.2,
                "brush": pg.mkBrush(int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2]), 140),
                "pen": pg.mkPen(int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2]), 100, width=0.3),
            }
            for i in range(int(x.size))
        ]
        sp = pg.ScatterPlotItem(pxMode=True)
        sp.setData(spots=spots)
        sp.setZValue(-20)
        pi.addItem(sp)
        return True
    except Exception:
        return False


def build_obs_drift_page(
    observations: List[OrientationObservation],
    position_correction: Tuple[float, float, float],
    parent=None,
    *,
    terrain_meta_utm: Optional[Dict[str, Any]] = None,
    terrain_path: str = "",
    terrain_palette: str = "terrain",
    terrain_shade_strength: float = 0.75,
    terrain_coast_enhance: bool = True,
) -> QWidget:
    """OBS 校正前后 + 炮点分布（可叠共用地形底图）。"""
    page = QWidget(parent)
    lay = QVBoxLayout(page)
    summary = build_obs_shift_summary(observations, position_correction)
    unit = unit_label(summary.coord_unit)
    az = summary.azimuth_from_north_deg
    az_txt = f"{az:.1f}°" if np.isfinite(az) else "—"
    n_shot = int(summary.shot_xy.shape[0]) if summary.shot_xy is not None else 0
    terr_name = Path(str(terrain_path or (terrain_meta_utm or {}).get("path", "") or "")).name

    head = QLabel(
        f"OBS×1　炮点×{n_shot}　"
        f"|Δxy|={summary.horizontal:.3f} {unit}　"
        f"漂移方位(北顺时针)={az_txt}　"
        f"dx={summary.dx:.3f}  dy={summary.dy:.3f}  dz={summary.dz:.3f} {unit}"
        + (f"　地形={terr_name}" if terr_name else ""),
        page,
    )
    head.setWordWrap(True)
    head.setStyleSheet("color:#0f172a; font-weight:600;")
    lay.addWidget(head)

    tip = QLabel(
        "灰圆=校正前 OBS；绿圆=校正后 OBS；红箭头=水平漂移；浅红三角=炮点。"
        "底图色标与位置 Map / 姿态预览一致（默认「地形」色带+hillshade）。"
        "若无底图请检查输入页地形路径是否有效。",
        page,
    )
    tip.setWordWrap(True)
    tip.setStyleSheet("color:#475569;")
    lay.addWidget(tip)

    fig_host = QWidget(page)
    fig_lay = QVBoxLayout(fig_host)
    fig_lay.setContentsMargins(0, 0, 0, 0)
    plot = pg.PlotWidget(background="w")
    plot.showGrid(x=True, y=True, alpha=0.15)
    plot.setLabel("bottom", f"X ({unit})")
    plot.setLabel("left", f"Y ({unit})")
    plot.setAspectLocked(True)
    plot.addLegend(offset=(10, 10))
    fig_lay.addWidget(plot)
    lay.addWidget(fig_host, stretch=2)
    terr_draw = terrain_meta_utm
    if isinstance(terrain_meta_utm, dict) and unit == "km":
        # 漂移点若为 km，将 UTM 米制地形缩到同轴
        try:
            terr_draw = dict(terrain_meta_utm)
            terr_draw["x"] = np.asarray(terrain_meta_utm.get("x", []), dtype=float) * 1e-3
            terr_draw["y"] = np.asarray(terrain_meta_utm.get("y", []), dtype=float) * 1e-3
        except Exception:
            terr_draw = terrain_meta_utm
    terr_ok = draw_terrain_utm_underlay(
        plot,
        terr_draw,
        palette=str(terrain_palette or "terrain"),
        shade_strength=float(terrain_shade_strength),
        coast_enhance=bool(terrain_coast_enhance),
    )
    draw_obs_shift_on_plot(plot, summary)
    if not terr_ok:
        miss = QLabel(
            "未叠加地形底图（无可用 UTM 地形；请确认输入页水深文件已加载）。",
            page,
        )
        miss.setStyleSheet("color:#b45309;")
        lay.addWidget(miss)
    attach_save_figures_button(lay, fig_host, default_stem="obs_drift", parent=page)

    table = QPlainTextEdit(page)
    table.setReadOnly(True)
    table.setMaximumHeight(140)
    lines = [summary.message or "", ""]
    if summary.points:
        p = summary.points[0]
        lines.append(
            f"OBS 校正前: ({p.x0:.3f}, {p.y0:.3f}, {p.z0:.3f})\n"
            f"OBS 校正后: ({p.x1:.3f}, {p.y1:.3f}, {p.z1:.3f})\n"
            f"参与观测: {p.n_obs}　炮点: {n_shot}　OBS端: {summary.obs_side}"
        )
    elif not summary.points:
        lines.append("无有效 OBS 坐标，无法绘制漂移。")
    table.setPlainText("\n".join(lines).strip())
    lay.addWidget(table, stretch=0)
    return page


def _az_to_xy(az_deg: float, length: float = 1.0) -> Tuple[float, float]:
    """地理方位（北顺时针）→ 平面坐标 (x=East, y=North)。"""
    rad = np.deg2rad(float(az_deg) % 360.0)
    return float(length * np.sin(rad)), float(length * np.cos(rad))


def _wrap_signed_deg(a: float) -> float:
    return ((float(a) + 180.0) % 360.0) - 180.0


def _draw_az_arrow(
    plot: pg.PlotWidget,
    az_deg: float,
    *,
    length: float,
    pen,
    name: str,
    head_frac: float = 0.18,
) -> None:
    x1, y1 = _az_to_xy(az_deg, length)
    line = pg.PlotDataItem([0.0, x1], [0.0, y1], pen=pen, name=name)
    line.setZValue(20)
    plot.addItem(line)
    horiz = max(float(np.hypot(x1, y1)), 1e-9)
    ux, uy = x1 / horiz, y1 / horiz
    head = max(length * head_frac, 0.06)
    c, s = float(np.cos(np.deg2rad(22.0))), float(np.sin(np.deg2rad(22.0)))
    bx1 = x1 - head * (ux * c - uy * s)
    by1 = y1 - head * (ux * s + uy * c)
    bx2 = x1 - head * (ux * c + uy * s)
    by2 = y1 - head * (-ux * s + uy * c)
    wing = pg.PlotDataItem([bx1, x1, bx2], [by1, y1, by2], pen=pen)
    wing.setZValue(21)
    plot.addItem(wing)


def _draw_rotation_arc(
    plot: pg.PlotWidget,
    az0: float,
    az1: float,
    *,
    radius: float,
    pen,
) -> None:
    """从 az0 转到 az1 的短弧（有符号最短路径）。"""
    d = _wrap_signed_deg(float(az1) - float(az0))
    if abs(d) < 1e-3:
        return
    n = max(8, int(abs(d) / 3.0) + 1)
    angs = np.linspace(float(az0), float(az0) + d, n, dtype=float)
    xs, ys = [], []
    for a in angs:
        x, y = _az_to_xy(a, radius)
        xs.append(x)
        ys.append(y)
    arc = pg.PlotDataItem(xs, ys, pen=pen)
    arc.setZValue(12)
    plot.addItem(arc)


def draw_azimuth_compare_on_plot(
    plot: pg.PlotWidget,
    az_before_deg: float,
    az_after_deg: float,
    *,
    tilt_deg: float = 0.0,
    ori_mean_aligned_deg: Optional[float] = None,
) -> None:
    """罗盘式原方位 / 校正后方位 /（可选）数据估方位箭头对比。"""
    plot.clear()
    plot.showGrid(x=True, y=True, alpha=0.12)
    plot.setAspectLocked(True)
    plot.setLabel("bottom", "East")
    plot.setLabel("left", "North")
    # clear() 会去掉图例，需重建
    plot.addLegend(offset=(8, 8))

    # 外圈与十字
    th = np.linspace(0.0, 2.0 * np.pi, 181)
    plot.plot(np.cos(th), np.sin(th), pen=pg.mkPen("#cbd5e1", width=1.2))
    plot.plot([-1.15, 1.15], [0.0, 0.0], pen=pg.mkPen("#e2e8f0", width=1.0))
    plot.plot([0.0, 0.0], [-1.15, 1.15], pen=pg.mkPen("#e2e8f0", width=1.0))

    for lab, az_lab in (("N", 0.0), ("E", 90.0), ("S", 180.0), ("W", 270.0)):
        lx, ly = _az_to_xy(az_lab, 1.12)
        ti = pg.TextItem(lab, color="#64748b", anchor=(0.5, 0.5))
        ti.setPos(lx, ly)
        ti.setZValue(5)
        plot.addItem(ti)

    az0 = float(az_before_deg) % 360.0
    az1 = float(az_after_deg) % 360.0
    _draw_az_arrow(
        plot,
        az0,
        length=0.78,
        pen=pg.mkPen("#94a3b8", width=2.6, style=Qt.PenStyle.DashLine),
        name=f"原方位(初值) {az0:.1f}°",
    )
    _draw_az_arrow(
        plot,
        az1,
        length=0.95,
        pen=pg.mkPen("#0f766e", width=3.0),
        name=f"校正后(解az) {az1:.1f}°",
    )
    if ori_mean_aligned_deg is not None and np.isfinite(float(ori_mean_aligned_deg)):
        mu = float(ori_mean_aligned_deg) % 360.0
        _draw_az_arrow(
            plot,
            mu,
            length=0.86,
            pen=pg.mkPen("#d97706", width=2.4, style=Qt.PenStyle.DotLine),
            name=f"数据估μ′ {mu:.1f}°",
        )
    _draw_rotation_arc(
        plot,
        az0,
        az1,
        radius=0.42,
        pen=pg.mkPen("#f59e0b", width=2.0),
    )

    dlt = _wrap_signed_deg(az1 - az0)
    tip = pg.TextItem(
        html=(
            f"<span style='color:#0f172a; font-size:11pt;'>"
            f"Δaz(解−初值) = {dlt:+.2f}°<br/>tilt = {float(tilt_deg):.2f}°"
            f"</span>"
        ),
        anchor=(0, 0),
    )
    tip.setPos(0.55, -1.05)
    tip.setZValue(30)
    plot.addItem(tip)

    plot.setXRange(-1.35, 1.35, padding=0.0)
    plot.setYRange(-1.35, 1.35, padding=0.0)


def build_azimuth_compare_page(
    az_before_deg: float,
    az_after_deg: float,
    *,
    tilt_deg: float = 0.0,
    ori_circ_mean_deg: Optional[float] = None,
    ori_mean_aligned_to_az_deg: Optional[float] = None,
    ori_circ_std_deg: Optional[float] = None,
    az_minus_ori_amb180_deg: Optional[float] = None,
    parent=None,
) -> QWidget:
    """原方位 vs 校正后方位（箭头旋转）；可选叠数据估方位 μ′。"""
    page = QWidget(parent)
    lay = QVBoxLayout(page)
    dlt = _wrap_signed_deg(float(az_after_deg) - float(az_before_deg))
    az0 = float(az_before_deg) % 360.0
    az1 = float(az_after_deg) % 360.0
    head_bits = [
        f"原方位(初值假定)={az0:.2f}°",
        f"校正后(解az，旋分量)={az1:.2f}°",
        f"Δaz={dlt:+.2f}°",
        f"tilt={float(tilt_deg):.2f}°",
    ]
    if ori_circ_mean_deg is not None and np.isfinite(float(ori_circ_mean_deg)):
        mu = float(ori_circ_mean_deg)
        mu_p = (
            float(ori_mean_aligned_to_az_deg)
            if ori_mean_aligned_to_az_deg is not None and np.isfinite(float(ori_mean_aligned_to_az_deg))
            else float("nan")
        )
        head_bits.append(f"数据估ORI均值 μ={mu:.2f}°")
        if np.isfinite(mu_p):
            head_bits.append(f"对齐az后 μ′={mu_p:.2f}°")
        if az_minus_ori_amb180_deg is not None and np.isfinite(float(az_minus_ori_amb180_deg)):
            head_bits.append(f"|az−μ′|={float(az_minus_ori_amb180_deg):.2f}°")
        if ori_circ_std_deg is not None and np.isfinite(float(ori_circ_std_deg)):
            head_bits.append(f"圆σ={float(ori_circ_std_deg):.2f}°")
    head = QLabel("　".join(head_bits), page)
    head.setWordWrap(True)
    head.setStyleSheet("color:#0f172a; font-weight:600;")
    lay.addWidget(head)

    tip = QLabel(
        "三者角色不同，勿直接加减："
        "灰虚线=原方位（校正前初值，默认 0°，不是测值）；"
        "青绿实线=校正后解方位 az（实际用来旋 R/T）；"
        "橙点线=多炮 ORI 圆均值对齐到 az 后的 μ′（数据估方位）。"
        "水平 PCA 有 180° 模糊，μ 与 az 表面可差约 180°，应看 |az−μ′|；"
        "圆σ 表示多炮离散，不是「校正量」。",
        page,
    )
    tip.setWordWrap(True)
    tip.setStyleSheet("color:#475569;")
    lay.addWidget(tip)

    fig_host = QWidget(page)
    fig_lay = QVBoxLayout(fig_host)
    fig_lay.setContentsMargins(0, 0, 0, 0)
    plot = pg.PlotWidget(background="w")
    fig_lay.addWidget(plot)
    lay.addWidget(fig_host, stretch=1)
    draw_azimuth_compare_on_plot(
        plot,
        float(az_before_deg),
        float(az_after_deg),
        tilt_deg=float(tilt_deg),
        ori_mean_aligned_deg=ori_mean_aligned_to_az_deg,
    )
    attach_save_figures_button(lay, fig_host, default_stem="azimuth_compare", parent=page)
    return page


def _finite_xy(xs: List[float], ys: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    xa = np.asarray(xs, dtype=float)
    ya = np.asarray(ys, dtype=float)
    m = np.isfinite(xa) & np.isfinite(ya)
    return xa[m], ya[m]


def build_ppol_distribution_page(
    observations: List[OrientationObservation],
    *,
    azimuth_deg: float,
    tilt_deg: float = 0.0,
    position_correction: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    parent=None,
) -> QWidget:
    """每道 ppol：ORI/残差分布 + POL_HOR/T能 统计。"""
    page = QWidget(parent)
    lay = QVBoxLayout(page)
    summary: PpolTraceSummary = compute_ppol_trace_results(
        observations,
        azimuth_deg=float(azimuth_deg),
        tilt_deg=float(tilt_deg),
        position_correction=position_correction,
    )

    head = QLabel(summary.message or "ppol 每道结果", page)
    head.setWordWrap(True)
    head.setStyleSheet("color:#0f172a; font-weight:600;")
    lay.addWidget(head)

    tip = QLabel(
        "ORI = BAZ_th(含位置改正) − BAZ_PCA。"
        "散点已消 180° 模糊对齐到解 az；绿线=解 az，橙线=μ′（数据均值对齐az）。"
        "能量比（右下）：按解方位旋到地理 R/T/Z 后，E_comp/E_tot；"
        "理想情况 R 高、T 低（能量进径向）；Z 随入射角变化。"
        "蓝点 |ORI−az|/90° 为同轴对照。四图横轴均为偏移距(km)。",
        page,
    )
    tip.setWordWrap(True)
    tip.setStyleSheet("color:#475569;")
    lay.addWidget(tip)

    fig_host = QWidget(page)
    grid = QGridLayout(fig_host)
    grid.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(fig_host, stretch=3)

    offs = [r.offset_km for r in summary.rows if r.valid]
    ori_al = [r.ori_aligned_deg for r in summary.rows if r.valid]
    res_az = [r.residual_to_az_deg for r in summary.rows if r.valid]
    pol_h = [r.pol_hor for r in summary.rows if r.valid]
    t_ratio = [r.t_energy_ratio for r in summary.rows if r.valid]
    r_ratio = [r.r_energy_ratio for r in summary.rows if r.valid]
    z_ratio = [r.z_energy_ratio for r in summary.rows if r.valid]

    # 1) offset vs ORI
    p_ori = pg.PlotWidget(background="w")
    p_ori.showGrid(x=True, y=True, alpha=0.15)
    p_ori.setLabel("bottom", "偏移距 (km)")
    p_ori.setLabel("left", "ORI→az (°)")
    p_ori.addLegend(offset=(8, 8))
    x, y = _finite_xy(offs, ori_al)
    if x.size:
        p_ori.plot(
            x,
            y,
            pen=None,
            symbol="o",
            symbolSize=7,
            symbolBrush=pg.mkBrush("#2563eb"),
            name="ORI/道",
        )
    mu_plot = (
        float(summary.ori_mean_aligned_to_az_deg)
        if np.isfinite(summary.ori_mean_aligned_to_az_deg)
        else float(summary.ori_circ_mean_deg)
    )
    if np.isfinite(mu_plot):
        p_ori.addItem(
            pg.InfiniteLine(
                pos=float(mu_plot),
                angle=0,
                pen=pg.mkPen("#f59e0b", width=2.0, style=Qt.PenStyle.DashLine),
                label="μ′(均值→az)",
            )
        )
    p_ori.addItem(
        pg.InfiniteLine(
            pos=float(summary.azimuth_deg) % 360.0,
            angle=0,
            pen=pg.mkPen("#059669", width=2.0),
            label="解 az",
        )
    )
    p_ori.setTitle("每道 ORI(对齐az) vs 偏移距")
    grid.addWidget(p_ori, 0, 0)

    # 2) residual histogram
    p_hist = pg.PlotWidget(background="w")
    p_hist.showGrid(x=True, y=True, alpha=0.15)
    p_hist.setLabel("bottom", "ORI − az (°，已消180°)")
    p_hist.setLabel("left", "道数")
    res = np.asarray(res_az, dtype=float)
    res = res[np.isfinite(res)]
    if res.size:
        bins = min(18, max(6, int(np.sqrt(res.size)) + 2))
        counts, edges = np.histogram(res, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = float(edges[1] - edges[0]) * 0.9 if edges.size > 1 else 1.0
        bar = pg.BarGraphItem(x=centers, height=counts, width=width, brush=pg.mkBrush(37, 99, 235, 160))
        p_hist.addItem(bar)
        p_hist.addItem(
            pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen("#dc2626", width=1.6, style=Qt.PenStyle.DashLine))
        )
    p_hist.setTitle("残差直方 (ORI−az，含180°对齐)")
    grid.addWidget(p_hist, 0, 1)

    # 3) POL_HOR vs offset
    p_pol = pg.PlotWidget(background="w")
    p_pol.showGrid(x=True, y=True, alpha=0.15)
    p_pol.setLabel("bottom", "偏移距 (km)")
    p_pol.setLabel("left", "POL_HOR")
    p_pol.setYRange(0.0, 1.05, padding=0.02)
    x2, y2 = _finite_xy(offs, pol_h)
    if x2.size:
        p_pol.plot(
            x2,
            y2,
            pen=None,
            symbol="s",
            symbolSize=7,
            symbolBrush=pg.mkBrush("#7c3aed"),
            name="POL_HOR",
        )
    if np.isfinite(summary.pol_hor_mean):
        p_pol.addItem(
            pg.InfiniteLine(
                pos=float(summary.pol_hor_mean),
                angle=0,
                pen=pg.mkPen("#7c3aed", width=1.6, style=Qt.PenStyle.DashLine),
            )
        )
    p_pol.setTitle("水平直线度 POL_HOR")
    grid.addWidget(p_pol, 1, 0)

    # 4) R/T/Z 能量比 + 方位残差 vs offset
    p_t = pg.PlotWidget(background="w")
    p_t.showGrid(x=True, y=True, alpha=0.15)
    p_t.setLabel("bottom", "偏移距 (km)")
    p_t.setLabel("left", "比值 (0–1)")
    p_t.setYRange(0.0, 1.05, padding=0.02)
    p_t.addLegend(offset=(8, 8))
    x_r, y_r = _finite_xy(offs, r_ratio)
    x_tt, y_tt = _finite_xy(offs, t_ratio)
    x_z, y_z = _finite_xy(offs, z_ratio)
    x_res, y_res = _finite_xy(offs, [abs(v) for v in res_az])
    if x_r.size:
        p_t.plot(
            x_r,
            y_r,
            pen=None,
            symbol="o",
            symbolSize=7,
            symbolBrush=pg.mkBrush("#059669"),
            name="R能量比 E_R/E_tot（宜高）",
        )
    if x_tt.size:
        p_t.plot(
            x_tt,
            y_tt,
            pen=None,
            symbol="t",
            symbolSize=7,
            symbolBrush=pg.mkBrush("#ea580c"),
            name="T能量比 E_T/E_tot（宜低）",
        )
    if x_z.size:
        p_t.plot(
            x_z,
            y_z,
            pen=None,
            symbol="s",
            symbolSize=6,
            symbolBrush=pg.mkBrush("#64748b"),
            name="Z能量比 E_Z/E_tot",
        )
    if x_res.size:
        # 残差绝对值归一到 0–1 便于同轴对比（/90）
        p_t.plot(
            x_res,
            np.clip(np.asarray(y_res, dtype=float) / 90.0, 0.0, 1.5),
            pen=None,
            symbol="x",
            symbolSize=7,
            symbolBrush=pg.mkBrush("#0ea5e9"),
            name="|ORI−az|/90°",
        )
    p_t.setTitle("R/T/Z 能量比 vs 偏移距（绿R宜高，橙T宜低）")
    grid.addWidget(p_t, 1, 1)
    attach_save_figures_button(lay, fig_host, default_stem="ppol_distribution", parent=page)

    # 明细表
    table = QPlainTextEdit(page)
    table.setReadOnly(True)
    table.setMaximumHeight(160)
    lines = [
        "trace\toffset_km\tBAZ_obs\tBAZ_th\tORI_al\tres_az\tPOL_HOR\tSNR_HOR\tR_ratio\tT_ratio\tZ_ratio",
    ]
    for r in summary.rows:
        if not r.valid:
            lines.append(
                f"{r.trace_idx}\t{r.offset_km:.4f}\t{r.baz_obs_deg:.2f}\t"
                f"—\t—\t—\t{r.pol_hor:.3f}\t{r.snr_hor:.3f}\t"
                f"{r.r_energy_ratio:.3f}\t{r.t_energy_ratio:.3f}\t{r.z_energy_ratio:.3f}"
            )
            continue
        lines.append(
            f"{r.trace_idx}\t{r.offset_km:.4f}\t{r.baz_obs_deg:.2f}\t{r.baz_th_deg:.2f}\t"
            f"{r.ori_aligned_deg:.2f}\t{r.residual_to_az_deg:+.2f}\t"
            f"{r.pol_hor:.3f}\t{r.snr_hor:.3f}\t"
            f"{r.r_energy_ratio:.3f}\t{r.t_energy_ratio:.3f}\t{r.z_energy_ratio:.3f}"
        )
    lines.append("")
    lines.append(
        f"统计: n={summary.n_valid}/{summary.n_total}  "
        f"解az={summary.azimuth_deg:.3f}°  "
        f"ORI_μ={summary.ori_circ_mean_deg:.3f}°  "
        f"μ′(对齐az)={summary.ori_mean_aligned_to_az_deg:.3f}°  "
        f"|az−μ|最短={summary.az_minus_ori_shortest_deg:.3f}°  "
        f"|az−μ′|(含180°)={summary.az_minus_ori_amb180_deg:.3f}°  "
        f"圆σ={summary.ori_circ_std_deg:.3f}°  "
        f"res_MAE={summary.residual_mae_deg:.3f}°  "
        f"res_RMS={summary.residual_rms_deg:.3f}°  "
        f"POL_mean={summary.pol_hor_mean:.3f}"
    )
    lines.append(
        "角色: 解az=旋分量用；ORI_μ=多炮数据估方位；μ′=μ消180°后贴az的一支；"
        "圆σ=多炮离散；原方位(初值)不在本表。"
    )
    table.setPlainText("\n".join(lines))
    lay.addWidget(table, stretch=1)
    return page


def build_diagnostics_page(result, parent=None) -> Optional[QWidget]:
    """迭代诊断页（无 history 时返回 None）。"""
    history = list(getattr(result, "iteration_history", []) or [])
    if not history:
        return None

    page = QWidget(parent)
    lay = QVBoxLayout(page)
    dx, dy, dz = getattr(result, "position_correction", (0.0, 0.0, 0.0))
    details = getattr(result, "details", {}) or {}
    t_prior = float(details.get("prior_time_shift_sec", 0.0))
    t_corr = float(details.get("tt_corr_sec", 0.0))
    t_final = float(details.get("time_shift_sec", 0.0))
    head = QLabel(
        f"az={float(getattr(result, 'azimuth_deg', 0.0)):.2f}°  "
        f"tilt={float(getattr(result, 'tilt_deg', 0.0)):.2f}°  "
        f"dx={float(dx):.3f} dy={float(dy):.3f} dz={float(dz):.3f}  "
        f"prior={t_prior:.3f}s  corr={t_corr:.3f}s  final={t_final:.3f}s  "
        f"J={float(getattr(result, 'objective', float('nan'))):.4f}  "
        f"iters={int(getattr(result, 'iterations', len(history)))}",
        page,
    )
    head.setWordWrap(True)
    head.setStyleSheet("color:#0f172a; font-weight:600;")
    lay.addWidget(head)

    fig_host = QWidget(page)
    grid = QGridLayout(fig_host)
    grid.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(fig_host, stretch=3)
    n_iters = len(history)
    x = np.arange(1, n_iters + 1, dtype=float)
    j = np.asarray([float(h.get("objective", np.nan)) for h in history], dtype=float)
    jtt = np.asarray([float(h.get("J_tt_n", np.nan)) for h in history], dtype=float)
    jpol = np.asarray([float(h.get("J_pol_n", np.nan)) for h in history], dtype=float)
    jsym = np.asarray([float(h.get("J_sym_n", np.nan)) for h in history], dtype=float)

    plot = pg.PlotWidget(background="w")
    plot.showGrid(x=True, y=True, alpha=0.16)
    plot.setLabel("left", "归一化目标函数")
    plot.setLabel("bottom", "迭代轮次")
    plot.addLegend(offset=(8, 8))
    plot.plot(x, j, pen=pg.mkPen("#2563eb", width=2.0), symbol="o", symbolSize=6, name="J")
    plot.plot(x, jtt, pen=pg.mkPen("#dc2626", width=1.8), symbol="t", symbolSize=5, name="J_tt")
    plot.plot(x, jpol, pen=pg.mkPen("#059669", width=1.8), symbol="s", symbolSize=5, name="J_pol")
    plot.plot(x, jsym, pen=pg.mkPen("#7c3aed", width=1.8), symbol="d", symbolSize=5, name="J_sym")
    plot.setTitle("目标函数迭代")
    _set_integer_iter_axis(plot, n_iters)
    grid.addWidget(plot, 0, 0)

    pol_plot = pg.PlotWidget(background="w")
    pol_plot.showGrid(x=True, y=True, alpha=0.16)
    pol_plot.setLabel("left", "极化质量 (0-1)")
    pol_plot.setLabel("bottom", "迭代轮次")
    pol_plot.setYRange(0.0, 1.05, padding=0.02)
    pol_plot.addLegend(offset=(8, 8))
    rect_m = np.asarray([float(h.get("rectilinearity_mean", np.nan)) for h in history], dtype=float)
    dom_m = np.asarray([float(h.get("dominant_energy_ratio_mean", np.nan)) for h in history], dtype=float)
    lin_m = np.asarray([float(h.get("linearity_mean", np.nan)) for h in history], dtype=float)
    pol_plot.plot(x, rect_m, pen=pg.mkPen("#0f766e", width=2.0), symbol="o", symbolSize=6, name="矩形度")
    pol_plot.plot(x, dom_m, pen=pg.mkPen("#b45309", width=1.8), symbol="t", symbolSize=5, name="主导占比")
    pol_plot.plot(x, lin_m, pen=pg.mkPen("#1d4ed8", width=1.8), symbol="s", symbolSize=5, name="线性度")
    pol_plot.setTitle("极化质量随迭代")
    _set_integer_iter_axis(pol_plot, n_iters)
    grid.addWidget(pol_plot, 0, 1)

    az_arr = np.asarray([float(h.get("azimuth_deg", np.nan)) for h in history], dtype=float)
    tilt_arr = np.asarray([float(h.get("tilt_deg", np.nan)) for h in history], dtype=float)
    dir_plot = pg.PlotWidget(background="w")
    dir_plot.showGrid(x=True, y=True, alpha=0.16)
    dir_plot.setLabel("left", "倾角 (°)")
    dir_plot.setLabel("bottom", "方位角 (°)")
    dir_plot.plot(
        az_arr,
        tilt_arr,
        pen=pg.mkPen("#7c3aed", width=2.0),
        symbol="o",
        symbolSize=7,
        symbolBrush=pg.mkBrush("#7c3aed"),
        name="az-tilt 轨迹",
    )
    dir_plot.setTitle("方位-倾角轨迹")
    grid.addWidget(dir_plot, 1, 0)

    def _dir_vec(az_deg: float, tilt_deg: float) -> np.ndarray:
        azr = np.deg2rad(float(az_deg))
        tiltr = np.deg2rad(float(tilt_deg))
        v = np.asarray(
            [np.cos(tiltr) * np.cos(azr), np.cos(tiltr) * np.sin(azr), np.sin(tiltr)],
            dtype=float,
        )
        n = float(np.linalg.norm(v))
        return v / n if np.isfinite(n) and n > 1e-12 else np.asarray([1.0, 0.0, 0.0])

    dir_change = np.full(len(history), np.nan, dtype=float)
    warn_deg, crit_deg = 5.0, 10.0
    for i in range(1, len(history)):
        if not (
            np.isfinite(az_arr[i - 1])
            and np.isfinite(tilt_arr[i - 1])
            and np.isfinite(az_arr[i])
            and np.isfinite(tilt_arr[i])
        ):
            continue
        c = float(
            np.clip(
                np.dot(_dir_vec(az_arr[i - 1], tilt_arr[i - 1]), _dir_vec(az_arr[i], tilt_arr[i])),
                -1.0,
                1.0,
            )
        )
        dir_change[i] = float(np.rad2deg(np.arccos(c)))

    chg_plot = pg.PlotWidget(background="w")
    chg_plot.showGrid(x=True, y=True, alpha=0.16)
    chg_plot.setLabel("left", "方向变化角 (°)")
    chg_plot.setLabel("bottom", "迭代轮次")
    chg_plot.addItem(
        pg.InfiniteLine(pos=warn_deg, angle=0, pen=pg.mkPen("#f59e0b", width=1.2, style=Qt.PenStyle.DashLine))
    )
    chg_plot.addItem(
        pg.InfiniteLine(pos=crit_deg, angle=0, pen=pg.mkPen("#ef4444", width=1.2, style=Qt.PenStyle.DashLine))
    )
    chg_plot.plot(x, dir_change, pen=pg.mkPen("#ea580c", width=2.0), name="方向变化")
    chg_plot.setTitle(f"相邻迭代方向变化（<{warn_deg:.0f}°稳定 / ≥{crit_deg:.0f}°高风险）")
    _set_integer_iter_axis(chg_plot, n_iters)
    grid.addWidget(chg_plot, 1, 1)
    attach_save_figures_button(lay, fig_host, default_stem="orientation_diagnostics", parent=page)

    txt = QPlainTextEdit(page)
    txt.setReadOnly(True)
    txt.setMaximumHeight(150)
    lines = ["iter\tJ\tJ_tt\tJ_pol\tJ_sym\taz\ttilt\tdt\tdx\tdy\tdz\tdir_chg"]
    for i, h in enumerate(history):
        dchg = float(dir_change[i]) if i < dir_change.size else float("nan")
        lines.append(
            f"{int(h.get('iter', 0))}\t"
            f"{float(h.get('objective', np.nan)):.5f}\t"
            f"{float(h.get('J_tt', np.nan)):.5f}\t"
            f"{float(h.get('J_pol', np.nan)):.5f}\t"
            f"{float(h.get('J_sym', np.nan)):.5f}\t"
            f"{float(h.get('azimuth_deg', np.nan)):.3f}\t"
            f"{float(h.get('tilt_deg', np.nan)):.3f}\t"
            f"{float(h.get('time_shift_sec', np.nan)):.4f}\t"
            f"{float(h.get('dx', np.nan)):.3f}\t"
            f"{float(h.get('dy', np.nan)):.3f}\t"
            f"{float(h.get('dz', np.nan)):.3f}\t"
            f"{dchg:.3f}"
        )
    txt.setPlainText("\n".join(lines))
    lay.addWidget(txt, stretch=1)
    return page


def show_orientation_result_bundle(
    observations: List[OrientationObservation],
    result,
    parent=None,
    *,
    activate: bool = True,
) -> Optional[QDialog]:
    """无工作台波形页时的统一结果窗：诊断/漂移/方位/ppol（页签）。"""
    dlg = QDialog(parent)
    dlg.setWindowTitle("姿态校正结果可视化")
    dlg.setModal(False)
    dlg.resize(1280, 760)
    lay = QVBoxLayout(dlg)
    tab = QTabWidget(dlg)
    lay.addWidget(tab)

    diag = build_diagnostics_page(result, parent=dlg)
    if diag is not None:
        tab.addTab(diag, "迭代诊断")

    pos = getattr(result, "position_correction", (0.0, 0.0, 0.0))
    pos_tuple = (float(pos[0]), float(pos[1]), float(pos[2]))
    terr_meta = None
    terr_path = ""
    try:
        parent_viewer = parent
        while parent_viewer is not None and not hasattr(parent_viewer, "ensure_shared_terrain_loaded"):
            parent_viewer = parent_viewer.parent() if hasattr(parent_viewer, "parent") else None
        if parent_viewer is not None and hasattr(parent_viewer, "ensure_shared_terrain_loaded"):
            try:
                parent_viewer.ensure_shared_terrain_loaded()
            except Exception:
                pass
            terr_meta = getattr(parent_viewer, "_orientation_terrain_meta_utm", None)
            terr_path = str(getattr(parent_viewer, "_orientation_terrain_path", "") or "")
    except Exception:
        terr_meta = None
    tab.addTab(
        build_obs_drift_page(
            observations,
            pos_tuple,
            parent=dlg,
            terrain_meta_utm=terr_meta,
            terrain_path=terr_path,
        ),
        "OBS 漂移图",
    )

    details = getattr(result, "details", {}) or {}
    az_before = float(details.get("initial_azimuth_deg", 0.0) or 0.0)
    az_sol = float(getattr(result, "azimuth_deg", 0.0))
    tilt_sol = float(getattr(result, "tilt_deg", 0.0))
    ppol_summary = compute_ppol_trace_results(
        observations,
        azimuth_deg=az_sol,
        tilt_deg=tilt_sol,
        position_correction=pos_tuple,
    )
    tab.addTab(
        build_azimuth_compare_page(
            az_before,
            az_sol,
            tilt_deg=tilt_sol,
            ori_circ_mean_deg=float(ppol_summary.ori_circ_mean_deg),
            ori_mean_aligned_to_az_deg=float(ppol_summary.ori_mean_aligned_to_az_deg),
            ori_circ_std_deg=float(ppol_summary.ori_circ_std_deg),
            az_minus_ori_amb180_deg=float(ppol_summary.az_minus_ori_amb180_deg),
            parent=dlg,
        ),
        "方位对比",
    )
    tab.addTab(
        build_ppol_distribution_page(
            observations,
            azimuth_deg=az_sol,
            tilt_deg=tilt_sol,
            position_correction=pos_tuple,
            parent=dlg,
        ),
        "ppol 每道分布",
    )

    btn_row = QHBoxLayout()
    btn_save_tab = QPushButton("保存当前页图…", dlg)
    btn_close = QPushButton("关闭", dlg)
    btn_row.addWidget(btn_save_tab)
    btn_row.addStretch(1)
    btn_row.addWidget(btn_close)
    lay.addLayout(btn_row)

    def _save_current_tab() -> None:
        w = tab.currentWidget()
        if w is None:
            return
        title = str(tab.tabText(tab.currentIndex()) or "orientation_tab")
        stem = "orientation_" + "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in title)
        save_widget_figure(
            w,
            parent=dlg,
            default_stem=stem.strip("_") or "orientation_tab",
            caption=f"保存当前页图（{title}）",
        )

    btn_save_tab.clicked.connect(_save_current_tab)
    btn_close.clicked.connect(dlg.close)
    show_modeless_dialog(dlg, activate=bool(activate))
    return dlg
