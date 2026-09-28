"""走时文件预览：ttimes.dat / tx.in；原生 pyqtgraph（对齐 vedit 走时图）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..dialog_utils import show_modeless_dialog, show_modeless_message
from ..services.obs_stations import parse_station_lis, resolve_station_lis_path
from ..services.paths import resolve_work_dir
from ..services.tt_plot_data import (
    TtPick,
    build_obs_catalog,
    build_phase_catalog,
    load_ttimes_picks,
    load_tx_in_picks,
    picks_to_arrays,
    reduce_traveltime,
)
from ..services.workflow import resolve_tx_convert_paths
from ..state.form_state import FormState
from ..widgets.obs_check_list import ObsCheckList
from ..widgets.simple_check_list import SimpleCheckList

# 与 vedit.gui.pg_ray_plot 震相色板同系，便于多 OBS 区分
_PALETTE = [
    QColor("#E60000"),
    QColor("#0055FF"),
    QColor("#009933"),
    QColor("#FF8800"),
    QColor("#9900CC"),
    QColor("#008888"),
    QColor("#FF1493"),
    QColor("#886600"),
    QColor("#222222"),
    QColor("#00AACC"),
    QColor("#CC0000"),
    QColor("#3366FF"),
    QColor("#1f77b4"),
    QColor("#d62728"),
    QColor("#2ca02c"),
    QColor("#9467bd"),
    QColor("#8c564b"),
    QColor("#e377c2"),
    QColor("#7f7f7f"),
    QColor("#bcbd22"),
]


def _color(i: int) -> QColor:
    return _PALETTE[int(i) % len(_PALETTE)]


class TravelTimePlotWindow(QWidget):
    """非模态走时图：pyqtgraph ViewBox 导航 + 可调 vred。"""

    def __init__(
        self,
        picks: list[TtPick],
        *,
        title: str,
        save_dir: str | None = None,
        kind: str = "ttimes",
        stations: list[tuple[int, float, float]] | None = None,
        state: FormState | None = None,
        persist_obs: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1000, 680)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._picks = list(picks)
        self._arr = picks_to_arrays(self._picks)
        self._save_dir = save_dir
        self._kind = kind  # "ttimes" | "tx"
        self._stations = list(stations or [])
        self._obs_catalog = build_obs_catalog(
            self._arr, self._stations or None
        )
        self._keep_view = False

        root = QVBoxLayout(self)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("折合速度 vred (km/s)："))
        self.spin_vred = QDoubleSpinBox()
        self.spin_vred.setRange(0.0, 20.0)
        self.spin_vred.setDecimals(3)
        self.spin_vred.setSingleStep(0.1)
        self.spin_vred.setValue(6.0)
        self.spin_vred.setToolTip(
            "t′ = t − |x−xobs|/vred（iphase）；x 为测线模型距离；设为 0 显示真走时"
        )
        bar.addWidget(self.spin_vred)
        self.chk_by_obs = QCheckBox("按 OBS 着色")
        self.chk_by_obs.setChecked(False)
        self.chk_by_obs.setToolTip(
            "按台站/OBS 的模型距离着色并标出位置；关闭时：ttimes 按折射/反射，tx.in 按震相号"
        )
        bar.addWidget(self.chk_by_obs)
        self.chk_mark_obs = QCheckBox("标 OBS 位置")
        self.chk_mark_obs.setChecked(True)
        self.chk_mark_obs.setToolTip("在各 OBS/台站的模型距离处画竖线")
        bar.addWidget(self.chk_mark_obs)
        bar.addStretch(1)
        btn_reset = QPushButton("复位")
        btn_reset.setToolTip("复位到数据范围（等同双击空白）")
        btn_reset.clicked.connect(self._reset_view)
        btn_save = QPushButton("保存图像…")
        btn_save.setToolTip("保存为 PNG / JPEG / TIFF / PDF / PS / EPS / SVG")
        btn_save.clicked.connect(self._save_png)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        bar.addWidget(btn_reset)
        bar.addWidget(btn_save)
        bar.addWidget(btn_close)
        root.addLayout(bar)

        split = QSplitter(Qt.Orientation.Horizontal)
        side = QWidget()
        side_lay = QVBoxLayout(side)
        side_lay.setContentsMargins(0, 0, 0, 0)
        side_lay.setSpacing(8)
        self.obs_list = ObsCheckList(
            state=state,
            persist_key="tx.obs_ids" if persist_obs else None,
            heading=(
                "显示 / 转换 OBS（与转换页同一列表）"
                if persist_obs
                else "显示 OBS（station.lis 号，可多选）"
            ),
        )
        self.obs_list.setToolTip(
            "第一列为 station.lis 的 OBS 号（与炮点 x 按 0.001 km 匹配）；"
            "对不上则显示 —。"
            + ("勾选同时决定转换哪些台站。" if persist_obs else "")
        )
        self.obs_list.set_records(self._obs_catalog)
        self.obs_list.selection_changed.connect(self._on_param_changed)

        phase_head = (
            "显示震相（0 折射 / 1 反射 / 2 水波 / 3 多次 / 4 折射台侧 / 5 反射台侧 / 6 PSP / 7 PPS / 8 PSS）"
            if kind == "ttimes"
            else "显示震相（tx.in phase）"
        )
        self.phase_list = SimpleCheckList(heading=phase_head)
        self.phase_list.setToolTip(
            "只绘制勾选的震相。ttimes：0=折射、1=反射、2=直达水波、3=水柱多次、4=折射台侧多次、5=反射台侧多次；tx.in：文件中的 phase 列。"
        )
        self.phase_list.set_records(build_phase_catalog(self._arr, kind=kind))
        self.phase_list.selection_changed.connect(self._on_param_changed)

        side_split = QSplitter(Qt.Orientation.Vertical)
        side_split.setChildrenCollapsible(False)
        side_split.addWidget(self.obs_list)
        side_split.addWidget(self.phase_list)
        side_split.setStretchFactor(0, 3)
        side_split.setStretchFactor(1, 2)
        side_split.setSizes([420, 280])
        side_lay.addWidget(side_split, stretch=1)
        split.addWidget(side)

        self.plot = pg.PlotWidget(background="w")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("bottom", "Model distance", units="km")
        self.plot.setLabel("left", "Traveltime", units="s")
        from ..plots.inv_monitor_pg import style_plot_ink

        style_plot_ink(self.plot)
        self.plot.invertY(True)
        try:
            self.plot.setMenuEnabled(False)
            vb = self.plot.getViewBox()
            vb.setMenuEnabled(False)
            vb.disableAutoRange()
        except Exception:
            pass
        split.addWidget(self.plot)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([260, 740])
        root.addWidget(split, stretch=1)

        tip = QLabel(
            f"拾取 {len(self._picks)} 点 · 左侧勾选 OBS / 震相"
            + ("（OBS 同时用于转换）" if persist_obs else "")
            + " · X=模型距离 · 滚轮缩放 · 左拖平移 · 右拖缩放 · 双击复位"
        )
        tip.setStyleSheet("color:#666; font-size:11px;")
        root.addWidget(tip)

        self.spin_vred.valueChanged.connect(self._on_param_changed)
        self.chk_by_obs.toggled.connect(self._on_param_changed)
        self.chk_mark_obs.toggled.connect(self._on_param_changed)
        self._redraw(fit=True)

    def _selected_obs_x(self) -> set[float] | None:
        return self.obs_list.selected_xobs()

    def _selected_phases(self) -> set[int] | None:
        return self.phase_list.selected_ids()

    def _on_param_changed(self, *_a) -> None:
        self._redraw(fit=False)

    def _reset_view(self) -> None:
        self._redraw(fit=True)

    def _view_range(self) -> tuple[list[float], list[float]] | None:
        try:
            xr, yr = self.plot.getViewBox().viewRange()
            return [float(xr[0]), float(xr[1])], [float(yr[0]), float(yr[1])]
        except Exception:
            return None

    def _apply_view(self, *, fit: bool, rng: tuple[list[float], list[float]] | None) -> None:
        vb = self.plot.getViewBox()
        try:
            self.plot.invertY(True)
        except Exception:
            pass
        if fit or rng is None:
            vb.enableAutoRange(enable=True)
            self.plot.autoRange()
        else:
            vb.setRange(xRange=rng[0], yRange=rng[1], padding=0)
        vb.disableAutoRange()
        self._keep_view = True

    def _add_scatter(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        color: QColor,
        name: str | None,
        size: float = 5.0,
    ) -> None:
        if x.size == 0:
            return
        kw = dict(
            x=np.asarray(x, dtype=float),
            y=np.asarray(y, dtype=float),
            size=size,
            brush=pg.mkBrush(color),
            pen=pg.mkPen(color.darker(120), width=0.4),
            pxMode=True,
        )
        if name:
            kw["name"] = name
        self.plot.addItem(pg.ScatterPlotItem(**kw))

    def _redraw(self, *, fit: bool = False) -> None:
        keep = (not fit) and self._keep_view
        rng = self._view_range() if keep else None
        self.plot.clear()
        try:
            self.plot.invertY(True)
            self.plot.getViewBox().disableAutoRange()
        except Exception:
            pass
        arr = self._arr
        if arr["t"].size == 0:
            self.plot.setTitle("无拾取点")
            return

        vred = float(self.spin_vred.value())
        sel = self._selected_obs_x()
        xkey = np.round(arr["shot_x"], 3)
        if sel is None:
            vis = np.ones(arr["t"].shape, dtype=bool)
        elif not sel:
            vis = np.zeros(arr["t"].shape, dtype=bool)
        else:
            vis = np.isin(xkey, list(sel))
        ph_sel = self._selected_phases()
        if ph_sel is not None:
            if not ph_sel:
                vis = np.zeros(arr["t"].shape, dtype=bool)
            else:
                vis = vis & np.isin(arr["code"], list(ph_sel))
        if not np.any(vis):
            if sel is not None and not sel:
                self.plot.setTitle("未选择 OBS")
            elif ph_sel is not None and not ph_sel:
                self.plot.setTitle("未选择震相")
            else:
                self.plot.setTitle("当前勾选无拾取点")
            self.plot.setLabel("left", "Traveltime", units="s")
            self._apply_view(fit=fit, rng=rng)
            return

        xm = arr["rcv_x"]
        tplot = np.asarray(reduce_traveltime(arr["t"], arr["offset"], vred), dtype=np.float64)
        color_rank = {float(r["xobs"]): i for i, r in enumerate(self._obs_catalog)}
        by_obs = self.chk_by_obs.isChecked()
        if by_obs:
            groups_sorted = [
                r
                for r in self._obs_catalog
                if np.any(vis & (xkey == float(r["xobs"])))
            ]
            n_obs = len(groups_sorted)
            use_legend = n_obs <= 24
            if use_legend:
                self.plot.addLegend(offset=(8, 8))
            for rec in groups_sorted:
                xobs = float(rec["xobs"])
                m = vis & (xkey == xobs)
                name = (
                    f"OBS {int(rec['obs_id'])}"
                    if rec.get("obs_id") is not None
                    else f"OBS {xobs:.2f} km"
                ) if use_legend else None
                self._add_scatter(
                    xm[m], tplot[m], color=_color(color_rank.get(xobs, 0)), name=name
                )
        elif self._kind == "ttimes":
            self.plot.addLegend(offset=(8, 8))
            m0 = vis & (arr["code"] == 0)
            m1 = vis & (arr["code"] == 1)
            other = vis & ~(arr["code"] == 0) & ~(arr["code"] == 1)
            self._add_scatter(xm[m0], tplot[m0], color=QColor("#1f77b4"), name="refr (code=0)")
            self._add_scatter(xm[m1], tplot[m1], color=QColor("#d62728"), name="refl (code=1)")
            self._add_scatter(xm[other], tplot[other], color=QColor("#7f7f7f"), name="other")
        else:
            phases = np.unique(arr["code"][vis])
            if len(phases) <= 24:
                self.plot.addLegend(offset=(8, 8))
            for i, ph in enumerate(phases):
                m = vis & (arr["code"] == ph)
                name = f"phase {int(ph)}" if len(phases) <= 24 else None
                self._add_scatter(xm[m], tplot[m], color=_color(i), name=name)

        if self.chk_mark_obs.isChecked() and arr["shot_x"].size:
            for xobs in np.unique(np.round(arr["shot_x"][vis], 3)):
                line = pg.InfiniteLine(
                    pos=float(xobs),
                    angle=90,
                    pen=pg.mkPen("#94a3b8", width=1, style=Qt.PenStyle.DashLine),
                    movable=False,
                )
                line.setZValue(-10)
                self.plot.addItem(line)

        ylab = "t" if vred <= 0 else f"t − |x−xobs|/vred  (vred={vred:g})"
        self.plot.setLabel("left", ylab, units="s")
        self.plot.setLabel("bottom", "Model distance", units="km")
        self.plot.setTitle(self.windowTitle())
        self._apply_view(fit=fit, rng=rng)

    def _save_png(self) -> None:
        from ..plots.export_figure import save_graphics_widget

        save_graphics_widget(
            self, self.plot, start_dir=self._save_dir or "", default_name="ttimes"
        )


def _stations_from_state(state: FormState, work: Path) -> list[tuple[int, float, float]]:
    try:
        sp = resolve_station_lis_path(state, work)
        if sp is None:
            return []
        return parse_station_lis(sp)
    except (OSError, ValueError):
        return []


def _show_picks_window(
    picks: list[TtPick],
    *,
    title: str,
    save_dir: str | None,
    kind: str,
    stations: list[tuple[int, float, float]] | None = None,
    state: FormState | None = None,
    persist_obs: bool = False,
) -> TravelTimePlotWindow:
    win = TravelTimePlotWindow(
        picks,
        title=title,
        save_dir=save_dir,
        kind=kind,
        stations=stations,
        state=state,
        persist_obs=persist_obs,
    )
    show_modeless_dialog(win, activate=True)
    return win


def open_ttimes_preview(parent: QWidget | None, state: FormState) -> None:
    """预览表单中的 ttimes.dat（tx.data_out）。"""
    try:
        work = resolve_work_dir(state.get_str("work_dir"))
        _, _, data_out, _ = resolve_tx_convert_paths(state, work)
    except Exception as e:
        show_modeless_message("预览 ttimes", str(e), icon=QMessageBox.Icon.Warning)
        return
    if not data_out.is_file():
        show_modeless_message(
            "预览 ttimes",
            f"文件不存在：\n{data_out}\n\n请先「运行 tx 转换」或确认输出路径。",
            icon=QMessageBox.Icon.Warning,
        )
        return
    try:
        picks = load_ttimes_picks(data_out)
        if not picks:
            raise ValueError("文件中无接收点")
        _show_picks_window(
            picks,
            title=f"ttimes — {data_out.name}",
            save_dir=str(data_out.parent),
            kind="ttimes",
            stations=_stations_from_state(state, work),
            state=state,
            persist_obs=False,
        )
    except Exception as e:
        show_modeless_message("预览 ttimes 失败", str(e), icon=QMessageBox.Icon.Critical)


def open_tx_in_preview(parent: QWidget | None, state: FormState) -> None:
    """预览表单中的一个或多个 tx.in。"""
    try:
        work = resolve_work_dir(state.get_str("work_dir"))
        _, tx_paths, _, _ = resolve_tx_convert_paths(state, work)
    except Exception as e:
        show_modeless_message("预览 tx.in", str(e), icon=QMessageBox.Icon.Warning)
        return
    missing = [p for p in tx_paths if not p.is_file()]
    if missing:
        show_modeless_message(
            "预览 tx.in",
            "以下文件不存在：\n" + "\n".join(str(p) for p in missing),
            icon=QMessageBox.Icon.Warning,
        )
        return
    try:
        picks = load_tx_in_picks(tx_paths)
        names = ", ".join(p.name for p in tx_paths[:3])
        if len(tx_paths) > 3:
            names += f" …(+{len(tx_paths) - 3})"
        _show_picks_window(
            picks,
            title=f"tx.in — {names}",
            save_dir=str(tx_paths[0].parent),
            kind="tx",
            stations=_stations_from_state(state, work),
            state=state,
            persist_obs=True,
        )
    except Exception as e:
        show_modeless_message("预览 tx.in 失败", str(e), icon=QMessageBox.Icon.Critical)
