"""蒙特卡洛：v.in 选界面 / 选层独立窗（速度底图 + 标注 + 蒙版）。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..dialog_utils import show_modeless_dialog, show_modeless_message
from ..plots.inv_monitor_model import (
    _overlay_mesh_geometry,
    _prepare_velocity_arrays,
    finish_figure_layout,
    imshow_velocity_field,
)
from ..plots.mpl_figure_window import MplNavCanvas
from ..state.form_state import FormState
from pyAOBS.modeling.tomo2d.tt_inverse_log_analysis import ensure_matplotlib_cjk_font


_ROLE_ROWS = (
    ("seafloor", "mc.vin_seafloor", "海底", False),
    ("basement", "mc.vin_basement", "基底", True),
    ("conrad", "mc.vin_conrad", "Conrad", True),
    ("moho", "mc.vin_moho", "莫霍", False),
)


class _VinPickPlot(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        ensure_matplotlib_cjk_font()
        fig = Figure(figsize=(7.2, 6.4), facecolor="w", layout="constrained")
        gs = GridSpec(1, 2, figure=fig, width_ratios=(24, 1), wspace=0.04)
        self.ax = fig.add_subplot(gs[0, 0])
        self.cax = fig.add_subplot(gs[0, 1])
        self._host = MplNavCanvas(fig, self)
        self._host.hint.setText("勾选界面画线 · 勾选层位蒙版")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._host)
        self.fig = fig

    def show_empty(self, message: str) -> None:
        self.ax.clear()
        self.cax.clear()
        self.cax.set_visible(False)
        self.ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=self.ax.transAxes,
            color="#64748b",
            wrap=True,
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self._host.canvas.draw_idle()

    def set_field(
        self,
        ds,
        mesh,
        extra,
        cmap_spec: str,
        title: str,
        *,
        reset_home: bool = True,
        cb_label: str = "km/s",
    ) -> None:
        xlim = self.ax.get_xlim() if not reset_home else None
        ylim = self.ax.get_ylim() if not reset_home else None
        self.ax.clear()
        self.cax.clear()
        self.cax.set_visible(True)
        prep = _prepare_velocity_arrays(ds, cmap_spec, mesh=mesh)
        xmin, xmax, zmin, zmax, _cb = imshow_velocity_field(
            self.ax,
            self.cax,
            self.fig,
            prep["data"],
            prep["x"],
            prep["z"],
            prep["cmap"],
            prep["lo"],
            prep["hi"],
            cb_label=cb_label,
        )
        _overlay_mesh_geometry(self.ax, mesh, extra)
        self.ax.set_title(title, color="black")
        self.ax.set_xlabel("模型距离 (km)", color="black")
        self.ax.set_ylabel("深度 (km)", color="black")
        self.ax.tick_params(colors="black")
        self.ax.grid(True, alpha=0.3)
        if xlim is not None:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        else:
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(zmax, zmin)
            self._host.remember_home_views()
        finish_figure_layout(self.fig)
        self._host.canvas.draw_idle()


class MonteCarloVinPickWindow(QWidget):
    """勾选海底/基底/Conrad/莫霍与扰动层，底图实时标注。"""

    def __init__(
        self,
        state: FormState,
        *,
        on_changed: Callable[[], None] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self._on_changed = on_changed
        self._zelt = None
        self._ds = None
        self._mesh = None
        self._updating = False
        self.setWindowTitle("v.in 选层")
        self.resize(1080, 720)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        bar = QHBoxLayout()
        from ..widgets.smesh_cmap_combo import SmeshCmapCombo

        self.cmap_combo = SmeshCmapCombo(
            self.state, on_changed=lambda *_a: self._paint(reset_home=False)
        )
        bar.addWidget(self.cmap_combo)
        bar.addStretch(1)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        bar.addWidget(btn_close)
        root.addLayout(bar)
        split = QSplitter(Qt.Orientation.Horizontal)

        self.plot = _VinPickPlot(self)
        split.addWidget(self.plot)

        side = QWidget()
        side_lay = QVBoxLayout(side)
        side_lay.setContentsMargins(6, 0, 0, 0)
        g_iface = QGroupBox("界面（勾选后在底图标注）")
        iface_lay = QVBoxLayout(g_iface)
        self._role_ck: dict[str, QCheckBox] = {}
        self._role_cmb: dict[str, QComboBox] = {}
        for role, _key, lab, optional in _ROLE_ROWS:
            row = QHBoxLayout()
            ck = QCheckBox(lab)
            ck.setChecked(not optional)
            if not optional:
                ck.setEnabled(False)
                ck.setToolTip("必选")
            cmb = QComboBox()
            cmb.setMinimumWidth(88)
            row.addWidget(ck)
            row.addWidget(cmb, stretch=1)
            iface_lay.addLayout(row)
            self._role_ck[role] = ck
            self._role_cmb[role] = cmb
            ck.toggled.connect(lambda *_a: self._on_ui_changed())
            cmb.currentTextChanged.connect(lambda *_a: self._on_ui_changed())
        side_lay.addWidget(g_iface)

        g_unit = QGroupBox("扰动层（勾选后底图蒙版）")
        unit_lay = QVBoxLayout(g_unit)
        self._unit_ck: dict[str, QCheckBox] = {}
        from ..services.mc_vin_layers import UNIT_LABEL, UNIT_ORDER

        for u in UNIT_ORDER:
            ck = QCheckBox(UNIT_LABEL[u])
            ck.setChecked(True)
            ck.toggled.connect(lambda *_a: self._on_ui_changed())
            unit_lay.addWidget(ck)
            self._unit_ck[u] = ck
        side_lay.addWidget(g_unit)
        side_lay.addStretch(1)
        hint = QLabel("浅虚线=未命名界面。基底/Conrad 不勾则没有沉积/下地壳。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#64748b;")
        side_lay.addWidget(hint)
        split.addWidget(side)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)
        root.addWidget(split, stretch=1)
        self.refresh(reset_home=True)

    def _marks_from_ui(self, n_ifaces: int):
        from pyAOBS.modeling.vedit.core.geo_ifaces import GeoIfaceMarks, parse_iface_label

        def _idx(role: str):
            if not self._role_ck[role].isChecked():
                return None
            return parse_iface_label(self._role_cmb[role].currentText())

        return GeoIfaceMarks(
            seafloor=_idx("seafloor"),
            basement=_idx("basement"),
            conrad=_idx("conrad"),
            moho=_idx("moho"),
        ).clamped(n_ifaces)

    def _units_from_ui(self, allowed: list[str]) -> list[str]:
        return [u for u, ck in self._unit_ck.items() if ck.isChecked() and u in allowed]

    def _on_ui_changed(self) -> None:
        if self._updating:
            return
        self._store_to_state()
        self._paint(reset_home=False)
        if callable(self._on_changed):
            self._on_changed()

    def _store_to_state(self) -> None:
        from ..services.mc_vin_layers import (
            available_units,
            format_stored_iface,
            format_vin_units,
        )

        if self._zelt is None:
            return
        n = len(getattr(self._zelt, "depth_nodes", []) or [])
        marks = self._marks_from_ui(n)
        self.state.set("mc.vin_seafloor", format_stored_iface(marks.seafloor))
        self.state.set("mc.vin_basement", format_stored_iface(marks.basement))
        self.state.set("mc.vin_conrad", format_stored_iface(marks.conrad))
        self.state.set("mc.vin_moho", format_stored_iface(marks.moho))
        allowed = available_units(marks)
        self.state.set("mc.vin_units", format_vin_units(self._units_from_ui(allowed)))

    def refresh(self, *, reset_home: bool = True) -> None:
        from pyAOBS.modeling.vedit.core.geo_ifaces import (
            format_iface_label,
            iface_option_labels,
        )

        from ..services.mc_vin_layers import (
            available_units,
            default_vin_marks,
            load_vin_pick_background,
            marks_from_state,
            parse_vin_units,
            resolve_mc_vin_path,
        )
        from ..services.paths import resolve_existing_file, resolve_work_dir

        work = resolve_work_dir(self.state.get_str("work_dir"))
        vin = resolve_mc_vin_path(self.state, work)
        if vin is None or not Path(vin).is_file():
            self._zelt = None
            self.plot.show_empty("未找到 v.in（请填本页或 gen_smesh 的 v.in）")
            return
        mesh_s = (
            (self.state.get_str("mc.base_mesh") or "").strip()
            or (self.state.get_str("inv.mesh") or "").strip()
            or (self.state.get_str("gen.smesh_out") or "").strip()
            or (self.state.get_str("fwd.smesh") or "").strip()
        )
        mesh_path = None
        if mesh_s:
            try:
                mesh_path = resolve_existing_file(mesh_s, work)
            except FileNotFoundError:
                cand = Path(mesh_s).expanduser()
                if not cand.is_absolute():
                    cand = work / mesh_s
                mesh_path = cand if cand.is_file() else None
        try:
            self._mesh, self._ds, self._zelt = load_vin_pick_background(vin, mesh_path)
        except Exception as exc:
            self._zelt = None
            self.plot.show_empty(f"无法绘制：{exc}")
            return
        n = len(getattr(self._zelt, "depth_nodes", []) or [])
        labels = iface_option_labels(n)
        marks = marks_from_state(self.state, n)
        defaults = default_vin_marks(n)
        cur = {
            "seafloor": marks.seafloor if marks.seafloor is not None else defaults.seafloor,
            "basement": marks.basement,
            "conrad": marks.conrad,
            "moho": marks.moho if marks.moho is not None else defaults.moho,
        }
        self._updating = True
        try:
            for role, _key, _lab, optional in _ROLE_ROWS:
                cmb = self._role_cmb[role]
                ck = self._role_ck[role]
                cmb.blockSignals(True)
                cmb.clear()
                cmb.addItems(labels or ["—"])
                text = format_iface_label(cur[role]) if cur[role] is not None else (labels[0] if labels else "—")
                ix = cmb.findText(text)
                cmb.setCurrentIndex(max(0, ix))
                cmb.blockSignals(False)
                if optional:
                    ck.blockSignals(True)
                    ck.setChecked(cur[role] is not None)
                    ck.blockSignals(False)
                    cmb.setEnabled(ck.isChecked())
            allowed = available_units(self._marks_from_ui(n))
            raw = (self.state.get_str("mc.vin_units") or "").strip()
            chosen = parse_vin_units(raw, allowed) if raw else list(allowed)
            for u, ck in self._unit_ck.items():
                on = u in allowed
                ck.setEnabled(on)
                ck.blockSignals(True)
                ck.setChecked(on and u in chosen)
                ck.blockSignals(False)
                if on:
                    from ..services.mc_vin_layers import unit_span_label

                    ck.setText(unit_span_label(self._marks_from_ui(n), u))
        finally:
            self._updating = False
        self._store_to_state()
        self._paint(reset_home=reset_home)

    def _paint(self, *, reset_home: bool) -> None:
        from ..services.mc_vin_layers import UNIT_LABEL, available_units, unit_span_label, vin_pick_overlays
        from ..services.paths import resolve_work_dir
        from ..services.smesh_plot_core import colorbar_label_for_cmap, resolve_plot_smesh_cmap

        if self._zelt is None or self._ds is None:
            return
        n = len(getattr(self._zelt, "depth_nodes", []) or [])
        marks = self._marks_from_ui(n)
        allowed = available_units(marks)
        units = self._units_from_ui(allowed)
        extra = vin_pick_overlays(self._zelt, marks, units)
        for role, cmb in self._role_cmb.items():
            cmb.setEnabled(self._role_ck[role].isChecked())
        work = resolve_work_dir(self.state.get_str("work_dir"))
        cmap = resolve_plot_smesh_cmap(self.state, work)
        self.plot.set_field(
            self._ds,
            self._mesh,
            extra,
            cmap,
            "v.in 速度  ·  界面标注 / 层位蒙版",
            reset_home=reset_home,
            cb_label=colorbar_label_for_cmap(cmap),
        )
        for u, ck in self._unit_ck.items():
            on = u in allowed
            ck.setEnabled(on)
            if on:
                ck.setText(unit_span_label(marks, u))
            else:
                ck.setText(UNIT_LABEL[u])
                if ck.isChecked():
                    self._updating = True
                    ck.blockSignals(True)
                    ck.setChecked(False)
                    ck.blockSignals(False)
                    self._updating = False


def open_monte_carlo_vin_pick(
    state: FormState,
    *,
    pull: Callable[[], None] | None = None,
    on_changed: Callable[[], None] | None = None,
    existing: MonteCarloVinPickWindow | None = None,
) -> MonteCarloVinPickWindow | None:
    if callable(pull):
        pull()
    from ..services.mc_vin_layers import resolve_mc_vin_path
    from ..services.paths import resolve_work_dir

    try:
        work = resolve_work_dir(state.get_str("work_dir"))
    except Exception as e:
        show_modeless_message("v.in 选层", str(e))
        return existing
    vin = resolve_mc_vin_path(state, work)
    if vin is None or not Path(vin).is_file():
        show_modeless_message("v.in 选层", "请先指定存在的 v.in（本页或 gen_smesh）。")
        return existing

    win = existing
    try:
        if win is not None and win.isVisible():
            win.refresh(reset_home=True)
            win.raise_()
            win.activateWindow()
            return win
    except RuntimeError:
        win = None

    win = MonteCarloVinPickWindow(state, on_changed=on_changed)
    try:
        win.refresh(reset_home=True)
    except Exception as e:
        show_modeless_message("v.in 选层", str(e))
        win.deleteLater()
        return None
    show_modeless_dialog(win, activate=True)
    return win
