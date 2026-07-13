"""
LIP Petrology Qt 主窗口 — REEBOX 熔融、岩性预设、H + V_LC 扫描。

布局：顶部 QToolBar + 紧凑参数条（三列横排）+ 全宽 matplotlib + 日志。
"""

from __future__ import annotations

import matplotlib
from pathlib import Path

for _backend in ("qtagg", "QtAgg", "Qt5Agg"):
    try:
        matplotlib.use(_backend)
        break
    except ValueError:
        continue

from .mpl_cjk import configure_matplotlib_cjk

configure_matplotlib_cjk()

import numpy as np
from matplotlib.figure import Figure

from .mpl_qt import FigureCanvasQTAgg, NavigationToolbar2QT

from PySide6.QtCore import Qt, QThread, Slot
from PySide6.QtGui import QAction, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from petrology.gui.preset_compare import (
    compare_lithology_presets,
    format_preset_compare_table,
    preset_compare_hints,
)
from petrology.melting.column import forward_melting_column
from petrology.melting.hvp_scan import scan_hvp_lip
from petrology.melting.lip_scan import scan_hvp_lip_phi
from petrology.fc.delta_vp import R3_DELTA_VP_WL_KW
from petrology.melting.pymelt_lithology_adapter import (
    LITHOLOGY_PRESETS,
    enriched_lithology_keys,
    list_lithology_presets,
    peridotite_lithology_keys,
    resolve_lithology_col_kwargs,
)

from .classic_curves_dialog import ClassicCurvesDialog
from .dialog_utils import show_modeless_dialog
from .formula_reference_dialog import show_formula_reference
from .katz2003_dialog import Katz2003Dialog
from .melting_schematic_dialog import show_melting_schematic_dialog
from .track_compare_dialog import TrackCompareDialog
from petrology.hvp.track_registry import (
    HVP_DISPLAY_BOTH,
    HVP_DISPLAY_CHOICES,
    HVP_DISPLAY_DEFAULT,
    HVP_DISPLAY_FIG12,
    HVP_DISPLAY_HVP,
    HVP_DISPLAY_MODERN,
)
from .hvp_plot import (
    ObservationPoint,
    build_fig12_linear_curves,
    build_modern_hvp_curves,
    linear_hvp_from_scan_grid,
    modern_hvp_from_scan_grid,
    plot_hvp_scan_panel,
)
from petrology.reference.galapagos_sallares_observations import FIG10_PANEL_LABELS
from petrology.hvp.observation_overlay import bulk_read_band, is_thick_crust
from petrology.imodel_bridge import CrustObservation, load_observation_json
from petrology.seismic.transect import TransectWindow, load_transect_windows_csv
from .param_tooltips import (
    PARAM_LABEL_TIPS,
    apply_action_tip,
    apply_group_tooltip,
    apply_labelled_param_tip,
    apply_tab_tooltips,
    apply_tip,
    apply_widget_tip,
)
from .reader_guide import show_reader_guide
from .forward_plot import plot_forward_result
from .app import LOG_FONT_PT, UI_FONT_PT
from .workers import BackgroundWorker, ProgressReport, run_in_thread

PARAMS_FONT_PT = 10
_PARAMS_DEFAULT_H = 158
_PLOT_DEFAULT_H = 480
_LOG_DEFAULT_H = 160


def _align_param_grid(grid: QGridLayout) -> None:
    grid.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    grid.setContentsMargins(6, 6, 6, 4)
    grid.setHorizontalSpacing(6)
    grid.setVerticalSpacing(4)


def _param_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return lbl


def _add_param_cells(
    grid: QGridLayout,
    row: int,
    start_col: int,
    items: list[tuple[str | None, QWidget]],
) -> int:
    col = start_col
    for label, widget in items:
        if label:
            lbl = _param_label(label)
            apply_labelled_param_tip(label, lbl, widget)
            grid.addWidget(
                lbl, row, col, alignment=Qt.AlignmentFlag.AlignLeft
            )
            col += 1
        grid.addWidget(widget, row, col, alignment=Qt.AlignmentFlag.AlignLeft)
        col += 1
    return col


def _stretch_param_grid(grid: QGridLayout, last_col: int) -> None:
    grid.setColumnStretch(last_col, 1)
    grid.setRowStretch(3, 1)

_MAIN_QSS = f"""
QMainWindow {{
    background-color: #f0f0f0;
    font-size: {UI_FONT_PT}pt;
}}
QWidget {{
    font-size: {UI_FONT_PT}pt;
}}
QGroupBox {{
    font-size: {UI_FONT_PT}pt;
    font-weight: bold;
    border: 1px solid #a0a0a0;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
}}
QToolBar {{
    font-size: {UI_FONT_PT}pt;
    background: #e4e4e4;
    border-bottom: 1px solid #b0b0b0;
    spacing: 8px;
    padding: 6px;
}}
QToolBar QToolButton {{
    font-size: {UI_FONT_PT}pt;
    padding: 4px 10px;
}}
QLineEdit, QComboBox {{
    font-size: {UI_FONT_PT}pt;
    min-height: 30px;
    padding: 2px 6px;
}}
QTextEdit {{
    font-size: {LOG_FONT_PT}pt;
}}
QStatusBar, QStatusBar QLabel {{
    font-size: {UI_FONT_PT - 1}pt;
}}
QWidget#LipParamsPanel {{
    font-size: {PARAMS_FONT_PT}pt;
}}
QWidget#LipParamsPanel QGroupBox {{
    font-size: {PARAMS_FONT_PT}pt;
    font-weight: bold;
    margin-top: 4px;
    padding-top: 2px;
}}
QWidget#LipParamsPanel QGroupBox::title {{
    padding: 0 4px;
}}
QWidget#LipParamsPanel QLineEdit,
QWidget#LipParamsPanel QComboBox {{
    font-size: {PARAMS_FONT_PT}pt;
    min-height: 22px;
    max-height: 24px;
    padding: 0 4px;
}}
QWidget#LipParamsPanel QCheckBox {{
    font-size: {PARAMS_FONT_PT - 1}pt;
    font-weight: normal;
    spacing: 4px;
}}
QWidget#LipParamsPanel QLabel {{
    font-size: {PARAMS_FONT_PT}pt;
}}
QSplitter::handle {{
    background-color: #c8c8c8;
}}
QSplitter::handle:vertical {{
    height: 6px;
    margin: 0 4px;
}}
QSplitter::handle:horizontal {{
    width: 6px;
    margin: 4px 0;
}}
QSplitter::handle:hover {{
    background-color: #7a9fd4;
}}
"""


class LipMainWindow(QMainWindow):
    """LIP 地幔熔融解释界面（PySide6）。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LIP Petrology — REEBOX melting")
        self.resize(1180, 780)
        self.setMinimumSize(960, 640)
        self.setStyleSheet(_MAIN_QSS)

        self._busy = False
        self._params_visible = True
        self._params_collapsed_h = _PARAMS_DEFAULT_H
        self._active_thread: QThread | None = None
        self._active_worker: BackgroundWorker | None = None
        self._last_scan_result = None
        self._last_scan_h_obs: float | None = None
        self._last_scan_v_lc: float | None = None
        self._last_scan_h_tol: float | None = None
        self._last_scan_b_km: float = 0.0
        # Cached H–Vp (eq.1) curves for instant display-mode switching
        self._hvp_linear_cache = None
        self._hvp_linear_cache_key: tuple | None = None
        # Cached Modern Fig.12-style curves (χ solids + χ=1,b family)
        self._hvp_modern_cache = None
        self._hvp_modern_cache_key: tuple | None = None
        self._imported_obs: CrustObservation | None = None
        self._imported_transect_windows: list[TransectWindow] | None = None

        self._build_menu()
        self._build_toolbar()
        self._build_central()
        self._build_status()
        self._on_backend_changed()
        self._set_busy(False, "就绪")

    # ------------------------------------------------------------------ UI
    def _build_menu(self) -> None:
        mb = self.menuBar()
        verify = mb.addMenu("验证")
        apply_tip(verify, "熔融参数化验证、经典曲线与公式查阅。")

        act_katz = QAction("Katz (2003) Fig.1–6…", self)
        act_katz.setShortcut("Ctrl+Shift+K")
        act_katz.triggered.connect(self._show_katz2003_figures)
        apply_action_tip(act_katz, "Katz (2003) Fig.1–6…")
        verify.addAction(act_katz)

        act_export_katz = QAction("导出 Katz Fig.1–6 PNG…", self)
        act_export_katz.triggered.connect(self._export_katz2003_figures)
        apply_action_tip(act_export_katz, "导出 Katz Fig.1–6 PNG…")
        verify.addAction(act_export_katz)

        verify.addSeparator()

        act_schematic = QAction("主动/被动熔融模式图…", self)
        act_schematic.setShortcut("Ctrl+Shift+M")
        act_schematic.triggered.connect(self._show_melting_schematic)
        apply_action_tip(act_schematic, "主动/被动熔融模式图…")
        verify.addAction(act_schematic)

        act_fc = QAction("分离结晶 Fig.2 / Fig.5…", self)
        act_fc.setShortcut("Ctrl+Shift+X")
        act_fc.triggered.connect(self._show_crystallization_figures)
        apply_action_tip(act_fc, "分离结晶 Fig.2 / Fig.5…")
        verify.addAction(act_fc)

        act_classic = QAction("经典熔融理论曲线…", self)
        act_classic.setShortcut("Ctrl+Shift+C")
        act_classic.triggered.connect(self._show_classic_curves)
        apply_action_tip(act_classic, "经典熔融理论曲线…")
        verify.addAction(act_classic)

        verify.addSeparator()
        act_formulas = QAction("公式与参数查阅…", self)
        act_formulas.setShortcut("Ctrl+Shift+F")
        act_formulas.triggered.connect(self._show_formula_reference)
        apply_action_tip(act_formulas, "公式与参数查阅…")
        verify.addAction(act_formulas)

        sallares_menu = verify.addMenu("Sallarès Galápagos")
        apply_tip(sallares_menu, "2005 GJI Fig.10：2D 湿熔融 + eq.(1) + GVP 观测。")
        for panel_key, panel_label in FIG10_PANEL_LABELS.items():
            act_panel = QAction(panel_label, self)
            act_panel.triggered.connect(
                lambda _checked=False, pk=panel_key: self._preview_sallares_galapagos(pk)
            )
            sallares_menu.addAction(act_panel)
        if sallares_menu.actions():
            sallares_menu.actions()[0].setShortcut("Ctrl+Shift+G")

        help_menu = mb.addMenu("帮助")
        apply_tip(help_menu, "读者指南与工作流说明。")
        act_guide = QAction("读者指南…\tF1", self)
        act_guide.triggered.connect(self._show_reader_guide)
        apply_action_tip(act_guide, "读者指南…")
        help_menu.addAction(act_guide)

        self._guide_shortcut = QShortcut(QKeySequence(Qt.Key.Key_F1), self)
        self._guide_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self._guide_shortcut.activated.connect(self._show_reader_guide)

    def _lip_context_for_katz(self) -> tuple[float, float]:
        try:
            tp_c = float(self._tp_edit.text())
        except ValueError:
            tp_c = 1350.0
        try:
            per_h2o = float(self._per_h2o_edit.text())
        except ValueError:
            per_h2o = 0.0
        return tp_c, per_h2o

    def _show_lithology_reference(self) -> None:
        from .lithology_reference_dialog import show_lithology_reference

        try:
            per_h2o = float(self._per_h2o_edit.text() or 0)
        except ValueError:
            per_h2o = 0.0
        try:
            pyr_h2o = float(self._pyr_h2o_edit.text() or 0)
        except ValueError:
            pyr_h2o = 0.0
        show_lithology_reference(
            preset=self._preset_combo.currentText(),
            per=self._per_combo.currentText(),
            pyr=self._pyr_combo.currentText(),
            backend=self._backend_combo.currentText(),
            per_h2o=per_h2o,
            pyr_h2o=pyr_h2o,
        )

    def _show_katz2003_figures(self) -> None:
        if self._busy:
            QMessageBox.information(self, "Katz (2003)", "请等待当前任务完成后再打开预览。")
            return
        tp_c, per_h2o = self._lip_context_for_katz()
        show_modeless_dialog(Katz2003Dialog(tp_c=tp_c, per_h2o_wt=per_h2o), singleton=True)

    def _show_melting_schematic(self) -> None:
        if self._busy:
            QMessageBox.information(self, "模式图", "请等待当前任务完成后再打开。")
            return
        show_melting_schematic_dialog()

    def _show_crystallization_figures(self) -> None:
        if self._busy:
            QMessageBox.information(self, "分离结晶", "请等待当前任务完成后再打开预览。")
            return
        from .crystallization_dialog import show_crystallization_dialog

        show_crystallization_dialog()

    def _export_katz2003_figures(self) -> None:
        if self._busy:
            QMessageBox.information(self, "Katz (2003)", "请等待当前任务完成后再导出。")
            return
        from pathlib import Path

        from petrology.validation.katz2003_workflow import FIG_DIR, export_katz2003_figures

        out_dir = QFileDialog.getExistingDirectory(
            self,
            "导出 Katz Fig.1–6",
            str(FIG_DIR),
        )
        if not out_dir:
            return
        try:
            paths = export_katz2003_figures(Path(out_dir))
        except Exception as exc:
            QMessageBox.critical(self, "Katz (2003)", f"导出失败:\n{exc}")
            return
        self._append_log(f"Katz (2003): 已导出 {len(paths)} 张图 → {out_dir}")
        QMessageBox.information(self, "Katz (2003)", f"已导出 {len(paths)} 张图至:\n{out_dir}")

    def _show_classic_curves(self) -> None:
        if self._busy:
            QMessageBox.information(self, "经典曲线", "请等待当前任务完成后再打开预览。")
            return
        try:
            tp_c = float(self._tp_edit.text())
        except ValueError:
            tp_c = 1350.0
        show_modeless_dialog(ClassicCurvesDialog(tp_c=tp_c), singleton=True)

    def _show_reader_guide(self) -> None:
        show_reader_guide(self)

    def _build_toolbar(self) -> None:
        tb = QToolBar("操作")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(tb)

        act_fwd = QAction("正演", self)
        act_fwd.triggered.connect(self._run_forward)
        apply_action_tip(act_fwd, "正演")
        tb.addAction(act_fwd)

        act_scan = QAction("Tp–χ 扫描", self)
        act_scan.triggered.connect(self._run_tp_chi_scan)
        apply_action_tip(act_scan, "Tp–χ 扫描")
        tb.addAction(act_scan)

        act_phi = QAction("Φ 扫描", self)
        act_phi.triggered.connect(self._run_phi_scan)
        apply_action_tip(act_phi, "Φ 扫描")
        tb.addAction(act_phi)

        act_cmp = QAction("预设对比", self)
        act_cmp.triggered.connect(self._run_preset_compare)
        apply_action_tip(act_cmp, "预设对比")
        tb.addAction(act_cmp)

        act_track = QAction("轨对比", self)
        act_track.triggered.connect(self._show_track_compare)
        apply_action_tip(act_track, "轨对比")
        tb.addAction(act_track)

        act_import = QAction("导入观测", self)
        act_import.triggered.connect(self._import_observation)
        apply_action_tip(act_import, "导入观测")
        tb.addAction(act_import)

        act_formulas = QAction("公式", self)
        act_formulas.triggered.connect(self._show_formula_reference)
        apply_action_tip(act_formulas, "公式")
        tb.addAction(act_formulas)

        tb.addSeparator()

        act_save = QAction("保存图", self)
        act_save.triggered.connect(self._save_figure)
        apply_action_tip(act_save, "保存图")
        tb.addAction(act_save)

        tb.addSeparator()

        self._toggle_params_act = QAction("收起参数", self)
        self._toggle_params_act.triggered.connect(self._toggle_params)
        apply_action_tip(self._toggle_params_act, "toggle_params")
        tb.addAction(self._toggle_params_act)
        apply_tip(tb, "主工具栏：正演、扫描、导入观测与图件保存。")

    def _build_central(self) -> None:
        self._main_split = QSplitter(Qt.Orientation.Vertical)
        self._main_split.setChildrenCollapsible(False)
        self.setCentralWidget(self._main_split)

        self._params_panel = QWidget()
        self._params_panel.setObjectName("LipParamsPanel")
        self._params_panel.setMinimumHeight(100)
        self._build_params(self._params_panel)
        self._main_split.addWidget(self._params_panel)

        plot_wrap = QWidget()
        pl = QVBoxLayout(plot_wrap)
        pl.setContentsMargins(0, 0, 0, 0)
        self._fig = Figure(figsize=(9, 4.2), dpi=100)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_title("(Tp, χ) 扫描 — 尚未运行")
        self._ax.set_xlabel("Tp (°C)")
        self._ax.set_ylabel("χ")
        self._apply_plot_fonts()
        self._canvas = FigureCanvasQTAgg(self._fig)
        tb_row = QWidget()
        tb_lay = QHBoxLayout(tb_row)
        tb_lay.setContentsMargins(0, 0, 0, 0)
        tb_lay.addWidget(NavigationToolbar2QT(self._canvas, tb_row))
        pl.addWidget(tb_row)
        pl.addWidget(self._canvas, stretch=1)
        plot_wrap.setMinimumHeight(180)
        self._main_split.addWidget(plot_wrap)

        log_grp = QGroupBox("输出")
        log_lay = QVBoxLayout(log_grp)
        self._out_tabs = QTabWidget()
        self._scan_table = QTextEdit()
        self._scan_table.setReadOnly(True)
        self._scan_table.setFont(QFont("Consolas", LOG_FONT_PT))
        self._read_table = QTextEdit()
        self._read_table.setReadOnly(True)
        self._read_table.setFont(QFont("Consolas", LOG_FONT_PT))
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", LOG_FONT_PT))
        self._out_tabs.addTab(self._scan_table, "扫描结果")
        self._out_tabs.addTab(self._read_table, "观测读图")
        self._out_tabs.addTab(self._log, "日志")
        apply_tab_tooltips(self._out_tabs)
        apply_group_tooltip(log_grp)
        apply_tip(plot_wrap, "主绘图区：Tp–χ 扫描、正演柱状图或 H–Vp / 可行区热图。")
        log_lay.addWidget(self._out_tabs)
        log_grp.setMinimumHeight(72)
        self._main_split.addWidget(log_grp)

        self._main_split.setStretchFactor(0, 0)
        self._main_split.setStretchFactor(1, 3)
        self._main_split.setStretchFactor(2, 1)
        self._main_split.setSizes(
            [_PARAMS_DEFAULT_H, _PLOT_DEFAULT_H, _LOG_DEFAULT_H]
        )

    def _build_params(self, parent: QWidget) -> None:
        outer = QVBoxLayout(parent)
        outer.setContentsMargins(4, 2, 4, 2)
        outer.setSpacing(0)

        h_split = QSplitter(Qt.Orientation.Horizontal)
        h_split.setChildrenCollapsible(False)
        outer.addWidget(h_split)

        obs = QGroupBox("观测锚点")
        og = QGridLayout(obs)
        _align_param_grid(og)
        self._h_edit = QLineEdit("30")
        self._vlc_edit = QLineEdit("7.0")
        self._b_edit = QLineEdit("0")
        self._phi_edit = QLineEdit("0.10")
        self._h_tol_edit = QLineEdit("5")
        for w, width in (
            (self._h_edit, 44),
            (self._vlc_edit, 52),
            (self._b_edit, 40),
            (self._phi_edit, 44),
            (self._h_tol_edit, 40),
        ):
            w.setFixedWidth(width)
        c0 = _add_param_cells(
            og,
            0,
            0,
            [("H", self._h_edit), ("V_LC", self._vlc_edit)],
        )
        c1 = _add_param_cells(
            og,
            1,
            0,
            [("b", self._b_edit), ("Φ", self._phi_edit)],
        )
        c2 = _add_param_cells(
            og,
            2,
            0,
            [("|ΔH|", self._h_tol_edit)],
        )
        _stretch_param_grid(og, max(c0, c1, c2))
        apply_group_tooltip(obs)
        obs.setMinimumWidth(240)
        h_split.addWidget(obs)

        model = QGroupBox("模型参数")
        mg = QGridLayout(model)
        _align_param_grid(mg)
        self._tp_edit = QLineEdit("1325")
        self._chi_edit = QLineEdit("6")
        self._engine_combo = QComboBox()
        self._engine_combo.addItems(["reebox", "kinzler_linear"])
        self._tp_min_edit = QLineEdit("1200")
        self._tp_max_edit = QLineEdit("1600")
        self._tp_step_edit = QLineEdit("25")
        self._chi_list_edit = QLineEdit("1,2,4,8,10,12,16")
        for w, width in (
            (self._tp_edit, 52),
            (self._chi_edit, 40),
            (self._tp_min_edit, 48),
            (self._tp_max_edit, 48),
            (self._tp_step_edit, 40),
        ):
            w.setFixedWidth(width)
        self._engine_combo.setFixedWidth(108)
        self._chi_list_edit.setMinimumWidth(100)
        tp_dash = QLabel("–")
        tp_dash.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._fast_scan_chk = QCheckBox("快速扫描")
        self._fast_scan_chk.setChecked(True)
        apply_widget_tip(self._fast_scan_chk, "fast_scan")
        self._bulk_bounds_chk = QCheckBox("bulk 界精修")
        apply_widget_tip(self._bulk_bounds_chk, "bulk_bounds")
        self._bulk_bounds_chk.toggled.connect(self._on_bulk_bounds_toggled)
        # Row 1: Tp / χ / engine / 快速扫描 / bulk 界精修
        mc0 = _add_param_cells(
            mg,
            0,
            0,
            [
                ("Tp", self._tp_edit),
                ("χ", self._chi_edit),
                ("engine", self._engine_combo),
                (None, self._fast_scan_chk),
                (None, self._bulk_bounds_chk),
            ],
        )
        self._hvp_mode_combo = QComboBox()
        for _key, label in HVP_DISPLAY_CHOICES:
            self._hvp_mode_combo.addItem(label, _key)
        default_idx = next(
            (i for i, (k, _) in enumerate(HVP_DISPLAY_CHOICES) if k == HVP_DISPLAY_DEFAULT),
            0,
        )
        self._hvp_mode_combo.setCurrentIndex(default_idx)
        self._hvp_mode_combo.currentIndexChanged.connect(self._on_hvp_display_changed)
        self._hvp_mode_combo.setMinimumWidth(120)
        self._read_band_chk = QCheckBox("Step-2 竖段")
        self._read_band_chk.setChecked(True)
        apply_widget_tip(self._read_band_chk, "read_band")
        self._read_band_chk.toggled.connect(self._on_hvp_display_changed)
        self._delta_vp_edit = QLineEdit("0.15")
        self._delta_vp_edit.setFixedWidth(44)
        self._thick_h_edit = QLineEdit("15")
        self._thick_h_edit.setFixedWidth(40)
        self._fig12_paper_axes_chk = QCheckBox("论文轴范围")
        self._fig12_paper_axes_chk.setChecked(True)
        apply_widget_tip(self._fig12_paper_axes_chk, "fig12_paper_axes")
        self._fig12_paper_axes_chk.toggled.connect(self._on_hvp_display_changed)
        # Row 2: H–Vp / Step-2 竖段 / ΔVp / 厚壳 / 论文轴范围
        mc1 = _add_param_cells(
            mg,
            1,
            0,
            [
                ("H–Vp", self._hvp_mode_combo),
                (None, self._read_band_chk),
                ("ΔVp", self._delta_vp_edit),
                ("厚壳 H>", self._thick_h_edit),
                (None, self._fig12_paper_axes_chk),
            ],
        )
        # Row 3: Tp scan range / step / χ list
        mc2 = _add_param_cells(
            mg,
            2,
            0,
            [
                ("Tp scan", self._tp_min_edit),
                (None, tp_dash),
                (None, self._tp_max_edit),
                ("step", self._tp_step_edit),
                ("χ list", self._chi_list_edit),
            ],
        )
        apply_tip(
            self._tp_max_edit,
            PARAM_LABEL_TIPS.get("Tp scan 上限", ""),
        )
        _stretch_param_grid(mg, max(mc0, mc1, mc2))
        apply_group_tooltip(model)
        model.setMinimumWidth(300)
        h_split.addWidget(model)

        lith = QGroupBox("岩性 (reebox)")
        lg = QGridLayout(lith)
        _align_param_grid(lg)
        self._backend_combo = QComboBox()
        self._backend_combo.addItems(["pymelt", "native"])
        self._backend_combo.currentTextChanged.connect(self._on_backend_changed)
        self._backend_combo.setFixedWidth(88)
        self._preset_combo = QComboBox()
        self._preset_combo.addItem("(custom)")
        self._preset_combo.addItems(list_lithology_presets())
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        self._per_combo = QComboBox()
        self._per_combo.setEditable(True)
        self._per_combo.addItems(peridotite_lithology_keys())
        self._pyr_combo = QComboBox()
        self._pyr_combo.setEditable(True)
        self._pyr_combo.addItems(enriched_lithology_keys())
        self._per_h2o_edit = QLineEdit("0")
        self._pyr_h2o_edit = QLineEdit("0")
        self._per_h2o_edit.setFixedWidth(40)
        self._pyr_h2o_edit.setFixedWidth(40)
        self._preset_combo.setMinimumWidth(140)
        self._per_combo.setMinimumWidth(100)
        self._pyr_combo.setMinimumWidth(100)
        lc0 = _add_param_cells(
            lg,
            0,
            0,
            [
                ("backend", self._backend_combo),
                ("preset", self._preset_combo),
            ],
        )
        lc1 = _add_param_cells(
            lg,
            1,
            0,
            [
                ("per", self._per_combo),
                ("pyr", self._pyr_combo),
            ],
        )
        lc2 = _add_param_cells(
            lg,
            2,
            0,
            [
                ("H₂O per", self._per_h2o_edit),
                ("H₂O pyr", self._pyr_h2o_edit),
            ],
        )
        _stretch_param_grid(lg, max(lc0, lc1, lc2))
        btn_lith = QPushButton("岩性说明…")
        btn_lith.setToolTip(
            "打开岩性 / preset 组成、固相线特点与适用场景说明（非模态）"
        )
        btn_lith.clicked.connect(self._show_lithology_reference)
        lg.addWidget(btn_lith, 3, 0, 1, 4)
        apply_group_tooltip(lith)
        lith.setMinimumWidth(280)
        h_split.addWidget(lith)
        # obs : model : lith — 收窄模型、放宽岩性
        h_split.setStretchFactor(0, 2)
        h_split.setStretchFactor(1, 3)
        h_split.setStretchFactor(2, 5)
        self._apply_greenland_startup_defaults()

    def _apply_greenland_startup_defaults(self) -> None:
        """Defaults from Greenland anchor K-section (greenland_kg1 + pymelt)."""
        self._backend_combo.setCurrentText("pymelt")
        self._on_backend_changed()
        idx = self._preset_combo.findText("greenland_kg1")
        if idx >= 0:
            self._preset_combo.setCurrentIndex(idx)
        else:
            self._on_preset_changed("greenland_kg1")

    def _build_status(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)
        self._progress = QProgressBar()
        self._progress.setMaximumWidth(180)
        self._progress.setTextVisible(False)
        self._progress.hide()
        sb.addPermanentWidget(self._progress)
        self._status_label = QLabel("就绪")
        sb.addWidget(self._status_label, stretch=1)

    def _toggle_params(self) -> None:
        self._params_visible = not self._params_visible
        sizes = self._main_split.sizes()
        if len(sizes) < 3:
            self._params_panel.setVisible(self._params_visible)
        elif self._params_visible:
            self._params_panel.show()
            restore = max(int(self._params_collapsed_h), _PARAMS_DEFAULT_H // 2)
            plot_log = max(sum(sizes[1:]), 1)
            self._main_split.setSizes([restore, int(plot_log * 0.75), int(plot_log * 0.25)])
        else:
            if sizes[0] > 0:
                self._params_collapsed_h = sizes[0]
            plot_log = max(sum(sizes[1:]), 1)
            self._params_panel.hide()
            self._main_split.setSizes([0, int(plot_log * 0.75), int(plot_log * 0.25)])
        self._toggle_params_act.setText("展开参数" if not self._params_visible else "收起参数")

    def _on_backend_changed(self, _text: str | None = None) -> None:
        native = self._backend_combo.currentText() == "native"
        for w in (self._preset_combo, self._per_combo, self._pyr_combo, self._per_h2o_edit, self._pyr_h2o_edit):
            w.setEnabled(not native)
        if native:
            self._preset_combo.setCurrentText("(custom)")

    def _on_preset_changed(self, name: str | None = None) -> None:
        name = name or self._preset_combo.currentText()
        if name in ("", "(custom)") or self._backend_combo.currentText() == "native":
            return
        if name not in LITHOLOGY_PRESETS:
            return
        p = LITHOLOGY_PRESETS[name]
        self._backend_combo.blockSignals(True)
        self._backend_combo.setCurrentText("pymelt")
        self._backend_combo.blockSignals(False)
        self._per_combo.setCurrentText(p.peridotite_key)
        self._pyr_combo.setCurrentText(p.pyroxenite_key)
        self._per_h2o_edit.setText(str(p.peridotite_h2o_wt))
        self._pyr_h2o_edit.setText(str(p.pyroxenite_h2o_wt))

    def _apply_plot_fonts(self) -> None:
        """Matplotlib 轴标签与刻度字号（与 Qt 控件协调）."""
        title = UI_FONT_PT
        label = UI_FONT_PT - 1
        tick = UI_FONT_PT - 2
        self._ax.set_title(self._ax.get_title(), fontsize=title)
        self._ax.set_xlabel(self._ax.get_xlabel(), fontsize=label)
        self._ax.set_ylabel(self._ax.get_ylabel(), fontsize=label)
        self._ax.tick_params(labelsize=tick)

    # ---------------------------------------------------------------- helpers
    def _append_log(self, msg: str) -> None:
        self._log.append(msg)

    def _set_busy(self, busy: bool, msg: str = "", *, indeterminate: bool = False) -> None:
        self._busy = busy
        if msg:
            self._status_label.setText(msg)
        if busy:
            self._progress.show()
            self._progress.setTextVisible(True)
            if indeterminate:
                self._progress.setRange(0, 0)
                self._progress.setFormat("")
            else:
                self._progress.setRange(0, 1)
                self._progress.setValue(0)
                self._progress.setFormat("0% (0/?)")
        else:
            self._progress.hide()
            self._progress.setRange(0, 100)
            self._progress.setValue(0)
        for tb in self.findChildren(QToolBar):
            for act in tb.actions():
                if act is self._toggle_params_act:
                    continue
                act.setEnabled(not busy)

    @Slot(int, int, str)
    def _on_progress(self, done: int, total: int, message: str) -> None:
        if total > 0:
            self._progress.setRange(0, total)
            val = min(max(done, 0), total)
            self._progress.setValue(val)
            pct = int(100 * val / total)
            self._progress.setFormat(f"{pct}% ({val}/{total})")
        elif done > 0:
            self._progress.setRange(0, 0)
        if message:
            self._status_label.setText(message)

    def _parse_floats_csv(self, text: str) -> list[float]:
        return [float(x.strip()) for x in text.split(",") if x.strip()]

    def _obs_kwargs(self) -> dict:
        return {
            "h_obs_km": float(self._h_edit.text()),
            "v_lc_obs_km_s": float(self._vlc_edit.text()),
            "b_km": float(self._b_edit.text()),
            "pyroxenite_frac": float(self._phi_edit.text()),
            "h_tolerance_km": float(self._h_tol_edit.text()),
        }

    def _lith_kwargs(self) -> dict:
        preset = self._preset_combo.currentText()
        preset_name = None if preset in ("", "(custom)") else preset
        if self._backend_combo.currentText() == "native":
            return resolve_lithology_col_kwargs(lithology_backend="native")
        return resolve_lithology_col_kwargs(
            lithology_backend="pymelt",
            lithology_preset=preset_name,
            peridotite_lith=self._per_combo.currentText(),
            pyroxenite_lith=self._pyr_combo.currentText(),
            peridotite_h2o_wt=float(self._per_h2o_edit.text() or 0),
            pyroxenite_h2o_wt=float(self._pyr_h2o_edit.text() or 0),
        )

    def _on_bulk_bounds_toggled(self, checked: bool) -> None:
        if checked:
            self._fast_scan_chk.setChecked(False)

    def _scan_performance_kwargs(self) -> dict:
        fast = self._fast_scan_chk.isChecked() and not self._bulk_bounds_chk.isChecked()
        engine = self._engine_combo.currentText()
        refine = 0 if fast else (4 if engine == "reebox" else 0)
        if self._bulk_bounds_chk.isChecked():
            refine = 12 if engine == "reebox" else 15
        return {
            "n_isentropic_steps": 20 if fast else 28,
            "refine_norm_vp": refine,
        }

    def _scan_kwargs(self) -> dict:
        engine = self._engine_combo.currentText()
        bulk_mode = self._bulk_bounds_chk.isChecked()
        return {
            **self._obs_kwargs(),
            **self._scan_performance_kwargs(),
            "melting_engine": engine,
            "tp_range_c": (float(self._tp_min_edit.text()), float(self._tp_max_edit.text())),
            "tp_step_c": float(self._tp_step_edit.text()),
            "chi_values": self._parse_floats_csv(self._chi_list_edit.text()),
            "delta_vp_engine": "wl1990",
            "delta_vp_wl_kw": dict(R3_DELTA_VP_WL_KW),
            "require_bulk_in_bounds": bulk_mode,
            "vp_bias_km_s": 0.0,
            **self._lith_kwargs(),
        }

    def _uses_burnman_refine(self) -> bool:
        perf = self._scan_performance_kwargs()
        return perf.get("refine_norm_vp", 0) > 0

    def _start_task(
        self,
        label: str,
        worker_fn,
        on_done,
        *,
        use_progress: bool = True,
        main_thread: bool = False,
    ) -> None:
        if self._busy:
            QMessageBox.information(self, "忙", "当前有任务在运行，请稍候。")
            return

        def _err(msg: str) -> None:
            self._active_thread = None
            self._active_worker = None
            self._set_busy(False, "错误")
            self._append_log(f"[{label}] 失败: {msg}")
            QMessageBox.critical(self, label, msg)

        def _finish(result) -> None:
            self._active_thread = None
            self._active_worker = None
            try:
                on_done(result)
            except Exception as exc:
                _err(str(exc))
                return
            self._set_busy(False, "就绪")

        if main_thread:
            self._set_busy(True, f"{label}…", indeterminate=not use_progress)
            self._append_log(
                f"[{label}] BurnMan 精修在主线程运行（后台 QThread 易触发 Segfault，界面会卡顿属正常）"
            )

            def report(done: int, total: int, message: str = "") -> None:
                self._on_progress(int(done), int(total), message or "")

            try:
                result = worker_fn(report) if use_progress else worker_fn()
            except Exception as exc:
                _err(str(exc))
                return
            _finish(result)
            return

        self._set_busy(True, f"{label}…", indeterminate=not use_progress)

        def task(report: ProgressReport):
            return worker_fn(report) if use_progress else worker_fn()

        thread, worker = run_in_thread(
            self,
            task,
            _finish,
            _err,
            on_progress=self._on_progress if use_progress else None,
            with_progress=use_progress,
        )
        self._active_thread = thread
        self._active_worker = worker

    def _hvp_display_mode(self) -> str:
        data = self._hvp_mode_combo.currentData()
        return str(data) if data else HVP_DISPLAY_MODERN

    def _fig12_use_paper_axes(self) -> bool:
        if not self._fig12_paper_axes_chk.isChecked():
            return False
        mode = self._hvp_display_mode()
        return mode in (HVP_DISPLAY_HVP, HVP_DISPLAY_FIG12, HVP_DISPLAY_BOTH)

    def _delta_vp_max(self) -> float:
        try:
            return float(self._delta_vp_edit.text())
        except ValueError:
            return 0.15

    def _thick_crust_h_min(self) -> float:
        try:
            return float(self._thick_h_edit.text())
        except ValueError:
            return 15.0

    def _observation_point(self, h_obs: float, v_lc: float) -> ObservationPoint:
        thick = is_thick_crust(h_obs, h_min_km=self._thick_crust_h_min())
        sigma = None
        if self._imported_obs is not None and self._imported_obs.v_lc_sigma_km_s is not None:
            sigma = self._imported_obs.v_lc_sigma_km_s
        return ObservationPoint(
            h_km=h_obs,
            v_lc_km_s=v_lc,
            v_lc_sigma_km_s=sigma,
            thick_crust=thick,
            label="观测",
        )

    def _update_read_tab(self, h_obs: float, v_lc: float) -> None:
        thick = is_thick_crust(h_obs, h_min_km=self._thick_crust_h_min())
        dv = self._delta_vp_max()
        v_lo, v_up = bulk_read_band(v_lc, delta_vp_max_km_s=dv)
        lines = [
            f"H = {h_obs:.2f} km",
            f"V_LC = {v_lc:.4f} km/s @ 600 MPa, 400°C",
            f"壳型: {'厚壳 (上界读图有效)' if thick else '薄壳 (V_LC 上界失效)'}",
            "",
            "Fig.15c 作图: (H, V_LC) 直接投 Fig.12a，不减 ΔVp",
            "",
        ]
        if thick:
            lines.extend(
                [
                    f"Step-2 读图: V_bulk,model ≤ {v_up:.4f} km/s",
                    f"           V_bulk 可低至 {v_lo:.4f} km/s (ΔVp ≤ {dv:.2f})",
                ]
            )
        else:
            lines.append("Step-2 读图: 跳过（孔隙/薄壳使上界失效）")
        if self._imported_obs is not None:
            lines.extend(
                [
                    "",
                    f"来源: {self._imported_obs.source}",
                    f"f_lower = {self._imported_obs.f_lower:.2f}",
                ]
            )
            if self._imported_obs.n_samples:
                lines.append(f"n_samples = {self._imported_obs.n_samples}")
        self._read_table.setPlainText("\n".join(lines))

    def _update_scan_results_tab(self, result) -> None:
        if not result or not result.points:
            self._scan_table.setPlainText("(无扫描结果)")
            return
        lines = [
            f"网格点数: {len(result.points)}",
            f"严格可行: {result.n_feasible}",
            "",
            f"{'Tp':>6} {'χ':>5} {'H':>7} {'Vp':>8} {'ΔH':>7} {'bulkOK':>7}",
            "-" * 48,
        ]
        show = sorted(result.points, key=lambda p: (p.chi, p.tp_c))
        for p in show[:40]:
            ok = "Y" if p.bulk_in_bounds else ("~" if p.feasible else "N")
            lines.append(
                f"{p.tp_c:6.0f} {p.chi:5.1f} {p.h_km:7.2f} {p.vp_bulk_km_s:8.3f} "
                f"{p.h_match_km:7.2f} {ok:>7}"
            )
        if len(show) > 40:
            lines.append(f"... ({len(show) - 40} more)")
        rng = result.tp_chi_ranges()
        if rng.get("tp_c"):
            lines.append(
                f"\n可行 Tp: {rng['tp_c'][0]:.0f}–{rng['tp_c'][1]:.0f}°C  "
                f"χ: {rng['chi'][0]:.1f}–{rng['chi'][1]:.1f}"
            )
        self._scan_table.setPlainText("\n".join(lines))

    def _show_formula_reference(self) -> None:
        show_formula_reference(self)

    def _show_track_compare(self) -> None:
        if self._last_scan_result is None:
            QMessageBox.information(self, "轨对比", "请先运行 Tp–χ 扫描。")
            return
        show_modeless_dialog(
            TrackCompareDialog(scan_result=self._last_scan_result),
            singleton=False,
        )

    def _import_observation(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入观测 (H, V_LC)",
            "",
            "JSON (*.json);;Transect windows CSV (*_windows.csv);;CSV (*.csv);;All (*.*)",
        )
        if not path:
            return
        try:
            p = Path(path)
            if self._looks_like_transect_windows_csv(p):
                self.import_transect_file(p)
            else:
                self.import_observation_file(path)
        except Exception as exc:
            QMessageBox.critical(self, "导入观测", str(exc))

    @staticmethod
    def _looks_like_transect_windows_csv(path: Path) -> bool:
        if path.name.endswith("_windows.csv"):
            return True
        import csv

        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            fields = {str(k or "").strip() for k in (reader.fieldnames or [])}
        return {"h_whole_km", "v_lc_km_s", "thick_crust"}.issubset(fields)

    def import_transect_file(self, path: str | Path) -> None:
        """Load imodel transect windows CSV and plot Fig.15c on Fig.12a."""
        windows = load_transect_windows_csv(path)
        if not windows:
            raise ValueError(f"未在 {path} 中找到有效滑窗记录")
        self._imported_transect_windows = windows
        thick = sum(1 for w in windows if w.thick_crust)
        self._append_log(
            f"已导入沿迹滑窗: {len(windows)} 窗 "
            f"(厚壳 {thick}, 薄壳 {len(windows) - thick}) ← {Path(path).name}"
        )
        if windows:
            mid = windows[len(windows) // 2]
            self._h_edit.setText(f"{mid.h_whole_km:g}")
            self._vlc_edit.setText(f"{mid.v_lc_km_s:.4f}")
        self._preview_fig15c_transect()

    def import_observation_file(self, path: str | Path) -> None:
        """Load imodel / single-point observation and refresh H–Vp plot."""
        p = Path(path)
        if self._looks_like_transect_windows_csv(p):
            self.import_transect_file(p)
            return
        if p.suffix.lower() == ".json":
            obs = load_observation_json(p)
        else:
            import csv

            with p.open(encoding="utf-8", newline="") as fh:
                row = next(csv.DictReader(fh))
            obs = CrustObservation(
                h_whole_km=float(row.get("h_whole_km", row.get("H", self._h_edit.text()))),
                v_lc_km_s=float(row.get("v_lc_km_s", row.get("V_LC", self._vlc_edit.text()))),
                v_lc_sigma_km_s=float(row["v_lc_sigma_km_s"]) if row.get("v_lc_sigma_km_s") else None,
                f_lower=float(row.get("f_lower", 0.7)),
                source=str(row.get("source", "csv")),
            )
        self._imported_obs = obs
        self._h_edit.setText(f"{obs.h_whole_km:g}")
        self._vlc_edit.setText(f"{obs.v_lc_km_s:.4f}")
        self._append_log(
            f"已导入观测: H={obs.h_whole_km:.2f} km  V_LC={obs.v_lc_km_s:.4f} km/s  "
            f"source={obs.source}"
        )
        self._update_read_tab(obs.h_whole_km, obs.v_lc_km_s)
        if self._last_scan_result is not None:
            self._plot_scan(
                self._last_scan_result,
                obs.h_whole_km,
                obs.v_lc_km_s,
                h_tolerance_km=self._last_scan_h_tol,
                b_km=self._last_scan_b_km,
            )

    def _preview_fig15c_transect(self) -> None:
        if self._busy:
            QMessageBox.information(self, "Fig.15", "请等待当前任务完成。")
            return
        windows = self._imported_transect_windows or []
        if not windows:
            QMessageBox.information(self, "Fig.15", "请先导入沿迹滑窗 windows CSV。")
            return
        from petrology.hvp.fig15_composite import plot_fig15_composite_on_axes

        self._fig.clear()
        gs = self._fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.35], hspace=0.42)
        ax_a = self._fig.add_subplot(gs[0])
        ax_b = self._fig.add_subplot(gs[1], sharex=ax_a)
        self._ax = self._fig.add_subplot(gs[2])
        import matplotlib.pyplot as plt

        plt.setp(ax_a.get_xticklabels(), visible=False)
        plot_fig15_composite_on_axes(
            (ax_a, ax_b, self._ax),
            windows=windows,
            transect_label="imported transect",
            h_min_km=self._thick_crust_h_min(),
            delta_vp_max_km_s=self._delta_vp_max(),
        )
        self._fig.subplots_adjust(top=0.96, bottom=0.07, left=0.12, right=0.88)
        self._apply_plot_fonts()
        self._canvas.draw_idle()
        self._append_log(f"已绘制 Fig.15 (a–c)（{len(windows)} 窗）")

    @Slot()
    def _on_hvp_display_changed(self, *_args) -> None:
        """Redraw immediately when H–Vp mode / paper-axes / read-band toggles."""
        if self._busy:
            return
        try:
            if self._last_scan_result is not None:
                h_obs = float(
                    self._last_scan_h_obs
                    if self._last_scan_h_obs is not None
                    else self._h_edit.text() or 30
                )
                v_lc = float(
                    self._last_scan_v_lc
                    if self._last_scan_v_lc is not None
                    else self._vlc_edit.text() or 7.0
                )
                self._plot_scan(
                    self._last_scan_result,
                    h_obs,
                    v_lc,
                    h_tolerance_km=self._last_scan_h_tol,
                    b_km=self._last_scan_b_km,
                )
            else:
                # No scan yet: still show default H–Vp so mode switch is visible
                self._plot_hvp_mode_preview()
        except Exception as exc:
            self._append_log(f"H–Vp 显示切换失败: {exc}")
            return
        from PySide6.QtWidgets import QApplication

        QApplication.processEvents()

    def _preview_sallares_galapagos(self, panel_key: str = "10a_ref") -> None:
        if self._busy:
            QMessageBox.information(self, "Sallarès", "请等待当前任务完成。")
            return
        from petrology.hvp.sallares_plot import plot_sallares_hvp_diagram

        self._fig.clear()
        self._ax = self._fig.add_subplot(111)
        panel = panel_key
        h_lim = (3.0, 35.0) if self._fig12_use_paper_axes() else (0.0, 65.0)
        vp_lim = (6.85, 7.50) if self._fig12_use_paper_axes() else (6.8, 7.6)
        results = plot_sallares_hvp_diagram(
            self._ax,
            panel_key=panel,
            vp_bias_km_s=0.0,
            h_lim=h_lim,
            vp_lim=vp_lim,
            label_fontsize=UI_FONT_PT - 2,
        )
        self._fig.tight_layout()
        self._apply_plot_fonts()
        self._canvas.draw_idle()
        n_curves = len({(r.tp_c, r.upwelling_x) for r in results})
        self._append_log(
            f"Sallarès Galápagos Fig.10 ({panel}): eq.(1) 曲线 {n_curves} 点 + "
            f"6 条 GVP 观测（Layer 3 @ 400°C/600 MPa）"
        )
        self._read_table.setPlainText(self._format_sallares_summary(results, panel))

    def _format_sallares_summary(self, results, panel_key: str) -> str:
        if not results:
            return "（无模型解）"
        lines = [
            f"Sallarès (2005) panel {panel_key}",
            f"  V_model range: {min(r.vp_bulk_km_s for r in results):.3f} – "
            f"{max(r.vp_bulk_km_s for r in results):.3f} km/s",
            f"  H range: {min(r.h_km for r in results):.1f} – {max(r.h_km for r in results):.1f} km",
            "",
            "厚壳观测 (~19 km, V_LC~6.85–6.90) vs pyrolite 曲线 (同 H 处 V~7.3+):",
            "  → H–V 反相关；热柱 + eq.(1) 难以同时解释厚壳与低速。",
            "",
            "锚点:",
        ]
        for tp, x in [(1300, 1), (1300, 8), (1300, 35), (1175, 35)]:
            match = [
                r
                for r in results
                if abs(r.tp_c - tp) < 1e-6 and abs(r.upwelling_x - x) < 1e-6
            ]
            if match:
                r = match[0]
                lines.append(
                    f"  Tp={tp:.0f}°C X={x:g}: H={r.h_km:.1f} km "
                    f"F̄={r.f_bar:.3f} P̄={r.p_bar_gpa:.2f} GPa V={r.vp_bulk_km_s:.3f} km/s"
                )
        return "\n".join(lines)

    # ---------------------------------------------------------------- plots
    def _reset_single_axes(self) -> None:
        self._fig.clear()
        self._ax = self._fig.add_subplot(111)

    def _modern_hvp_build_params(self) -> dict:
        """Kwargs for ``build_modern_hvp_curves`` / ``modern_hvp_from_scan_grid``."""
        perf = self._scan_performance_kwargs()
        return {
            "col_kw": {
                **self._lith_kwargs(),
                "n_isentropic_steps": int(perf.get("n_isentropic_steps", 28)),
                "vp_bias_km_s": 0.0,
            },
            "pyroxenite_frac": float(self._phi_edit.text() or 0.10),
            "melting_engine": self._engine_combo.currentText(),
        }

    def _hvp_linear_cache_for(self, modern_points) -> tuple:
        """Return cached eq.(1) H–Vp curves; recompute only when Tp grid changes."""
        if not modern_points:
            key = ("default",)
            if self._hvp_linear_cache_key != key or self._hvp_linear_cache is None:
                import numpy as np

                tp_vals = np.arange(1200.0, 1600.0 + 0.5, 10.0)
                self._hvp_linear_cache = build_fig12_linear_curves(tp_values_c=tp_vals)
                self._hvp_linear_cache_key = key
            return self._hvp_linear_cache

        tp_key = tuple(sorted({round(float(p.tp_c), 6) for p in modern_points}))
        key = ("scan", tp_key)
        if self._hvp_linear_cache_key != key or self._hvp_linear_cache is None:
            self._hvp_linear_cache = linear_hvp_from_scan_grid(
                modern_points, vp_bias_km_s=0.0, include_chi1_b_family=True
            )
            self._hvp_linear_cache_key = key
        return self._hvp_linear_cache

    def _hvp_modern_cache_for(self, modern_points) -> tuple:
        """Return Modern Fig.12-style curves (χ solids + χ=1,b dashes)."""
        import numpy as np

        build = self._modern_hvp_build_params()
        lith = build["col_kw"]
        param_key = (
            build["melting_engine"],
            round(float(build["pyroxenite_frac"]), 6),
            lith.get("lithology_backend"),
            self._preset_combo.currentText(),
            lith.get("peridotite_lith"),
            lith.get("pyroxenite_lith"),
            round(float(lith.get("peridotite_h2o_wt", 0.0) or 0.0), 6),
            round(float(lith.get("pyroxenite_h2o_wt", 0.0) or 0.0), 6),
            int(lith.get("n_isentropic_steps", 28)),
        )
        if not modern_points:
            key = ("default",) + param_key
            if self._hvp_modern_cache_key != key or self._hvp_modern_cache is None:
                # Coarser Tp than linear default — Modern forward is slower
                tp_vals = np.arange(1200.0, 1600.0 + 0.5, 50.0)
                self._hvp_modern_cache = build_modern_hvp_curves(
                    tp_values_c=tp_vals,
                    **build,
                )
                self._hvp_modern_cache_key = key
            return self._hvp_modern_cache

        tp_key = tuple(sorted({round(float(p.tp_c), 6) for p in modern_points}))
        chi_key = tuple(sorted({round(float(p.chi), 6) for p in modern_points}))
        key = ("scan", tp_key, chi_key) + param_key
        if self._hvp_modern_cache_key != key or self._hvp_modern_cache is None:
            self._hvp_modern_cache = modern_hvp_from_scan_grid(
                modern_points,
                include_chi1_b_family=True,
                **build,
            )
            self._hvp_modern_cache_key = key
        return self._hvp_modern_cache

    def _plot_hvp_mode_preview(self) -> None:
        """Show H–Vp / Modern / both without a prior scan (mode-switch feedback)."""
        from petrology.gui.hvp_plot import plot_hvp_fig12_style, plot_hvp_dual_track

        try:
            h_obs = float(self._h_edit.text())
            v_lc = float(self._vlc_edit.text())
        except ValueError:
            h_obs, v_lc = 30.0, 7.0

        display_mode = self._hvp_display_mode()
        linear, chi1_b = self._hvp_linear_cache_for([])
        self._fig.clear()
        self._ax = self._fig.add_subplot(111)
        obs_pt = self._observation_point(h_obs, v_lc)

        if display_mode in (HVP_DISPLAY_HVP, HVP_DISPLAY_FIG12):
            plot_hvp_fig12_style(
                self._ax,
                linear,
                chi1_b_lines=chi1_b,
                tp_range_c=(1200.0, 1600.0),
                label_fontsize=UI_FONT_PT - 3,
                use_paper_axis_limits=self._fig12_use_paper_axes(),
            )
            self._ax.set_title("H–Vp 图（尚未扫描；显示默认计算轨）")
        elif display_mode == HVP_DISPLAY_BOTH:
            modern, modern_chi1_b = self._hvp_modern_cache_for([])
            plot_hvp_dual_track(
                self._ax,
                modern,
                linear,
                chi1_b_lines=chi1_b,
                modern_chi1_b_lines=modern_chi1_b,
                tp_range_c=(1200.0, 1600.0),
                label_fontsize=UI_FONT_PT - 3,
                use_paper_axis_limits=self._fig12_use_paper_axes(),
            )
            self._ax.set_title("H–Vp 双轨（尚未扫描；默认计算 + Modern）")
        else:
            modern, modern_chi1_b = self._hvp_modern_cache_for([])
            plot_hvp_fig12_style(
                self._ax,
                modern,
                chi1_b_lines=modern_chi1_b,
                tp_range_c=(1200.0, 1600.0),
                label_fontsize=UI_FONT_PT - 3,
                use_paper_axis_limits=self._fig12_use_paper_axes(),
            )
            self._ax.set_title("H–Vp（Modern REEBOX；尚未扫描，默认网格）")

        from petrology.hvp.observation_overlay import plot_observation_on_hvp

        plot_observation_on_hvp(
            self._ax,
            obs_pt,
            delta_vp_max_km_s=self._delta_vp_max(),
            h_min_km=self._thick_crust_h_min(),
            show_read_band=self._read_band_chk.isChecked(),
        )
        self._ax.set_xlabel("H (km)")
        self._ax.set_ylabel("Bulk Vp (km/s) @ 600 MPa, 400°C")
        self._ax.legend(loc="best", fontsize=UI_FONT_PT - 3)
        self._fig.tight_layout()
        self._apply_plot_fonts()
        self._canvas.draw_idle()

    def _plot_scan(
        self,
        result,
        h_obs: float,
        v_lc: float,
        *,
        h_tolerance_km: float | None = None,
        b_km: float = 0.0,
    ) -> None:
        self._last_scan_result = result
        self._last_scan_h_obs = h_obs
        self._last_scan_v_lc = v_lc
        self._last_scan_h_tol = h_tolerance_km
        self._last_scan_b_km = b_km

        self._fig.clear()
        ax_hv, ax_tc = self._fig.subplots(1, 2)
        self._ax = ax_hv

        display_mode = self._hvp_display_mode()
        obs_pt = self._observation_point(h_obs, v_lc)
        linear_cache = self._hvp_linear_cache_for(result.points)
        modern_cache = None
        if display_mode in (HVP_DISPLAY_MODERN, HVP_DISPLAY_BOTH):
            modern_cache = self._hvp_modern_cache_for(result.points)
        if result.points:
            tp_u = sorted({p.tp_c for p in result.points})
            tp_range = (tp_u[0], tp_u[-1]) if tp_u else None
            plot_hvp_scan_panel(
                ax_hv,
                result.points,
                display_mode=display_mode,
                b_km=b_km,
                tp_range_c=tp_range,
                label_fontsize=UI_FONT_PT - 3,
                vp_bias_km_s=0.0,
                use_paper_axis_limits=self._fig12_use_paper_axes(),
                observation=obs_pt,
                delta_vp_max_km_s=self._delta_vp_max(),
                thick_crust_h_min_km=self._thick_crust_h_min(),
                show_observation_read_band=self._read_band_chk.isChecked(),
                linear_cache=linear_cache,
                modern_cache=modern_cache,
            )
        elif display_mode in (HVP_DISPLAY_HVP, HVP_DISPLAY_FIG12, HVP_DISPLAY_BOTH):
            # Empty scan but still show cached / default H–Vp track
            from petrology.gui.hvp_plot import plot_hvp_fig12_style

            linear, chi1_b = linear_cache
            plot_hvp_fig12_style(
                ax_hv,
                linear,
                chi1_b_lines=chi1_b,
                label_fontsize=UI_FONT_PT - 3,
                use_paper_axis_limits=self._fig12_use_paper_axes(),
            )
            from petrology.hvp.observation_overlay import plot_observation_on_hvp

            plot_observation_on_hvp(
                ax_hv,
                obs_pt,
                delta_vp_max_km_s=self._delta_vp_max(),
                h_min_km=self._thick_crust_h_min(),
                show_read_band=self._read_band_chk.isChecked(),
            )

        show_feasible = display_mode in (HVP_DISPLAY_MODERN, HVP_DISPLAY_BOTH)
        if show_feasible and result.feasible:
            ax_hv.scatter(
                [p.h_km for p in result.feasible],
                [p.vp_bulk_km_s for p in result.feasible],
                s=36,
                c="crimson",
                zorder=5,
                label=f"可行 Modern ({result.n_feasible})",
            )
        elif show_feasible and h_tolerance_km is not None:
            h_ok = [p for p in result.points if abs(p.h_match_km) <= h_tolerance_km]
            if h_ok:
                ax_hv.scatter(
                    [p.h_km for p in h_ok],
                    [p.vp_bulk_km_s for p in h_ok],
                    s=18,
                    c="#2ca02c",
                    alpha=0.55,
                    zorder=4,
                    label=f"|ΔH|≤{h_tolerance_km:g} ({len(h_ok)})",
                )
        ax_hv.set_xlabel("H (km)")
        ax_hv.set_ylabel("Bulk Vp (km/s) @ 600 MPa, 400°C")
        titles = {
            HVP_DISPLAY_MODERN: "H–Vp（Modern REEBOX）",
            HVP_DISPLAY_HVP: "H–Vp 图",
            HVP_DISPLAY_FIG12: "H–Vp 图",
            HVP_DISPLAY_BOTH: "H–Vp 双轨（计算 + Modern）",
        }
        ax_hv.set_title(titles.get(display_mode, "H–Vp 图"))
        ax_hv.legend(loc="best", fontsize=UI_FONT_PT - 3)

        if result.points:
            tp_u = sorted({p.tp_c for p in result.points})
            chi_u = sorted({p.chi for p in result.points})
            grid = np.full((len(chi_u), len(tp_u)), np.nan)
            for p in result.points:
                val = 1.0 if p.feasible else 0.0
                if not p.feasible and h_tolerance_km is not None and abs(p.h_match_km) <= h_tolerance_km:
                    val = 0.5
                grid[chi_u.index(p.chi), tp_u.index(p.tp_c)] = val
            ax_tc.imshow(
                grid,
                aspect="auto",
                origin="lower",
                cmap="RdYlGn",
                vmin=0,
                vmax=1,
                extent=[tp_u[0] - 5, tp_u[-1] + 5, chi_u[0] - 0.5, chi_u[-1] + 0.5],
            )
        ax_tc.set_xlabel("Tp (°C)")
        ax_tc.set_ylabel("χ")
        ax_tc.set_title(f"(Tp, χ) 可行区 n={result.n_feasible}")

        self._fig.tight_layout()
        self._apply_plot_fonts()
        self._canvas.draw_idle()
        self._update_scan_results_tab(result)
        self._update_read_tab(h_obs, v_lc)

    # ---------------------------------------------------------------- actions
    def _run_forward(self) -> None:
        tp_text = self._tp_edit.text()
        chi_text = self._chi_edit.text()
        phi_text = self._phi_edit.text()
        fwd_kw = {
            "tp_c": float(tp_text),
            "chi": float(chi_text),
            "melting_engine": self._engine_combo.currentText(),
            "b_km": float(self._b_edit.text()),
            "pyroxenite_frac": float(phi_text),
            "n_isentropic_steps": 28,
            **self._lith_kwargs(),
        }

        def worker(report: ProgressReport):
            report(0, 1, "正演计算…")
            col = forward_melting_column(**fwd_kw)
            report(1, 1, "正演完成")
            return col

        def done(col):
            melt = col.pooled_melt_wt
            sio2 = melt.get("SiO2", float("nan"))
            self._append_log(
                f"正演 Tp={tp_text}°C χ={chi_text} Φ={phi_text}\n"
                f"  P0={col.p0_gpa:.2f} Pf={col.pf_gpa:.2f} GPa  H={col.h_km:.1f} km  "
                f"Fbar={col.fbar:.3f} Fmax={col.f_max:.3f}\n"
                f"  Vp(eq1)={col.vp_bulk_eq1_km_s:.3f} km/s  SiO2={sio2:.1f} wt%"
            )
            self._fig.clear()
            plot_forward_result(
                self._fig,
                col,
                tp_c=float(tp_text),
                chi=float(chi_text),
                phi=float(phi_text),
                b_km=float(self._b_edit.text()),
                melting_engine=self._engine_combo.currentText(),
                lith_kwargs=self._lith_kwargs(),
                title_fontsize=UI_FONT_PT,
                card_fontsize=UI_FONT_PT - 2,
                note_fontsize=UI_FONT_PT - 3,
            )
            self._ax = getattr(self._fig, "_lip_primary_ax", self._fig.axes[0] if self._fig.axes else self._fig.add_subplot(111))
            self._apply_plot_fonts()
            self._canvas.draw_idle()
            self._status_label.setText("正演完成")

        self._start_task("正演", worker, done)

    def _run_tp_chi_scan(self) -> None:
        kw = self._scan_kwargs()

        def worker(report: ProgressReport):
            return scan_hvp_lip(
                verbose=False,
                progress_callback=report,
                **kw,
            )

        def done(result):
            self._plot_scan(
                result,
                kw["h_obs_km"],
                kw["v_lc_obs_km_s"],
                h_tolerance_km=kw["h_tolerance_km"],
                b_km=kw["b_km"],
            )
            self._append_log(
                f"Tp–χ 扫描: {len(result.points)} 点, 可行 {result.n_feasible}\n"
                f"  engine={kw['melting_engine']} lith={self._backend_combo.currentText()} "
                f"refine={kw['refine_norm_vp']} steps={kw['n_isentropic_steps']} "
                f"H–Vp={self._hvp_mode_combo.currentText()}"
            )
            rng = result.tp_chi_ranges()
            if rng["tp_c"]:
                self._append_log(
                    f"  可行 Tp: {rng['tp_c'][0]:.0f}–{rng['tp_c'][1]:.0f}°C  "
                    f"χ: {rng['chi'][0]:.1f}–{rng['chi'][1]:.1f}"
                )
            self._status_label.setText(f"扫描完成 — 可行 {result.n_feasible}")

        self._start_task("Tp–χ 扫描", worker, done, main_thread=self._uses_burnman_refine())

    def _run_phi_scan(self) -> None:
        kw = self._scan_kwargs()
        phi_kw = {k: v for k, v in kw.items() if k != "pyroxenite_frac"}
        phi_vals = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]

        def worker(report: ProgressReport):
            return scan_hvp_lip_phi(
                phi_values=phi_vals,
                verbose=False,
                progress_callback=report,
                **phi_kw,
            )

        def done(scan):
            self._reset_single_axes()
            xs, ys = [], []
            for sl in scan.slices:
                pt = sl.best_pareto
                if pt is None:
                    continue
                xs.append(sl.pyroxenite_frac)
                ys.append(max(0.0, pt.vp_bulk_km_s - kw["v_lc_obs_km_s"]))
            if xs:
                self._ax.plot(xs, ys, "o-", lw=1.5)
            self._ax.axhline(0, color="0.5", ls="--", lw=0.8)
            self._ax.set_xlabel("Pyroxenite fraction Φ")
            self._ax.set_ylabel("Vp excess (km/s)")
            self._ax.set_title(f"Φ 扫描 — 总可行 {scan.n_feasible_total}")
            self._apply_plot_fonts()
            self._canvas.draw_idle()
            self._append_log(f"Φ 扫描: 总可行 {scan.n_feasible_total}")
            for sl in scan.slices:
                pt = sl.best_pareto
                if pt:
                    self._append_log(
                        f"  Φ={sl.pyroxenite_frac:.2f}  n={sl.result.n_feasible}  "
                        f"best Tp={pt.tp_c:.0f} χ={pt.chi:.1f} H={pt.h_km:.1f} Vp={pt.vp_bulk_km_s:.3f}"
                    )
            self._status_label.setText(f"Φ 扫描 — 可行 {scan.n_feasible_total}")

        self._start_task("Φ 扫描", worker, done, main_thread=self._uses_burnman_refine())

    def _run_preset_compare(self) -> None:
        kw = self._obs_kwargs()
        perf = self._scan_performance_kwargs()
        cmp_kw = {
            **kw,
            "tp_range_c": (float(self._tp_min_edit.text()), float(self._tp_max_edit.text())),
            "tp_step_c": float(self._tp_step_edit.text()),
            "chi_values": tuple(self._parse_floats_csv(self._chi_list_edit.text())),
            "melting_engine": self._engine_combo.currentText(),
            **perf,
        }

        def worker(report: ProgressReport):
            return compare_lithology_presets(
                verbose=False,
                progress_callback=report,
                **cmp_kw,
            )

        def done(rows):
            fast = self._fast_scan_chk.isChecked()
            perf_note = (
                f"快速模式 refine=0 steps={perf['n_isentropic_steps']}"
                if fast
                else f"refine={perf['refine_norm_vp']} steps={perf['n_isentropic_steps']}"
            )
            log = f"预设对比 ({perf_note}):\n" + format_preset_compare_table(rows)
            hints = preset_compare_hints(
                rows,
                pyroxenite_frac=kw["pyroxenite_frac"],
                h_tolerance_km=kw["h_tolerance_km"],
                refine_norm_vp=perf["refine_norm_vp"],
                fast_scan=fast,
            )
            if hints:
                log += "\n\n说明:\n" + hints
            self._append_log(log)

            self._reset_single_axes()
            names = [r.preset.replace("__native__", "native") for r in rows]
            use_h_only = all(r.n_feasible == 0 for r in rows) and any(r.n_h_match > 0 for r in rows)
            if use_h_only:
                counts = [r.n_h_match for r in rows]
                xlabel = f"仅 H 匹配点数 (|ΔH|≤{kw['h_tolerance_km']:g} km)"
                title = "岩性预设对比（严格可行=0，改示 n_H）"
            else:
                counts = [r.n_feasible for r in rows]
                xlabel = "严格可行 (Tp, χ) 点数"
                title = "岩性预设对比"
            colors = ["#2ca02c" if c > 0 else "#999999" for c in counts]
            self._ax.barh(names, counts, color=colors)
            self._ax.set_xlabel(xlabel)
            self._ax.set_title(title)
            self._ax.invert_yaxis()
            self._apply_plot_fonts()
            self._canvas.draw_idle()
            self._status_label.setText("预设对比完成")

        self._start_task("预设对比", worker, done, main_thread=self._uses_burnman_refine())

    def _save_figure(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存图",
            "",
            "PNG (*.png);;PDF (*.pdf);;All (*.*)",
        )
        if not path:
            return
        self._fig.savefig(path, dpi=150, bbox_inches="tight")
        self._append_log(f"已保存图: {path}")
