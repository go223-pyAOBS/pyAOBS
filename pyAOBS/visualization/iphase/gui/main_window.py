"""iphase Qt 主窗：工区 + 阶段页签（输入 / 走时图 / 输出）。

对齐 idata / relocation：顶部仅工区动作；输入与输出分页；绘图在「走时图」。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QTabBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from pyAOBS.utils.open_path import open_path_in_file_manager
from pyAOBS.utils.qt_combo import connect_combo_deferred, configure_combo_list_view

from ._business import IPhaseBusinessMixin, _StatusVar, _Var
from .dialog_utils import show_modeless_message
from .help_dialog import install_help_shortcut, show_help_dialog
from .panels import InputPanel, OutputPanel, PlotPanel
from .project import IphaseProject
from .services.file_result import DEFAULT_PSP_PHASE
from .services.workdir_layout import OUTPUTS_DIR
from .styles import apply_iphase_chrome, apply_iphase_font, set_path_badge


class IPhaseMainWindow(IPhaseBusinessMixin, QMainWindow):
    def __init__(self, argv: list[str] | None = None) -> None:
        QMainWindow.__init__(self)
        self.setWindowTitle("iphase — 震相分析工区")
        self.resize(1400, 900)

        self.project = IphaseProject()
        self._widget_refreshers: list[Callable[[], None]] = []
        self._param_line_flushers: list[Callable[[], None]] = []
        self._loading_project = False

        self.files: list[Path] = []
        self.results = []
        self.theory_mode = _Var("1D")
        self.psp_export_mode = _Var("picked")
        self.share_y_var = _Var(True)
        self.h_cr = _Var(2.0)
        self.vp_cr = _Var(3.5)
        self.vs_cr = _Var(1.5)
        self.window_points = _Var(11)
        self.smooth_dense_half_win = _Var("5")
        self.psp_phase_id = _Var(DEFAULT_PSP_PHASE)
        self.obs_mark_y = _Var(0.2)
        self.section_field_mode = _Var("point")
        self.picked_policy = _Var("插值")
        self.strict_diff_pair = _Var(False)
        self.force_recompute = _Var(False)
        self.theory2d_auto_fallback = _Var(True)
        self.seafloor_path: Path | None = None
        self.seafloor_x: np.ndarray | None = None
        self.seafloor_z: np.ndarray | None = None
        self.shot_depth_path: Path | None = None
        self.shot_depth_x: np.ndarray | None = None
        self.shot_depth_z: np.ndarray | None = None
        self.shot_depth_ids: list[str] | None = None
        self.pss_inversion_ready: bool = False
        self.pss_profile_by_file: dict[str, object] = {}
        self.theory2d_notice: str = ""
        self.status_var = _StatusVar("请先新建或打开工区")
        self.pois_left_var = _Var("")
        self.pois_right_var = _Var("")
        self.pois_first_val: float | None = None
        self.pps_pss_ratio = _Var(0.6)
        self.equi_write_equiv_psp = _Var(False)
        self.use_rin_enabled_filter = _Var(True)
        self._theory2d_left_branch = None
        self._theory2d_right_branch = None
        self._t2d_side_cache: dict = {}
        self._rin_editor_proc: subprocess.Popen | None = None
        self._rin_editor_path: Path | None = None
        self._rin_editor_last_mtime_ns: int | None = None
        self._rin_editor_poll_job = None
        self._gui_state_file = (
            Path(os.environ.get("PYAOBS_GUI_STATE_FILE", "").strip())
            if os.environ.get("PYAOBS_GUI_STATE_FILE", "").strip()
            else None
        )
        self._run_inputs_dir = (
            Path(os.environ.get("PYAOBS_RUN_INPUTS_DIR", "").strip())
            if os.environ.get("PYAOBS_RUN_INPUTS_DIR", "").strip()
            else None
        )
        self._startup_argv = list(argv if argv is not None else sys.argv)
        self._startup_opened_project = False

        self._build_ui()
        apply_iphase_chrome(self)
        install_help_shortcut(self)
        self._update_title()
        QTimer.singleShot(0, self._startup_restore)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self._hide_menubar()

        self.act_new_proj = QAction("新建工区", self)
        self.act_new_proj.triggered.connect(self._new_project)
        self.act_open_proj = QAction("打开工区", self)
        self.act_open_proj.triggered.connect(self._open_project)
        self.act_save_proj = QAction("保存工区", self)
        self.act_save_proj.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.act_save_proj.setToolTip("保存工区 (Ctrl+Shift+S)")
        self.act_save_proj.triggered.connect(self._save_project)

        # 文档 §9：走时 / 存图（不进工具栏，仅快捷键；须 addAction 才会生效）
        self.act_open_tx = QAction("添加走时文件", self)
        self.act_open_tx.setShortcut(QKeySequence.StandardKey.Open)  # Ctrl+O
        self.act_open_tx.setToolTip("添加走时文件 tx.in (Ctrl+O)")
        self.act_open_tx.triggered.connect(self._shortcut_add_tx)
        self.act_save_fig = QAction("保存走时图", self)
        self.act_save_fig.setShortcut(QKeySequence.StandardKey.Save)  # Ctrl+S
        self.act_save_fig.setToolTip("保存当前走时图 (Ctrl+S)")
        self.act_save_fig.triggered.connect(self.save_figure)

        self.act_help = QAction("帮助", self)
        self.act_help.setShortcut(QKeySequence.StandardKey.HelpContents)
        self.act_help.setToolTip("帮助 (F1)")
        self.act_help.triggered.connect(
            lambda: QTimer.singleShot(0, lambda: show_help_dialog(activate=True))
        )
        self.act_quit = QAction("退出", self)
        self.act_quit.setShortcut(QKeySequence.StandardKey.Quit)  # Ctrl+Q
        self.act_quit.setToolTip("退出 (Ctrl+Q)")
        self.act_quit.triggered.connect(self.close)

        for act in (
            self.act_new_proj,
            self.act_open_proj,
            self.act_save_proj,
            self.act_open_tx,
            self.act_save_fig,
            self.act_help,
            self.act_quit,
        ):
            act.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            self.addAction(act)

        tb = QToolBar("主工具")
        tb.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)
        title = QLabel("iphase")
        title.setObjectName("IphaseAppTitle")
        tb.addWidget(title)
        tb.addAction(self.act_new_proj)
        tb.addAction(self.act_open_proj)
        tb.addAction(self.act_save_proj)
        tb.addSeparator()
        tb.addAction(self.act_help)
        tb.addAction(self.act_quit)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        self.stage_bar = QTabBar()
        self.stage_bar.setObjectName("IphaseStageBar")
        self.stage_bar.setExpanding(False)
        for name in ("1 输入", "2 走时图", "3 输出"):
            self.stage_bar.addTab(name)
        self.stack = QStackedWidget()
        self.stage_bar.currentChanged.connect(self.stack.setCurrentIndex)

        self.panel_input = InputPanel()
        self.panel_plot = PlotPanel()
        self.panel_output = OutputPanel()
        self.stack.addWidget(self.panel_input)
        self.stack.addWidget(self.panel_plot)
        self.stack.addWidget(self.panel_output)

        # 走时图面板持有画布与参数条
        self.param_strip = self.panel_plot.param_strip
        self.plot = self.panel_plot.plot
        self.fig = self.plot.fig
        self.canvas = self.plot.canvas
        self._populate_param_strip()

        self.panel_input.request_apply.connect(self._apply_inputs_to_plot)
        self.panel_input.btn_rin_editor.clicked.connect(self.open_rin_phase_groups_editor)
        self.panel_input.paths_changed.connect(self._mark_dirty)
        self.panel_plot.request_theory2d.connect(self.run_theory2d_forward_all)
        self.panel_plot.request_inversion.connect(self.show_local_inversion_result)
        self.panel_plot.request_diagnostics.connect(self.show_1d_diagnostics)
        self.panel_plot.request_rin_editor.connect(self.open_rin_phase_groups_editor)
        self.panel_plot.request_export_tx.connect(self.show_export_tx_select)
        self.panel_output.request_save_figure.connect(self.save_figure)
        self.panel_output.request_export_psp.connect(self.export_psp_files)
        self.panel_output.request_open_outputs.connect(self._open_outputs_folder)
        self.panel_output.psp_mode_changed.connect(self._on_psp_mode_changed)

        root.addWidget(self.stage_bar)
        root.addWidget(self.stack, stretch=1)
        self.setCentralWidget(central)

        self.setStatusBar(QStatusBar())
        self._path_badge = QLabel("未打开工区")
        set_path_badge(self._path_badge, active=False, text="未打开工区")
        self.statusBar().addWidget(self._path_badge, stretch=0)
        self._status_label = QLabel("请先新建或打开工区")
        self.status_var.bind_label(self._status_label)
        self.statusBar().addWidget(self._status_label, stretch=1)

    def _hide_menubar(self) -> None:
        try:
            mb = self.menuBar()
            mb.clear()
            mb.setVisible(False)
            mb.setMaximumHeight(0)
        except Exception:
            pass

    def _mark_dirty(self) -> None:
        if self._loading_project:
            return
        if self.project.workdir:
            self.project.dirty = True
            self._update_title()

    def _wrap_change(self, fn: Callable) -> Callable:
        def _inner(*args, **kwargs):
            self._mark_dirty()
            return fn(*args, **kwargs)

        return _inner

    def _populate_param_strip(self) -> None:
        strip = self.param_strip
        strip.clear_widgets()
        self._widget_refreshers.clear()
        self._param_line_flushers.clear()

        tips = {
            "时差模式": "理论时差计算方式：\n• 1D：薄层公式（厚度·Vp·Vs）\n• 2D：RAYINVR 正演 tx.out\n• 2Dequi：等效构造 tx_2Dequiv.in",
            "共享y轴": "多文件时统一右上（观测时差）、左下（理论vs观测）子图的纵轴范围。",
            "严格配对": "勾选后 PPS−PPP / PSS−PSP 要求同道严格匹配；\n未勾选时在后相位点处用前相位拟合曲线取值（默认）。",
            "强制重算": "忽略部分缓存，强制重新计算理论时差与相关量。",
            "2D失败回退": "2D / 2Dequi 理论失败时自动回退到 1D，避免图空白。",
            "按r.in保留组过滤": "仅保留 r.in 中 ivray 且 nray>0 的相位组对应拾取（第二阶段过滤）。",
            "校正策略": "差分对构造策略：\n• 插值：受控插值\n• 严格：严格配对\n• 宽松：放宽匹配",
            "厚度": "1D 地壳厚度初值 h（km），用于理论时差与 1D 反演。",
            "Vp": "1D P 波速度初值（km/s）。",
            "Vs": "1D S 波速度初值（km/s）。",
            "线性窗口": "LocalLinear 拟合窗口点数（7–15），影响斜率/路径因子估计。",
            "平滑半窗": "密点平滑半窗宽度；0 表示关闭平滑。",
            "二维参数场": "1D 反演剖面填充场：\n• point：逐点估计\n• eff：有效场",
            "PSP相位号": "输入 / 导出 PSP 震相的 phase id。",
            "OBS标注Y(s)": "主图 OBS 三角标注的纵坐标（秒）。",
            "pois左支": "写回左支 pois 并正演；仅刷新 OBS 左侧理论 PPP/PPS 点与左支时差（右侧缓存保留）。",
            "pois右支": "写回右支 pois 并正演；仅刷新 OBS 右侧理论 PPP/PPS 点与右支时差（左侧缓存保留）。",
            "PPS/PSS": "2Dequi 等效构造使用的 PPS/PSS 比值。",
            "2Dequi写等效PSP": "勾选后写入等效 PSP，并在 2Dequi 模式下自动触发正演。",
        }

        def _tip(name: str) -> str:
            return tips.get(name, "")

        def _lab(text: str, tip_key: str | None = None) -> QLabel:
            lab = QLabel(text)
            tip = _tip(tip_key or text)
            if tip:
                lab.setToolTip(tip)
            return lab

        def _bind_combo(combo: QComboBox, var: _Var, values: list, on_change, *, tip: str = "") -> None:
            configure_combo_list_view(combo)
            combo.addItems([str(v) for v in values])
            combo.setCurrentText(str(var.get()))
            combo.setMaximumWidth(110)
            if tip:
                combo.setToolTip(tip)
            on_change = self._wrap_change(on_change)

            def _sync(_idx: int = -1) -> None:
                text = combo.currentText()
                cur = var.get()
                if isinstance(cur, bool):
                    var.set(text.lower() in ("1", "true", "yes"))
                elif isinstance(cur, int) and not isinstance(cur, bool):
                    try:
                        var.set(int(text))
                    except ValueError:
                        var.set(text)
                elif isinstance(cur, float):
                    try:
                        var.set(float(text))
                    except ValueError:
                        var.set(text)
                else:
                    var.set(text)
                on_change()

            connect_combo_deferred(combo, _sync)

            def _refresh() -> None:
                combo.blockSignals(True)
                combo.setCurrentText(str(var.get()))
                combo.blockSignals(False)

            self._widget_refreshers.append(_refresh)

        def _bind_check(cb: QCheckBox, var: _Var, on_change, *, tip: str = "") -> None:
            cb.setChecked(bool(var.get()))
            if tip:
                cb.setToolTip(tip)
            on_change = self._wrap_change(on_change)

            def _sync(_state: int = 0) -> None:
                var.set(bool(cb.isChecked()))
                on_change()

            cb.stateChanged.connect(_sync)

            def _refresh() -> None:
                cb.blockSignals(True)
                cb.setChecked(bool(var.get()))
                cb.blockSignals(False)

            self._widget_refreshers.append(_refresh)

        def _bind_line(
            edit: QLineEdit, var: _Var, on_change, *, as_float=False, as_int=False, tip: str = ""
        ) -> None:
            edit.setText(str(var.get()))
            edit.setMaximumWidth(72)
            if tip:
                edit.setToolTip(tip)
            on_change = self._wrap_change(on_change)

            def _apply() -> None:
                text = edit.text().strip()
                try:
                    if as_int:
                        var.set(int(float(text)))
                    elif as_float:
                        var.set(float(text))
                    else:
                        var.set(text)
                except Exception:
                    edit.setText(str(var.get()))
                    return
                on_change()

            def _flush_only() -> None:
                """仅把框内文字写入 var，不触发正演（供按钮点击前批量提交）。"""
                text = edit.text().strip()
                try:
                    if as_int:
                        var.set(int(float(text)))
                    elif as_float:
                        var.set(float(text))
                    else:
                        var.set(text)
                except Exception:
                    edit.setText(str(var.get()))

            edit.editingFinished.connect(_apply)
            self._param_line_flushers.append(_flush_only)

            def _refresh() -> None:
                edit.blockSignals(True)
                edit.setText(str(var.get()))
                edit.blockSignals(False)

            self._widget_refreshers.append(_refresh)

        def _add_line(
            group: str,
            label: str,
            var: _Var,
            *,
            as_float: bool = False,
            as_int: bool = False,
            slot=None,
            width: int = 56,
            equi_only: bool = False,
        ) -> QLineEdit:
            tip = _tip(label)
            lab = _lab(label)
            strip.add_group_widget(group, lab, equi_only=equi_only)
            edit = QLineEdit()
            edit.setMaximumWidth(width)
            _bind_line(edit, var, slot, as_float=as_float, as_int=as_int, tip=tip)
            strip.add_group_widget(group, edit, equi_only=equi_only)
            return edit

        # ---- 通用 ----
        strip.add_group_widget("common", _lab("时差模式"))
        cb_mode = QComboBox()
        _bind_combo(
            cb_mode,
            self.theory_mode,
            ["1D", "2D", "2Dequi"],
            self._on_theory_mode_ui_changed,
            tip=_tip("时差模式"),
        )
        strip.add_group_widget("common", cb_mode)

        for text, var, slot in (
            ("共享y轴", self.share_y_var, self.redraw),
            ("严格配对", self.strict_diff_pair, self._reload_if_loaded),
            ("强制重算", self.force_recompute, self.redraw),
            ("按r.in保留组过滤", self.use_rin_enabled_filter, self._reload_if_loaded),
        ):
            cb = QCheckBox(text)
            _bind_check(cb, var, slot, tip=_tip(text))
            strip.add_group_widget("common", cb)

        strip.add_group_widget("common", _lab("校正策略"))
        cb_pick = QComboBox()
        _bind_combo(cb_pick, self.picked_policy, ["插值", "严格", "宽松"], self.redraw, tip=_tip("校正策略"))
        strip.add_group_widget("common", cb_pick)

        _add_line("common", "PSP相位号", self.psp_phase_id, as_int=True, slot=self._reload_if_loaded, width=56)
        _add_line("common", "OBS标注Y(s)", self.obs_mark_y, as_float=True, slot=self.redraw, width=56)
        strip.add_group_stretch("common")

        # ---- 1D ----
        _add_line("1D", "厚度", self.h_cr, as_float=True, slot=self._on_model_param_changed, width=56)
        _add_line("1D", "Vp", self.vp_cr, as_float=True, slot=self._on_model_param_changed, width=56)
        _add_line("1D", "Vs", self.vs_cr, as_float=True, slot=self._on_model_param_changed, width=56)

        strip.add_group_widget("1D", _lab("线性窗口"))
        cb_wp = QComboBox()
        _bind_combo(
            cb_wp, self.window_points, [7, 9, 11, 13, 15], self._on_model_param_changed, tip=_tip("线性窗口")
        )
        strip.add_group_widget("1D", cb_wp)

        strip.add_group_widget("1D", _lab("平滑半窗"))
        cb_sm = QComboBox()
        _bind_combo(
            cb_sm,
            self.smooth_dense_half_win,
            ["0", "3", "5", "8", "10"],
            self._on_model_param_changed,
            tip=_tip("平滑半窗"),
        )
        strip.add_group_widget("1D", cb_sm)

        strip.add_group_widget("1D", _lab("二维参数场"))
        cb_sf = QComboBox()
        _bind_combo(cb_sf, self.section_field_mode, ["point", "eff"], self.redraw, tip=_tip("二维参数场"))
        strip.add_group_widget("1D", cb_sf)
        strip.add_group_stretch("1D")

        # ---- 2D / 2Dequi（同一行；2Dequi 时标题为【2Dequi】并显示等效参数）----
        cb_fb = QCheckBox("2D失败回退")
        _bind_check(cb_fb, self.theory2d_auto_fallback, self.redraw, tip=_tip("2D失败回退"))
        strip.add_group_widget("fwd", cb_fb)
        self._pois_left_edit = _add_line(
            "fwd", "pois左支", self.pois_left_var, slot=self._write_pois_left_and_run, width=110
        )
        self._pois_right_edit = _add_line(
            "fwd", "pois右支", self.pois_right_var, slot=self._write_pois_right_and_run, width=110
        )
        _add_line(
            "fwd",
            "PPS/PSS",
            self.pps_pss_ratio,
            as_float=True,
            slot=self._on_pps_pss_ratio_changed,
            width=48,
            equi_only=True,
        )
        cb_equi = QCheckBox("2Dequi写等效PSP")
        _bind_check(cb_equi, self.equi_write_equiv_psp, self._on_equi_equiv_psp_changed, tip=_tip("2Dequi写等效PSP"))
        strip.add_group_widget("fwd", cb_equi, equi_only=True)

        btn_save_pois = QPushButton("保存 pois")
        btn_save_pois.setToolTip(
            "同时保存左支与右支 pois 到走时目录下的 pois_branches.json（并写入工区 analysis）。"
        )
        btn_save_pois.clicked.connect(self.save_pois_branches)
        strip.add_group_widget("fwd", btn_save_pois)
        strip.add_group_stretch("fwd")

        self._update_param_groups_visibility()

    def _update_param_groups_visibility(self) -> None:
        self.param_strip.set_mode_visibility(str(self.theory_mode.get()))

    def _on_theory_mode_ui_changed(self) -> None:
        self._update_param_groups_visibility()
        self._on_theory_mode_changed()

    def _refresh_param_widgets(self) -> None:
        for fn in self._widget_refreshers:
            try:
                fn()
            except Exception:
                pass
        self._update_param_groups_visibility()

    # ------------------------------------------------------------------ stages / IO panels
    def _shortcut_add_tx(self) -> None:
        """Ctrl+O：切到输入页并弹出添加走时文件。"""
        try:
            self.stage_bar.setCurrentIndex(0)
        except Exception:
            pass
        self.panel_input._browse_add_tx()

    def _apply_inputs_to_plot(self) -> None:
        raw = self.panel_input.tx_paths()
        files = [Path(p) for p in raw if Path(p).is_file()]
        missing = [p for p in raw if not Path(p).is_file()]
        if not files:
            show_modeless_message(
                "缺少走时文件",
                "请在输入页添加至少一个有效的 tx.in。",
                icon=QMessageBox.Icon.Warning,
            )
            return
        self.files = files

        sea = self.panel_input.seafloor_path()
        if sea:
            if Path(sea).is_file():
                self._load_seafloor_depth_from_file(Path(sea), show_error=True)
            else:
                show_modeless_message("地形文件不存在", sea, icon=QMessageBox.Icon.Warning)
        else:
            self.seafloor_path = None
            self.seafloor_x = None
            self.seafloor_z = None

        obs = self.panel_input.obs_depth_path()
        if obs:
            if Path(obs).is_file():
                self._load_shot_depth_from_file(Path(obs), show_error=True)
            else:
                show_modeless_message("OBS/炮点深度文件不存在", obs, icon=QMessageBox.Icon.Warning)
        else:
            self.shot_depth_path = None
            self.shot_depth_x = None
            self.shot_depth_z = None
            self.shot_depth_ids = None

        rin = self.panel_input.rin_path()
        self._rin_editor_path = Path(rin) if rin else None

        self._refresh_input_survey_map()
        self._load_and_draw()
        msg = f"已加载 {len(files)} 个走时文件"
        if missing:
            msg += f"；跳过不存在 {len(missing)} 个"
        self.panel_input.set_status(msg)
        self.status_var.set(msg)
        self.stage_bar.setCurrentIndex(1)
        self._mark_dirty()

    def _schedule_survey_refresh(self) -> None:
        """延迟刷新输入页工区剖面。"""
        QTimer.singleShot(50, self._refresh_input_survey_map)

    def _refresh_input_survey_map(self) -> None:
        """用主窗已加载的水深/OBS 数组刷新输入页剖面。"""
        self.panel_input.update_survey_map(
            seafloor_x=self.seafloor_x,
            seafloor_z=self.seafloor_z,
            obs_x=self.shot_depth_x,
            obs_z=self.shot_depth_z,
            obs_labels=self.shot_depth_ids,
        )

    def _on_psp_mode_changed(self, mode: str) -> None:
        self.psp_export_mode.set(str(mode))
        self._mark_dirty()

    def _open_outputs_folder(self) -> None:
        if not self.project.workdir:
            show_modeless_message(
                "未打开工区",
                "请先新建或打开工区，outputs/ 位于工区目录下。",
                icon=QMessageBox.Icon.Warning,
            )
            return
        out = Path(self.project.workdir) / OUTPUTS_DIR
        out.mkdir(parents=True, exist_ok=True)
        ok, msg = open_path_in_file_manager(out)
        if ok:
            self.panel_output.set_status(f"已打开：{out}")
            self.statusBar().showMessage(f"已打开：{out}", 4000)
        else:
            show_modeless_message("打开文件夹", msg, icon=QMessageBox.Icon.Warning)
            self.panel_output.set_status(msg)

    def _sync_input_panel(self) -> None:
        """把主窗运行态路径推到输入页（打开工区 / 新建后用）。"""
        self.panel_input.sync_from_paths(
            tx_files=self.files,
            seafloor=self.seafloor_path,
            obs_depth=self.shot_depth_path,
            rin=self._rin_editor_path,
        )
        if self.project.workdir:
            self.panel_input.set_status(f"工区：{self.project.workdir}")
            self.panel_output.set_status(f"工区：{self.project.workdir}")
        self.panel_output.set_psp_mode(str(self.psp_export_mode.get()))

    def _mirror_panel_paths_to_runtime(self) -> None:
        """把输入页当前路径同步到运行态，避免保存后再 sync 把界面冲掉。"""
        sea = self.panel_input.seafloor_path().strip()
        obs = self.panel_input.obs_depth_path().strip()
        rin = self.panel_input.rin_path().strip()

        if sea:
            old = str(self.seafloor_path) if self.seafloor_path else ""
            if old != sea:
                self.seafloor_path = Path(sea)
                self.seafloor_x = None
                self.seafloor_z = None
        else:
            self.seafloor_path = None
            self.seafloor_x = None
            self.seafloor_z = None

        if obs:
            old = str(self.shot_depth_path) if self.shot_depth_path else ""
            if old != obs:
                self.shot_depth_path = Path(obs)
                self.shot_depth_x = None
                self.shot_depth_z = None
                self.shot_depth_ids = None
        else:
            self.shot_depth_path = None
            self.shot_depth_x = None
            self.shot_depth_z = None
            self.shot_depth_ids = None

        self._rin_editor_path = Path(rin) if rin else None

    # ------------------------------------------------------------------ project
    def _update_title(self) -> None:
        title = "iphase — 震相分析工区"
        if self.project.workdir:
            mark = " *" if self.project.dirty else ""
            title += f" [{self.project.name or Path(self.project.workdir).name}]{mark}"
        self.setWindowTitle(title)

    def _update_path_badge(self) -> None:
        if self.project.workdir:
            name = self.project.name or Path(self.project.workdir).name
            if self.files:
                extra = self.files[0].name if len(self.files) == 1 else f"{len(self.files)} 个 tx.in"
                text = f"{name} · {extra}"
            else:
                text = name
            set_path_badge(self._path_badge, active=True, text=text)
            return
        if not self.files:
            set_path_badge(self._path_badge, active=False, text="未打开工区")
            return
        text = self.files[0].name if len(self.files) == 1 else f"{len(self.files)} 个 tx.in"
        set_path_badge(self._path_badge, active=True, text=text)

    def _confirm_discard_project(self) -> bool:
        if not self.project.dirty:
            return True
        ans = QMessageBox.question(
            self,
            "未保存",
            "当前工区有未保存更改，继续将丢弃，是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return ans == QMessageBox.StandardButton.Yes

    def _ui_to_project(self) -> None:
        # 输入页路径优先写回工程
        tx_from_panel = self.panel_input.tx_paths()
        if tx_from_panel:
            self.files = [Path(p) for p in tx_from_panel if p]

        a = self.project.analysis
        a.theory_mode = str(self.theory_mode.get())
        a.psp_export_mode = str(self.psp_export_mode.get())
        a.share_y = bool(self.share_y_var.get())
        a.window_points = int(self.window_points.get())
        a.smooth_half_win = str(self.smooth_dense_half_win.get())
        a.psp_phase_id = int(self.psp_phase_id.get())
        a.picked_policy = str(self.picked_policy.get())
        a.strict_diff_pair = bool(self.strict_diff_pair.get())
        a.force_recompute = bool(self.force_recompute.get())
        a.theory2d_auto_fallback = bool(self.theory2d_auto_fallback.get())
        a.use_rin_enabled_filter = bool(self.use_rin_enabled_filter.get())
        a.h_cr = float(self.h_cr.get())
        a.vp_cr = float(self.vp_cr.get())
        a.vs_cr = float(self.vs_cr.get())
        a.obs_mark_y = float(self.obs_mark_y.get())
        a.pois_left = str(self.pois_left_var.get())
        a.pois_right = str(self.pois_right_var.get())
        a.pps_pss_ratio = float(self.pps_pss_ratio.get())
        a.equi_write_equiv_psp = bool(self.equi_write_equiv_psp.get())
        a.section_field_mode = str(self.section_field_mode.get())

        wf = self.project.workflow
        wf.tx_files = [self.project.to_rel_or_abs(str(p)) for p in self.files]
        self._mirror_panel_paths_to_runtime()
        sea = str(self.seafloor_path) if self.seafloor_path else ""
        obs = str(self.shot_depth_path) if self.shot_depth_path else ""
        rin = str(self._rin_editor_path) if self._rin_editor_path else ""
        # 若运行态为空，仍以输入页为准（兼容未点「加载」仅填路径）
        sea = sea or self.panel_input.seafloor_path().strip()
        obs = obs or self.panel_input.obs_depth_path().strip()
        rin = rin or self.panel_input.rin_path().strip()
        wf.seafloor_path = self.project.to_rel_or_abs(sea)
        wf.shot_depth_path = self.project.to_rel_or_abs(obs)
        wf.rin_path = self.project.to_rel_or_abs(rin)

    def _project_to_ui(self) -> None:
        self._loading_project = True
        try:
            a = self.project.analysis
            self.theory_mode.set(a.theory_mode)
            self.psp_export_mode.set(a.psp_export_mode)
            self.share_y_var.set(bool(a.share_y))
            self.window_points.set(int(a.window_points))
            self.smooth_dense_half_win.set(str(a.smooth_half_win))
            self.psp_phase_id.set(int(a.psp_phase_id))
            self.picked_policy.set(str(a.picked_policy))
            self.strict_diff_pair.set(bool(a.strict_diff_pair))
            self.force_recompute.set(bool(a.force_recompute))
            self.theory2d_auto_fallback.set(bool(a.theory2d_auto_fallback))
            self.use_rin_enabled_filter.set(bool(a.use_rin_enabled_filter))
            self.h_cr.set(float(a.h_cr))
            self.vp_cr.set(float(a.vp_cr))
            self.vs_cr.set(float(a.vs_cr))
            self.obs_mark_y.set(float(a.obs_mark_y))
            self.pois_left_var.set(str(a.pois_left))
            self.pois_right_var.set(str(a.pois_right))
            self.pps_pss_ratio.set(float(a.pps_pss_ratio))
            self.equi_write_equiv_psp.set(bool(a.equi_write_equiv_psp))
            self.section_field_mode.set(str(a.section_field_mode))
            self._refresh_param_widgets()

            wf = self.project.workflow
            files: list[Path] = []
            for item in wf.tx_files:
                p = Path(self.project.abs_or_join(item))
                if p.is_file():
                    files.append(p)
            self.files = files

            sea = self.project.abs_or_join(wf.seafloor_path)
            if sea and Path(sea).is_file():
                self._load_seafloor_depth_from_file(Path(sea), show_error=False)
            else:
                self.seafloor_path = None
                self.seafloor_x = None
                self.seafloor_z = None

            shot = self.project.abs_or_join(wf.shot_depth_path)
            if shot and Path(shot).is_file():
                self._load_shot_depth_from_file(Path(shot), show_error=False)
            else:
                self.shot_depth_path = None
                self.shot_depth_x = None
                self.shot_depth_z = None
                self.shot_depth_ids = None

            rin = self.project.abs_or_join(wf.rin_path)
            self._rin_editor_path = Path(rin) if rin else None

            self._sync_input_panel()
            self._refresh_input_survey_map()

            if self.files:
                self._load_and_draw()
                self.stage_bar.setCurrentIndex(1)
            else:
                self.results = []
                if hasattr(self, "fig"):
                    self.fig.clear()
                    self.canvas.draw_idle()
                self.status_var.set("工区已打开，请在输入页添加走时文件")
                self.stage_bar.setCurrentIndex(0)
            self._update_path_badge()
            self._update_title()
        finally:
            self._loading_project = False

    def _new_project(self) -> None:
        if not self._confirm_discard_project():
            return
        path = QFileDialog.getExistingDirectory(self, "选择新建工区目录", "")
        if not path:
            return
        name, ok = QInputDialog.getText(self, "工区名称", "名称：", text=Path(path).name)
        if not ok:
            return
        try:
            self.project = IphaseProject.create_new(path, name=str(name or ""))
            self.project.save()
        except Exception as exc:
            show_modeless_message("新建失败", str(exc), icon=QMessageBox.Icon.Critical)
            return
        self.files = []
        self.results = []
        self.seafloor_path = None
        self.seafloor_x = None
        self.seafloor_z = None
        self.shot_depth_path = None
        self.shot_depth_x = None
        self.shot_depth_z = None
        self.shot_depth_ids = None
        self._rin_editor_path = None
        if hasattr(self, "fig"):
            self.fig.clear()
            self.canvas.draw_idle()
        self._sync_input_panel()
        self._refresh_input_survey_map()
        self.stage_bar.setCurrentIndex(0)
        self.status_var.set(f"已新建工区：{self.project.workdir}")
        self._update_path_badge()
        self._update_title()

    def _open_project(self) -> None:
        if not self._confirm_discard_project():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开 iphase 工区",
            "",
            "iphase project (iphase_project.json);;JSON (*.json);;All (*.*)",
        )
        if not path:
            return
        try:
            self.project = IphaseProject.load(path)
            self.project.ensure_workdir()
        except Exception as exc:
            show_modeless_message("打开失败", str(exc), icon=QMessageBox.Icon.Critical)
            return
        self._project_to_ui()
        self.status_var.set(f"已打开工区：{self.project.workdir}")

    def _try_open_startup_project(self) -> bool:
        candidates: list[str] = []
        env = os.environ.get("PYAOBS_IPHASE_PROJECT", "").strip()
        if env:
            candidates.append(env)
        for a in self._startup_argv[1:]:
            s = str(a).strip()
            if not s or s.startswith("-"):
                continue
            candidates.append(s)
        for raw in candidates:
            jp = IphaseProject.resolve_open_path(raw)
            if jp:
                try:
                    self.project = IphaseProject.load(jp)
                    self.project.ensure_workdir()
                except Exception as exc:
                    self.status_var.set(f"启动打开工区失败：{exc}")
                    continue
                self._project_to_ui()
                self.status_var.set(f"已打开工区：{self.project.workdir}")
                return True
            p = Path(raw)
            if p.is_dir():
                try:
                    self.project = IphaseProject.create_new(str(p.resolve()))
                except Exception:
                    continue
                self.files = []
                self._sync_input_panel()
                self.status_var.set(f"已绑定工区目录（尚未保存工程 JSON）：{p}")
                self._update_path_badge()
                self._update_title()
                return True
        return False

    def _startup_restore(self) -> None:
        if self._try_open_startup_project():
            self._startup_opened_project = True
            return
        self._restore_workbench_state()

    def _save_project(self) -> None:
        if not self.project.workdir:
            path = QFileDialog.getExistingDirectory(self, "选择工区保存目录", "")
            if not path:
                return
            name, ok = QInputDialog.getText(self, "工区名称", "名称：", text=Path(path).name)
            if not ok:
                return
            self.project.workdir = str(Path(path).resolve())
            self.project.name = str(name or Path(path).name)
            try:
                self.project.ensure_workdir()
            except Exception as exc:
                show_modeless_message("保存工区失败", str(exc), icon=QMessageBox.Icon.Critical)
                return
        self._ui_to_project()
        try:
            jp = self.project.save()
        except Exception as exc:
            show_modeless_message("保存工区失败", str(exc), icon=QMessageBox.Icon.Critical)
            return
        self._update_title()
        self._update_path_badge()
        # 不要用过期运行态冲掉输入页；路径已在 _ui_to_project 里镜像
        if self.project.workdir:
            self.panel_input.set_status(f"工区：{self.project.workdir}")
            self.panel_output.set_status(f"工区：{self.project.workdir}")
        self.status_var.set(f"工区已保存：{jp}")
        self.statusBar().showMessage("工区已保存", 3000)

    def _load_and_draw(self) -> None:
        plot = getattr(self, "plot", None)
        if plot is not None and hasattr(plot, "clear_saved_views"):
            try:
                plot.clear_saved_views()
            except Exception:
                pass
        super()._load_and_draw()
        self._mark_dirty()
        self._update_path_badge()

    def closeEvent(self, event: QCloseEvent) -> None:
        try:
            if self.project.workdir and self.project.dirty:
                ans = QMessageBox.question(
                    self,
                    "退出",
                    "工区有未保存更改，是否保存后退出？",
                    QMessageBox.StandardButton.Save
                    | QMessageBox.StandardButton.Discard
                    | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Save,
                )
                if ans == QMessageBox.StandardButton.Cancel:
                    event.ignore()
                    return
                if ans == QMessageBox.StandardButton.Save:
                    self._save_project()
                    if self.project.dirty:
                        event.ignore()
                        return
            self._save_workbench_state()
        except Exception:
            pass
        event.accept()


def run_iphase_app(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv
    # WSL/Ubuntu Wayland 下最大化主窗 + 独立提示框易触发 xdg_wm_base 协议崩溃；
    # 未指定 QT_QPA_PLATFORM 时优先 xcb（可 export QT_QPA_PLATFORM=wayland 覆盖）。
    try:
        from pyAOBS.utils.qt_platform import prefer_xcb_on_wayland

        prefer_xcb_on_wayland()
    except Exception:
        pass
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication.instance() or QApplication(argv)
    apply_iphase_font(app)
    win = IPhaseMainWindow(argv)
    win.show()
    return int(app.exec())
