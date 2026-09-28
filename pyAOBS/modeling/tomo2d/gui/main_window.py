"""tomo2d Qt 主窗：顶栏 + 9 流程页签 + 预览/日志。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QCloseEvent, QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..param_hints import get_param_tooltip
from .dialog_utils import file_dialog_options, show_modeless_message
from .dialogs.help_dialog import install_help_shortcut, open_program_help
from .dialogs.inv_analysis_dialog import open_inv_analysis_dialog
from .dialogs.inv_monitor_dialog import open_inv_monitor_dialog
from .dialogs.model_compare_dialog import open_model_compare_dialog
from .dialogs.model_picker_dialog import open_model_picker_dialog
from .dialogs.qc_result_dialog import open_mc_profile_dialog
from .dialogs.smesh_plot import plot_smesh_velocity_qt
from .dialogs.ttimes_plot import open_ttimes_preview, open_tx_in_preview
from .services.smesh_plot_core import CMD_TAB_PLOT_SOURCES, lookup_plot_sources_for_tab
from .services.ui_prefs import restore_window_layout, save_window_layout
from .panels.checkerboard_tab import CheckerboardTab
from .panels.edit_smesh_tab import EditSmeshTab
from .panels.gen_damp_tab import GenDampTab
from .panels.gen_dcorr_tab import GenDcorrTab
from .panels.gen_smesh_tab import GenSmeshTab
from .panels.gen_vcorr_tab import GenVcorrTab
from .panels.log_panel import LogPanel
from .panels.monte_carlo_tab import MonteCarloTab
from .panels.output_panel import OutputPanel, list_output_tree
from .panels.pipeline_tab import PipelineTab
from .panels.preview_panel import PreviewPanel
from .panels.stat_smesh_tab import StatSmeshTab
from .panels.top_chrome import TopChromePanel
from .panels.tt_forward_tab import TtForwardTab
from .panels.tt_inverse_tab import TtInverseTab
from .panels.tx_convert_tab import TxConvertTab
from .panels.wave2d_tab import Wave2dTab
from .services.workflow_bridge import (
    apply_fwd_outputs_to_inv,
    fill_inv_from_upstream,
    sync_ray_params,
)
from .project import Tomo2dProject
from .services import (
    append_gui_file_log,
    apply_manifest_to_state,
    collect_run_env,
    format_parallel_status_line,
    get_tomo,
    load_last_project_path,
    load_workbench_profile,
    project_path_from_env,
    normalize_file_path_vars,
    project_json_path,
    read_profile_json,
    resolve_work_dir,
    save_workbench_profile,
    sync_fwd_smesh_from_inv_if_missing,
    validate_work_dir,
)
from .services.workflow import (
    PreparedRun,
    prepare_edit_smesh,
    prepare_gen_damp,
    prepare_gen_dcorr,
    prepare_gen_smesh,
    prepare_gen_vcorr,
    prepare_pipeline_simple,
    prepare_stat_smesh,
    prepare_tt_forward,
    prepare_tt_inverse,
    prepare_tx_convert,
    preview_edit_smesh,
    preview_gen_damp,
    preview_gen_dcorr,
    preview_gen_smesh,
    preview_gen_vcorr,
    preview_pipeline,
    preview_stat_smesh,
    preview_tt_forward,
    preview_tt_inverse,
    preview_tx_convert,
)
from .services.qc_workflows import (
    preview_checkerboard,
    preview_monte_carlo,
    run_checkerboard_test,
    run_monte_carlo,
)
from .services.wave2d_gather import prepare_wave2d, preview_wave2d
from .state.form_state import FormState
from .styles import apply_tomo2d_chrome
from .workers import start_command_worker


def _stream_log_tag(stream: str, text: str) -> str:
    """stderr 上既有状态提示也有真正错误；勿一律标 [err]。"""
    if stream != "stderr":
        return "out"
    low = (text or "").strip().lower()
    if any(
        k in low
        for k in (
            "error(",
            "error:",
            " invalid",
            "failed",
            "fatal",
            "usage:",
            "can't open",
            "cannot ",
            "mismatch",
            "too many options",
        )
    ):
        return "err"
    return "info"


class Tomo2DMainWindow(QMainWindow):
    def __init__(self, argv: list[str] | None = None) -> None:
        super().__init__()
        self.setWindowTitle("TOMO2D Workflow GUI (Qt)")
        self.resize(1360, 860)
        self._startup_argv = list(argv if argv is not None else sys.argv)

        self.state = FormState()
        from .services.bin_defaults import default_bin_path

        self.state.set("bin_path", default_bin_path())
        self.state.set("work_dir", str(Path.cwd()))
        self.state.set("gui.write_file_log", True)
        from .services.io_defaults import ensure_quiet_io_defaults

        ensure_quiet_io_defaults(self.state)

        self.project = Tomo2dProject()
        self._suppress_dirty = False
        self._run_thread = None
        self._run_worker = None
        self._active_tomo = None
        self._job_streamed = False
        self._tabs: list[Any] = []
        self._run_elapsed_timer: QTimer | None = None
        self._run_elapsed_start: float | None = None
        self._run_elapsed_title = ""

        self._build_ui()
        from .services.model_compare import register_form_apply_hook

        register_form_apply_hook(self._on_model_form_written)
        self.setAcceptDrops(True)
        self.menuBar().hide()
        apply_tomo2d_chrome(self)
        install_help_shortcut(self)
        self._refresh_window_title()
        self.statusBar().showMessage(
            "就绪 — 可新建/打开工区；smesh 可拖入，或在资源管理器复制后到绘制窗 Ctrl+V（F1 帮助）"
        )
        QTimer.singleShot(150, self._restore_workbench_state)

    def _try_open_startup_project(self) -> bool:
        candidates: list[str] = []
        env_p = project_path_from_env()
        if env_p is not None:
            candidates.append(str(env_p))
        for a in self._startup_argv[1:]:
            s = str(a).strip()
            if not s or s.startswith("-"):
                continue
            candidates.append(s)
        for raw in candidates:
            jp = Tomo2dProject.resolve_open_path(raw)
            if jp:
                try:
                    self._load_project_path(jp)
                    return bool(self.project.is_open)
                except Exception:
                    continue
            p = Path(raw)
            if p.is_dir():
                try:
                    self.project = Tomo2dProject.create_new(str(p.resolve()))
                    self._apply_project_to_ui()
                    self.log.log(
                        f"已绑定工区目录（尚未保存工程 JSON）：{p}",
                        write_file=False,
                    )
                    return True
                except Exception:
                    continue
        return False

    def _build_ui(self) -> None:
        self.chrome = TopChromePanel(self.state, self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.chrome)
        self.chrome.new_project_requested.connect(self.new_project)
        self.chrome.open_project_requested.connect(self.open_project)
        self.chrome.save_project_requested.connect(self.save_project)
        self.chrome.save_profile_requested.connect(self.save_profile)
        self.chrome.load_profile_requested.connect(self.load_profile)
        self.chrome.plot_smesh_requested.connect(self.plot_smesh)
        self.chrome.inv_analysis_requested.connect(self.open_inv_analysis)
        self.chrome.inv_monitor_requested.connect(self.open_inv_monitor)
        self.chrome.model_picker_requested.connect(self.open_model_picker)
        self.chrome.model_compare_requested.connect(self.open_model_compare)
        self.chrome.help_requested.connect(self.open_help)
        self.chrome.exit_requested.connect(self.close)
        self._inv_monitor = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        self._main_split = QSplitter(Qt.Orientation.Horizontal)

        # 左：竖排命令导航（文字横向完整显示）
        self.nav = QListWidget()
        self.nav.setObjectName("tomoCmdNav")
        self.nav.setMinimumWidth(168)
        self.nav.setMaximumWidth(220)
        self.nav.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.nav.setMouseTracking(True)

        # 中：参数面板（与右侧预览/日志可拖曳调宽，不设最大宽度）
        form_col = QWidget()
        form_col.setObjectName("tomoFormCol")
        form_col.setMinimumWidth(360)
        form_lay = QVBoxLayout(form_col)
        form_lay.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedWidget()
        self.tabs = self.stack  # 兼容 currentIndex / count

        self.gen_tab = GenSmeshTab(self.state)
        self.fwd_tab = TtForwardTab(self.state)
        self.damp_tab = GenDampTab(self.state)
        self.vcorr_tab = GenVcorrTab(self.state)
        self.dcorr_tab = GenDcorrTab(self.state)
        self.inv_tab = TtInverseTab(self.state)
        self.stat_tab = StatSmeshTab(self.state)
        self.edit_tab = EditSmeshTab(self.state)
        self.pipe_tab = PipelineTab(self.state)
        self.tx_tab = TxConvertTab(self.state)
        self.cb_tab = CheckerboardTab(self.state)
        self.mc_tab = MonteCarloTab(self.state)
        self.wave_tab = Wave2dTab(self.state)
        self._tabs = [
            self.gen_tab,
            self.fwd_tab,
            self.damp_tab,
            self.vcorr_tab,
            self.dcorr_tab,
            self.inv_tab,
            self.stat_tab,
            self.edit_tab,
            self.pipe_tab,
            self.tx_tab,
            self.cb_tab,
            self.mc_tab,
            self.wave_tab,
        ]
        titles = [
            ("1) gen_smesh", "tab.gen_smesh"),
            ("2) tt_forward", "tab.tt_forward"),
            ("3) gen_damp", "tab.gen_damp"),
            ("4) gen_vcorr", "tab.gen_vcorr"),
            ("5) gen_dcorr", "tab.gen_dcorr"),
            ("6) tt_inverse", "tab.tt_inverse"),
            ("7) stat_smesh", "tab.stat_smesh"),
            ("8) edit_smesh_HHB", "tab.edit_smesh"),
            ("9) pipeline", "tab.pipeline"),
            ("10) tx.in→tomo2d", "tab.tx_convert"),
            ("11) 棋盘格测试", "tab.checkerboard"),
            ("12) 蒙特卡洛", "tab.monte_carlo"),
            ("13) wave2d", "tab.wave2d"),
        ]
        if len(titles) != len(CMD_TAB_PLOT_SOURCES) or len(titles) != len(self._tabs):
            raise RuntimeError("命令页签与 CMD_TAB_PLOT_SOURCES 数量不一致")
        for (t, hint_key), src, w in zip(titles, CMD_TAB_PLOT_SOURCES, self._tabs):
            item = QListWidgetItem(t)
            tip = get_param_tooltip(hint_key)
            if tip:
                item.setToolTip(tip)
            self.nav.addItem(item)
            w.setProperty("cmd_tab_id", src.tab_id)
            self.stack.addWidget(w)
        self.nav.setCurrentRow(0)
        self.nav.currentRowChanged.connect(self._on_nav_row)
        form_lay.addWidget(self.stack, stretch=1)

        # 右：预览 / 日志 / 输出文件（不设最大宽度，以便与参数区拖曳分配）
        right = QWidget()
        right.setObjectName("tomoRightDock")
        right.setMinimumWidth(280)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        self.preview = PreviewPanel()
        self.log = LogPanel()
        self.log.set_file_sink(self._file_log_line)
        self.output_panel = OutputPanel()
        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(self.preview, "预览")
        self.right_tabs.addTab(self.log, "日志")
        self.right_tabs.addTab(self.output_panel, "输出")
        right_lay.addWidget(self.right_tabs, stretch=1)

        self._main_split.addWidget(self.nav)
        self._main_split.addWidget(form_col)
        self._main_split.addWidget(right)
        self._main_split.setChildrenCollapsible(False)
        self._main_split.setStretchFactor(0, 0)
        self._main_split.setStretchFactor(1, 3)
        self._main_split.setStretchFactor(2, 2)
        self._main_split.setSizes([180, 640, 480])
        root.addWidget(self._main_split)
        self.setStatusBar(QStatusBar())
        restore_window_layout(
            self, "main_window", splitters={"main": self._main_split}
        )

        self.gen_tab.preview_requested.connect(self.preview_gen_smesh)
        self.gen_tab.run_requested.connect(self.run_gen_smesh)
        self.fwd_tab.preview_requested.connect(self.preview_tt_forward)
        self.fwd_tab.run_requested.connect(self.run_tt_forward)
        self.fwd_tab.bridge_to_inv_requested.connect(self.bridge_fwd_to_inv)
        self.inv_tab.fill_upstream_requested.connect(self.fill_inv_from_upstream)
        self.inv_tab.sync_ray_from_fwd_requested.connect(self.sync_inv_ray_from_fwd)
        self.inv_tab.go_damp_requested.connect(lambda: self.nav.setCurrentRow(2))
        self.inv_tab.go_vcorr_requested.connect(lambda: self.nav.setCurrentRow(3))
        self.inv_tab.go_dcorr_requested.connect(lambda: self.nav.setCurrentRow(4))
        self.output_panel.refresh_requested.connect(self.refresh_output_panel)
        self.output_panel.open_inputs_requested.connect(
            lambda: self._open_work_subdir("inputs")
        )
        self.output_panel.open_outputs_requested.connect(
            lambda: self._open_work_subdir("outputs")
        )
        self.output_panel.open_runs_requested.connect(
            lambda: self._open_work_subdir("runs")
        )
        self.output_panel.plot_smesh_requested.connect(self.plot_smesh)
        self.output_panel.inv_analysis_requested.connect(self.open_inv_analysis)
        self.output_panel.model_picker_requested.connect(self.open_model_picker)
        self.output_panel.model_compare_requested.connect(self.open_model_compare)
        self.output_panel.file_activated.connect(self._open_output_file)
        QTimer.singleShot(0, self.refresh_output_panel)
        self.damp_tab.preview_requested.connect(self.preview_gen_damp)
        self.damp_tab.run_requested.connect(self.run_gen_damp)
        self.vcorr_tab.preview_requested.connect(self.preview_gen_vcorr)
        self.vcorr_tab.run_requested.connect(self.run_gen_vcorr)
        self.dcorr_tab.preview_requested.connect(self.preview_gen_dcorr)
        self.dcorr_tab.run_requested.connect(self.run_gen_dcorr)
        self.inv_tab.preview_requested.connect(self.preview_tt_inverse)
        self.inv_tab.run_requested.connect(self.run_tt_inverse)
        self.stat_tab.preview_requested.connect(self.preview_stat_smesh)
        self.stat_tab.run_requested.connect(self.run_stat_smesh)
        self.edit_tab.preview_requested.connect(self.preview_edit_smesh)
        self.edit_tab.run_requested.connect(self.run_edit_smesh)
        self.pipe_tab.preview_requested.connect(self.preview_pipeline)
        self.pipe_tab.run_requested.connect(self.run_pipeline)
        self.tx_tab.preview_requested.connect(self.preview_tx_convert)
        self.tx_tab.run_requested.connect(self.run_tx_convert)
        self.tx_tab.plot_tx_in_requested.connect(self.plot_tx_in)
        self.tx_tab.plot_ttimes_requested.connect(self.plot_ttimes)
        self.cb_tab.preview_requested.connect(self.preview_checkerboard)
        self.cb_tab.run_requested.connect(self.run_checkerboard)
        self.mc_tab.preview_requested.connect(self.preview_monte_carlo)
        self.mc_tab.run_requested.connect(self.run_monte_carlo)
        self.wave_tab.preview_requested.connect(self.preview_wave2d)
        self.wave_tab.run_requested.connect(self.run_wave2d)

    def _on_nav_row(self, row: int) -> None:
        if row < 0:
            return
        if 0 <= row < self.stack.count():
            self.stack.setCurrentIndex(row)

    def _refresh_window_title(self) -> None:
        if self.project.is_open:
            mark = " *" if self.project.dirty else ""
            self.setWindowTitle(
                f"TOMO2D — {self.project.name}{mark} [{self.project.workdir}]"
            )
        else:
            self.setWindowTitle("TOMO2D Workflow GUI (Qt)")

    def _sync_dynamic_tabs(self) -> None:
        for t in (
            self.gen_tab,
            self.damp_tab,
            self.vcorr_tab,
            self.dcorr_tab,
            self.stat_tab,
            self.edit_tab,
        ):
            if hasattr(t, "_sync"):
                t._sync()
            elif hasattr(t, "_sync_dynamic"):
                t._sync_dynamic()

    def _apply_project_to_ui(self) -> None:
        self._suppress_dirty = True
        try:
            self.project.apply_to_form_state(self.state)
            normalize_file_path_vars(
                self.state, self._all_file_keys(), warn_outside=False
            )
            self._push_all()
            self._sync_dynamic_tabs()
            self.project.dirty = False
            self._refresh_window_title()
        finally:
            self._suppress_dirty = False

    def _capture_ui_to_project(self) -> None:
        self._pull_all()
        normalize_file_path_vars(
            self.state, self._all_file_keys(), warn_outside=False
        )
        self.project.capture_from_form_state(self.state)

    def _confirm_discard_project(self) -> bool:
        if not self.project.dirty:
            return True
        ans = QMessageBox.question(
            self,
            "切换工区",
            "当前工区有未保存更改，继续将丢失。继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return ans == QMessageBox.StandardButton.Yes

    def new_project(self) -> None:
        if not self._confirm_discard_project():
            return
        initial = self.state.get_str("work_dir") or str(Path.cwd())
        path = QFileDialog.getExistingDirectory(
            self,
            "选择新建工区目录",
            initial,
            file_dialog_options(QFileDialog.Option.ShowDirsOnly),
        )
        if not path:
            return
        name, ok = QInputDialog.getText(
            self, "工区名称", "名称：", text=Path(path).name
        )
        if not ok:
            return
        try:
            self._pull_all()
            self.project = Tomo2dProject.create_new(path, name=str(name or ""))
            self.project.bin_path = self.state.get_str("bin_path")
            self.project.capture_from_form_state(self.state)
            self.project.workdir = str(Path(path).resolve())
            self.project.profile["work_dir"] = self.project.workdir
            self.project.save()
            self._apply_project_to_ui()
            self.log.log(f"已新建工区：{self.project.workdir}")
            self.statusBar().showMessage(f"工区已创建：{self.project.name}")
        except Exception as exc:
            show_modeless_message(
                "新建工区失败", str(exc), icon=QMessageBox.Icon.Critical
            )

    def open_project(self) -> None:
        if not self._confirm_discard_project():
            return
        initial = self.state.get_str("work_dir") or str(Path.cwd())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开 tomo2d 工区",
            initial,
            "tomo2d project (tomo2d_project.json);;JSON (*.json);;所有文件 (*)",
            options=file_dialog_options(),
        )
        if not path:
            return
        self._load_project_path(path)

    def _load_project_path(self, path: str) -> None:
        try:
            self.project = Tomo2dProject.load(path)
            self.project.ensure_workdir()
            self._apply_project_to_ui()
            self.log.log(f"已打开工区：{self.project.workdir}")
            self.statusBar().showMessage(f"工区：{self.project.name}")
        except Exception as exc:
            show_modeless_message(
                "打开工区失败", str(exc), icon=QMessageBox.Icon.Critical
            )

    def save_project(self) -> None:
        if not self.project.is_open:
            show_modeless_message(
                "保存工区",
                "请先「新建工区」或「打开工区」。\n"
                "也可使用「保存配置」导出独立 JSON。",
                icon=QMessageBox.Icon.Warning,
            )
            return
        try:
            self._capture_ui_to_project()
            out = self.project.save()
            self._refresh_window_title()
            self.log.log(f"已保存工区：{out}")
            self.statusBar().showMessage(f"已保存：{out}")
        except Exception as exc:
            show_modeless_message(
                "保存工区失败", str(exc), icon=QMessageBox.Icon.Critical
            )

    def _file_log_line(self, text: str) -> None:
        try:
            self.chrome.binder.pull_from_widgets()
        except Exception:
            pass
        append_gui_file_log(
            self.state.get_str("work_dir") or ".",
            text,
            enabled=self.state.get_bool("gui.write_file_log", True),
        )

    def _start_elapsed(self, title: str) -> None:
        import time

        self._cancel_elapsed()
        self._run_elapsed_title = title
        self._run_elapsed_start = time.monotonic()
        self._run_elapsed_timer = QTimer(self)
        self._run_elapsed_timer.setInterval(500)
        self._run_elapsed_timer.timeout.connect(self._tick_elapsed)
        self._run_elapsed_timer.start()
        self.statusBar().showMessage(f"运行 {title}… 0.0s")

    def _tick_elapsed(self) -> None:
        import time

        if self._run_elapsed_start is None:
            return
        sec = time.monotonic() - self._run_elapsed_start
        self.statusBar().showMessage(
            f"运行 {self._run_elapsed_title}… {sec:.1f}s"
        )

    def _cancel_elapsed(self) -> None:
        if self._run_elapsed_timer is not None:
            self._run_elapsed_timer.stop()
            self._run_elapsed_timer.deleteLater()
            self._run_elapsed_timer = None
        self._run_elapsed_start = None

    def _finish_elapsed(self, title: str, ok: bool) -> None:
        import time

        started = self._run_elapsed_start
        self._cancel_elapsed()
        if started is None:
            self.statusBar().showMessage(f"{title} {'完成' if ok else '失败'}")
            return
        sec = time.monotonic() - started
        self.statusBar().showMessage(
            f"{title} {'完成' if ok else '失败'}（{sec:.1f}s）"
        )
        self.log.log(f"[{title}] 耗时 {sec:.1f}s", write_file=True)

    def _restore_workbench_state(self) -> None:
        if self._try_open_startup_project():
            return
        last = load_last_project_path()
        if last and Path(last).exists():
            try:
                self._load_project_path(last)
                self.log.log(
                    f"已恢复上次工区：{self.project.workdir}", write_file=False
                )
                return
            except Exception as exc:
                self.log.log(f"恢复工区失败，改用表单配置: {exc}", write_file=False)
        profile = load_workbench_profile()
        if not profile:
            return
        try:
            self._suppress_dirty = True
            self.state.apply_mapping(profile)
            normalize_file_path_vars(
                self.state, self._all_file_keys(), warn_outside=False
            )
            self._push_all()
            self._sync_dynamic_tabs()
            self.log.log("已恢复上次 TOMO2D GUI 配置", write_file=False)
        except Exception as exc:
            self.log.log(f"恢复 TOMO2D GUI 配置失败: {exc}", write_file=False)
        finally:
            self._suppress_dirty = False

    def _save_workbench_state(self) -> None:
        try:
            self._pull_all()
            normalize_file_path_vars(
                self.state, self._all_file_keys(), warn_outside=False
            )
            last = ""
            if self.project.is_open:
                last = project_json_path(self.project.workdir)
            save_workbench_profile(
                self.state.to_profile_dict(), last_project=last or None
            )
        except Exception:
            pass

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self.project.is_open and self.project.dirty:
            ans = QMessageBox.question(
                self,
                "关闭",
                "工区有未保存更改，是否保存后再退出？",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes,
            )
            if ans == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if ans == QMessageBox.StandardButton.Yes:
                try:
                    self._capture_ui_to_project()
                    self.project.save()
                except Exception as exc:
                    show_modeless_message(
                        "保存工区失败", str(exc), icon=QMessageBox.Icon.Critical
                    )
                    event.ignore()
                    return
        self._save_workbench_state()
        try:
            save_window_layout(
                self, "main_window", splitters={"main": self._main_split}
            )
        except Exception:
            pass
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except Exception:
            pass
        super().closeEvent(event)

    def _pull_all(self) -> None:
        self.chrome.binder.pull_from_widgets()
        for t in self._tabs:
            t.pull()
        if self.project.is_open and not getattr(self, "_suppress_dirty", False):
            if not self.project.dirty:
                self.project.dirty = True
                self._refresh_window_title()

    def _push_all(self) -> None:
        self.chrome.binder.push_to_widgets()
        if hasattr(self.chrome, "cmap_combo"):
            self.chrome.cmap_combo.sync_from_state()
        if hasattr(self.chrome, "parallel"):
            self.chrome.parallel._sync_enabled()
        for t in self._tabs:
            if hasattr(t, "binder"):
                t.binder.push_to_widgets()
            if hasattr(t, "_grav_body"):
                t._grav_body.binder.push_to_widgets()
            if hasattr(t, "on_state_pushed"):
                t.on_state_pushed()

    def _all_file_keys(self) -> list[str]:
        keys: list[str] = []
        keys.extend(self.chrome.binder.file_keys())
        for t in self._tabs:
            keys.extend(t.binder.file_keys())
            if hasattr(t, "_grav_body"):
                keys.extend(t._grav_body.binder.file_keys())
        return list(dict.fromkeys(keys))

    def _prepare_state(self) -> Path:
        self._pull_all()
        normalize_file_path_vars(self.state, self._all_file_keys(), warn_outside=False)
        self._push_all()
        return validate_work_dir(self.state.get_str("work_dir"))

    def _tomo(self, work: Path):
        tomo = get_tomo(self.state.get_str("bin_path"))
        tomo.proc_cwd = str(work)
        tomo.run_env = collect_run_env(self.state)
        self._active_tomo = tomo
        return tomo

    def _busy(self) -> bool:
        if self._run_thread is not None:
            show_modeless_message(
                "忙碌中", "已有任务在运行", icon=QMessageBox.Icon.Warning
            )
            return True
        return False

    def _start_job(
        self,
        title: str,
        work: Path,
        fn: Callable[[], Any],
        *,
        finally_fn: Callable[[Any, Any], None] | None = None,
    ) -> None:
        tomo = self._active_tomo
        worker_box: dict[str, Any] = {"w": None}
        self.log.log(f"开始 {title} @ {work}")
        if tomo is not None and tomo.run_env:
            pline = format_parallel_status_line(tomo.run_env)
            if pline:
                self.log.log(pline)
        self._start_elapsed(title)
        self._job_streamed = False

        def job():
            if tomo is not None:
                w = worker_box["w"]

                def _emit(stream: str, line: str) -> None:
                    if w is not None:
                        w.output_line.emit(stream, line)

                tomo.stream_output_line = _emit
            try:
                return fn()
            finally:
                if tomo is not None:
                    tomo.stream_output_line = None

        def on_line(stream: str, text: str) -> None:
            if not text:
                return
            self._job_streamed = True
            tag = _stream_log_tag(stream, text)
            self.log.log(f"[{tag}] {text}", write_file=False)
            mon = getattr(self, "_inv_monitor", None)
            if mon is not None:
                try:
                    mon.note_stream_line(stream, text)
                except RuntimeError:
                    pass

        def done(ok: bool, log: str, err: object) -> None:
            self._run_thread = None
            self._run_worker = None
            if finally_fn is not None:
                try:
                    finally_fn(log if ok else None, None if ok else err)
                except Exception as e:
                    self.log.log(f"收尾失败: {e}")
            # 正演写回反演后刷新控件 + 提示
            hint = self.state.get_str("gui._fwd_bridge_hint")
            if hint:
                self.state.set("gui._fwd_bridge_hint", "")
                try:
                    self._push_all()
                except Exception:
                    pass
                for line in hint.splitlines():
                    self.log.log(line)
                show_modeless_message(
                    "合成数据",
                    hint,
                    icon=QMessageBox.Icon.Information,
                )
            if ok:
                # 已流式刷过则不再整段 dump；未流式时保留摘要
                if log and not self._job_streamed:
                    self.log.log(log[:4000], write_file=False)
                self.log.log(f"{title} 完成")
                self._finish_elapsed(title, True)
                self.refresh_output_panel()
                if any(k in title for k in ("tt_forward", "tt_inverse", "pipeline")):
                    try:
                        self.right_tabs.setCurrentWidget(self.output_panel)
                    except Exception:
                        pass
            else:
                detail = str(err)
                if hasattr(err, "stderr") and getattr(err, "stderr", None):
                    detail = f"{detail}\n\n{getattr(err, 'stderr', '')[:2000]}"
                show_modeless_message(
                    "执行失败", detail, icon=QMessageBox.Icon.Critical
                )
                self.log.log(f"{title} 失败: {err}")
                self._finish_elapsed(title, False)
                self.refresh_output_panel()

        self._run_thread, self._run_worker = start_command_worker(
            self, job, done, on_output_line=on_line, autostart=False
        )
        worker_box["w"] = self._run_worker
        self._run_thread.start()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        from .dialogs.smesh_plot import classify_dropped_plot_files

        smesh, ifaces = classify_dropped_plot_files(event, loose_iface=False)
        if smesh is not None or ifaces:
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        from .dialogs.smesh_plot import classify_dropped_plot_files

        smesh, ifaces = classify_dropped_plot_files(event, loose_iface=False)
        if smesh is not None or ifaces:
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        from .dialogs.smesh_plot import classify_dropped_plot_files

        smesh, ifaces = classify_dropped_plot_files(event, loose_iface=False)
        if smesh is None and not ifaces:
            super().dropEvent(event)
            return
        event.acceptProposedAction()
        try:
            self._pull_all()
        except Exception:
            pass
        plot_smesh_velocity_qt(
            self, self.state, path=smesh, overlay_ifaces=ifaces or None
        )
        names = []
        if smesh is not None:
            names.append(smesh.name)
        names.extend(p.name for p in ifaces)
        self.log.log("[绘制 smesh] 拖入 " + "、".join(names))

    def current_cmd_tab_id(self) -> str:
        w = self.stack.currentWidget()
        if w is None:
            return ""
        return str(w.property("cmd_tab_id") or "")

    def plot_smesh(self) -> None:
        try:
            self._pull_all()
        except Exception:
            pass
        try:
            work = resolve_work_dir(self.state.get_str("work_dir"))
        except Exception:
            work = Path.cwd()
        hit = lookup_plot_sources_for_tab(
            self.state, work, self.current_cmd_tab_id()
        )
        start = str(hit.browse_start) if hit.browse_start is not None else ""
        if hit.smesh_path is None:
            plot_smesh_velocity_qt(
                self,
                self.state,
                skip_guess=True,
                browse_start=start,
            )
            self.log.log(f"[绘制 smesh] {hit.nav_name} 未找到网格，已打开图窗")
            return
        ifaces = [hit.refl_path] if hit.refl_path is not None else []
        plot_smesh_velocity_qt(
            self,
            self.state,
            path=hit.smesh_path,
            overlay_ifaces=ifaces,
            browse_start=start,
        )
        note = f"[绘制 smesh] {hit.nav_name} ← {hit.smesh_path.name}"
        if hit.refl_path is not None:
            note = f"{note} · 界面 {hit.refl_path.name}"
        self.log.log(note)
        refl_msg = hit.missing_refl_message()
        if refl_msg:
            show_modeless_message(
                "叠加界面", refl_msg, icon=QMessageBox.Icon.Warning
            )

    def open_inv_analysis(self) -> None:
        try:
            self._pull_all()
        except Exception:
            pass
        open_inv_analysis_dialog(self.state, self)

    def open_inv_monitor(self) -> None:
        try:
            self._pull_all()
        except Exception:
            pass
        self._inv_monitor = open_inv_monitor_dialog(self.state, self, running=False)

    def open_model_picker(self, spec=None) -> None:
        try:
            self._pull_all()
        except Exception:
            pass
        open_model_picker_dialog(self.state, None, spec=spec)

    def open_model_compare(self) -> None:
        try:
            self._pull_all()
        except Exception:
            pass
        dlg = open_model_compare_dialog(self.state, None)
        dlg.applied.connect(
            self._on_model_compare_applied, Qt.ConnectionType.UniqueConnection
        )

    def _on_model_form_written(self, key: str, path: str) -> None:
        """速度图/统计/挑选写回 FormState 后刷新控件。"""
        try:
            self._push_all()
        except Exception:
            pass
        self.log.log(f"[写回表单] {key} ← {path}")

    def _on_model_compare_applied(self, _key: str, _path: str) -> None:
        try:
            self._push_all()
        except Exception:
            pass

    def refresh_output_panel(self) -> None:
        try:
            work = resolve_work_dir(self.state.get_str("work_dir"))
        except Exception:
            work = Path(".")
        self.output_panel.set_listing(work, list_output_tree(work))

    def _open_work_subdir(self, name: str) -> None:
        work = resolve_work_dir(self.state.get_str("work_dir"))
        p = work / name
        target = p if p.exists() else work
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))

    def _open_output_file(self, path: str) -> None:
        p = Path(path)
        if p.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p.resolve())))

    def bridge_fwd_to_inv(self) -> None:
        try:
            self._pull_all()
        except Exception:
            pass
        notes = apply_fwd_outputs_to_inv(
            self.state, overwrite=False, sync_ray=True, sync_refl=True
        )
        self._push_all()
        msg = "\n".join(notes) if notes else "无空位可填（目标字段已有值；可先清空再试）"
        for n in notes:
            self.log.log(f"[正演→反演] {n}")
        show_modeless_message("合成数据", msg, icon=QMessageBox.Icon.Information)
        self.nav.setCurrentRow(5)

    def fill_inv_from_upstream(self) -> None:
        try:
            self._pull_all()
        except Exception:
            pass
        notes = fill_inv_from_upstream(self.state, overwrite=False)
        self._push_all()
        show_modeless_message(
            "上游填充",
            "\n".join(notes),
            icon=QMessageBox.Icon.Information,
        )
        for n in notes:
            self.log.log(f"[上游填充] {n}")

    def sync_inv_ray_from_fwd(self) -> None:
        try:
            self._pull_all()
        except Exception:
            pass
        notes = sync_ray_params(self.state, direction="fwd_to_inv", overwrite=True)
        self._push_all()
        show_modeless_message(
            "同步 -N",
            "\n".join(notes) if notes else "无变化",
            icon=QMessageBox.Icon.Information,
        )

    def open_help(self) -> None:
        idx = self.tabs.currentIndex()
        section_map = {
            0: "gen_smesh",
            1: "tt_forward",
            2: "gen_damp",
            3: "gen_vcorr",
            4: "gen_dcorr",
            5: "tt_inverse",
            6: "stat_smesh",
            7: "edit_smesh / edit_smesh_HHB",
        }
        # 对齐 vedit/idata：默认 GUI HELP.md；在对应流程页时跳到 TomoHelp 章节
        open_program_help(self, default_section=section_map.get(idx))

    # ----- profile -----
    def save_profile(self) -> None:
        try:
            self._prepare_state()
        except Exception as e:
            show_modeless_message("保存配置", str(e), icon=QMessageBox.Icon.Warning)
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存配置",
            "",
            "JSON (*.json);;所有文件 (*)",
            options=file_dialog_options(),
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state.to_profile_dict(), f, ensure_ascii=False, indent=2)
        self.log.log(f"配置已保存: {path}")

    def load_profile(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "加载配置（全界面 JSON 或 tt_inverse manifest.json）",
            "",
            "JSON (*.json);;所有文件 (*)",
            options=file_dialog_options(),
        )
        if not path:
            return
        try:
            data = read_profile_json(path)
        except Exception as e:
            show_modeless_message("加载配置失败", str(e), icon=QMessageBox.Icon.Warning)
            return
        try:
            if data.get("kind") == "pyaobs.tomo2d.tt_inverse":
                notes = apply_manifest_to_state(
                    self.state, data, path, file_keys=self._all_file_keys()
                )
                self._offer_sync_work_dir(data)
                normalize_file_path_vars(
                    self.state, self._all_file_keys(), warn_outside=False
                )
                sync_fwd_smesh_from_inv_if_missing(self.state)
                self._push_all()
                for n in notes:
                    self.log.log(n)
                self.log.log(f"已从 manifest 加载: {path}")
            else:
                self.state.apply_mapping(data)
                normalize_file_path_vars(
                    self.state, self._all_file_keys(), warn_outside=False
                )
                self._push_all()
                self._sync_dynamic_tabs()
                self.log.log(f"配置已加载: {path}")
        except Exception as e:
            show_modeless_message("加载失败", str(e), icon=QMessageBox.Icon.Critical)

    def _offer_sync_work_dir(self, manifest: dict) -> None:
        wd_m = manifest.get("work_dir")
        if not wd_m:
            return
        try:
            p = Path(str(wd_m)).expanduser().resolve()
        except OSError:
            return
        if not p.is_dir():
            return
        cur = resolve_work_dir(self.state.get_str("work_dir"))
        if cur == p:
            return
        # 短交互：Yes/No 门禁允许模态
        ans = QMessageBox.question(
            self,
            "同步工作目录",
            f"manifest 工作目录：\n{p}\n\n当前：\n{cur}\n\n是否同步？",
        )
        if ans == QMessageBox.StandardButton.Yes:
            self.state.set("work_dir", str(p))


    def _do_preview(self, title: str, text: str) -> None:
        self.preview.set_preview(text)
        self.statusBar().showMessage(f"已预览 {title}")

    def _do_run(
        self, work: Path, prep: PreparedRun, *, push_state: bool = False
    ) -> None:
        if self._busy():
            return
        if push_state:
            self._push_all()
        if prep.preview_text:
            self.preview.set_preview(prep.preview_text)
        for n in prep.notes:
            self.log.log(n, write_file=True)
        self._start_job(prep.title, work, prep.job, finally_fn=prep.finally_fn)

    def preview_gen_smesh(self) -> None:
        try:
            work = self._prepare_state()
            self._do_preview(
                "gen_smesh", preview_gen_smesh(self.state, work, self._tomo(work))
            )
        except Exception as e:
            show_modeless_message("预览失败", str(e), icon=QMessageBox.Icon.Warning)

    def run_gen_smesh(self) -> None:
        try:
            work = self._prepare_state()
            self._do_run(
                work, prepare_gen_smesh(self.state, work, self._tomo(work))
            )
        except Exception as e:
            show_modeless_message(
                "运行前检查失败", str(e), icon=QMessageBox.Icon.Warning
            )

    def preview_tt_forward(self) -> None:
        try:
            work = self._prepare_state()
            self._do_preview(
                "tt_forward", preview_tt_forward(self.state, work, self._tomo(work))
            )
        except Exception as e:
            show_modeless_message("预览失败", str(e), icon=QMessageBox.Icon.Warning)

    def run_tt_forward(self) -> None:
        try:
            work = self._prepare_state()
            self._do_run(
                work, prepare_tt_forward(self.state, work, self._tomo(work))
            )
        except Exception as e:
            show_modeless_message(
                "运行前检查失败", str(e), icon=QMessageBox.Icon.Warning
            )

    def preview_gen_damp(self) -> None:
        try:
            work = self._prepare_state()
            self._do_preview(
                "gen_damp", preview_gen_damp(self.state, work, self._tomo(work))
            )
        except Exception as e:
            show_modeless_message("预览失败", str(e), icon=QMessageBox.Icon.Warning)

    def run_gen_damp(self) -> None:
        try:
            work = self._prepare_state()
            self._do_run(
                work, prepare_gen_damp(self.state, work, self._tomo(work))
            )
        except Exception as e:
            show_modeless_message(
                "运行前检查失败", str(e), icon=QMessageBox.Icon.Warning
            )

    def preview_gen_vcorr(self) -> None:
        try:
            work = self._prepare_state()
            self._do_preview(
                "gen_vcorr", preview_gen_vcorr(self.state, work, self._tomo(work))
            )
        except Exception as e:
            show_modeless_message("预览失败", str(e), icon=QMessageBox.Icon.Warning)

    def run_gen_vcorr(self) -> None:
        try:
            work = self._prepare_state()
            self._do_run(
                work, prepare_gen_vcorr(self.state, work, self._tomo(work))
            )
        except Exception as e:
            show_modeless_message(
                "运行前检查失败", str(e), icon=QMessageBox.Icon.Warning
            )

    def preview_gen_dcorr(self) -> None:
        try:
            work = self._prepare_state()
            self._do_preview(
                "gen_dcorr", preview_gen_dcorr(self.state, work, self._tomo(work))
            )
        except Exception as e:
            show_modeless_message("预览失败", str(e), icon=QMessageBox.Icon.Warning)

    def run_gen_dcorr(self) -> None:
        try:
            work = self._prepare_state()
            self._do_run(
                work, prepare_gen_dcorr(self.state, work, self._tomo(work))
            )
        except Exception as e:
            show_modeless_message(
                "运行前检查失败", str(e), icon=QMessageBox.Icon.Warning
            )

    def preview_tt_inverse(self) -> None:
        try:
            work = self._prepare_state()
            self._do_preview(
                "tt_inverse", preview_tt_inverse(self.state, work, self._tomo(work))
            )
        except Exception as e:
            show_modeless_message("预览失败", str(e), icon=QMessageBox.Icon.Warning)

    def run_tt_inverse(self) -> None:
        try:
            work = self._prepare_state()
            prep = prepare_tt_inverse(
                self.state,
                work,
                self._tomo(work),
                gui_profile=self.state.to_profile_dict(),
            )
            self._attach_inv_monitor(prep)
            self._do_run(work, prep)
        except Exception as e:
            show_modeless_message(
                "运行前检查失败", str(e), icon=QMessageBox.Icon.Warning
            )

    def _attach_inv_monitor(self, prep: PreparedRun) -> None:
        """打开/改绑反演监视窗；收尾时通知归类完成后再刷一次。"""
        spec = getattr(prep, "monitor_spec", None)
        if spec is None:
            return
        dlg = open_inv_monitor_dialog(
            self.state, self, spec=spec, running=True
        )
        self._inv_monitor = dlg
        prev_finally = prep.finally_fn

        def _wrapped(res, err):
            try:
                if prev_finally is not None:
                    prev_finally(res, err)
            finally:
                try:
                    if spec.run_dir is not None:
                        self.state.set(
                            "gui.last_tt_inverse_run", str(Path(spec.run_dir).resolve())
                        )
                except Exception:
                    pass
                try:
                    dlg.mark_run_finished()
                except RuntimeError:
                    pass

        prep.finally_fn = _wrapped

    def preview_stat_smesh(self) -> None:
        try:
            work = self._prepare_state()
            self._do_preview(
                "stat_smesh", preview_stat_smesh(self.state, work, self._tomo(work))
            )
        except Exception as e:
            show_modeless_message("预览失败", str(e), icon=QMessageBox.Icon.Warning)

    def run_stat_smesh(self) -> None:
        try:
            work = self._prepare_state()
            self._do_run(
                work, prepare_stat_smesh(self.state, work, self._tomo(work))
            )
        except Exception as e:
            show_modeless_message(
                "运行前检查失败", str(e), icon=QMessageBox.Icon.Warning
            )

    def preview_edit_smesh(self) -> None:
        try:
            work = self._prepare_state()
            self._do_preview(
                "edit_smesh", preview_edit_smesh(self.state, work, self._tomo(work))
            )
        except Exception as e:
            show_modeless_message("预览失败", str(e), icon=QMessageBox.Icon.Warning)

    def run_edit_smesh(self) -> None:
        try:
            work = self._prepare_state()
            self._do_run(
                work, prepare_edit_smesh(self.state, work, self._tomo(work))
            )
        except Exception as e:
            show_modeless_message(
                "运行前检查失败", str(e), icon=QMessageBox.Icon.Warning
            )

    def preview_pipeline(self) -> None:
        try:
            work = self._prepare_state()
            text = preview_pipeline(self.state, work, self._tomo(work))
            self._push_all()
            self._do_preview("pipeline", text)
        except Exception as e:
            show_modeless_message("预览失败", str(e), icon=QMessageBox.Icon.Warning)

    def run_pipeline(self) -> None:
        try:
            work = self._prepare_state()
            prep = prepare_pipeline_simple(self.state, work, self._tomo(work))
            self._do_run(work, prep, push_state=True)
        except Exception as e:
            show_modeless_message(
                "运行前检查失败", str(e), icon=QMessageBox.Icon.Warning
            )

    def plot_tx_in(self) -> None:
        try:
            self._prepare_state()
        except Exception as e:
            show_modeless_message("预览 tx.in", str(e), icon=QMessageBox.Icon.Warning)
            return
        open_tx_in_preview(self, self.state)

    def plot_ttimes(self) -> None:
        try:
            self._prepare_state()
        except Exception as e:
            show_modeless_message("预览 ttimes", str(e), icon=QMessageBox.Icon.Warning)
            return
        open_ttimes_preview(self, self.state)

    def preview_tx_convert(self) -> None:
        try:
            work = self._prepare_state()
            self._do_preview("tx_convert", preview_tx_convert(self.state, work))
        except Exception as e:
            show_modeless_message("预览失败", str(e), icon=QMessageBox.Icon.Warning)

    def run_tx_convert(self) -> None:
        try:
            work = self._prepare_state()
            self._do_run(work, prepare_tx_convert(self.state, work))
        except Exception as e:
            show_modeless_message(
                "运行前检查失败", str(e), icon=QMessageBox.Icon.Warning
            )

    def preview_checkerboard(self) -> None:
        try:
            work = self._prepare_state()
            self._do_preview(
                "checkerboard",
                preview_checkerboard(self.state, work, self._tomo(work)),
            )
        except Exception as e:
            show_modeless_message("预览失败", str(e), icon=QMessageBox.Icon.Warning)

    def run_checkerboard(self) -> None:
        if self._busy():
            return
        try:
            work = self._prepare_state()
            tomo = self._tomo(work)
            result_box: dict = {}

            def job():
                r = run_checkerboard_test(self.state, work, tomo)
                result_box["r"] = r
                return "\n".join(r.messages)

            def finally_fn(log, err):
                r = result_box.get("r")
                if r is None:
                    return
                self.preview.set_preview(
                    f"棋盘格完成 @ {r.run_dir}\n"
                    + "\n".join(f"{k}: {v}" for k, v in r.artifacts.items())
                )
                anom = r.artifacts.get("recovered_anomaly_pct")
                if anom:
                    try:
                        self.state.set("gui.plot_smesh_hint", anom)
                    except Exception:
                        pass
                try:
                    self.cb_tab.open_result_window(r.run_dir)
                except Exception as e:
                    self.log.log(f"打开棋盘结果图失败: {e}")

            prep = PreparedRun(
                title="checkerboard",
                job=job,
                preview_text=preview_checkerboard(self.state, work, tomo),
                finally_fn=finally_fn,
            )
            self._do_run(work, prep)
        except Exception as e:
            show_modeless_message(
                "运行前检查失败", str(e), icon=QMessageBox.Icon.Warning
            )

    def preview_monte_carlo(self) -> None:
        try:
            work = self._prepare_state()
            self._do_preview(
                "monte_carlo",
                preview_monte_carlo(self.state, work, self._tomo(work)),
            )
        except Exception as e:
            show_modeless_message("预览失败", str(e), icon=QMessageBox.Icon.Warning)

    def run_monte_carlo(self) -> None:
        if self._busy():
            return
        try:
            work = self._prepare_state()
            tomo = self._tomo(work)
            result_box: dict = {}

            def job():
                r = run_monte_carlo(self.state, work, tomo)
                result_box["r"] = r
                return "\n".join(r.messages)

            def finally_fn(log, err):
                r = result_box.get("r")
                if r is None:
                    return
                self.preview.set_preview(
                    f"蒙特卡洛完成 @ {r.run_dir}\n"
                    + "\n".join(f"{k}: {v}" for k, v in r.artifacts.items())
                )
                try:
                    self.mc_tab.open_result_window(r.run_dir)
                except Exception as e:
                    self.log.log(f"蒙特卡洛结果图打开失败: {e}")
                    prof = r.artifacts.get("mean_profile")
                    if prof:
                        try:
                            open_mc_profile_dialog(prof, self)
                        except Exception as e2:
                            self.log.log(f"剖面图打开失败: {e2}")

            prep = PreparedRun(
                title="monte_carlo",
                job=job,
                preview_text=preview_monte_carlo(self.state, work, tomo),
                finally_fn=finally_fn,
            )
            self._do_run(work, prep)
        except Exception as e:
            show_modeless_message(
                "运行前检查失败", str(e), icon=QMessageBox.Icon.Warning
            )

    def preview_wave2d(self) -> None:
        try:
            work = self._prepare_state()
            self._do_preview("wave2d", preview_wave2d(self.state, work))
        except Exception as e:
            show_modeless_message("预览失败", str(e), icon=QMessageBox.Icon.Warning)

    def run_wave2d(self) -> None:
        try:
            work = self._prepare_state()
            prep = prepare_wave2d(self.state, work)
            shown = getattr(prep, "shown_png", [])

            def _open_png(_log, err) -> None:
                if err is not None or not shown:
                    return
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(shown[-1])))

            prep.finally_fn = _open_png
            self._do_run(work, prep)
        except Exception as e:
            show_modeless_message(
                "运行前检查失败", str(e), icon=QMessageBox.Icon.Warning
            )


def run_application(argv: list[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(
        argv if argv is not None else sys.argv
    )
    win = Tomo2DMainWindow(argv if argv is not None else sys.argv)
    win.show()
    return app.exec()
