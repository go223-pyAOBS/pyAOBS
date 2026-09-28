# -*- coding: utf-8 -*-
"""idata 主窗：工程化工区 + 转换 / 道头 / 几何。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..project import DEFAULT_GEOM, IdataProject
from .dialog_utils import show_modeless_message
from .help_dialog import install_help_shortcut, show_help_dialog
from ..services.workdir_layout import PROJECT_JSON
from .panels import ConvertPanel, GeometryPanel, HeaderPanel
from .services.convert_runner import ConvertRunner
from .services.env_context import IdataEnvContext
from .services.segy_dataset import SegyDataset


class IdataMainWindow(QMainWindow):
    def __init__(self, env: Optional[IdataEnvContext] = None, argv: Optional[list[str]] = None) -> None:
        super().__init__()
        self.env = env or IdataEnvContext()
        self.project = IdataProject()
        self.dataset = SegyDataset()
        self.runner = ConvertRunner(self.env, self)
        self._loading = False
        self.setWindowTitle("idata — 数据转换与道头编辑工区")
        self.resize(1280, 860)
        self._build_ui()
        self._hide_menubar()
        install_help_shortcut(self)
        self._update_title()
        self.env.audit(
            "idata_started",
            run_id=self.env.run_id,
            project_root=self.env.project_root,
        )
        self.append_log(
            "idata 工程化工区：新建/打开/保存工区 → 1 转换 → 2 道头 → 3 几何。\n"
            f"工程文件：{PROJECT_JSON}；目录 inputs/ outputs/ convert/。\n"
            "几何约定：炮=sx/sy，OBS=gx/gy。转换后端：processors/raw2sac。\n"
            "帮助：工具栏「帮助」（F1）打开完整文档（含几何字段与关于）。"
        )
        self._startup_argv = None if argv is None else list(argv)
        QTimer.singleShot(0, self._startup_open_project)

    def _build_ui(self) -> None:
        self.act_new_proj = QAction("新建工区", self)
        self.act_new_proj.triggered.connect(self._new_project)
        self.act_open_proj = QAction("打开工区", self)
        self.act_open_proj.triggered.connect(self._open_project)
        self.act_save_proj = QAction("保存工区", self)
        self.act_save_proj.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.act_save_proj.triggered.connect(self._save_project)

        self.act_open = QAction("打开 SEGY/SU…", self)
        self.act_open.setShortcut(QKeySequence.StandardKey.Open)
        self.act_open.triggered.connect(self.open_segy)
        self.act_save = QAction("保存数据", self)
        self.act_save.setShortcut(QKeySequence.StandardKey.Save)
        self.act_save.triggered.connect(self.save_segy)
        self.act_save_as = QAction("数据另存为…", self)
        self.act_save_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.act_save_as.triggered.connect(self.save_segy_as)
        self.act_export_su = QAction("导出为 SU…", self)
        self.act_export_su.triggered.connect(self.export_su)
        self.act_quit = QAction("退出", self)
        self.act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        self.act_quit.triggered.connect(self.close)
        self.act_clear_log = QAction("清空日志", self)
        self.act_clear_log.triggered.connect(self.clear_log)
        self.act_help = QAction("帮助", self)
        self.act_help.setShortcut(QKeySequence.StandardKey.HelpContents)
        self.act_help.triggered.connect(lambda: QTimer.singleShot(0, lambda: show_help_dialog(activate=True)))

        tb = QToolBar("主工具")
        tb.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)
        tb.addAction(self.act_new_proj)
        tb.addAction(self.act_open_proj)
        tb.addAction(self.act_save_proj)
        # 帮助 / 退出紧跟「保存工区」（对齐 zplotpy / relocation）
        tb.addSeparator()
        tb.addAction(self.act_help)
        tb.addAction(self.act_quit)
        tb.addSeparator()
        tb.addAction(self.act_open)
        tb.addAction(self.act_save)
        tb.addAction(self.act_save_as)
        tb.addAction(self.act_export_su)
        tb.addSeparator()
        tb.addAction(self.act_clear_log)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)

        self.stage_bar = QTabBar()
        self.stage_bar.setExpanding(False)
        for title in ("1 转换", "2 道头编辑", "3 工区几何"):
            self.stage_bar.addTab(title)
        self.stack = QStackedWidget()
        self.stage_bar.currentChanged.connect(self._on_stage_changed)

        self.panel_convert = ConvertPanel(self.env, self.runner)
        self.panel_header = HeaderPanel()
        self.panel_geom = GeometryPanel()
        self.stack.addWidget(self.panel_convert)
        self.stack.addWidget(self.panel_header)
        self.stack.addWidget(self.panel_geom)

        top = QWidget()
        top_l = QVBoxLayout(top)
        top_l.setContentsMargins(0, 0, 0, 0)
        top_l.setSpacing(2)
        top_l.addWidget(self.stage_bar)
        top_l.addWidget(self.stack, stretch=1)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setPlaceholderText("执行日志…")

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(self.log)
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 1)
        split.setSizes([640, 160])
        root.addWidget(split)

        self.setStatusBar(QStatusBar())
        self._file_label = QLabel("未打开工区")
        self.statusBar().addWidget(self._file_label, stretch=1)
        self.statusBar().showMessage("就绪")

        self.runner.log.connect(self.append_log)
        self.runner.status.connect(self.statusBar().showMessage)
        self.panel_convert.request_load_segy.connect(self.load_segy_path)
        self.panel_convert.fields_changed.connect(self._on_convert_fields_changed)
        self.panel_header.selection_changed.connect(self.panel_geom.highlight_trace)
        self.panel_header.dataset_modified.connect(self._on_dataset_modified)
        self.panel_header.geom_mode_changed.connect(self._on_geom_mode_changed)
        self.panel_header.log_message.connect(self.append_log)
        self.panel_geom.geom_mode_changed.connect(self._on_geom_mode_from_map)
        self.panel_geom.trace_selected.connect(self._on_geom_trace)
        self.panel_geom.log_message.connect(self.append_log)
        # 启动默认：约定（炮=sx/sy，OBS=gx/gy）
        self.panel_header.set_geom_mode(DEFAULT_GEOM)
        self.panel_geom.set_geom_mode(DEFAULT_GEOM)
        self.project.workflow.geom = DEFAULT_GEOM

    def _hide_menubar(self) -> None:
        """取消顶部「文件 / 帮助」菜单栏，动作已迁到工具栏。"""
        try:
            mb = self.menuBar()
            mb.clear()
            mb.setVisible(False)
            mb.setMaximumHeight(0)
        except Exception:
            pass

    # ---- project ----
    def _update_title(self) -> None:
        title = "idata — 数据转换与道头编辑工区"
        if self.project.workdir:
            mark = " *" if self.project.dirty else ""
            title += f" [{self.project.name or Path(self.project.workdir).name}]{mark}"
        self.setWindowTitle(title)

    def _bind_env_to_project(self) -> None:
        if self.project.workdir:
            self.env.bind_workdir(self.project.workdir)
        else:
            self.env.bind_workdir(None)

    def _panels_to_project(self) -> None:
        if self._loading:
            return
        self.project.convert.fields = self.panel_convert.collect_state_fields()
        self.project.convert.convert_tab_index = self.panel_convert.convert_tab_index()
        self.project.workflow.stage_index = int(self.stage_bar.currentIndex())
        self.project.workflow.geom = self.panel_header.geom_mode()
        if self.dataset.is_open and self.dataset.path is not None:
            self.project.workflow.current_data = str(self.dataset.path)
        self.project.dirty = True
        self._update_title()

    def _project_to_panels(self) -> None:
        self._loading = True
        try:
            self.panel_convert.restore_state_fields(self.project.convert.fields)
            self.panel_convert.set_convert_tab_index(self.project.convert.convert_tab_index)
            geom = str(self.project.workflow.geom or DEFAULT_GEOM).lower()
            if geom in ("literal_segy", "约定", ""):
                geom = DEFAULT_GEOM
            self.panel_header.set_geom_mode(geom)
            self.panel_geom.set_geom_mode(geom)
            stage = int(self.project.workflow.stage_index or 0)
            self.stage_bar.setCurrentIndex(max(0, min(stage, self.stage_bar.count() - 1)))
        finally:
            self._loading = False
        self._update_title()
        self._update_file_status()

    def _confirm_discard_project(self) -> bool:
        if not self.project.dirty and self.dataset.dirty_count == 0:
            return True
        ans = QMessageBox.question(
            self,
            "未保存",
            "当前工区或道头有未保存更改，继续将丢弃，是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return ans == QMessageBox.StandardButton.Yes

    def _new_project(self) -> None:
        if not self._confirm_discard_project():
            return
        parent = self.env.default_open_initial_dir()
        path = QFileDialog.getExistingDirectory(self, "选择新建工区目录", parent)
        if not path:
            return
        name, ok = QInputDialog.getText(
            self, "工区名称", "名称：", text=Path(path).name
        )
        if not ok:
            return
        try:
            self.project = IdataProject.create_new(path, name=str(name or ""))
            self.project.save()
        except Exception as exc:
            show_modeless_message("新建失败", str(exc), icon=QMessageBox.Icon.Critical)
            return
        self.dataset.clear()
        self.panel_header.set_dataset(self.dataset)
        self.panel_geom.set_dataset(self.dataset)
        self._bind_env_to_project()
        self._project_to_panels()
        self.append_log(f"新建工区：{self.project.workdir}")
        self.env.audit("project_new", workdir=self.project.workdir)

    def _open_project(self) -> None:
        if not self._confirm_discard_project():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开 idata 工区",
            self.env.default_open_initial_dir(),
            "idata project (idata_project.json);;JSON (*.json);;All (*.*)",
        )
        if not path:
            return
        try:
            self.project = IdataProject.load(path)
            self.project.ensure_workdir()
        except Exception as exc:
            show_modeless_message("打开失败", str(exc), icon=QMessageBox.Icon.Critical)
            return
        self._after_project_opened()
        self.append_log(f"打开工区：{self.project.workdir}")
        self.env.audit("project_open", workdir=self.project.workdir)

    def _after_project_opened(self) -> None:
        self._bind_env_to_project()
        self._project_to_panels()
        data = self.project.abs_or_join(self.project.workflow.current_data)
        if data and Path(data).is_file():
            self.load_segy_path(data, switch_stage=False)
        else:
            self.dataset.clear()
            self.panel_header.set_dataset(self.dataset)
            self.panel_geom.set_dataset(self.dataset)
            self._update_file_status()

    def _startup_open_project(self) -> None:
        import os

        argv = self._startup_argv
        candidates: list[str] = []
        env = os.environ.get("PYAOBS_IDATA_PROJECT", "").strip()
        if env:
            candidates.append(env)
        if argv:
            for a in argv[1:]:
                s = str(a).strip()
                if not s or s.startswith("-"):
                    continue
                candidates.append(s)
        for raw in candidates:
            jp = IdataProject.resolve_open_path(raw)
            if jp:
                try:
                    self.project = IdataProject.load(jp)
                    self.project.ensure_workdir()
                    self._after_project_opened()
                    self.append_log(f"打开工区：{self.project.workdir}")
                    return
                except Exception as exc:
                    self.append_log(f"启动打开工区失败：{exc}")
                    continue
            p = Path(raw)
            if p.is_dir():
                try:
                    self.project = IdataProject.create_new(str(p.resolve()))
                    self._after_project_opened()
                    self.append_log(f"已绑定工区目录（尚未保存工程 JSON）：{p}")
                    return
                except Exception as exc:
                    self.append_log(f"启动绑定工区失败：{exc}")

    def _save_project(self) -> None:
        if not self.project.workdir:
            parent = self.env.default_save_initial_dir()
            path = QFileDialog.getExistingDirectory(self, "选择工区保存目录", parent)
            if not path:
                return
            name, ok = QInputDialog.getText(
                self, "工区名称", "名称：", text=Path(path).name
            )
            if not ok:
                return
            self.project.workdir = str(Path(path).resolve())
            self.project.name = str(name or Path(path).name)
            self.project.ensure_workdir()
            self._bind_env_to_project()
        self._panels_to_project()
        try:
            jp = self.project.save()
        except Exception as exc:
            show_modeless_message("保存工区失败", str(exc), icon=QMessageBox.Icon.Critical)
            return
        self._update_title()
        self.append_log(f"已保存工区：{jp}")
        self.statusBar().showMessage("工区已保存", 3000)
        self.env.audit("project_save", path=jp)

    # ---- log / stages ----
    def append_log(self, text: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{stamp}] {text}")

    def clear_log(self) -> None:
        self.log.clear()
        self.append_log("日志已清空")
        self.env.audit("log_cleared")

    def _on_stage_changed(self, idx: int) -> None:
        self.stack.setCurrentIndex(idx)
        self.env.audit("tab_changed", tab=self.stage_bar.tabText(idx))
        if not self._loading:
            self.project.workflow.stage_index = int(idx)
            self.project.dirty = True
            self._update_title()
        if idx == 2:
            # 切回几何：同步解释模式，但保留缩放/平移；并按当前道高亮炮点
            mode = self.panel_header.geom_mode()
            if self.panel_geom.geom_mode() != mode:
                self.panel_geom.set_geom_mode(mode, reset_view=False)
            row = self.panel_header.current_row()
            if row >= 0:
                self.panel_geom.highlight_trace(row)

    def _on_convert_fields_changed(self) -> None:
        if self._loading:
            return
        self._panels_to_project()

    def _on_geom_mode_changed(self, mode: str) -> None:
        self.panel_geom.set_geom_mode(mode, reset_view=False)
        if not self._loading:
            self.project.workflow.geom = mode
            self.project.dirty = True
            self._update_title()

    def _on_geom_mode_from_map(self, mode: str) -> None:
        self.panel_header.set_geom_mode(mode)
        if not self._loading:
            self.project.workflow.geom = mode
            self.project.dirty = True
            self._update_title()

    # ---- data ----
    def open_segy(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开 SEGY / SU",
            self.env.default_open_initial_dir(),
            "Seismic (*.su *.segy *.sgy);;All (*.*)",
        )
        if path:
            self.load_segy_path(path)

    def load_segy_path(self, path: str, *, switch_stage: bool = True) -> None:
        self.append_log(f"[数据] 正在打开… {path}")
        try:
            self.dataset.open(path)
        except Exception as exc:
            show_modeless_message("打开失败", str(exc), icon=QMessageBox.Icon.Critical)
            self.append_log(f"[数据] 打开失败: {exc}")
            return
        self.panel_header.set_dataset(self.dataset)
        self.panel_geom.set_dataset(self.dataset)
        self.panel_geom.set_geom_mode(self.panel_header.geom_mode(), reset_view=False)
        self.project.workflow.current_data = str(self.dataset.path)
        suf = Path(path).suffix.lower()
        if suf == ".su":
            self.project.workflow.last_su = str(self.dataset.path)
        elif suf in (".segy", ".sgy"):
            self.project.workflow.last_segy = str(self.dataset.path)
        self.project.dirty = True
        self._update_file_status()
        self._update_title()
        kind = "SU" if self.dataset.is_su else "SEGY"
        self.append_log(
            f"[数据] 已打开 {kind}  {self.dataset.path}  "
            f"ntraces={self.dataset.ntraces}  endian={self.dataset.endian}  "
            f"sample_format={self.dataset.sample_format}  "
            f"geom={self.panel_header.geom_mode()}"
        )
        if switch_stage:
            self.stage_bar.setCurrentIndex(1)
        self.env.audit("segy_opened", path=str(self.dataset.path), ntraces=self.dataset.ntraces)

    def export_su(self) -> None:
        if not self.dataset.is_open:
            show_modeless_message("导出 SU", "请先打开 SEGY/SU 文件。")
            return
        default_name = ""
        if self.dataset.path is not None:
            default_name = str(
                Path(self.env.default_save_initial_dir())
                / f"{self.dataset.path.stem}.su"
            )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出为 SU",
            default_name or self.env.default_save_initial_dir(),
            "SU (*.su);;All (*.*)",
        )
        if not path:
            return
        target = self.env.rewrite_save_target(path)
        try:
            dest = self.dataset.export_su(target, endian="little")
        except Exception as exc:
            show_modeless_message("导出失败", str(exc), icon=QMessageBox.Icon.Critical)
            self.append_log(f"[数据] 导出 SU 失败: {exc}")
            return
        self.project.workflow.last_su = str(dest)
        self.project.dirty = True
        self._update_title()
        self.append_log(
            f"[数据] 已导出 SU  {dest}  "
            f"（IEEE float, little-endian, ntraces={self.dataset.ntraces}）"
        )
        self.env.audit("segy_exported_su", path=str(dest))
        show_modeless_message("导出 SU", f"已写入：\n{dest}")

    def save_segy(self) -> None:
        if not self.dataset.is_open:
            show_modeless_message("保存", "未打开文件。")
            return
        dirty_before = self.dataset.dirty_count
        self.append_log(f"[数据] 正在保存道头… dirty={dirty_before}")
        try:
            dest = self.dataset.save()
        except Exception as exc:
            show_modeless_message("保存失败", str(exc), icon=QMessageBox.Icon.Critical)
            self.append_log(f"[数据] 保存失败: {exc}")
            return
        # 同文件原地保存：勿 set_dataset（会重载全部道集样本）
        self.panel_header.notify_saved()
        self.panel_geom.refresh(reset_view=False)
        row = self.panel_header.current_row()
        if row >= 0:
            self.panel_geom.highlight_trace(row)
        self.project.workflow.current_data = str(dest)
        self._update_file_status()
        self.append_log(
            f"[数据] 已保存（原地写道头）  {dest}  "
            f"写入脏道≈{dirty_before}  当前dirty={self.dataset.dirty_count}"
        )
        self.env.audit("segy_saved", path=str(dest))

    def save_segy_as(self) -> None:
        if not self.dataset.is_open:
            show_modeless_message("另存为", "未打开文件。")
            return
        src_su = bool(self.dataset.is_su)
        if src_su:
            filt = (
                "SU 同格式 (*.su);;"
                "SEGY 转换 (*.segy *.sgy);;"
                "All (*.*)"
            )
            hint = "当前为 SU：选 .segy/.sgy 将真正转换为 SEGY（IEEE）；选 .su 为同格式另存"
        else:
            filt = (
                "SEGY 同格式 (*.segy *.sgy);;"
                "SU 转换 (*.su);;"
                "All (*.*)"
            )
            hint = "当前为 SEGY：选 .su 将真正转换为 SU（IEEE）；选 .segy/.sgy 为同格式另存"
        self.statusBar().showMessage(hint, 8000)
        default_dir = self.env.default_save_initial_dir()
        if self.dataset.path is not None:
            stem = self.dataset.path.stem
            default_name = str(
                Path(default_dir) / f"{stem}{'.su' if src_su else '.segy'}"
            )
        else:
            default_name = default_dir
        path, _ = QFileDialog.getSaveFileName(
            self,
            "另存为",
            default_name,
            filt,
        )
        if not path:
            return
        target = self.env.rewrite_save_target(path)
        dest_path = Path(target)
        dest_su = dest_path.suffix.lower() == ".su"
        cross = dest_su != src_su
        if cross:
            self.append_log(
                f"[数据] 另存为将做格式转换： "
                f"{'SU' if src_su else 'SEGY'} → {'SU' if dest_su else 'SEGY'}  → {target}"
            )
        else:
            self.append_log(
                f"[数据] 另存为（同格式拷贝）→ {target}"
            )
        try:
            dest = self.dataset.save(target)
        except Exception as exc:
            show_modeless_message("保存失败", str(exc), icon=QMessageBox.Icon.Critical)
            self.append_log(f"[数据] 另存为失败: {exc}")
            return
        self.panel_header.set_dataset(self.dataset)
        self.panel_geom.set_dataset(self.dataset)
        self.project.workflow.current_data = str(dest)
        self.project.dirty = True
        self._update_file_status()
        self._update_title()
        kind = "SU" if self.dataset.is_su else "SEGY"
        self.append_log(
            f"[数据] 已另存为 {kind}  {dest}  "
            f"ntraces={self.dataset.ntraces}  endian={self.dataset.endian}  "
            f"sample_format={self.dataset.sample_format}"
            + ("  （已转换）" if cross else "  （同格式）")
        )
        self.env.audit(
            "segy_saved_as",
            path=str(dest),
            cross_format=cross,
            is_su=self.dataset.is_su,
        )

    def _on_dataset_modified(self) -> None:
        self._update_file_status()
        self.panel_geom.refresh(reset_view=False)
        row = self.panel_header.current_row()
        if row >= 0:
            self.panel_geom.highlight_trace(row)

    def _on_geom_trace(self, row: int) -> None:
        # 炮点先高亮，再跳到道头/道集（纯导航，不写日志）
        self.panel_geom.highlight_trace(row)
        self.stage_bar.setCurrentIndex(1)
        self.panel_header.select_row(row)

    def _update_file_status(self) -> None:
        parts = []
        if self.project.workdir:
            parts.append(f"工区: {self.project.name or Path(self.project.workdir).name}")
        else:
            parts.append("未打开工区")
        if self.dataset.is_open:
            dirty = self.dataset.dirty_count
            parts.append(
                f"{self.dataset.path.name} n={self.dataset.ntraces} "
                f"endian={self.dataset.endian} dirty={dirty}"
            )
            self.statusBar().showMessage(f"脏道数: {dirty}" if dirty else "数据已同步")
        else:
            parts.append("未打开数据")
            self.statusBar().showMessage("就绪")
        self._file_label.setText("  |  ".join(parts))

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        need = self.project.dirty or self.dataset.dirty_count > 0
        if need:
            ans = QMessageBox.question(
                self,
                "未保存更改",
                "工区或道头有未保存更改，仍要退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self.env.audit("idata_closed", active_jobs=self.runner.active_jobs)
        event.accept()


def run_idata_app(argv: Optional[list[str]] = None) -> int:
    import sys

    argv = argv if argv is not None else sys.argv
    app = QApplication.instance() or QApplication(argv)
    win = IdataMainWindow(argv=argv)
    win.show()
    return app.exec()
