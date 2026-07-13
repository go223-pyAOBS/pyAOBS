"""PySide6 Workbench main window."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.project_manager import ProjectContext, ProjectError, ProjectManager
from ..core.run_manager import RunManager
from ..core.state_store import StateStore
from ..petrology_launcher import launch_lip_from_workbench_state
from ..plugins.base import PluginValidationError
from ..plugins.tomo2d_templates import TEMPLATE_CUSTOM
from ..registry.builtin_tools import create_builtin_registry
from ..shell.logic.gui_form_commands import find_latest_imodel_gui_state
from ..shell.logic.helpers import (
    normalize_payload_for_plugin,
    should_auto_preflight,
)
from ..shell.logic.project_tree import scan_tree_nodes
from ..shell.logic.run_history import scan_run_history
from .history_tab import RunHistoryTab
from .runner_tab import RunnerTab
from .styles import (
    apply_workbench_chrome,
    apply_workbench_font,
    section_title,
    set_project_badge_style,
    status_bar_message_label,
)


class _RunNodeWorker(QThread):
    finished = Signal(dict)

    def __init__(
        self,
        run_manager: RunManager,
        project: ProjectContext,
        spec,
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self._run_manager = run_manager
        self._project = project
        self._spec = spec
        self._cancel_event = cancel_event

    def run(self) -> None:
        result: dict[str, str] = {
            "run_id": "",
            "status": "failed",
            "return_code": "",
            "manifest_file": "",
            "error": "",
        }
        try:
            run = self._run_manager.run_command(
                project=self._project,
                node_id=self._spec.node_id,
                command=self._spec.command,
                params=self._spec.params,
                inputs=self._spec.inputs,
                env=self._spec.env,
                cwd=self._spec.cwd,
                cancel_event=self._cancel_event,
            )
            result["run_id"] = run.run_id
            result["manifest_file"] = str(run.manifest_file)
            try:
                manifest = json.loads(run.manifest_file.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
            result["status"] = str(manifest.get("status", "unknown"))
            result["return_code"] = str(manifest.get("return_code", ""))
        except Exception as exc:
            result["error"] = str(exc)
        self.finished.emit(result)


class WorkbenchMainWindow(QMainWindow):
    """Project workbench — PySide6 shell."""

    tool_registry = None  # set in __init__ for runner_tab access

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("pyAOBS Workbench")
        self.resize(1320, 820)

        self.project_manager = ProjectManager()
        self.run_manager = RunManager()
        self.state_store = StateStore()
        self.tool_registry = create_builtin_registry()
        WorkbenchMainWindow.tool_registry = self.tool_registry

        self.current_project: ProjectContext | None = None
        self._run_id_to_dir: dict[str, Path] = {}
        self._project_path_to_item: dict[str, QTreeWidgetItem] = {}
        self._single_run_active = False
        self._single_run_cancel_event: threading.Event | None = None
        self._run_worker: _RunNodeWorker | None = None
        self._tree_scan_token = 0
        self._history_scan_token = 0

        self._apply_style()
        self._build_toolbar()
        self._build_body()
        self._build_status()
        apply_workbench_chrome(self)
        self._set_status("就绪")

    def _apply_style(self) -> None:
        app = QApplication.instance()
        if app is not None:
            apply_workbench_font(app)

    def _build_toolbar(self) -> None:
        tb = QToolBar("主工具栏")
        tb.setMovable(False)
        self.addToolBar(tb)

        title = QLabel("pyAOBS Workbench")
        title.setObjectName("WorkbenchAppTitle")
        tb.addWidget(title)

        for text, slot in (
            ("新建项目", self._create_project),
            ("打开项目", self._open_project),
            ("保存界面状态", self._save_ui_state),
            ("恢复界面状态", self._restore_ui_state),
            ("刷新", self._refresh_all),
        ):
            act = QAction(text, self)
            act.triggered.connect(slot)
            tb.addAction(act)
        tb.addSeparator()
        self._project_label = QLabel("当前项目：未打开")
        set_project_badge_style(self._project_label, project_open=False)
        tb.addWidget(self._project_label)

    def _build_body(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        split = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left.setObjectName("WorkbenchSidePanel")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(10, 10, 10, 10)
        ll.addWidget(section_title("项目资源树"))
        self._project_tree = QTreeWidget()
        self._project_tree.setAlternatingRowColors(True)
        self._project_tree.setHeaderHidden(True)
        self._project_tree.itemDoubleClicked.connect(self._on_tree_double_click)
        ll.addWidget(self._project_tree, stretch=1)
        split.addWidget(left)

        tab_wrap = QWidget()
        tab_wrap.setObjectName("WorkbenchTabPanel")
        tab_lay = QVBoxLayout(tab_wrap)
        tab_lay.setContentsMargins(6, 6, 6, 6)
        self._tabs = QTabWidget()
        self._history_tab = RunHistoryTab()
        self._runner_tab = RunnerTab(self.tool_registry.ids(), self.tool_registry)
        self._tabs.addTab(self._history_tab, "运行历史")
        self._tabs.addTab(self._runner_tab, "运行节点")
        tab_lay.addWidget(self._tabs)
        split.addWidget(tab_wrap)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        lay.addWidget(split)

        self._history_tab.open_run_folder.connect(self._open_run_folder)
        self._history_tab.prefill_runner.connect(self._runner_tab.set_run_payload)
        self._history_tab.delete_run.connect(self._delete_run)
        self._history_tab.bulk_delete_non_success.connect(self._bulk_delete_non_success)
        self._history_tab.rerun_failed.connect(self._rerun_selected_failed)
        self._history_tab.batch_rerun_failed.connect(self._batch_rerun_failed_filtered)

        self._runner_tab.run_requested.connect(self._run_selected_node)
        self._runner_tab.cancel_run.connect(self._cancel_run)
        self._runner_tab.apply_gui_form.connect(self._on_apply_gui_form)
        self._runner_tab.apply_template.connect(self._on_apply_template)
        self._runner_tab.fill_example.connect(self._fill_plugin_example)
        self._runner_tab.launch_lip_bridge.connect(self._launch_lip_bridge)
        self._runner_tab.refresh_petrology_bridge.connect(
            self._runner_tab.update_petrology_bridge_label
        )
        self._runner_tab.fill_petrology_from_imodel.connect(
            self._fill_petrology_from_latest_imodel
        )

    def _build_status(self) -> None:
        sb = self.statusBar()
        self._status_host = QWidget()
        sh = QHBoxLayout(self._status_host)
        sh.setContentsMargins(0, 0, 0, 0)
        self._status = status_bar_message_label("就绪", "info")
        sh.addWidget(self._status, stretch=1)
        sb.addWidget(self._status_host, stretch=1)
        self._task_status = status_bar_message_label("", "info")
        sb.addPermanentWidget(self._task_status)

    def _set_status(self, text: str, level: str = "info") -> None:
        """level: info | success | failed | running | warn"""
        self._status.setText(text)
        suffix = {
            "success": "Success",
            "failed": "Failed",
            "running": "Running",
            "warn": "Warn",
        }.get(level, "")
        self._status.setObjectName(
            "WorkbenchStatusBarMessage" + suffix if suffix else "WorkbenchStatusBarMessage"
        )
        self._status.style().unpolish(self._status)
        self._status.style().polish(self._status)

    def _set_task_status(self, text: str, level: str = "info") -> None:
        self._task_status.setText(text)
        suffix = {
            "success": "Success",
            "failed": "Failed",
            "running": "Running",
            "warn": "Warn",
        }.get(level, "")
        self._task_status.setObjectName(
            "WorkbenchStatusBarMessage" + suffix if suffix else "WorkbenchStatusBarMessage"
        )
        self._task_status.style().unpolish(self._task_status)
        self._task_status.style().polish(self._task_status)

    def _update_task_bar(self) -> None:
        if self._single_run_active:
            self._set_task_status("单任务: 运行中", "running")
        else:
            self._set_task_status("单任务: 空闲", "info")

    # ---- project ----
    def _create_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择项目目录（若不存在会自动创建）")
        if not path:
            return
        root = Path(path).resolve()
        overwrite = False
        if (root / "project.yaml").exists():
            overwrite = (
                QMessageBox.question(
                    self,
                    "项目已存在",
                    f"{root}\n已存在 project.yaml。\n是否覆盖初始化？",
                )
                == QMessageBox.StandardButton.Yes
            )
            if not overwrite:
                return
        try:
            self.current_project = self.project_manager.create_project(
                root, name=root.name, overwrite=overwrite
            )
        except ProjectError as exc:
            QMessageBox.critical(self, "创建项目失败", str(exc))
            return
        self._on_project_opened(f"已创建项目：{root}")

    def _open_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择 pyAOBS 项目目录")
        if not path:
            return
        root = Path(path).resolve()
        try:
            self.current_project = self.project_manager.open_project(root)
        except ProjectError as exc:
            QMessageBox.critical(self, "打开项目失败", str(exc))
            return
        self._restore_ui_state(silent=True)
        self._on_project_opened(f"已打开项目：{root}")

    def _on_project_opened(self, status: str) -> None:
        assert self.current_project is not None
        set_project_badge_style(
            self._project_label,
            project_open=True,
            path_text=str(self.current_project.root.name),
        )
        self._project_label.setToolTip(str(self.current_project.root))
        self._refresh_all()
        self._set_status(status, "success")

    def _refresh_all(self) -> None:
        if self.current_project is None:
            return
        self._refresh_project_tree()
        self._refresh_run_history()

    def _refresh_project_tree(self) -> None:
        if self.current_project is None:
            return
        self._tree_scan_token += 1
        token = self._tree_scan_token
        root_path = self.current_project.root

        def worker() -> None:
            nodes = scan_tree_nodes(root_path)
            QTimer.singleShot(0, lambda: self._apply_tree_scan(root_path, token, nodes))

        threading.Thread(target=worker, daemon=True).start()

    @Slot()
    def _apply_tree_scan(self, root_path: Path, token: int, nodes: list) -> None:
        if token != self._tree_scan_token or self.current_project is None:
            return
        if self.current_project.root.resolve() != root_path.resolve():
            return
        self._project_tree.clear()
        self._project_path_to_item.clear()
        root_item = QTreeWidgetItem([f"{root_path.name}/"])
        root_item.setData(0, Qt.ItemDataRole.UserRole, str(root_path.resolve()))
        self._project_tree.addTopLevelItem(root_item)
        self._project_path_to_item[str(root_path.resolve())] = root_item
        root_item.setExpanded(True)
        parents: dict[int, QTreeWidgetItem] = {0: root_item}
        for node in nodes:
            if node.get("placeholder"):
                depth = int(node.get("depth", 1))
                parent = parents.get(depth - 1, root_item)
                QTreeWidgetItem(parent, [str(node["placeholder"])])
                continue
            path_obj = node.get("path")
            if path_obj is None:
                continue
            depth = int(node.get("depth", 1))
            is_dir = bool(node.get("is_dir", False))
            parent = parents.get(depth - 1, root_item)
            label = Path(path_obj).name + ("/" if is_dir else "")
            item = QTreeWidgetItem(parent, [label])
            key = str(Path(path_obj).resolve())
            item.setData(0, Qt.ItemDataRole.UserRole, key)
            self._project_path_to_item[key] = item
            if is_dir:
                parents[depth] = item

    def _selected_tree_path(self) -> Path | None:
        item = self._project_tree.currentItem()
        if item is None:
            return None
        raw = item.data(0, Qt.ItemDataRole.UserRole)
        if not raw:
            return None
        return Path(str(raw))

    def _on_tree_double_click(self) -> None:
        p = self._selected_tree_path()
        if p is None or not p.exists():
            return
        if p.is_dir():
            subprocess.Popen(["explorer", str(p)])
        else:
            os.startfile(str(p))  # noqa: S606 — Windows file open

    def _refresh_run_history(self) -> None:
        if self.current_project is None:
            return
        self._history_scan_token += 1
        token = self._history_scan_token
        root = self.current_project.root

        def worker() -> None:
            records, statuses, nodes, _trunc = scan_run_history(root)
            QTimer.singleShot(
                0,
                lambda: self._apply_history_scan(records, statuses, nodes, token),
            )

        threading.Thread(target=worker, daemon=True).start()

    def _apply_history_scan(
        self,
        records: list,
        statuses: set[str],
        nodes: set[str],
        token: int,
    ) -> None:
        if token != self._history_scan_token:
            return

        self._run_id_to_dir = {
            str(r["run_id"]): Path(r["run_dir"]) for r in records if r.get("run_dir")
        }
        self._history_tab.set_records(records, statuses=statuses, nodes=nodes)

    # ---- run node ----
    def _on_apply_gui_form(self) -> None:
        msg = self._runner_tab.apply_gui_quick_form()
        self._set_status(msg)

    def _on_apply_template(self) -> None:
        if self._runner_tab.plugin_id() != "tomo2d.shell":
            self._set_status("当前插件不使用 TOMO2D 模板。")
            return
        tid = self._runner_tab._template.currentText().strip()
        if tid == TEMPLATE_CUSTOM:
            self._set_status("当前是自定义模板，直接编辑命令区即可。")
            return
        try:
            payload = self._runner_tab.build_template_payload()
        except ValueError as exc:
            QMessageBox.critical(self, "模板参数错误", str(exc))
            return
        self._runner_tab.apply_template_payload(payload)
        self._set_status("已根据模板生成命令。")

    def _fill_plugin_example(self) -> None:
        pid = self._runner_tab.plugin_id()
        if pid == "tomo2d.shell":
            self._on_apply_template()
            return
        self._on_apply_gui_form()

    def _run_selected_node(self) -> None:
        if self.current_project is None:
            QMessageBox.warning(self, "未打开项目", "请先新建或打开项目。")
            return
        if self._single_run_active:
            QMessageBox.information(self, "任务进行中", "已有节点任务正在运行。")
            return
        plugin_id = self._runner_tab.plugin_id()
        if not plugin_id:
            QMessageBox.warning(self, "未选择插件", "请先选择插件。")
            return

        payload = self._runner_tab.collect_run_payload()
        if should_auto_preflight(plugin_id):
            QMessageBox.information(
                self,
                "预检",
                "TOMO2D shell 节点：请确认命令区参数后运行。\n"
                "（完整预检对话框将在后续版本对齐 Tk 版。）",
            )

        payload = normalize_payload_for_plugin(plugin_id, payload)
        plugin = self.tool_registry.get(plugin_id)
        try:
            spec = plugin.build_spec(self.current_project.root, payload)
        except PluginValidationError as exc:
            QMessageBox.critical(self, "节点配置错误", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self, "插件错误", str(exc))
            return

        self._single_run_active = True
        self._single_run_cancel_event = threading.Event()
        self._runner_tab.set_run_buttons_enabled(running=True)
        self._update_task_bar()
        self._set_status(f"运行中：{' '.join(spec.command)}", "running")

        self._run_worker = _RunNodeWorker(
            self.run_manager,
            self.current_project,
            spec,
            self._single_run_cancel_event,
        )
        self._run_worker.finished.connect(self._on_run_finished)
        self._run_worker.start()

    @Slot(dict)
    def _on_run_finished(self, result: dict) -> None:
        self._single_run_active = False
        self._single_run_cancel_event = None
        self._runner_tab.set_run_buttons_enabled(running=False)
        self._update_task_bar()
        self._refresh_all()
        if result.get("error"):
            QMessageBox.critical(self, "运行异常", result["error"])
            self._set_status("运行失败：执行异常", "failed")
            return
        run_id = result.get("run_id", "")
        status = result.get("status", "unknown")
        code = result.get("return_code", "")
        if status == "success":
            self._set_status(f"运行完成：{run_id}（return_code={code}）", "success")
        elif status == "cancelled":
            self._set_status(f"运行已取消：{run_id}", "warn")
        else:
            QMessageBox.warning(
                self,
                "运行失败",
                f"节点运行失败。\nrun_id: {run_id}\nmanifest: {result.get('manifest_file', '')}",
            )
            self._set_status(f"运行失败：{run_id}（return_code={code}）", "failed")

    def _cancel_run(self) -> None:
        if not self._single_run_active or self._single_run_cancel_event is None:
            QMessageBox.information(self, "无运行任务", "当前没有可取消的节点任务。")
            return
        self._single_run_cancel_event.set()
        self._update_task_bar()
        self._set_status("已请求取消当前节点任务…", "warn")

    def _open_run_folder(self, run_id: str) -> None:
        run_dir = self._run_id_to_dir.get(run_id)
        if run_dir and run_dir.is_dir():
            subprocess.Popen(["explorer", str(run_dir)])

    def _delete_run(self, run_id: str) -> None:
        run_dir = self._run_id_to_dir.get(run_id)
        if run_dir is None or not run_dir.is_dir():
            return
        if (
            QMessageBox.question(self, "确认删除", f"删除工作区？\n{run_dir}")
            != QMessageBox.StandardButton.Yes
        ):
            return
        try:
            shutil.rmtree(run_dir)
        except Exception as exc:
            QMessageBox.critical(self, "删除失败", str(exc))
            return
        self._refresh_run_history()
        self._set_status(f"已删除工作区：{run_id}")

    def _bulk_delete_non_success(self) -> None:
        if not self._run_id_to_dir:
            return
        targets = []
        for run_id, run_dir in self._run_id_to_dir.items():
            manifest = run_dir / "manifest.json"
            if not manifest.is_file():
                continue
            try:
                st = str(json.loads(manifest.read_text(encoding="utf-8")).get("status", ""))
            except Exception:
                continue
            if st != "success":
                targets.append(run_dir)
        if not targets:
            QMessageBox.information(self, "无需清理", "没有非 success 的工作区。")
            return
        if (
            QMessageBox.question(self, "确认", f"删除 {len(targets)} 个非 success 工作区？")
            != QMessageBox.StandardButton.Yes
        ):
            return
        for d in targets:
            try:
                shutil.rmtree(d)
            except Exception:
                pass
        self._refresh_run_history()

    def _rerun_selected_failed(self) -> None:
        run_id = self._history_tab.selected_run_id()
        if not run_id:
            QMessageBox.warning(self, "未选择", "请先选择一条运行记录。")
            return
        run_dir = self._run_id_to_dir.get(run_id)
        if run_dir is None:
            return
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            QMessageBox.warning(self, "读取失败", "无法读取 manifest。")
            return
        cmd = manifest.get("command") or []
        payload = {
            "node_id": str(manifest.get("node_id", "")),
            "executable": str(cmd[0]) if cmd else "",
            "args": " ".join(str(x) for x in cmd[1:]) if len(cmd) > 1 else "",
            "cwd": str(manifest.get("cwd", "")),
            "env_text": "",
            "inputs_text": "",
        }
        self._runner_tab.set_run_payload(payload)
        self._tabs.setCurrentWidget(self._runner_tab)
        self._run_selected_node()

    def _batch_rerun_failed_filtered(self) -> None:
        QMessageBox.information(
            self,
            "批量复跑",
            "Qt 版 Workbench 当前支持单条失败复跑。\n"
            "完整批量队列 UI 将在下一迭代对齐 Tk 版。",
        )

    def _fill_petrology_from_latest_imodel(self) -> None:
        if self.current_project is None:
            return
        state_path = find_latest_imodel_gui_state(self.current_project.root)
        if state_path is None:
            QMessageBox.information(
                self,
                "未找到 imodel 会话",
                "当前项目 runs/ 下没有带 gui_state 的 imodel.gui 运行记录。",
            )
            return
        self._runner_tab.set_petrology_state_path(str(state_path))
        self._set_status(f"已填充最近 imodel gui_state：{state_path.name}")

    def _launch_lip_bridge(self) -> None:
        text = self._runner_tab._petrology_state.text().strip()
        if not text:
            QMessageBox.warning(self, "LIP Petrology", "请先指定 imodel gui_state JSON。")
            return
        state_path = Path(text)
        if not state_path.is_file():
            QMessageBox.critical(self, "LIP Petrology", f"gui_state 不存在:\n{text}")
            return
        try:
            launch_lip_from_workbench_state(state_path)
        except Exception as exc:
            QMessageBox.critical(self, "LIP Petrology", f"启动失败:\n{exc}")
            return
        self._runner_tab.update_petrology_bridge_label()
        self._set_status("已在独立进程启动 LIP Petrology。")

    # ---- ui state ----
    def _save_ui_state(self) -> None:
        if self.current_project is None:
            QMessageBox.warning(self, "未打开项目", "请先新建或打开项目。")
            return
        tab_idx = self._tabs.currentIndex()
        tab_text = self._tabs.tabText(tab_idx)
        state = {
            "window_geometry": self.saveGeometry().toHex().data().decode(),
            "selected_project_path": str(self.current_project.root),
            "right_tab_text": tab_text,
            "selected_run_id": self._history_tab.selected_run_id(),
            **self._history_tab.filter_state(),
            **self._runner_tab.ui_state_fragment(),
        }
        sel = self._selected_tree_path()
        if sel is not None:
            state["selected_tree_path"] = str(sel)
        self.state_store.save_ui_state(self.current_project, state)
        self._set_status("已保存界面状态到 state/ui_state.json")

    def _restore_ui_state(self, silent: bool = False) -> None:
        if self.current_project is None:
            if not silent:
                QMessageBox.warning(self, "未打开项目", "请先打开项目。")
            return
        state = self.state_store.load_ui_state(self.current_project)
        if not state:
            return
        geo = state.get("window_geometry")
        if geo:
            try:
                from PySide6.QtCore import QByteArray

                self.restoreGeometry(QByteArray.fromHex(str(geo).encode()))
            except Exception:
                pass
        self._history_tab.apply_filter_state(state)
        self._runner_tab.apply_ui_state_fragment(state)
        tab = str(state.get("right_tab_text", "") or "")
        for i in range(self._tabs.count()):
            if self._tabs.tabText(i) == tab:
                self._tabs.setCurrentIndex(i)
                break
        rid = str(state.get("selected_run_id", "") or "")
        if rid:
            self._history_tab.select_run_id(rid)
        if not silent:
            self._set_status("已恢复界面状态。")

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._single_run_active:
            if (
                QMessageBox.question(
                    self,
                    "任务运行中",
                    "节点任务仍在运行，确定退出？",
                )
                != QMessageBox.StandardButton.Yes
            ):
                event.ignore()
                return
        event.accept()

    def run(self) -> None:
        """Tk-compatible entry: start Qt event loop."""
        self.show()
        QApplication.instance().exec()
