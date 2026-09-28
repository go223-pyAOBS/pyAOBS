# -*- coding: utf-8 -*-
"""zplotpy 工程主窗（工区 + 阶段页 + 存盘）。

布局：顶部 新建/打开/保存 → 阶段 Tab → 主区 → 底部日志。
阶段：1 输入 → 2 波形/拾取 → 3 输出。姿态联合反演请用独立 ``processors.relocation.gui``。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..project import ZplotProject
from ..services.workdir_layout import prepare_workdir
from .help_dialog import install_help_shortcut, show_help_dialog
from .panels import InputPanel, OutputPanel, WorkbenchPanel
from .workbench_state import (
    gui_state_file_from_env,
    load_zplotpy_section,
    project_path_from_env,
    save_zplotpy_section,
)


class _FlexPane(QWidget):
    def __init__(self, min_w: int = 160, min_h: int = 100, parent=None) -> None:
        super().__init__(parent)
        self._min = QSize(min_w, min_h)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return self._min


class ZplotProjectWindow(QMainWindow):
    """工程化 zplotpy GUI。"""

    def __init__(self, argv: Optional[list[str]] = None) -> None:
        super().__init__()
        self.setWindowTitle("pyAOBS — zplotpy 波形工区")
        self.resize(1280, 900)
        self.project = ZplotProject()
        self._startup_argv = list(argv if argv is not None else sys.argv)
        self._gui_state_file = gui_state_file_from_env()
        self._startup_opened_project = False
        self._loading_panels = False
        self._build_ui()
        self._hide_menubar()
        install_help_shortcut(self)
        self._sync_panels_from_project()
        self.append_log(
            "流程：新建/打开工区 → 输入数据 → 波形/拾取 → 输出/保存工程。\n"
            "保存工区会自动写出查看器参数（outputs/viewer_params.json）；打开并加载数据时自动恢复。\n"
            "姿态联合反演请另启：python -m pyAOBS.processors.relocation.gui\n"
            "帮助：工具栏「帮助」（F1 / H）打开完整文档；窗口非模态可对照操作。"
        )
        QTimer.singleShot(0, self._startup_restore)

    # ---- UI ----
    def _build_ui(self) -> None:
        tb = QToolBar("主工具")
        tb.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)
        act_new = QAction("新建工区", self)
        act_new.triggered.connect(self._new_project)
        act_open = QAction("打开工区", self)
        act_open.triggered.connect(self._open_project)
        act_save = QAction("保存工区", self)
        act_save.triggered.connect(self._save_project)
        tb.addAction(act_new)
        tb.addAction(act_open)
        tb.addAction(act_save)

        # 帮助 / 退出放在工具栏「保存工区」后（单按钮打开 HELP.md）
        tb.addSeparator()
        act_help = QAction("帮助", self)
        act_help.triggered.connect(lambda: QTimer.singleShot(0, self._help_manual))
        tb.addAction(act_help)
        act_quit = QAction("退出", self)
        act_quit.triggered.connect(self.close)
        tb.addAction(act_quit)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        self.stage_bar = QTabBar()
        self.stage_bar.setObjectName("ZplotStageBar")
        self.stage_bar.setExpanding(False)
        self.stack = QStackedWidget()
        for title in ("1 输入", "2 波形/拾取", "3 输出"):
            self.stage_bar.addTab(title)
        self.stage_bar.currentChanged.connect(self._on_stage_changed)

        self.panel_input = InputPanel()
        self.panel_workbench = WorkbenchPanel()
        self.panel_output = OutputPanel()

        self.stack.addWidget(self.panel_input)
        self.stack.addWidget(self.panel_workbench)
        self.stack.addWidget(self.panel_output)

        top = _FlexPane(200, 200)
        top_l = QVBoxLayout(top)
        top_l.setContentsMargins(0, 0, 0, 0)
        top_l.setSpacing(2)
        top_l.addWidget(self.stage_bar)
        top_l.addWidget(self.stack, stretch=1)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(4000)
        self.log.setPlaceholderText("输出信息…")

        self.waveop_list_host = QWidget()
        waveop_host_lay = QVBoxLayout(self.waveop_list_host)
        waveop_host_lay.setContentsMargins(4, 4, 4, 4)
        waveop_host_lay.setSpacing(2)
        self.lbl_waveop_empty = QLabel("加载波形工作台后显示 V 段列表")
        self.lbl_waveop_empty.setStyleSheet("color:#64748b;")
        self.lbl_waveop_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        waveop_host_lay.addWidget(self.lbl_waveop_empty)

        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.setMinimumHeight(80)
        self.bottom_tabs.addTab(self.log, "输出")
        self.bottom_tabs.addTab(self.waveop_list_host, "V段")

        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_splitter.setObjectName("ZplotProjectSplitter")
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(8)
        self.main_splitter.addWidget(top)
        self.main_splitter.addWidget(self.bottom_tabs)
        self.main_splitter.setStretchFactor(0, 5)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([720, 180])
        root.addWidget(self.main_splitter)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("就绪")

        self.panel_input.project_changed.connect(self._panels_to_project)
        self.panel_output.project_changed.connect(self._panels_to_project)
        self.panel_input.request_load_workbench.connect(self._load_into_workbench)
        self.panel_output.request_export_all.connect(self._export_all_outputs)
        self.panel_output.request_open_outputs.connect(self._open_outputs_folder)
        self.panel_workbench.workbench_ready.connect(self._on_workbench_ready)
        # 空闲时预创建波形台，减少首次点「波形/拾取」时的等待与闪烁
        QTimer.singleShot(0, self._prefetch_workbench)

    def _prefetch_workbench(self) -> None:
        try:
            self.panel_workbench.ensure_viewer()
        except Exception:
            pass

    def _hide_menubar(self) -> None:
        """取消顶部「文件 / 帮助」菜单栏，动作已迁到工具栏。"""
        try:
            mb = self.menuBar()
            mb.clear()
            mb.setVisible(False)
            mb.setMaximumHeight(0)
        except Exception:
            pass

    def _on_stage_changed(self, idx: int) -> None:
        idx = int(idx)
        # 先准备好波形台再切页，避免先露出空白页再突然填入查看器
        if idx == 1:
            if not self._loading_panels:
                try:
                    self.panel_input.apply_to_project(self.project)
                except Exception:
                    pass
            self.panel_workbench.ensure_viewer()
            self._push_shared_terrain_into_viewer()
        self.stack.setCurrentIndex(idx)
        self.project.workflow.stage_index = idx
        if idx == 2:
            self._pull_workbench_into_project()

            def _reload() -> None:
                self.panel_output.load_from_project(self.project)

            self._with_panels_loading(_reload)

    # ---- log / sync ----
    def append_log(self, text: str) -> None:
        self.log.appendPlainText(str(text).rstrip())

    def _with_panels_loading(self, fn) -> None:
        self._loading_panels = True
        try:
            fn()
        finally:
            self._loading_panels = False

    def _panels_to_project(self) -> None:
        if self._loading_panels:
            return
        self.panel_input.apply_to_project(self.project)
        self.panel_output.apply_to_project(self.project)
        self._push_shared_terrain_into_viewer()
        self._pull_workbench_into_project()

    def _sync_panels_from_project(self) -> None:
        def _load() -> None:
            self.panel_input.load_from_project(self.project)
            self.panel_output.load_from_project(self.project)

        self._with_panels_loading(_load)
        title = "pyAOBS — zplotpy 波形工区"
        if self.project.workdir:
            title += f" [{self.project.name or Path(self.project.workdir).name}]"
        self.setWindowTitle(title)
        try:
            stage = int(self.project.workflow.stage_index or 0)
            stage = max(0, min(self.stage_bar.count() - 1, stage))
            self.stage_bar.setCurrentIndex(stage)
        except Exception:
            pass

    def _resolve_input_path(self, raw: str) -> str:
        s = str(raw or "").strip()
        if not s:
            return ""
        if os.path.isfile(s):
            return s
        joined = self.project.abs_or_join(s)
        if joined and os.path.isfile(joined):
            return joined
        return joined or s

    def _push_shared_terrain_into_viewer(self, *, log_ok: bool = False) -> None:
        viewer = self.panel_workbench.viewer
        if viewer is None:
            return
        raw = str(self.project.inputs.terrain_path or "").strip()
        path = self._resolve_input_path(raw) if raw else ""
        try:
            viewer._orientation_terrain_path = path
            viewer._orientation_geom = str(self.project.inputs.geom or "obs")
            if path and hasattr(viewer, "ensure_shared_terrain_loaded"):
                viewer.ensure_shared_terrain_loaded(path or None, force_reload=False)
                if log_ok:
                    self.append_log(f"已同步水深：{path}")
        except Exception as exc:
            self.append_log(f"同步水深失败：{exc}")

    def _pull_workbench_into_project(self) -> None:
        viewer = self.panel_workbench.viewer
        if viewer is None:
            return
        try:
            sels = getattr(viewer, "waveform_selections", None) or []
            self.project.waveform_selections = [dict(s) for s in sels]
        except Exception:
            pass
        try:
            corr = getattr(viewer, "_waveop_corrected_ttrue", {}) or {}
            self.project.waveop_corrected_ttrue = [
                {"trace_idx": int(k[0]), "pick_word": int(k[1]), "t_true": float(v)}
                for k, v in sorted(corr.items(), key=lambda kv: (int(kv[0][1]), int(kv[0][0])))
            ]
        except Exception:
            pass
        try:
            if getattr(viewer, "spin_apick", None) is not None:
                self.project.workflow.current_apick = int(viewer.spin_apick.value())
        except Exception:
            pass

    def _push_project_into_workbench(self) -> None:
        viewer = self.panel_workbench.viewer
        if viewer is None:
            return
        self._push_shared_terrain_into_viewer()
        try:
            if self.project.waveform_selections:
                viewer.waveform_selections = [dict(s) for s in self.project.waveform_selections]
        except Exception:
            pass
        try:
            corr = {}
            for item in self.project.waveop_corrected_ttrue or []:
                key = (int(item.get("trace_idx", -1)), int(item.get("pick_word", 1)))
                corr[key] = float(item.get("t_true", 0.0))
            viewer._waveop_corrected_ttrue = corr
        except Exception:
            pass
        try:
            if getattr(viewer, "spin_apick", None) is not None:
                viewer.spin_apick.setValue(int(self.project.workflow.current_apick))
        except Exception:
            pass

    # ---- project IO ----
    def _new_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择空目录作为工区根")
        if not path:
            return
        self.project = ZplotProject.create_new(path)
        self._sync_panels_from_project()
        self.append_log(f"已新建工区：{path}")
        self.statusBar().showMessage(f"工区：{path}", 5000)

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开工区 JSON",
            "",
            "Zplotpy project (zplotpy_project.json);;JSON (*.json);;All (*)",
        )
        if not path:
            return
        self._load_project_file(path, prompt_load_data=True)

    def _load_project_file(self, path: str, *, prompt_load_data: bool) -> bool:
        try:
            self.project = ZplotProject.load(path)
            prepare_workdir(self.project)
            self._sync_panels_from_project()
            self.append_log(f"已打开工程：{path}")
            self._try_load_sidecar_outputs()
            if self.project.inputs.dfile:
                load_now = True
                if prompt_load_data:
                    ans = QMessageBox.question(
                        self,
                        "加载数据",
                        "是否立即将工程中的 Z 数据加载到波形工作台？",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes,
                    )
                    load_now = ans == QMessageBox.StandardButton.Yes
                if load_now:
                    self._load_into_workbench(silent=not prompt_load_data)
            self._save_workbench_state()
            return True
        except Exception as exc:
            if prompt_load_data:
                QMessageBox.warning(self, "打开失败", str(exc))
            self.append_log(f"打开失败：{exc}")
            return False

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
            jp = ZplotProject.resolve_open_path(raw)
            if jp:
                return self._load_project_file(jp, prompt_load_data=False)
            p = Path(raw)
            if p.is_dir():
                try:
                    self.project = ZplotProject.create_new(str(p.resolve()))
                    self._sync_panels_from_project()
                    self.append_log(f"已绑定工区目录（尚未保存工程 JSON）：{p}")
                    return True
                except Exception:
                    continue
        return False

    def _startup_restore(self) -> None:
        if self._try_open_startup_project():
            self._startup_opened_project = True
            return
        if self._gui_state_file is None:
            return
        sec = load_zplotpy_section(self._gui_state_file)
        for key in ("project_json", "workdir"):
            raw = str(sec.get(key, "") or "").strip()
            if not raw:
                continue
            jp = ZplotProject.resolve_open_path(raw)
            if jp and self._load_project_file(jp, prompt_load_data=False):
                self.append_log(f"已从会话恢复工区：{jp}")
                return

    def _save_workbench_state(self) -> None:
        if self._gui_state_file is None or not self.project.workdir:
            return
        try:
            from ..services.workdir_layout import project_json_path

            jp = project_json_path(self.project.workdir)
            save_zplotpy_section(
                self._gui_state_file,
                workdir=self.project.workdir,
                project_json=jp if os.path.isfile(jp) else "",
            )
        except Exception:
            pass

    def _save_project(self) -> None:
        self._panels_to_project()
        if not self.project.workdir:
            path = QFileDialog.getExistingDirectory(self, "选择工区目录以保存")
            if not path:
                return
            self.project.workdir = path
            if not self.project.name:
                self.project.name = Path(path).name
        try:
            prepare_workdir(self.project)
            self.project.workflow.stage_index = int(self.stage_bar.currentIndex())
            self._export_all_outputs(silent=True)
            out = self.project.save()
            self._sync_panels_from_project()
            self._save_workbench_state()
            self.append_log(f"工程已保存：{out}")
            self.statusBar().showMessage(f"已保存 {out}", 5000)
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", str(exc))
            self.append_log(f"保存失败：{exc}")

    def _try_load_sidecar_outputs(self) -> None:
        """打开工程后，若 outputs 有独立文件则补齐内存快照。"""
        wp = self.project.abs_or_join(self.project.workflow.waveop_path)
        if wp and os.path.isfile(wp) and not self.project.waveform_selections:
            try:
                with open(wp, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                sels = payload.get("waveform_selections") or []
                if isinstance(sels, list):
                    self.project.waveform_selections = [dict(s) for s in sels]
                corr = payload.get("waveop_corrected_ttrue") or []
                if isinstance(corr, list):
                    self.project.waveop_corrected_ttrue = [dict(x) for x in corr]
                self.append_log(f"已读取 V段：{wp}")
            except Exception as exc:
                self.append_log(f"读取 waveop 失败：{exc}")
        vp = self.project.abs_or_join(self.project.workflow.viewer_params_path)
        if vp and os.path.isfile(vp):
            self.append_log(f"发现查看器参数：{vp}（加载数据后自动应用）")

    # ---- workbench / export ----
    def _load_into_workbench(self, *args, silent: bool = False) -> None:
        self._panels_to_project()
        dfile = self._resolve_input_path(self.project.inputs.dfile)
        if not dfile or not os.path.isfile(dfile):
            if not silent:
                QMessageBox.information(self, "加载数据", "请先在输入页指定有效的 .z 文件")
            return
        viewer = self.panel_workbench.ensure_viewer()
        self.stage_bar.setCurrentIndex(1)
        self._push_project_into_workbench()
        hfile = self._resolve_input_path(self.project.inputs.hfile)
        rfile = self._resolve_input_path(self.project.inputs.rfile)
        try:
            viewer._dfile = dfile
            viewer._hfile = hfile if hfile and os.path.isfile(hfile) else None
            viewer._rfile = rfile if rfile and os.path.isfile(rfile) else None
            viewer._orientation_geom = str(self.project.inputs.geom or "obs")
            if hasattr(viewer, "_update_file_open_status_label"):
                viewer._update_file_open_status_label()
            viewer._load_data()
            if getattr(viewer, "loaded", None) is None:
                raise RuntimeError("数据加载后 loaded 仍为空，请检查 .z 文件")
            # 应用 viewer_params sidecar（若有）
            vp = self.project.abs_or_join(self.project.workflow.viewer_params_path)
            if vp and os.path.isfile(vp) and hasattr(viewer, "_apply_ui_parameters"):
                try:
                    with open(vp, "r", encoding="utf-8") as f:
                        conf = json.load(f)
                    params = conf.get("parameters") if isinstance(conf, dict) else None
                    if isinstance(params, dict):
                        viewer._apply_ui_parameters(params)
                        self.append_log(f"已应用查看器参数：{vp}")
                except Exception as exc:
                    self.append_log(f"应用查看器参数失败：{exc}")
            # 恢复 picks
            pp = self.project.abs_or_join(self.project.workflow.picks_path)
            if pp and os.path.isfile(pp) and getattr(viewer, "pick_manager", None) is not None:
                try:
                    if hasattr(viewer.pick_manager, "load_picks"):
                        ok = viewer.pick_manager.load_picks(pp)
                        if ok:
                            self.append_log(f"已恢复拾取：{pp}")
                            if hasattr(viewer, "request_render"):
                                viewer.request_render(delay_ms=30)
                except Exception as exc:
                    self.append_log(f"恢复拾取失败：{exc}")
            self.append_log(f"已加载数据：{dfile}")
            if self.project.waveform_selections:
                try:
                    viewer.waveform_selections = [dict(s) for s in self.project.waveform_selections]
                    if hasattr(viewer, "request_render"):
                        viewer.request_render(delay_ms=30)
                    self.append_log(f"已恢复 V段 {len(self.project.waveform_selections)} 条")
                except Exception as exc:
                    self.append_log(f"恢复 V段失败：{exc}")
        except Exception as exc:
            if not silent:
                QMessageBox.warning(self, "加载失败", str(exc))
            self.append_log(f"加载失败：{exc}")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._save_workbench_state()
        event.accept()

    def _on_workbench_ready(self) -> None:
        viewer = self.panel_workbench.viewer
        if viewer is None:
            return
        viewer._shared_terrain_path_provider = self._shared_terrain_path_for_viewer
        self._push_shared_terrain_into_viewer()
        if hasattr(viewer, "relocate_waveop_list"):
            empty = getattr(self, "lbl_waveop_empty", None)
            if empty is not None:
                empty.setParent(None)
                self.lbl_waveop_empty = None
            viewer.relocate_waveop_list(self.waveop_list_host)
            if hasattr(viewer, "on_waveop_list_changed"):
                viewer.on_waveop_list_changed(self._on_waveop_list_count)
            try:
                n = len(viewer._current_apick_waveform_selections())  # type: ignore[attr-defined]
            except Exception:
                n = 0
            self._on_waveop_list_count(n)

    def _shared_terrain_path_for_viewer(self) -> str:
        return self._resolve_input_path(self.project.inputs.terrain_path)

    def _on_waveop_list_count(self, n: int) -> None:
        host = getattr(self, "waveop_list_host", None)
        tabs = getattr(self, "bottom_tabs", None)
        if host is None or tabs is None:
            return
        idx = tabs.indexOf(host)
        if idx < 0:
            return
        tabs.setTabText(idx, f"V段 ({int(n)})" if int(n) > 0 else "V段")

    def _export_all_outputs(self, silent: bool = False) -> None:
        self._panels_to_project()
        if not self.project.workdir:
            if not silent:
                QMessageBox.information(self, "导出", "请先新建或打开工区")
            return
        prepare_workdir(self.project)
        written = []
        viewer = self.panel_workbench.viewer
        # waveop
        wp = self.project.abs_or_join(self.project.workflow.waveop_path)
        try:
            os.makedirs(os.path.dirname(wp), exist_ok=True)
            payload = {
                "version": 1,
                "kind": "waveop_state",
                "dfile": self.project.inputs.dfile,
                "hfile": self.project.inputs.hfile,
                "rfile": self.project.inputs.rfile,
                "waveform_selections": list(self.project.waveform_selections or []),
                "waveop_corrected_ttrue": list(self.project.waveop_corrected_ttrue or []),
            }
            with open(wp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            written.append(wp)
        except Exception as exc:
            self.append_log(f"写 waveop 失败：{exc}")
        # viewer params
        vp = self.project.abs_or_join(self.project.workflow.viewer_params_path)
        if viewer is not None and hasattr(viewer, "_collect_ui_parameters"):
            try:
                os.makedirs(os.path.dirname(vp), exist_ok=True)
                payload = {
                    "version": 1,
                    "dfile": self.project.inputs.dfile,
                    "hfile": self.project.inputs.hfile,
                    "rfile": self.project.inputs.rfile,
                    "parameters": viewer._collect_ui_parameters(),
                }
                with open(vp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                written.append(vp)
            except Exception as exc:
                self.append_log(f"写查看器参数失败：{exc}")
        # picks
        pp = self.project.abs_or_join(self.project.workflow.picks_path)
        if viewer is not None and getattr(viewer, "pick_manager", None) is not None:
            try:
                os.makedirs(os.path.dirname(pp), exist_ok=True)
                ok = viewer.pick_manager.save_picks(pp, format="zplot")
                if ok:
                    written.append(pp)
            except Exception as exc:
                self.append_log(f"写 picks 失败：{exc}")

        msg = "已导出：\n" + "\n".join(written) if written else "未写出任何产物"
        self.append_log(msg.replace("\n", " | "))
        self.panel_output.set_status(msg)
        if not silent and written:
            self.statusBar().showMessage(f"已导出 {len(written)} 个文件", 4000)

    def _open_outputs_folder(self) -> None:
        if not self.project.workdir:
            QMessageBox.information(self, "输出", "请先新建或打开工区")
            return
        out_dir = os.path.join(self.project.workdir, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        from pyAOBS.utils.open_path import open_path_in_file_manager

        ok, msg = open_path_in_file_manager(out_dir)
        if ok:
            self.statusBar().showMessage(f"已打开：{out_dir}", 4000)
            self.append_log(f"已打开 outputs：{out_dir}")
        else:
            QMessageBox.information(self, "输出", msg)

    def _help_manual(self) -> None:
        """完整帮助（含快捷键与关于）：非模态 Markdown（docs/HELP.md）。"""
        show_help_dialog(activate=True)


def main() -> int:
    import os as _os
    import warnings

    import pyqtgraph as pg

    warnings.filterwarnings(
        "ignore",
        message=r"This figure includes Axes that are not compatible with tight_layout.*",
        category=UserWarning,
    )
    _os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    existing = QApplication.instance()
    created_here = existing is None
    app = existing if existing is not None else QApplication(sys.argv)
    pg.setConfigOptions(antialias=False, useOpenGL=True)
    win = ZplotProjectWindow(list(sys.argv))
    win.show()
    if created_here:
        return app.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
