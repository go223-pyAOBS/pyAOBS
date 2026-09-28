# -*- coding: utf-8 -*-
"""OBS 姿态校正工程主窗（对齐 RTM：工区 + 阶段页 + 存盘）。

布局：顶部 新建/打开/保存 → 阶段 Tab → 主区 → 底部「输出 | V段」页签。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction
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

from ..project import RelocationProject
from ..services.models import AttitudeSolution
from ..services.workdir_layout import prepare_workdir
from .help_dialog import install_help_shortcut, show_help_dialog
from .panels import AttitudePanel, InputPanel, OutputPanel, WorkbenchPanel


class _FlexPane(QWidget):
    def __init__(self, min_w: int = 160, min_h: int = 100, parent=None) -> None:
        super().__init__(parent)
        self._min = QSize(min_w, min_h)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return self._min


class RelocationProjectWindow(QMainWindow):
    """工程化姿态校正 GUI。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("pyAOBS — OBS 姿态校正工区")
        self.resize(1280, 900)
        self.project = RelocationProject()
        self._loading_panels = False
        self._build_ui()
        self._hide_menubar()
        install_help_shortcut(self)
        self._sync_panels_from_project()
        self.append_log(
            "流程：新建/打开工区 → 输入数据 → 波形/拾取（V选波）→ 姿态校正 → 输出/保存工程。\n"
            "底栏页签「输出 | V段」；帮助：工具栏「帮助」（F1）打开完整文档（非模态）。\n"
            "几何默认 geom=obs（与 RTM 一致）。保存工程会写出 meta/relocation_project.json 与 outputs/*。"
        )

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
        self.stage_bar.setObjectName("RelocationStageBar")
        self.stage_bar.setExpanding(False)
        self.stack = QStackedWidget()
        for title in ("1 输入", "2 波形/拾取", "3 姿态校正", "4 输出"):
            self.stage_bar.addTab(title)
        self.stage_bar.currentChanged.connect(self._on_stage_changed)

        self.panel_input = InputPanel()
        self.panel_workbench = WorkbenchPanel()
        self.panel_attitude = AttitudePanel()
        self.panel_output = OutputPanel()

        self.stack.addWidget(self.panel_input)
        self.stack.addWidget(self.panel_workbench)
        self.stack.addWidget(self.panel_attitude)
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
        self.bottom_tabs.addTab(self.log, "输出")
        self.bottom_tabs.addTab(self.waveop_list_host, "V段")

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(self.bottom_tabs)
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 1)
        split.setSizes([720, 160])
        root.addWidget(split)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("就绪")

        self.panel_input.project_changed.connect(self._panels_to_project)
        self.panel_attitude.project_changed.connect(self._panels_to_project)
        self.panel_output.project_changed.connect(self._panels_to_project)
        self.panel_input.request_load_workbench.connect(self._load_into_workbench)
        self.panel_attitude.request_open_workbench.connect(lambda: self.stage_bar.setCurrentIndex(1))
        self.panel_attitude.request_run.connect(self._run_attitude)
        self.panel_attitude.request_preview.connect(self._preview_attitude_solution)
        self.panel_output.request_export_all.connect(self._export_all_outputs)
        self.panel_output.request_open_outputs.connect(self._open_outputs_folder)
        self.panel_workbench.workbench_ready.connect(self._on_workbench_ready)

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
        self.stack.setCurrentIndex(int(idx))
        if int(idx) == 1:
            # 进入波形台前把面板姿态参数（含预置 shift）推入 viewer
            if not self._loading_panels:
                try:
                    self.panel_attitude.apply_to_project(self.project)
                except Exception:
                    pass
                try:
                    self.panel_input.apply_to_project(self.project)
                except Exception:
                    pass
            self.panel_workbench.ensure_viewer()
            self._push_attitude_ui_into_viewer()
            # 同步输入页地形，供位置 Map / 姿态共用自动加载
            self._push_shared_terrain_into_viewer()
        if int(idx) in (2, 3):
            # 从波形台拉回 UI/解，并刷新姿态面板（避免保存时用旧 spin 覆盖）
            self._pull_workbench_into_project(pull_attitude_ui=True)

            def _reload() -> None:
                self.panel_attitude.load_from_project(self.project)
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
        self.panel_attitude.apply_to_project(self.project)
        self.panel_output.apply_to_project(self.project)
        # 面板为 attitude_ui 权威来源：先推入波形台，再 pull 时勿用旧 viewer 默认值覆盖
        self._push_attitude_ui_into_viewer()
        # 输入页改地形后立即同步到工作台（位置 Map 打开时可直接叠加）
        self._push_shared_terrain_into_viewer()
        self._pull_workbench_into_project(pull_attitude_ui=False)

    def _sync_panels_from_project(self) -> None:
        def _load() -> None:
            self.panel_input.load_from_project(self.project)
            self.panel_attitude.load_from_project(self.project)
            self.panel_output.load_from_project(self.project)

        self._with_panels_loading(_load)
        title = f"pyAOBS — OBS 姿态校正工区"
        if self.project.workdir:
            title += f" [{self.project.name or Path(self.project.workdir).name}]"
        self.setWindowTitle(title)

    def _push_attitude_ui_into_viewer(self) -> None:
        """把工程中的姿态 UI 参数写入波形台（含预置走时 shift）。"""
        viewer = self.panel_workbench.viewer
        if viewer is None:
            return
        try:
            ui_dict = self.project.attitude_ui.to_dict()
            try:
                ui_dict["freqlo"] = float(viewer.spin_freqlo.value())
                ui_dict["freqhi"] = float(viewer.spin_freqhi.value())
                ui_dict["npoles"] = float(viewer.spin_npoles.value())
                ui_dict["izerop"] = 1.0 if viewer.chk_zerop.isChecked() else 0.0
            except Exception:
                pass
            viewer._orientation_ui_params.update(ui_dict)
        except Exception:
            pass

    def _pull_workbench_into_project(self, *, pull_attitude_ui: bool = True) -> None:
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
        if pull_attitude_ui:
            try:
                ui = getattr(viewer, "_orientation_ui_params", None) or {}
                if isinstance(ui, dict) and ui:
                    from ..services.models import AttitudeUiParams

                    # 与工程已有参数合并，避免 viewer 缺字段时用默认 0 冲掉面板值
                    merged = dict(self.project.attitude_ui.to_dict())
                    merged.update({k: float(v) for k, v in ui.items() if v is not None})
                    self.project.attitude_ui = AttitudeUiParams.from_dict(merged)
            except Exception:
                pass
        try:
            sol = getattr(viewer, "_orientation_current_solution", None) or {}
            last = getattr(viewer, "_orientation_last_applied_solution", None) or {}
            # 接受修正后 current 可能已归零，优先用 last_applied 写入工程
            if isinstance(last, dict) and any(
                abs(float(last.get(k, 0) or 0)) > 1e-9
                for k in (
                    "azimuth_deg",
                    "tilt_deg",
                    "dx",
                    "dy",
                    "dz",
                    "time_shift_sec",
                )
            ):
                sol = last
            if isinstance(sol, dict) and sol:
                self.project.attitude_solution = AttitudeSolution.from_dict(sol)
                self.panel_attitude.set_solution_from_project(self.project)
        except Exception:
            pass
        try:
            if getattr(viewer, "spin_apick", None) is not None:
                self.project.workflow.current_apick = int(viewer.spin_apick.value())
        except Exception:
            pass
        try:
            geom = str(getattr(viewer, "_orientation_geom", "obs") or "obs")
            self.project.inputs.geom = geom
        except Exception:
            pass

    def _shared_terrain_path_for_viewer(self) -> str:
        """给工作台解析最新地形路径（先同步输入面板；多候选探测实际文件）。"""
        try:
            if not self._loading_panels:
                self.panel_input.apply_to_project(self.project)
        except Exception:
            pass
        raw = str(self.project.inputs.terrain_path or "").strip()
        if not raw:
            return ""
        candidates: list[str] = []
        try:
            candidates.append(str(self.project.abs_or_join(raw) or "").strip())
        except Exception:
            pass
        candidates.append(raw)
        if not os.path.isabs(raw):
            dfile = str(self.project.inputs.dfile or "").strip()
            if dfile:
                candidates.append(os.path.normpath(os.path.join(os.path.dirname(dfile), raw)))
            wd = str(self.project.workdir or "").strip()
            if wd:
                candidates.append(os.path.normpath(os.path.join(wd, raw)))
                candidates.append(os.path.normpath(os.path.join(wd, "inputs", os.path.basename(raw))))
        seen = set()
        for c in candidates:
            c = str(c or "").strip()
            if not c or c in seen:
                continue
            seen.add(c)
            try:
                if os.path.isfile(c):
                    return c
            except Exception:
                continue
        # 找不到文件时仍返回 abs_or_join 结果，便于状态提示
        try:
            return str(self.project.abs_or_join(raw) or raw).strip()
        except Exception:
            return raw

    def _push_shared_terrain_into_viewer(self, *, log_ok: bool = False) -> None:
        viewer = self.panel_workbench.viewer
        if viewer is None:
            return
        try:
            viewer._shared_terrain_path_provider = self._shared_terrain_path_for_viewer
        except Exception:
            pass
        terr = self._shared_terrain_path_for_viewer()
        if not terr:
            return
        try:
            prev = str(getattr(viewer, "_orientation_terrain_path", "") or "")
            map_meta = getattr(viewer, "_location_map_terrain_meta", None)
            already = (
                prev == terr
                and isinstance(map_meta, dict)
                and str(map_meta.get("path", "")) == terr
            )
            if os.path.isfile(terr) and hasattr(viewer, "ensure_shared_terrain_loaded"):
                ok = viewer.ensure_shared_terrain_loaded(terr)
                if log_ok or (ok and not already):
                    if ok:
                        self.append_log(f"已共用输入页地形：{Path(terr).name}")
                    else:
                        self.append_log(f"地形自动加载未完成：{Path(terr).name}")
            else:
                viewer._orientation_terrain_path = terr
        except Exception as exc:
            self.append_log(f"设置地形路径提示：{exc}")

    def _push_project_into_workbench(self) -> None:
        viewer = self.panel_workbench.ensure_viewer()
        # 确保保存姿态结果回调已挂上
        try:
            viewer._orientation_solution_persist_cb = self._on_viewer_persist_orientation_solution
            viewer._orientation_ui_persist_cb = self._on_viewer_persist_orientation_ui
            viewer._shared_terrain_path_provider = self._shared_terrain_path_for_viewer
        except Exception:
            pass
        try:
            viewer._orientation_geom = str(self.project.inputs.geom or "obs")
        except Exception:
            pass
        self._push_attitude_ui_into_viewer()
        try:
            from ..services.models import AttitudeUiParams

            # 仅用主图带通覆盖频带字段，勿用 viewer 整表覆盖（防 prior 被默认 0 冲掉）
            ui_dict = dict(self.project.attitude_ui.to_dict())
            for k in ("freqlo", "freqhi", "npoles", "izerop"):
                if k in (getattr(viewer, "_orientation_ui_params", None) or {}):
                    ui_dict[k] = float(viewer._orientation_ui_params[k])
            self.project.attitude_ui = AttitudeUiParams.from_dict(ui_dict)
            viewer._orientation_ui_params.update(ui_dict)
        except Exception:
            pass
        try:
            viewer._orientation_current_solution.update(self.project.attitude_solution.to_dict())
        except Exception:
            pass
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

    def _apply_attitude_preview_to_viewer(self, viewer=None) -> bool:
        """把工程中的姿态解应用到波形台主图预览。"""
        viewer = viewer or self.panel_workbench.viewer
        if viewer is None:
            return False
        sol = self.project.attitude_solution.to_dict()
        if not hasattr(viewer, "apply_saved_orientation_preview"):
            try:
                viewer._orientation_current_solution.update(sol)
            except Exception:
                pass
            return False
        ok = bool(viewer.apply_saved_orientation_preview(sol, enabled=True))
        if ok:
            self.append_log(
                f"已开启姿态解预览：az={sol.get('azimuth_deg', 0):.2f}° "
                f"tilt={sol.get('tilt_deg', 0):.2f}°"
            )
        return ok

    # ---- project IO ----
    def _new_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择空目录作为工区根")
        if not path:
            return
        name = Path(path).name or "untitled"
        self.project = RelocationProject(name=name, workdir=path)
        prepare_workdir(self.project)
        self._sync_panels_from_project()
        self.append_log(f"已新建工区：{path}")
        self.statusBar().showMessage(f"工区：{path}", 5000)

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开工区 JSON",
            "",
            "Relocation project (relocation_project.json);;JSON (*.json);;All (*)",
        )
        if not path:
            return
        try:
            self.project = RelocationProject.load(path)
            prepare_workdir(self.project)
            self._sync_panels_from_project()
            self.append_log(f"已打开工程：{path}")
            # 尝试恢复产物快照
            self._try_load_sidecar_outputs()
            if self.project.inputs.dfile:
                ans = QMessageBox.question(
                    self,
                    "加载数据",
                    "是否立即将工程中的 Z 数据加载到波形工作台？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if ans == QMessageBox.StandardButton.Yes:
                    self._load_into_workbench()
        except Exception as exc:
            QMessageBox.warning(self, "打开失败", str(exc))
            self.append_log(f"打开失败：{exc}")

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
            self._export_all_outputs(silent=True)
            out = self.project.save()
            self._sync_panels_from_project()
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
        sp = self.project.abs_or_join(self.project.workflow.solution_path)
        if sp and os.path.isfile(sp):
            try:
                with open(sp, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                sol = payload.get("attitude_solution") or payload
                self.project.attitude_solution = AttitudeSolution.from_dict(sol)
                # attitude_ui 以主工程 JSON 为准，勿用 sidecar 旧快照覆盖预置 shift 等
                self.panel_attitude.load_from_project(self.project)
                sol = self.project.attitude_solution
                self.append_log(
                    f"已读取姿态解：{sp} | az={sol.azimuth_deg:.2f}° "
                    f"tilt={sol.tilt_deg:.2f}° dx={sol.dx:.3f} dy={sol.dy:.3f} dz={sol.dz:.3f} "
                    f"prior={sol.prior_tt_shift_sec:.3f}s corr={sol.tt_corr_sec:.3f}s "
                    f"final={sol.time_shift_sec:.3f}s"
                )
                # 推入初值；主图预览仅在用户点击「预览当前解」时开启
                if self.panel_workbench.viewer is not None:
                    self._push_project_into_workbench()
            except Exception as exc:
                self.append_log(f"读取姿态解失败：{exc}")

    # ---- workbench / run ----
    def _load_into_workbench(self) -> None:
        self._panels_to_project()
        dfile = self.project.inputs.dfile.strip()
        if not dfile or not os.path.isfile(dfile):
            QMessageBox.information(self, "加载数据", "请先在输入页指定有效的 .z 文件")
            return
        viewer = self.panel_workbench.ensure_viewer()
        self.stage_bar.setCurrentIndex(1)
        self._push_project_into_workbench()
        hfile = self.project.inputs.hfile.strip() or ""
        rfile = self.project.inputs.rfile.strip() or ""
        try:
            viewer._dfile = dfile
            viewer._hfile = hfile if hfile and os.path.isfile(hfile) else None
            viewer._rfile = rfile if rfile and os.path.isfile(rfile) else None
            if hasattr(viewer, "_update_file_open_status_label"):
                viewer._update_file_open_status_label()
            viewer._load_data()
            if getattr(viewer, "loaded", None) is None:
                raise RuntimeError("数据加载后 loaded 仍为空，请检查 .z 文件")
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
            QMessageBox.warning(self, "加载失败", str(exc))
            self.append_log(f"加载失败：{exc}")

    def _on_workbench_ready(self) -> None:
        viewer = self.panel_workbench.viewer
        if viewer is None:
            return
        viewer._orientation_solution_persist_cb = self._on_viewer_persist_orientation_solution
        viewer._orientation_ui_persist_cb = self._on_viewer_persist_orientation_ui
        viewer._shared_terrain_path_provider = self._shared_terrain_path_for_viewer
        self._push_attitude_ui_into_viewer()
        self._push_shared_terrain_into_viewer()
        # V 段列表迁到主窗底部「V段」页签，腾出参数条高度
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

    def _on_waveop_list_count(self, n: int) -> None:
        host = getattr(self, "waveop_list_host", None)
        tabs = getattr(self, "bottom_tabs", None)
        if host is None or tabs is None:
            return
        idx = tabs.indexOf(host)
        if idx < 0:
            return
        tabs.setTabText(idx, f"V段 ({int(n)})" if int(n) > 0 else "V段")

    def _on_viewer_persist_orientation_ui(self, ui: dict) -> None:
        """波形台姿态对话框改参后，同步到工程与姿态面板（含预置走时 shift）。"""
        try:
            from ..services.models import AttitudeUiParams

            merged = dict(self.project.attitude_ui.to_dict())
            for k, v in (ui or {}).items():
                if v is None:
                    continue
                try:
                    merged[k] = float(v)
                except (TypeError, ValueError):
                    continue
            self.project.attitude_ui = AttitudeUiParams.from_dict(merged)

            def _load() -> None:
                self.panel_attitude.load_from_project(self.project)

            self._with_panels_loading(_load)
        except Exception as exc:
            self.append_log(f"同步姿态 UI 参数失败：{exc}")

    def _on_viewer_persist_orientation_solution(self, sol: dict) -> None:
        """波形台「保存姿态结果」：写入工程内存 + outputs JSON，不改 .z/.hdr。"""
        try:
            self.project.attitude_solution = AttitudeSolution.from_dict(sol or {})
            self.panel_attitude.set_solution_from_project(self.project)
        except Exception as exc:
            self.append_log(f"更新工程姿态解失败：{exc}")
            raise
        if not self.project.workdir:
            # 无工区时至少更新面板；提示用户保存工区
            self.append_log("姿态解已记入工程内存；请先新建/打开工区后再落盘")
            QMessageBox.information(
                self,
                "保存姿态结果",
                "姿态解已记入工程。当前无工区目录，请「新建/打开工区」后再次保存或点「保存工区」。",
            )
            return
        prepare_workdir(self.project)
        sp = self.project.abs_or_join(self.project.workflow.solution_path)
        os.makedirs(os.path.dirname(sp), exist_ok=True)
        payload = {
            "attitude_solution": self.project.attitude_solution.to_dict(),
            "attitude_ui": self.project.attitude_ui.to_dict(),
            "terrain_path": self.project.inputs.terrain_path,
            "geom": self.project.inputs.geom,
        }
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        # 同步主工程 JSON 内嵌字段
        try:
            self.project.save()
        except Exception as exc:
            self.append_log(f"工程 JSON 同步提示：{exc}")
        self.append_log(f"姿态结果已保存：{sp}（未修改 .z/.hdr）")
        self.statusBar().showMessage(f"姿态结果已保存 {Path(sp).name}", 5000)

    def _preview_attitude_solution(self) -> None:
        """打开校正结果图（三分量/极化），不是整剖面主图预览。"""
        self._panels_to_project()
        viewer = self.panel_workbench.ensure_viewer()
        self.stage_bar.setCurrentIndex(1)
        self._push_project_into_workbench()
        if getattr(viewer, "loaded", None) is None:
            QMessageBox.information(self, "预览姿态解", "请先加载数据到波形工作台")
            return
        sol = self.project.attitude_solution.to_dict()
        if hasattr(viewer, "show_orientation_solution_result_figures"):
            ok = viewer.show_orientation_solution_result_figures(
                sol, apply_main_section_preview=False
            )
            if not ok:
                QMessageBox.information(
                    self,
                    "预览姿态解",
                    "当前无有效姿态解，或无法用现有 V 段生成结果图。请先运行校正。",
                )
            return
        QMessageBox.information(self, "预览姿态解", "当前工作台不支持结果图预览")

    def _run_attitude(self) -> None:
        self._panels_to_project()
        viewer = self.panel_workbench.viewer
        if viewer is None or getattr(viewer, "loaded", None) is None:
            QMessageBox.information(self, "姿态校正", "请先加载数据到波形工作台，并完成 V 选波")
            self.stage_bar.setCurrentIndex(1)
            return
        self._push_project_into_workbench()
        self._push_shared_terrain_into_viewer(log_ok=True)
        try:
            if hasattr(viewer, "_run_attitude_correction_placeholder"):
                viewer._run_attitude_correction_placeholder()
            elif hasattr(viewer, "_open_attitude_correction_dialog"):
                viewer._open_attitude_correction_dialog()
            else:
                raise RuntimeError("工作台无姿态校正入口")
            self.append_log("已打开姿态校正对话框（在工作台内完成运行）")
            self.stage_bar.setCurrentIndex(1)
        except Exception as exc:
            QMessageBox.warning(self, "姿态校正", str(exc))
            self.append_log(f"姿态校正失败：{exc}")

    def _export_all_outputs(self, silent: bool = False) -> None:
        self._panels_to_project()
        if not self.project.workdir:
            if not silent:
                QMessageBox.information(self, "导出", "请先新建或打开工区")
            return
        prepare_workdir(self.project)
        written = []
        # solution
        sp = self.project.abs_or_join(self.project.workflow.solution_path)
        try:
            os.makedirs(os.path.dirname(sp), exist_ok=True)
            payload = {
                "attitude_solution": self.project.attitude_solution.to_dict(),
                "attitude_ui": self.project.attitude_ui.to_dict(),
                "terrain_path": self.project.inputs.terrain_path,
                "geom": self.project.inputs.geom,
            }
            with open(sp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            written.append(sp)
        except Exception as exc:
            self.append_log(f"写姿态解失败：{exc}")
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
        # viewer params (best-effort)
        viewer = self.panel_workbench.viewer
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
    app = existing if existing is not None else QApplication(sys.argv)
    pg.setConfigOptions(antialias=True, background="w", foreground="k")
    win = RelocationProjectWindow()
    win.show()
    if existing is None:
        return int(app.exec())
    return 0
