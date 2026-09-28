# -*- coding: utf-8 -*-
"""OBS RTM 主窗口：顶部工具栏 + 顶部阶段页签 + 主区全宽绘图 + 底部日志。"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np
from PySide6.QtCore import (
    Q_ARG,
    QMetaObject,
    QProcess,
    QProcessEnvironment,
    QSize,
    Qt,
    QThread,
    QTimer,
    Slot,
)
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


class _FlexPane(QWidget):
    """降低 minimumSizeHint，避免子面板撑死外层 QSplitter 无法拖动。"""

    def __init__(self, min_w: int = 160, min_h: int = 100, parent=None) -> None:
        super().__init__(parent)
        self._min = QSize(min_w, min_h)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return self._min

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(900, 480)

from .panels import DataPanel, GeometryPanel, PreprocessPanel, RtmPanel, VelocityPanel
from .project import ObsRtmProject, list_shot_rsf, list_shot_rsf_by_indices
from .styles import apply_obs_rtm_chrome, apply_obs_rtm_font, section_title
from .services.batch_preprocess import process_rtm_shots
from .services.geometry import (
    apply_obs_x_shift,
    check_landing,
    check_offset_sign_consistency,
    load_xz_txt,
    suggest_grid_from_shots,
)
from .services.montage import (
    build_offset_montage,
    highlight_index_for_path,
    montage_offset_span_km,
    typical_montage_dx_km,
    used_path_index_map,
)
from .services.preprocess import (
    apply_bandpass_only,
    apply_display_gain,
    apply_mute_only,
    bandpass_backend,
    last_vel_mute_stats,
    time_axis,
)
from .services.rsf_io import parse_rsf_header
from .services.model_import import (
    convert_model_to_rsf,
    load_velocity_for_preview,
    suggest_grid_from_model_file,
)
from .services.rtm_job import (
    build_impulse_scons_cmd,
    build_rtm_run_cmd,
    format_obs_stack_label,
    load_image_for_preview,
    obs_stack_staleness,
    prepare_impulse_workdir,
    prepare_scons_workdir,
    rebuild_obs_image_stack,
    stack_shot_images,
    sync_obs_stack_manifest,
    try_export_img_lap_npy,
)
from .services.su_import import build_su_to_shots_cmd
from .services import preview_disk_cache as _pdisk
from .services.velocity import (
    bath_on_grid,
    build_builtin_1d_velocity,
    build_velocity_model,
    read_vel_rsf,
)
from .workers import start_worker


class ObsRtmMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("pyAOBS — OBS / Madagascar 偏移工区")
        self.resize(1200, 1080)
        self.project = ObsRtmProject()
        self._busy = False
        self._busy_cursor_depth = 0
        self._rtm_proc: Optional[QProcess] = None
        # 道集预览缓存：按数据源×炮子集分槽；切换 shots/proc/mute 命中缓存不重拼
        self._mont_caches: dict = {}
        self._mont_cache: Optional[dict] = None
        self._mont_cache_max = 6
        # 几何预览缓存键：文件 mtime + 网格/模式；切页未变则不重绘
        self._geom_preview_key: Optional[tuple] = None
        # 速度预览缓存键：(path, vin_dx, vin_dz) 或 builtin_1d 参数元组
        self._vel_preview_key: Optional[tuple] = None
        self._vel_preview_pending: Optional[bool] = None  # busy 时排队再刷
        # 用户模型预览体缓存：GUI 内一维↔文件切换免重读盘
        self._vel_file_preview_cache: Optional[dict] = None
        # 内置一维预览缓存：参数键未变则免重算
        self._vel_builtin_preview_cache: Optional[dict] = None
        # 偏移页当前底图对应的速度预览键（同键则只刷炮点/界面，不重绘体）
        self._rtm_vel_display_key: Optional[tuple] = None
        # 盘上 vel.rsf 由哪种速度来源写出（file / builtin_1d）；与面板来源不一致时偏移预览跟速度页
        self._vel_rsf_source: Optional[str] = None
        # 用户模型写出 vel 时的签名：(abs_path, vin_dx, vin_dz)；换模型/改栅格即过期
        self._vel_rsf_file_sig: Optional[tuple] = None
        # 自动生成 vel 完成后继续： "run" | "prepare" | None
        self._rtm_pending_after_vel: Optional[str] = None
        # 偏移页 Interfaces：缓存 (path, mtime, zelt_or_None)
        self._zelt_overlay_cache: Optional[tuple] = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(80)
        self._preview_timer.timeout.connect(self._preview_gather_now)
        # 多边形闭合后同步 RTM 炮集（合并短时间内多次 selection_changed）
        self._rtm_scope_timer = QTimer(self)
        self._rtm_scope_timer.setSingleShot(True)
        self._rtm_scope_timer.setInterval(120)
        self._rtm_scope_timer.timeout.connect(self._flush_rtm_scope_from_prep)
        # 打开/同步时暂停预览；拼图中防 processEvents 重入导致读盘两次
        self._preview_suspend = 0
        self._preview_pending = False
        self._montage_busy = False
        self._rtm_heartbeat = QTimer(self)
        self._rtm_heartbeat.setInterval(30000)
        self._rtm_heartbeat.timeout.connect(self._on_rtm_heartbeat)
        self._rtm_t0: Optional[float] = None
        self._rtm_mode: str = ""
        self._rtm_last_out_t: Optional[float] = None
        self._wfl_anim_timer = QTimer(self)
        self._wfl_anim_timer.timeout.connect(self._on_wfl_anim_tick)
        self._wfl_anim: Optional[dict] = None
        self._impulse_meta: Optional[dict] = None  # 最近一次脉冲成像参数
        self._closing = False
        self._loading_panels = False  # 同步面板时禁止回写，避免用旧控件值污染工程

        self._build_ui()
        apply_obs_rtm_chrome(self)
        self._build_menu()
        self._sync_panels_from_project(auto_preview=False)
        self.append_log(
            "流程：工区 → 几何 → 导入 → 预处理 → 速度 → RTM。\n"
            "偏移页默认 Madagascar awefd2d（scons）；大网格请加大 jsnap。"
        )

    def _build_ui(self) -> None:
        tb = QToolBar("主工具")
        tb.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)
        act_new = QAction("新建工区", self)
        act_new.setToolTip("选择空目录作为工区根，初始化工程参数")
        act_new.setStatusTip(act_new.toolTip())
        act_new.triggered.connect(self._new_project)
        act_open = QAction("打开工区", self)
        act_open.setToolTip("打开工区 JSON，同步各阶段面板与炮列表")
        act_open.setStatusTip(act_open.toolTip())
        act_open.triggered.connect(self._open_project)
        act_save = QAction("保存工区", self)
        act_save.setToolTip("把当前各面板参数写入工区 JSON")
        act_save.setStatusTip(act_save.toolTip())
        act_save.triggered.connect(self._save_project)
        tb.addAction(act_new)
        tb.addAction(act_open)
        tb.addAction(act_save)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # 上：阶段页签 + 参数/绘图（全宽）；下：日志通栏
        vsplit = QSplitter(Qt.Orientation.Vertical)
        vsplit.setChildrenCollapsible(False)
        vsplit.setOpaqueResize(True)
        vsplit.setHandleWidth(6)
        root.addWidget(vsplit, stretch=1)
        self._vsplit = vsplit

        top_wrap = _FlexPane(200, 120)
        top_l = QVBoxLayout(top_wrap)
        top_l.setContentsMargins(0, 0, 0, 0)
        top_l.setSpacing(4)

        stage_wrap = QWidget()
        stage_wrap.setObjectName("ObsRtmStageBar")
        stage_l = QVBoxLayout(stage_wrap)
        stage_l.setContentsMargins(4, 4, 4, 2)
        stage_l.setSpacing(0)
        self.stage_list = QTabBar()
        self.stage_list.setExpanding(False)
        self.stage_list.setDocumentMode(False)
        self.stage_list.setDrawBase(True)
        self.stage_list.setUsesScrollButtons(True)
        _stage_tips = (
            ("1. 数据加载", "SU → 按炮 RSF（su_to_shots）；设工区与分量"),
            ("2. 工区几何", "炮检几何与成像网格；预览/建议网格/检查落点"),
            ("3. 预处理 / 道集范围", "调带通增益；应用选道→shots_proc/；RTM 用当前选道"),
            ("4. 速度模型", "层析速度 → tomo_vel 中间体 / vel.rsf 成像速度"),
            ("5. 偏移作业", "OBS 为源互易 RTM（Madagascar awefd2d）/ 自定义与成像预览"),
        )
        for title, tip in _stage_tips:
            idx = self.stage_list.addTab(title)
            self.stage_list.setTabToolTip(idx, tip)
        self.stage_list.setCurrentIndex(0)
        self.stage_list.currentChanged.connect(self._on_stage)
        stage_l.addWidget(self.stage_list)
        top_l.addWidget(stage_wrap)

        self.stack = QStackedWidget()
        self.stack.setMinimumHeight(0)
        self.stack.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.panel_data = DataPanel()
        self.panel_geom = GeometryPanel()
        self.panel_prep = PreprocessPanel()
        self.panel_vel = VelocityPanel()
        self.panel_rtm = RtmPanel()
        for p in (
            self.panel_data,
            self.panel_geom,
            self.panel_prep,
            self.panel_vel,
            self.panel_rtm,
        ):
            p.setMinimumHeight(0)
            p.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            self.stack.addWidget(p)
        top_l.addWidget(self.stack, stretch=1)
        vsplit.addWidget(top_wrap)

        log_wrap = _FlexPane(200, 80)
        log_wrap.setObjectName("ObsRtmLogPanel")
        log_l = QVBoxLayout(log_wrap)
        log_l.setContentsMargins(8, 6, 8, 6)
        log_l.setSpacing(4)
        log_hdr = section_title("输出信息")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("作业日志…")
        self.log.setToolTip("作业与预览日志（只读）；导入/预处理/RTM 输出显示于此")
        self.log.setMinimumHeight(60)
        self.log.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        log_l.addWidget(log_hdr)
        log_l.addWidget(self.log, stretch=1)
        vsplit.addWidget(log_wrap)

        vsplit.setStretchFactor(0, 3)
        vsplit.setStretchFactor(1, 2)
        # 默认主区更高；日志区保留可读高度（勿过小导致显示不全）
        vsplit.setSizes([780, 220])

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("就绪")

        # SpinBox 稳健粘贴（Ctrl+V / 右键）；显示后再补一次（lineEdit 延迟创建）
        try:
            from PySide6.QtCore import QTimer

            from .dialog_utils import enable_spinbox_paste

            enable_spinbox_paste(self)
            QTimer.singleShot(0, lambda: enable_spinbox_paste(self))
        except Exception:
            pass

        self.panel_data.request_import.connect(self._run_import)
        self.panel_data.project_changed.connect(self._panels_to_project)
        self.panel_geom.request_check.connect(self._run_geom_check)
        self.panel_geom.request_check_offset_sign.connect(self._run_offset_sign_check)
        self.panel_geom.request_suggest_grid.connect(self._suggest_grid)
        self.panel_geom.request_preview.connect(self._preview_geometry)
        self.panel_geom.request_apply_obs_x.connect(self._apply_obs_x)
        self.panel_geom.project_changed.connect(self._panels_to_project)
        self.panel_prep.project_changed.connect(self._panels_to_project)
        self.panel_prep.request_preview.connect(self._schedule_preview)
        self.panel_prep.request_montage.connect(
            lambda *_: self._preview_montage(force_reload=False, switch_stage=True)
        )
        self.panel_prep.request_apply_mute.connect(self._apply_mute_selection)
        self.panel_prep.request_preview_proc.connect(
            lambda: self._preview_montage(
                force_reload=False, switch_stage=True, full_axes=True
            )
        )
        self.panel_prep.request_refresh_shots.connect(
            lambda: self._refresh_shot_list(auto_preview=True, force_reload=True)
        )
        self.panel_prep.shot_changed.connect(self._on_shot_changed_preview)
        self.panel_prep.request_style_only.connect(self._preview_style_only)
        self.panel_prep.selection_changed.connect(self._on_prep_selection_changed)
        self.panel_prep.impulse_point_picked.connect(self._on_impulse_point_picked)
        self.panel_prep.canvas.status.connect(self.append_log)
        self.panel_vel.project_changed.connect(self._panels_to_project)
        self.panel_vel.request_build.connect(self._build_velocity)
        self.panel_vel.request_preview_tomo.connect(self._preview_tomo)
        self.panel_vel.request_convert.connect(self._convert_tomo_rsf)
        self.panel_vel.request_sync_grid_from_model.connect(self._sync_grid_from_model)
        self.panel_vel.iface_sync_requested.connect(self._on_vel_iface_sync)
        self.panel_rtm.canvas.iface_changed.connect(self._on_rtm_iface_changed)
        self.panel_rtm.project_changed.connect(self._panels_to_project)
        self.panel_rtm.request_run.connect(self._run_rtm)
        self.panel_rtm.request_stop.connect(self._stop_rtm)
        self.panel_rtm.request_stack.connect(self._stack_rtm)
        self.panel_rtm.request_preview.connect(self._preview_rtm_image)
        self.panel_rtm.request_preview_vel.connect(
            lambda: self._preview_rtm_vel(silent=False)
        )
        self.panel_rtm.request_preview_wfl.connect(self._preview_rtm_wfl)
        self.panel_rtm.request_play_wfl.connect(self._play_rtm_wfl)
        self.panel_rtm.request_stop_wfl.connect(self._stop_rtm_wfl)
        self.panel_rtm.request_prepare_scons.connect(self._prepare_scons)
        self.panel_rtm.request_sync_time_from_shot.connect(self._sync_rtm_time_from_shot)
        self.panel_rtm.ed_vel.editingFinished.connect(
            lambda: self._preview_rtm_vel(silent=True)
        )

    def _build_menu(self) -> None:
        m = self.menuBar().addMenu("文件")
        a = m.addAction("新建工区", self._new_project)
        a.setToolTip("选择空目录作为工区根，初始化工程参数")
        a = m.addAction("打开…", self._open_project)
        a.setToolTip("打开工区 JSON，同步各阶段面板与炮列表")
        a = m.addAction("保存", self._save_project)
        a.setToolTip("把当前各面板参数写入工区 JSON")
        m.addSeparator()
        a = m.addAction("退出", self.close)
        a.setToolTip("关闭 OBS RTM 主窗口")
        h = self.menuBar().addMenu("帮助")
        a = h.addAction("关于框架", self._about)
        a.setToolTip("显示流程与依赖说明")

    def _on_stage(self, row: int) -> None:
        if 0 <= row < self.stack.count():
            self.stack.setCurrentIndex(row)
            if row == 1:
                # 切回几何：文件/网格未变则保留已绘，不刷状态栏
                self._ensure_geom_preview()
            if row == 2:
                # 切回预处理：炮列表与画布未变则跳过重绘
                self._ensure_gather_preview()
            if row == 3:
                # 切回速度页：模型/栅格未变则复用已绘速度图
                self._ensure_vel_preview()
            if row == 4:
                # 偏移页：同步单像下拉；底图与速度页预览同步（不写 vel.rsf）
                try:
                    self._refresh_rtm_img_source_combo()
                except Exception:
                    pass
                self._ensure_rtm_vel_preview()
                # 速度页已设的 B/S/M 同步到偏移画布（含 Interfaces 着色）
                self._apply_iface_to_canvas(self.panel_rtm.canvas)

    @staticmethod
    def _qt_alive(obj) -> bool:
        if obj is None:
            return False
        try:
            from shiboken6 import isValid

            return bool(isValid(obj))
        except Exception:
            try:
                obj.objectName()
                return True
            except RuntimeError:
                return False

    def append_log(self, text: str) -> None:
        """线程安全：禁止在工作线程直接写 QPlainTextEdit。"""
        if self._closing or not self._qt_alive(self):
            return
        msg = str(text)
        if QThread.currentThread() is not self.thread():
            QMetaObject.invokeMethod(
                self,
                "_append_log_ui",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, msg),
            )
            return
        self._append_log_ui(msg)

    @Slot(str)
    def _append_log_ui(self, text: str) -> None:
        if self._closing or not self._qt_alive(self):
            return
        try:
            self.log.appendPlainText(text)
        except RuntimeError:
            pass

    def _make_rtm_process_env(self) -> QProcessEnvironment:
        """
        RTM 子进程环境：继承 shell，并去掉会卡住 scons 的 IDE jobserver 变量。
        """
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        # Cursor/VS Code/外层 make 常注入 MAKEFLAGS=--jobserver-auth=…
        # scons 会一直等不存在的 jobserver → 只见 Building、数分钟无产物
        stripped = []
        for key in (
            "MAKEFLAGS",
            "MFLAGS",
            "MAKELEVEL",
            "GNUMAKEFLAGS",
            "SCONSFLAGS",
        ):
            if env.contains(key):
                val = env.value(key) or ""
                env.remove(key)
                stripped.append("%s=%s" % (key, val[:80]))
        if stripped:
            self.append_log(
                "已清除会卡住 scons 的环境变量: %s" % "; ".join(stripped)
            )

        # 波场体强制写 Linux 本地盘（勿落在 /mnt/d）
        dp = (env.value("DATAPATH") or "").strip()
        if (not dp) or dp.rstrip("/").startswith("/mnt/"):
            dp = "/var/tmp/obs_rtm_qt/"
            try:
                os.makedirs(dp, exist_ok=True)
            except OSError:
                dp = "/tmp/obs_rtm_qt/"
                try:
                    os.makedirs(dp, exist_ok=True)
                except OSError:
                    pass
            if not dp.endswith(os.sep) and not dp.endswith("/"):
                dp = dp + "/"
            env.insert("DATAPATH", dp)
            self.append_log("DATAPATH → %s（避免波场写到 /mnt/d）" % dp)
        else:
            if not dp.endswith("/") and not dp.endswith(os.sep):
                dp = dp + "/"
                env.insert("DATAPATH", dp)

        omp = (env.value("OMP_NUM_THREADS") or "").strip()
        try:
            ncpu = int(os.cpu_count() or 1)
        except Exception:
            ncpu = 1
        if not omp:
            if ncpu > 1:
                env.insert("OMP_NUM_THREADS", str(ncpu))
                omp = str(ncpu)
            else:
                omp = "(default)"
        elif omp == "1" and ncpu > 1:
            self.append_log(
                "注意: OMP_NUM_THREADS=1，awefd2d 将单核偏慢；"
                "终端若更快请 unset 或设为 %d" % ncpu
            )
        self.append_log(
            "RTM 环境: OMP_NUM_THREADS=%s  cpu_count=%d  DATAPATH=%s"
            % (omp, ncpu, env.value("DATAPATH") or "(unset)")
        )
        return env

    def _rtm_child_diag(self) -> str:
        """心跳诊断：作业目录里的 .rsf + 是否已有 sfawefd2d。"""
        parts: List[str] = []
        try:
            if str(self._rtm_mode or "") == "impulse":
                from .services.impulse_gather import impulse_run_dir

                run = impulse_run_dir(self.project)
            else:
                from .services.rtm_job import rtm_run_dir

                run = rtm_run_dir(self.project)
            names = sorted(
                n
                for n in os.listdir(run)
                if n.endswith(".rsf") or n.startswith("SConstruct")
            )
            parts.append("文件[%s]" % (",".join(names[:14]) or "无"))
        except OSError:
            pass
        awefd: List[str] = []
        try:
            for name in os.listdir("/proc"):
                if not name.isdigit():
                    continue
                try:
                    with open("/proc/%s/cmdline" % name, "rb") as f:
                        raw = f.read()
                except OSError:
                    continue
                if not raw:
                    continue
                cmd = raw.replace(b"\x00", b" ").decode("utf-8", "replace")
                if "awefd2d" in cmd:
                    awefd.append("pid=%s" % name)
                    if len(awefd) >= 3:
                        break
        except OSError:
            pass
        parts.append(
            "awefd2d[%s]" % (",".join(awefd) if awefd else "未启动")
        )
        # 若父进程环境曾带 jobserver，提示一眼可辨
        try:
            mf = os.environ.get("MAKEFLAGS") or ""
            if "jobserver" in mf:
                parts.append("警告:本进程MAKEFLAGS含jobserver")
        except Exception:
            pass
        return " | ".join(parts) if parts else ""

    def _shutdown_bg_process(self, *, wait_ms: int = 4000) -> None:
        """关闭窗口 / 停止作业时：断开信号并杀掉仍在跑的 QProcess。"""
        try:
            self._rtm_heartbeat.stop()
        except Exception:
            pass
        self._rtm_t0 = None
        proc = self._rtm_proc
        self._rtm_proc = None
        if proc is None:
            return
        try:
            proc.readyReadStandardOutput.disconnect()
        except Exception:
            pass
        try:
            proc.finished.disconnect()
        except Exception:
            pass
        try:
            if proc.state() != QProcess.ProcessState.NotRunning:
                # 先 terminate，再 kill，尽量带走 scons 子进程
                proc.terminate()
                if not proc.waitForFinished(max(500, wait_ms // 2)):
                    proc.kill()
                    proc.waitForFinished(max(500, wait_ms // 2))
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.deleteLater()
        except Exception:
            pass

    def closeEvent(self, event) -> None:  # noqa: N802
        """关窗时绝不能让 QProcess 在仍运行时被 GC 销毁。"""
        self._closing = True
        self._shutdown_bg_process(wait_ms=5000)
        try:
            self.panel_rtm.set_running(False)
        except Exception:
            pass
        super().closeEvent(event)

    def _notify(self, msg: str, *, log: bool = False, process_events: bool = True) -> None:
        """状态栏即时反馈，避免用户误以为界面卡死。"""
        self.statusBar().showMessage(msg)
        if log:
            self.append_log(msg)
        if process_events:
            QApplication.processEvents()

    def _commit_spinbox_edits(self) -> None:
        """把尚未回车确认的 SpinBox 文本提交进 value()（KeyboardTracking=False 时尤甚）。"""
        from PySide6.QtWidgets import QAbstractSpinBox

        for w in self.findChildren(QAbstractSpinBox):
            try:
                w.interpretText()
            except Exception:
                pass

    def _panels_to_project(self) -> None:
        if self._loading_panels or self._closing:
            return
        self._commit_spinbox_edits()
        self.panel_data.apply_to_project(self.project)
        self.panel_geom.apply_to_project(self.project)
        self.panel_prep.apply_to_project(self.project)
        self.panel_vel.apply_to_project(self.project)
        self.panel_rtm.apply_to_project(self.project)

    def _with_panels_loading(self, fn) -> None:
        """在回调内禁止 project_changed → apply 回写。"""
        self._loading_panels = True
        try:
            fn()
        finally:
            self._loading_panels = False

    def _suspend_preview(self) -> None:
        """暂停定时预览（打开工程 / 同步面板时避免排队二次拼图）。"""
        self._preview_suspend += 1
        self._preview_timer.stop()

    def _resume_preview(self, *, flush: bool = False) -> None:
        self._preview_suspend = max(0, int(self._preview_suspend) - 1)
        if self._preview_suspend > 0:
            return
        if flush and self._preview_pending:
            self._preview_pending = False
            self._schedule_preview()
        else:
            self._preview_pending = False

    def _sync_panels_from_project(self, *, auto_preview: bool = True) -> None:
        # load_* 会改控件并发出 project_changed；若此时回写会用「尚未 load 的面板」默认值覆盖 JSON
        self._suspend_preview()
        self._rtm_scope_timer.stop()
        try:
            self._with_panels_loading(
                lambda: (
                    self.panel_data.load_from_project(self.project),
                    self.panel_geom.load_from_project(self.project),
                    self.panel_prep.load_from_project(self.project),
                    self.panel_vel.load_from_project(self.project),
                    self.panel_rtm.load_from_project(self.project),
                )
            )
            # 面板已还原：立刻同步 RTM 炮表/黄星（勿等 selection 定时器）
            self._refresh_rtm_shot_markers()
            self._refresh_shot_list(auto_preview=auto_preview)
        finally:
            # 不 flush：由调用方显式 auto_preview / 打开流程负责一次预览
            self._resume_preview(flush=False)

    def _set_busy(self, busy: bool, msg: str = "") -> None:
        """busy + 等待光标 + 状态栏说明。"""
        if busy:
            if not self._busy:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                self._busy_cursor_depth += 1
            self._busy = True
            self._notify(msg or "请稍候…（界面未卡死，正在处理）", log=False)
        else:
            self._busy = False
            if self._busy_cursor_depth > 0:
                QApplication.restoreOverrideCursor()
                self._busy_cursor_depth -= 1
            self.statusBar().showMessage(msg or "就绪")
            QApplication.processEvents()

    def _invalidate_preview_cache(self, *, forget_shot_list: bool = False) -> None:
        """清空拼图缓存。默认保留 _last_shot_paths，避免切页被当成「炮列表变了」再拼一次。"""
        self._mont_caches.clear()
        self._mont_cache = None
        if forget_shot_list:
            self._last_shot_paths = None
        self._geom_preview_key = None
        self._vel_preview_key = None
        try:
            from .services.polygon_mute import invalidate_offset_cache

            if self.project.workdir:
                invalidate_offset_cache(self.project.workdir)
        except Exception:
            pass

    def _store_mont_cache(self, load_key: tuple, cache: dict) -> None:
        """按 load_key 分槽缓存；优先保留各数据源的全炮槽；并落盘供重开复用。"""
        self._mont_caches[load_key] = cache
        self._mont_cache = cache
        try:
            if self.project.workdir:
                _pdisk.save_mont(
                    self.project.workdir,
                    load_key,
                    cache,
                    max_slots=int(self._mont_cache_max),
                )
        except Exception:
            pass
        if len(self._mont_caches) <= int(self._mont_cache_max):
            return
        # load_key: (workdir, src, stride, maxtr, filt_sig, ...)
        # 超限时先丢子集槽，再丢非当前源
        extras = [k for k in self._mont_caches if k != load_key]
        cur_src = load_key[1] if len(load_key) > 1 else None

        def _evict_rank(k: tuple) -> tuple:
            filt = k[4] if len(k) > 4 else None
            src = k[1] if len(k) > 1 else None
            # 子集优先删；其它数据源次之；同数据源全炮尽量留
            return (filt is None, src == cur_src, str(k))

        extras.sort(key=_evict_rank)  # False 在前 → 子集先删
        while len(self._mont_caches) > int(self._mont_cache_max) and extras:
            self._mont_caches.pop(extras.pop(0), None)

    def _load_mont_cache_from_disk(self, load_key: tuple) -> Optional[dict]:
        """从工区 ``.obs_rtm_preview_cache`` 恢复拼图；失败返回 None。"""
        if not self.project.workdir:
            return None
        try:
            cache = _pdisk.load_mont(self.project.workdir, load_key)
        except Exception:
            return None
        if cache is None:
            return None
        self._mont_caches[load_key] = cache
        self._mont_cache = cache
        return cache

    def _invalidate_mont_caches_for_sources(self, *sources: str) -> None:
        """写出定稿后丢弃对应数据源缓存（目录 mtime 不可靠）。"""
        want = {str(s) for s in sources if s}
        if not want:
            return
        drop = [k for k in self._mont_caches if len(k) > 1 and str(k[1]) in want]
        for k in drop:
            self._mont_caches.pop(k, None)
        if self._mont_cache is not None:
            lk = self._mont_cache.get("load_key")
            if lk is not None and len(lk) > 1 and str(lk[1]) in want:
                self._mont_cache = None
        try:
            if self.project.workdir:
                _pdisk.invalidate_mont_sources(self.project.workdir, list(want))
        except Exception:
            pass

    def _remap_mont_caches_after_obs_x(self, delta: float) -> None:
        """
        OBS x 应用后：平移各槽 offs（model x），并按新几何重写 load_key。
        振幅 / mute / 增益层保留（相对 offset 不变，无需重读盘）。
        """
        g = self.project.geometry
        geom_s = str(getattr(g, "geom", "") or "")
        obs_x = float(getattr(g, "obs_x_km", 0.0) or 0.0)
        sign = float(getattr(g, "offset_sign", 1.0) or 1.0)
        mt_sx = self._file_mtime(self.project.path(self.project.shots_xz))
        mt_ox = self._file_mtime(self.project.path(self.project.obs_xz))
        mt_off = self._file_mtime(self.project.path(self.project.offsets_txt))
        try:
            d = float(delta)
        except (TypeError, ValueError):
            d = 0.0

        new_map: dict = {}
        for old_key, cache in list(self._mont_caches.items()):
            if not isinstance(cache, dict):
                continue
            offs = cache.get("offs")
            if offs is not None and abs(d) > 1e-15:
                cache["offs"] = np.asarray(offs, dtype=float) + d
            if isinstance(old_key, tuple) and len(old_key) >= 12:
                new_key = (
                    old_key[0],
                    old_key[1],
                    old_key[2],
                    old_key[3],
                    old_key[4],
                    geom_s,
                    obs_x,
                    sign,
                    mt_sx,
                    mt_ox,
                    mt_off,
                    old_key[11],
                )
            else:
                new_key = old_key
            cache["load_key"] = new_key
            new_map[new_key] = cache
        self._mont_caches = new_map
        if self._mont_cache is not None:
            lk = self._mont_cache.get("load_key")
            if lk in new_map:
                self._mont_cache = new_map[lk]
            elif lk is not None:
                # 当前槽曾游离于分槽表：补平移并登记
                offs = self._mont_cache.get("offs")
                if offs is not None and abs(d) > 1e-15:
                    self._mont_cache["offs"] = np.asarray(offs, dtype=float) + d
                self._mont_caches[lk] = self._mont_cache
        # OBS x 变更后按新键落盘，避免重开仍命中旧几何槽
        try:
            if self.project.workdir:
                for k, c in self._mont_caches.items():
                    _pdisk.save_mont(
                        self.project.workdir,
                        k,
                        c,
                        max_slots=int(self._mont_cache_max),
                    )
        except Exception:
            pass

    def _ensure_gather_preview(self) -> None:
        """进入预处理页：炮列表未变且画布已有图则跳过重绘。"""
        self._refresh_shot_list(auto_preview=True, force_reload=False)

    def _refresh_shot_list(
        self, *, auto_preview: bool = True, force_reload: bool = False
    ) -> None:
        """刷新炮列表；仅当炮集变化或 force_reload 时清空拼图缓存并重拼。"""
        if not self.project.workdir:
            self._invalidate_preview_cache()
            self.panel_prep.set_shot_list([])
            return
        paths = list_shot_rsf(self.project)
        prev = getattr(self, "_last_shot_paths", None)
        changed = force_reload or (list(paths) != list(prev or []))
        self._last_shot_paths = list(paths)
        self.panel_prep.set_shot_list(paths)

        if changed:
            self._mont_caches.clear()
            self._mont_cache = None
            self._notify("扫描炮集已完成：找到 %d 炮" % len(paths), log=True)
        else:
            self.statusBar().showMessage("shots: %d（缓存）" % len(paths))

        if not auto_preview or not paths:
            if not paths:
                self.append_log("扫描炮集已完成：尚无 shot_*.rsf")
            return

        if self.panel_prep.preview_all_shots():
            if self._mont_cache is not None and not changed:
                # 画布已有图：切页不再重绘；仅画布空时从缓存恢复一次
                if self.panel_prep.canvas._data is not None:
                    self.statusBar().showMessage(
                        "道集预览已就绪（缓存，未重绘）· %d 炮" % len(paths)
                    )
                    return
                self._preview_montage(switch_stage=False, force_reload=False)
                self.append_log("道集预览拼装已完成（缓存恢复）· %d 炮" % len(paths))
                self.statusBar().showMessage(
                    "道集预览已就绪（缓存，未重拼）· %d 炮" % len(paths)
                )
                return
            self._notify(
                "正在拼装道集预览（炮多时需数秒，请稍候）…",
                log=True,
            )
            self.panel_prep.canvas.clear("正在拼装道集预览，请稍候…")
            self._preview_gather()
            self.append_log("道集预览拼装已完成 · %d 炮" % len(paths))
        elif changed:
            self._preview_gather()
            self.append_log("单炮预览已完成")
        elif self.panel_prep.canvas._data is None:
            self._preview_gather()
            self.append_log("道集预览已完成")
        else:
            self.statusBar().showMessage(
                "道集预览已就绪（缓存，未重绘）· %d 炮" % len(paths)
            )

    def _schedule_preview(self) -> None:
        """增益旋钮等频繁触发时合并到一次刷新。"""
        if self._preview_suspend > 0 or self._loading_panels or self._montage_busy:
            self._preview_pending = True
            return
        self._preview_timer.start()

    def _preview_gather_now(self) -> None:
        if self._preview_suspend > 0 or self._loading_panels or self._montage_busy:
            self._preview_pending = True
            return
        self._preview_gather()

    def _on_shot_changed_preview(self) -> None:
        """换炮：拼图缓存命中时只改高亮（绝不重读 RSF）。"""
        self._apply_shot_highlight()

    def _montage_highlight_for_shot(self, sel: Optional[str]) -> Optional[int]:
        """在缓存 montage 中定位选中炮；不在列表时按 offset 最近邻，不触发重载。"""
        cache = self._mont_cache
        if cache is None:
            load_key = self._montage_load_key()
            cache = self._mont_caches.get(load_key)
            if cache is not None:
                self._mont_cache = cache
        if not sel or cache is None:
            return None
        used = cache.get("used") or []
        index_map = cache.get("used_index")
        if index_map is None:
            index_map = used_path_index_map(used)
            cache["used_index"] = index_map
        hl = highlight_index_for_path(used, sel, index_map=index_map)
        if hl is not None:
            return hl
        offs = cache.get("offs")
        if offs is None:
            return None
        try:
            ishot = int(
                os.path.basename(sel).replace("shot_", "").replace(".rsf", "")
            )
            # 与拼图横轴一致：模型测线坐标；勿在此路径触发读盘拼图
            from .services.montage import montage_model_x_km

            off = float(montage_model_x_km(self.project, ishot))
            arr = np.asarray(offs, dtype=float)
            if arr.size == 0:
                return None
            return int(np.argmin(np.abs(arr - off)))
        except Exception:
            return None

    def _apply_shot_highlight(self) -> None:
        """换炮：拼图缓存命中时只改高亮（全炮/手选子集同逻辑，不重读 RSF）。"""
        load_key = self._montage_load_key()
        cache = self._mont_caches.get(load_key)
        if cache is None:
            self._schedule_preview()
            return
        self._mont_cache = cache
        sel = self.panel_prep.current_shot_path()
        # 临时让 _montage_highlight_for_shot 用当前槽
        hl = self._montage_highlight_for_shot(sel)
        title = cache.get("title_base", "道集")
        note = ""
        if sel:
            title = "%s · 选中 %s" % (title, os.path.basename(sel))
            used = cache.get("used") or []
            index_map = cache.get("used_index")
            if (
                highlight_index_for_path(used, sel, index_map=index_map) is None
                and hl is not None
            ):
                note = "（未在拼图中，高亮最近道；可减小 stride / 增大 max）"
        self.panel_prep.canvas.set_highlight_idx(hl)
        if self.panel_prep.canvas.plot is not None:
            self.panel_prep.canvas.plot.setTitle(title)
        self.statusBar().showMessage(
            "%s  hl=%s%s" % (title, str(hl) if hl is not None else "-", note)
        )

    def _preview_style_only(self) -> None:
        """仅 dscale / 显示模式 / pclip：不重算增益。"""
        if self.panel_prep.canvas._data is None:
            self._schedule_preview()
            return
        self._panels_to_project()
        self.panel_prep.canvas.update_display_style(
            dscale=self.panel_prep.dscale(),
            pclip=self.panel_prep.pclip(),
            display_mode=self.panel_prep.display_mode(),
            display_vred=self.panel_prep.display_vred(),
        )

    def _montage_load_key(self) -> tuple:
        """读盘拼图键；含炮子集、数据源与工区几何（横轴=model x）。"""
        filt = self.panel_prep.montage_shot_filter()
        filt_sig = tuple(int(i) for i in filt) if filt is not None else None
        g = self.project.geometry
        src = self.panel_prep.montage_source()
        if src == "shots_proc":
            src_dir = self.project.path(self.project.shots_proc_dir)
        elif src == "shots_mute":
            src_dir = self.project.path(
                getattr(self.project, "shots_mute_dir", None) or "shots_mute"
            )
        else:
            src_dir = self.project.path(self.project.shots_dir)
        return (
            self.project.workdir,
            str(src),
            int(self.panel_prep.montage_stride()),
            int(self.panel_prep.montage_max()),
            filt_sig,
            str(getattr(g, "geom", "") or ""),
            float(getattr(g, "obs_x_km", 0.0) or 0.0),
            float(getattr(g, "offset_sign", 1.0) or 1.0),
            self._file_mtime(self.project.path(self.project.shots_xz)),
            self._file_mtime(self.project.path(self.project.obs_xz)),
            self._file_mtime(self.project.path(self.project.offsets_txt)),
            self._file_mtime(src_dir),
        )

    def _mute_proc_key(self) -> tuple:
        return self._mute_proc_key_for(bool(self.panel_prep.chk_show_proc.isChecked()))

    def _mute_proc_key_for(self, mute_on: bool) -> tuple:
        """mute 层缓存键：预览只 bake 速度 mute；多边形由画布叠层。"""
        p = self.project.preprocess
        return (
            "velmute_v7",  # tm+|x|/vm；默认切深；反选切浅（无 mute-vred）
            bool(mute_on),
            bool(p.use_mute),
            float(p.tmute),
            float(p.vmute),
            float(getattr(p, "mute_tp", 0.15) or 0.0),
            bool(getattr(p, "vel_mute_invert", False)),
        )

    def _bp_proc_key(self) -> tuple:
        p = self.project.preprocess
        return (
            bool(p.use_bandpass),
            float(p.freqlo),
            float(p.freqhi),
        )

    def _gain_cache_key(self) -> tuple:
        p = self.project.preprocess
        return (
            bool(p.use_gain),
            int(p.iscale),
            float(p.amp),
            float(p.rcor),
            float(p.sf),
            float(p.tvg),
            float(p.pvg),
            float(p.clip),
        )

    def _new_project(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择空目录作为工区")
        if not d:
            return
        self.project = ObsRtmProject(name=os.path.basename(d), workdir=d)
        self.project.ensure_workdir()
        self._geom_preview_key = None
        self._vel_preview_key = None
        self._vel_file_preview_cache = None
        self._vel_builtin_preview_cache = None
        self._rtm_vel_display_key = None
        self._vel_rsf_source = None
        self._vel_rsf_file_sig = None
        self._rtm_pending_after_vel = None
        self._sync_panels_from_project(auto_preview=False)
        self.append_log("新建工区: %s" % d)
        self.statusBar().showMessage("新建工区就绪")

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "打开工区 JSON", "", "Project (*.json);;All (*.*)"
        )
        if not path:
            return
        self._set_busy(True, "正在打开工区 JSON…")
        self._suspend_preview()
        try:
            self.project = ObsRtmProject.load(path)
            from .services.workdir_layout import prepare_workdir

            # 纠正 /mnt/d 等失效 workdir；旧工区自动分层迁移
            prepare_workdir(
                self.project, migrate=True, log=self.append_log, json_path=path
            )
            self._geom_preview_key = None
            self._vel_preview_key = None
            self._vel_file_preview_cache = None
            self._vel_builtin_preview_cache = None
            self._rtm_vel_display_key = None
            self._vel_rsf_source = None
            self._vel_rsf_file_sig = None
            self._rtm_pending_after_vel = None
            self._zelt_overlay_cache = None
            # 内存槽清空：道集/速度改从工区 .obs_rtm_preview_cache 恢复
            self._mont_caches.clear()
            self._mont_cache = None
            self._last_shot_paths = None
            self._notify(
                "已读取工程，正在同步面板…", log=True, process_events=False
            )
            # 先同步控件，再切到预处理页；全程抑制定时预览，只在最后拼一次
            self._sync_panels_from_project(auto_preview=False)
            self.append_log("面板同步已完成")
            self.stage_list.blockSignals(True)
            self.stage_list.setCurrentIndex(2)
            self.stack.setCurrentIndex(2)
            self.stage_list.blockSignals(False)
            self._notify(
                "正在扫描炮集并加载道集预览…", log=True, process_events=False
            )
            self._refresh_shot_list(auto_preview=True)
            self.append_log("扫描炮集并加载道集预览已完成")
            self.append_log("已打开: %s" % path)
            n = len(getattr(self.panel_prep, "_shot_paths", None) or [])
            done = (
                "工区已打开 · %d 炮，道集预览就绪" % n
                if n
                else "工区已打开（尚无 shots，请先导入 SU）"
            )
            self._set_busy(False, done)
            # busy 结束后再预热速度：优先工区磁盘缓存，未命中则后台重建
            try:
                self._notify(
                    "正在恢复速度预览（优先工区缓存）…",
                    log=True,
                    process_events=False,
                )
                self._preview_tomo(force=False)
            except Exception:
                pass
        except Exception as exc:
            self._set_busy(False, "打开失败")
            QMessageBox.warning(self, "打开失败", str(exc))
        finally:
            self._preview_pending = False
            self._resume_preview(flush=False)

    def _save_project(self) -> None:
        try:
            if not self.project.workdir:
                d = QFileDialog.getExistingDirectory(self, "选择工区目录以保存工程")
                if not d:
                    return
                self.project.workdir = d
                self.project.name = os.path.basename(d) or self.project.name
                self.project.ensure_workdir()
            # 必须先保证 workdir，再从面板回写（勿先 _new_project，否则参数被清空）
            self._panels_to_project()
            # 把当前多边形/手选炮集写入 rtm.shot_list，保证重开后偏移预览能高亮
            self._sync_rtm_shots_from_prep(log=False)
            self._panels_to_project()
            path = self.project.save()
            v = self.project.velocity
            r = self.project.rtm
            tmax = (int(r.nt) - 1) * float(r.dt)
            fs = (1.0 / float(r.dt)) if float(r.dt) > 0 else 0.0
            self.append_log(
                "已保存: %s\n"
                "  velocity.vel_source=%s preset=%s v0=%g\n"
                "  rtm T=%.3fs fs=%.3fHz → nt=%d dt=%g  first=%d max=%s"
                % (
                    path,
                    getattr(v, "vel_source", ""),
                    getattr(v, "v1d_preset", ""),
                    float(getattr(v, "v1d_v0", 0)),
                    tmax,
                    fs,
                    int(r.nt),
                    float(r.dt),
                    int(getattr(r, "first_shot", 0)),
                    str(r.max_shot) if int(r.max_shot) > 0 else "all",
                )
            )
            self.statusBar().showMessage("已保存 %s" % os.path.basename(path))
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", str(exc))

    def _run_import(self) -> None:
        """用独立 QProcess 跑 su_to_shots，避免子进程崩溃拖垮 GUI。"""
        if self._busy:
            return
        if self._rtm_proc is not None and self._rtm_proc.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "导入", "已有后台进程在运行，请先停止")
            return
        self._panels_to_project()
        try:
            self.project.ensure_workdir()
        except Exception as exc:
            QMessageBox.warning(self, "工区", str(exc))
            return
        if not self.project.su_path:
            QMessageBox.information(self, "导入", "请先选择 SU 文件")
            return
        if not self.project.geometry.component and not self.project.geometry.trid:
            QMessageBox.information(
                self,
                "导入",
                "4C SU 请先选分量（推荐 hydro），否则会尝试加载全部分量导致内存暴涨。",
            )
            return

        try:
            cmd = build_su_to_shots_cmd(self.project)
        except Exception as exc:
            QMessageBox.warning(self, "导入", str(exc))
            return

        self._set_busy(True, "正在导入 SU（后台进程运行中，请看下方日志）…")
        self.append_log("--- 导入 SU（独立进程）---\n$ " + " ".join(cmd))
        self._rtm_proc = QProcess(self)
        self._rtm_proc.setWorkingDirectory(self.project.workdir)
        self._rtm_proc.setProgram(cmd[0])
        self._rtm_proc.setArguments(cmd[1:])
        self._rtm_proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._rtm_proc.setProcessEnvironment(QProcessEnvironment.systemEnvironment())

        def _on_out() -> None:
            if self._closing or self._rtm_proc is None:
                return
            try:
                data = bytes(self._rtm_proc.readAllStandardOutput()).decode(
                    "utf-8", "replace"
                )
            except RuntimeError:
                return
            for line in data.splitlines():
                self.append_log(line)

        def _on_fin(code: int, _status) -> None:
            self._rtm_proc = None
            if self._closing or not self._qt_alive(self):
                return
            try:
                self._set_busy(False, "导入进程结束，正在整理炮列表/预览…")
                self._notify("导入结束，正在扫描炮集并预览…", log=True)
                self._refresh_shot_list(auto_preview=True)
                n = len(list_shot_rsf(self.project))
                self.append_log("扫描炮集并预览已完成")
                self.append_log("su_to_shots 退出码=%s，shots=%d" % (code, n))
            except RuntimeError:
                return
            try:
                self.project.save()
            except Exception:
                pass
            if int(code) != 0:
                QMessageBox.warning(
                    self,
                    "导入失败",
                    "su_to_shots 退出码=%s（若曾 Segmentation fault，请确认选了 hydro 分量并已更新脚本）。"
                    % code,
                )
                return
            if n:
                try:
                    self.panel_geom.sync_obs_x_from_file(self.project)
                    self._panels_to_project()
                except Exception as exc:
                    self.append_log("同步 OBS x 跳过: %s" % exc)
                try:
                    self._suggest_grid(silent=True, reason="导入后自动")
                except Exception as exc:
                    self.append_log("自动建议网格跳过: %s" % exc)
                try:
                    self._preview_geometry()
                except Exception as exc:
                    self.append_log("几何预览跳过: %s" % exc)
                self.stage_list.setCurrentIndex(1)  # 先到几何页确认

        self._rtm_proc.readyReadStandardOutput.connect(_on_out)
        self._rtm_proc.finished.connect(_on_fin)
        self._rtm_proc.start()
        if not self._rtm_proc.waitForStarted(5000):
            self._set_busy(False)
            self._shutdown_bg_process(wait_ms=500)
            QMessageBox.warning(self, "导入", "无法启动 su_to_shots 进程")

    def _run_geom_check(self) -> None:
        """进程内落点检查（不刷数千行日志，避免 WSL/Qt 段错误）。"""
        if self._busy:
            return
        self._panels_to_project()
        if not self.project.workdir:
            QMessageBox.information(self, "几何检查", "请先设置工区")
            return
        try:
            text, n_shot_out, n_obs_out = check_landing(self.project)
            self.append_log(text)
            n_out = n_shot_out + n_obs_out
            if n_out:
                QMessageBox.warning(
                    self,
                    "落点检查",
                    "有 %d 个点在网格外（炮 OUT=%d, OBS OUT=%d）。\n"
                    "详见日志与工区 diag/geom_check.txt。"
                    % (n_out, n_shot_out, n_obs_out),
                )
            else:
                QMessageBox.information(
                    self,
                    "落点检查",
                    "全部炮/OBS 落在网格内。\n明细已写 diag/geom_check.txt。",
                )
        except Exception as exc:
            self.append_log("落点检查失败: %s" % exc)
            QMessageBox.warning(self, "几何检查", str(exc))

    def _run_offset_sign_check(self) -> None:
        """核对 shots_xz−OBS 与 offsets×offset_sign；评估是否应翻转符号。"""
        if self._busy:
            return
        self._panels_to_project()
        if not self.project.workdir:
            QMessageBox.information(self, "offset 符号", "请先设置工区")
            return
        try:
            text, stats = check_offset_sign_consistency(self.project)
            self.append_log(text)
            status = str(stats.get("status") or "")
            if status == "ok":
                QMessageBox.information(self, "offset 符号检查", text)
            else:
                QMessageBox.warning(self, "offset 符号检查", text)
        except Exception as exc:
            self.append_log("offset 符号检查失败: %s" % exc)
            QMessageBox.warning(self, "offset 符号检查", str(exc))

    def _suggest_grid(self, silent: bool = False, reason: str = "") -> None:
        """由炮/OBS 范围写 ox/nx（保留当前 dx/dz）；silent=导入后自动调用。"""
        self._panels_to_project()
        shots = load_xz_txt(self.project.path(self.project.shots_xz))
        obs = load_xz_txt(self.project.path(self.project.obs_xz))
        if not shots:
            if not silent:
                QMessageBox.information(self, "网格", "请先导入 SU，生成 shots_xz.txt")
            return
        g0 = self.project.grid
        self.project.grid = suggest_grid_from_shots(
            shots,
            obs=obs or None,
            pad_km=10.0,
            dx=float(g0.dx) if g0.dx > 0 else 0.5,
            dz=float(g0.dz) if g0.dz > 0 else 0.25,
            zmax=40.0,
        )
        self._with_panels_loading(
            lambda: self.panel_geom.load_from_project(self.project)
        )
        g = self.project.grid
        tag = reason or "建议网格"
        self.append_log(
            "%s: ox=%.3f dx=%.4f nx=%d | oz=%.3f dz=%.4f nz=%d"
            % (tag, g.ox, g.dx, g.nx, g.oz, g.dz, g.nz)
        )
        try:
            self._panels_to_project()
            self.project.save()
        except Exception:
            pass
        if not silent:
            self._preview_geometry()
        # 内置一维随网格走；OBS x 路径会自行再调一次（此处跳过避免重复）
        if (
            self._vel_source() == "builtin_1d"
            and "OBS x" not in (reason or "")
        ):
            self._sync_builtin_vel_to_grid(reason=tag)
        else:
            # 用户速度：几何变了不必重读 tomo 预览，但须刷新炮点/OBS 叠层
            self._keep_vel_preview_key_if_valid()
            self._refresh_vel_canvas_geom()

    @staticmethod
    def _file_mtime(path: str) -> float:
        try:
            if os.path.isfile(path) or os.path.isdir(path):
                return float(os.path.getmtime(path))
            return -1.0
        except OSError:
            return -1.0

    def _geom_preview_cache_key(self) -> Optional[tuple]:
        if not self.project.workdir:
            return None
        g = self.project.grid
        geom = self.project.geometry
        return (
            os.path.normpath(os.path.abspath(self.project.workdir)),
            self._file_mtime(self.project.path(self.project.shots_xz)),
            self._file_mtime(self.project.path(self.project.obs_xz)),
            float(g.ox),
            float(g.dx),
            int(g.nx),
            float(g.oz),
            float(g.dz),
            int(g.nz),
            str(getattr(geom, "geom", "") or ""),
            float(getattr(geom, "obs_x_km", 0.0) or 0.0),
        )

    def _apply_obs_x(self, *, quiet: bool = False) -> bool:
        """offset 模式：平移 shots/OBS，并按新炮点更新成像网格 ox/nx。"""
        if self._busy:
            return False
        self._panels_to_project()
        if str(self.project.geometry.geom) != "offset":
            if not quiet:
                QMessageBox.information(
                    self, "OBS x", "仅 offset 模式可编辑并应用 OBS x。"
                )
            return False
        if not self.project.workdir:
            if not quiet:
                QMessageBox.information(self, "OBS x", "请先设置工区并导入 SU")
            return False
        new_x = float(self.panel_geom.sp_obs_x.value())
        try:
            old, new, delta = apply_obs_x_shift(
                self.project, new_x, log=self.append_log
            )
        except Exception as exc:
            if not quiet:
                QMessageBox.warning(self, "应用 OBS x 失败", str(exc))
            else:
                self.append_log("应用 OBS x 失败: %s" % exc)
            return False
        self.panel_geom.sp_obs_x.blockSignals(True)
        self.panel_geom.sp_obs_x.setValue(new)
        self.panel_geom.sp_obs_x.blockSignals(False)
        # 网格与 OBS/炮点同坐标系平移：ox ← ox+Δ（nx/dx 不变）
        # 例：ox=-100、OBS 0→100 → ox=0，成像窗跟着测线走
        vel_src = self._vel_source()
        ox_before = float(self.project.grid.ox)
        if abs(delta) > 1e-12:
            self.project.grid.ox = ox_before + float(delta)
            self.append_log(
                "OBS x 平移 Δ=%+.6f km → 网格 ox: %.6f → %.6f（nx/dx 不变）"
                % (float(delta), ox_before, float(self.project.grid.ox))
            )
        # 内置一维：再按炮/OBS 范围重估 ox/nx（可微调垫宽）
        if vel_src == "builtin_1d":
            try:
                self._suggest_grid(silent=True, reason="OBS x 更新后")
            except Exception as exc:
                self.append_log("OBS x 后建议网格跳过: %s" % exc)
                self._with_panels_loading(
                    lambda: self.panel_geom.load_from_project(self.project)
                )
        else:
            self._with_panels_loading(
                lambda: self.panel_geom.load_from_project(self.project)
            )
        try:
            self.project.save()
        except Exception:
            pass
        # 几何热更新：平移缓存横轴 / 画布 model x，保留振幅；不重读炮集
        self._remap_mont_caches_after_obs_x(delta)
        shift_poly = str(self.panel_prep.poly_x_mode() or "offset") == "offset"
        try:
            self.panel_prep.canvas.shift_model_x(
                delta, shift_polygon=shift_poly
            )
            if shift_poly and abs(float(delta)) > 1e-15:
                self.panel_prep.sync_mute_to_project(self.project)
        except Exception as exc:
            self.append_log("道集横轴热更新跳过: %s" % exc)
        self._geom_preview_key = None
        self._preview_geometry()
        # 内置一维：盘上 vel.rsf 须与新网格一致
        if vel_src == "builtin_1d":
            self._sync_builtin_vel_to_grid(reason="OBS x 更新后")
        else:
            self._keep_vel_preview_key_if_valid()
        # 炮点/OBS 散点热更新（速度体坐标随 model x）
        self._refresh_vel_canvas_geom()
        if quiet:
            return True
        g = self.project.grid
        extra = (
            "已按新网格重写 vel.rsf（内置一维）。"
            if vel_src == "builtin_1d"
            else "用户模型：ox 已随 OBS 平移；运行 RTM 时将按新网格生成 vel.rsf。"
        )
        QMessageBox.information(
            self,
            "OBS x",
            "OBS x：%.6f → %.6f km（Δ=%+.6f）。\n"
            "已按 offsets 重建 shots_xz，网格 ox=%.3f nx=%d。\n"
            "拼图横轴已热更新（未重读炮集）。\n%s"
            % (old, new, delta, g.ox, g.nx, extra),
        )
        return True

    def _ensure_geom_preview(self) -> None:
        """进入几何页：文件/网格未变且已绘过则跳过重绘。"""
        self._panels_to_project()
        key = self._geom_preview_cache_key()
        if key is None:
            return
        if self._geom_preview_key == key:
            self.statusBar().showMessage("几何预览已就绪（缓存，未重绘）")
            return
        self._preview_geometry()

    def _preview_geometry(self) -> None:
        self._panels_to_project()
        # offset：旋钮 OBS x 与文件不一致时先写入并更新网格，再预览
        if str(self.project.geometry.geom) == "offset" and self.project.workdir:
            try:
                from .services.geometry import current_obs_x_km

                want = float(self.panel_geom.sp_obs_x.value())
                have = float(current_obs_x_km(self.project))
                if abs(want - have) > 1e-6:
                    self.append_log(
                        "预览前自动应用 OBS x：%.6f → %.6f km" % (have, want)
                    )
                    self._apply_obs_x(quiet=True)
                    return  # _apply_obs_x 内已预览
            except Exception as exc:
                self.append_log("预览前应用 OBS x 跳过: %s" % exc)
        shots = load_xz_txt(self.project.path(self.project.shots_xz))
        obs = load_xz_txt(self.project.path(self.project.obs_xz))
        g = self.project.grid
        geom = self.project.geometry
        if not shots and not obs:
            self.panel_geom.canvas.clear("无 shots_xz / obs_xz（请先导入 SU）")
            self._geom_preview_key = None
            return
        # offset：蓝虚线=旋钮 OBS x；其它模式不画（OBS 见绿点）
        mark = (
            float(self.panel_geom.sp_obs_x.value())
            if str(getattr(geom, "geom", "") or "") == "offset"
            else None
        )
        self.panel_geom.canvas.show_geometry(
            shots=shots or None,
            obs=obs or None,
            ox=g.ox, dx=g.dx, nx=g.nx,
            oz=g.oz, dz=g.dz, nz=g.nz,
            obs_x_mark=mark,
        )
        self._geom_preview_key = self._geom_preview_cache_key()
        # 速度页若正显示用户/一维预览，同步炮检叠层
        self._refresh_vel_canvas_geom()
        self.append_log(
            "几何预览: shots=%d OBS=%d  grid ox=%.3f nx=%d"
            % (len(shots), len(obs), g.ox, g.nx)
        )

    def _preview_gather(self) -> None:
        """统一走 offset 拼图：全炮 或 已追加手选子集（每炮一道）。"""
        self._preview_montage(switch_stage=False, force_reload=False)

    def _preview_montage(
        self,
        *,
        switch_stage: bool = True,
        force_reload: bool = False,
        full_axes: bool = False,
    ) -> None:
        """按 offset 拼多炮。

        缓存分层（禁止重复滤波/增益）：
          raw → muted=mute(raw) → prep_raw=bp+gain(raw)
            → prep_muted=gain(mute(bp(raw)))（与写出一致，先 bp 再 mute）
        full_axes：横轴全炮 model x、纵轴全时程（定稿预览用）。
        """
        if self._montage_busy:
            # processEvents / 定时器重入：排队，勿并行读盘
            self._preview_pending = True
            return
        self._panels_to_project()
        if not self.project.workdir:
            if switch_stage:
                QMessageBox.information(self, "montage", "请先设置工区并导入 SU")
            return
        sel = self.panel_prep.current_shot_path()
        held_busy = False
        self._montage_busy = True
        try:
            shot_filter = self.panel_prep.montage_shot_filter()
            mont_src = self.panel_prep.montage_source()
            # 换拼图源（原始/定稿/mute）时取消脉冲拾取点——坐标与振幅已失效
            prev_mont = getattr(self, "_impulse_mont_src", None)
            if prev_mont is not None and str(prev_mont) != str(mont_src):
                self._clear_impulse_pick(
                    reason="已切换拼图数据源（%s→%s），已清除脉冲拾取点"
                    % (prev_mont, mont_src)
                )
            self._impulse_mont_src = mont_src
            # 定稿/仅mute：与默认「全炮拼图」同一炮子集键，便于切换数据源命中缓存；
            # 视窗仍强制全炮×全时。盘上已 purge，目录即当前选道结果。
            if mont_src in ("shots_proc", "shots_mute"):
                full_axes = True
            if shot_filter is not None and not shot_filter:
                tip = "请先追加手选炮，或勾选「全部炮拼图」"
                self.panel_prep.canvas.clear(tip, reset_view=False)
                self.statusBar().showMessage(tip)
                return
            load_key = self._montage_load_key()
            prev_key = (
                None
                if self._mont_cache is None
                else self._mont_cache.get("load_key")
            )
            if force_reload:
                self._mont_caches.pop(load_key, None)
            cache = self._mont_caches.get(load_key)
            from_disk = False
            if cache is None and not force_reload:
                cache = self._load_mont_cache_from_disk(load_key)
                from_disk = cache is not None
            need_load = cache is None
            from_cache = not need_load
            if need_load:
                if not self._busy:
                    self._set_busy(
                        True,
                        "正在读炮集并拼图（首次较慢，请勿重复点击）…",
                    )
                    held_busy = True
                else:
                    # 打开工程时已 busy：勿 processEvents，否则定时预览会重入再读一遍
                    self._notify(
                        "正在读炮集并拼图…", log=True, process_events=False
                    )
                tip = (
                    "正在拼装手选 %d 炮…" % len(shot_filter)
                    if shot_filter is not None
                    else "正在拼装全部炮道集，请稍候…"
                )
                mont_src = self.panel_prep.montage_source()
                if mont_src == "shots_proc":
                    shot_dir = self.project.path(self.project.shots_proc_dir)
                    tip = tip.replace("拼装", "拼装定稿 ")
                elif mont_src == "shots_mute":
                    shot_dir = self.project.path(
                        getattr(self.project, "shots_mute_dir", None)
                        or "shots_mute"
                    )
                    tip = tip.replace("拼装", "拼装仅mute ")
                else:
                    shot_dir = None
                # 重读拼图时重适配 X（手选子集按几何偏移距铺开，勿沿用全炮视窗）
                self.panel_prep.canvas.clear(tip, reset_view=True)
                gather, offs, d1, o1, used = build_offset_montage(
                    self.project,
                    stride=self.panel_prep.montage_stride(),
                    max_traces=self.panel_prep.montage_max(),
                    force_include=None if shot_filter is not None else sel,
                    shot_indices=shot_filter,
                    shot_dir=shot_dir,
                )
                self.append_log(
                    "读炮集并拼图已完成：%d 道 / %d 炮%s 源=%s"
                    % (
                        gather.shape[1],
                        len(used),
                        "（手选子集）" if shot_filter is not None else "",
                        mont_src,
                    )
                )
                times = time_axis(gather.shape[0], d1, o1)
                cache = {
                    "load_key": load_key,
                    "raw": gather,
                    "offs": offs,
                    "d1": d1,
                    "o1": o1,
                    "used": used,
                    "used_index": used_path_index_map(used),
                    "times": times,
                    "mute_key": None,
                    "muted": None,
                    "prep_raw_key": None,
                    "prep_raw": None,
                    "prep_muted_key": None,
                    "prep_muted": None,
                    "gain_muted_key": None,
                    "gain_muted": None,
                    "disk_prep_key": None,
                    "disk_prep": None,
                    "disk_gain_key": None,
                    "disk_gain": None,
                }
                self._store_mont_cache(load_key, cache)
            else:
                self._mont_cache = cache
                if prev_key != load_key or from_disk:
                    self.append_log(
                        "拼图缓存命中：%s（%s）"
                        % (
                            (
                                "手选 %d 炮" % len(shot_filter)
                                if shot_filter is not None
                                else "全部炮"
                            ),
                            "工区磁盘" if from_disk else "未重读盘",
                        )
                    )

            # 切换全炮↔手选槽时重适配视窗；同槽刷新则尽量保视窗
            # 定稿预览：始终重置为全炮×全时
            scope_switch = bool(full_axes) or (prev_key != load_key)

            prep = self.project.preprocess
            times = cache["times"]
            # offs = 拼图横轴 model_x；mute/增益用相对 OBS 的 offset
            offs = cache["offs"]
            from .services.montage import _obs_x_ref_km

            obs_x = float(_obs_x_ref_km(self.project))
            rel_offs = np.asarray(offs, dtype=float) - obs_x
            chain = self.panel_prep.preview_chain_mode()
            apply_prep = self.panel_prep.use_apply_preprocess()
            mute_on = bool(self.panel_prep.chk_show_proc.isChecked())
            vel_mute_on = bool(prep.use_mute)
            bp_key = self._bp_proc_key()
            gkey = self._gain_cache_key()
            mute_key = self._mute_proc_key_for(mute_on)

            def _ensure_prep_raw():
                prep_raw_key = (bp_key, gkey)
                if (
                    cache.get("prep_raw_key") != prep_raw_key
                    or cache.get("prep_raw") is None
                ):
                    self._notify("正在应用带通+增益(原始)…", log=not need_load)
                    tmp = apply_bandpass_only(cache["raw"], times, prep)
                    cache["prep_raw"] = apply_display_gain(
                        tmp, times, rel_offs, prep
                    )
                    cache["prep_raw_key"] = prep_raw_key
                    if prep.use_bandpass and not need_load:
                        self.append_log(
                            "拼图带通(raw): %.3g–%.3g Hz  backend=%s"
                            % (prep.freqlo, prep.freqhi, bandpass_backend())
                        )

            def _ensure_muted():
                # 预览：速度 mute bake；多边形留给画布，避免二次 mute 抹掉波形
                if cache.get("mute_key") != mute_key or cache.get("muted") is None:
                    self._notify("正在应用 mute…", log=not need_load)
                    cache["muted"] = apply_mute_only(
                        cache["raw"], times, rel_offs, prep, skip_poly=True
                    )
                    cache["mute_key"] = mute_key
                    cache["prep_muted_key"] = None
                    cache["prep_muted"] = None
                    cache["gain_muted_key"] = None
                    cache["gain_muted"] = None
                    if vel_mute_on:
                        st = last_vel_mute_stats() or {}
                        ntr_s = int(st.get("ntr", 0))
                        alive = int(
                            st.get(
                                "alive_traces",
                                ntr_s - int(st.get("fully_muted", 0)),
                            )
                        )
                        cut = str(st.get("cut", "deep"))
                        msg = (
                            "速度 mute: tm=%.3g vm=%.2g %s | "
                            "存活道 %d/%d (%.0f%%) | 能量 %.0f%% | 整道静音 %d"
                            % (
                                float(st.get("tmute", prep.tmute)),
                                float(st.get("vmute", prep.vmute)),
                                "切深" if cut == "deep" else "切浅",
                                alive,
                                ntr_s,
                                100.0
                                * float(
                                    st.get("alive_ratio", alive / max(1, ntr_s))
                                ),
                                100.0 * float(st.get("energy_ratio", 0.0)),
                                int(st.get("fully_muted", 0)),
                            )
                        )
                        x_t = float(st.get("x_line_at_T_km", -1.0))
                        if x_t >= 0:
                            msg += " | 线达 T=%.1fs 于 |x|≈%.0f km" % (
                                float(st.get("tmax", 0.0)),
                                x_t,
                            )
                        self.append_log(msg)
                        self.statusBar().showMessage(msg, 12000)

            def _ensure_prep_muted():
                # 先带通再 mute 再增益（避免 mute→滤波 Gibbs 假震相）；多边形画布叠
                prep_muted_key = (mute_key, bp_key, gkey, "bp_mute_gain")
                if (
                    cache.get("prep_muted_key") != prep_muted_key
                    or cache.get("prep_muted") is None
                ):
                    self._notify(
                        "正在应用 带通→mute→增益…", log=not need_load
                    )
                    tmp = apply_bandpass_only(cache["raw"], times, prep)
                    muted_bp = apply_mute_only(
                        tmp, times, rel_offs, prep, skip_poly=True
                    )
                    cache["prep_muted"] = apply_display_gain(
                        muted_bp, times, rel_offs, prep
                    )
                    cache["prep_muted_key"] = prep_muted_key

            def _ensure_gain_muted():
                """mute 后只加显示增益（不带通），便于 mute 预览时试增益。"""
                _ensure_muted()
                gain_muted_key = (mute_key, gkey, "gain_only")
                if (
                    cache.get("gain_muted_key") != gain_muted_key
                    or cache.get("gain_muted") is None
                ):
                    cache["gain_muted"] = apply_display_gain(
                        cache["muted"], times, rel_offs, prep
                    )
                    cache["gain_muted_key"] = gain_muted_key

            # 须同时勾选「启用多边形 mute」；仅 canvas 仍开启时不算
            canvas_poly = bool(mute_on and self.panel_prep.poly_mute_active())
            # 速度 mute / 多边形：只要「显示 mute 效果」开就进 mute 预览链
            want_mute_view = bool(mute_on and (vel_mute_on or canvas_poly))
            # mute 预览链与脉冲拾取互斥：启用 mute 时清掉脉冲点
            if want_mute_view or mont_src in ("shots_mute", "shots_proc"):
                try:
                    had_pick = bool(
                        self.panel_prep.canvas.impulse_mode()
                        or getattr(self.panel_prep.canvas, "_impulse_pick", None)
                    )
                except Exception:
                    had_pick = False
                if had_pick:
                    self._clear_impulse_pick(
                        reason="已启用 mute/定稿道集预览，已清除脉冲拾取点"
                    )
            use_gain = bool(getattr(prep, "use_gain", True))
            mont_src_now = self.panel_prep.montage_source()
            # 多边形边缘过渡与速度 mute tp 同步到画布
            try:
                self.panel_prep.canvas.set_mute_tp(
                    float(getattr(prep, "mute_tp", 0.15) or 0.0)
                )
            except Exception:
                pass
            # 原则：shots_proc 定稿预览 = 盘上原样，禁止再 mute/带通/增益
            if mont_src_now == "shots_proc":
                gather = cache["raw"]
                tag = " [shots_proc 定稿·原样]"
                apply_mute_preview = False
            elif mont_src_now == "shots_mute":
                # shots_mute = mute(raw)；可叠显示增益试看，禁止再带通
                apply_mute_preview = False
                if use_gain:
                    disk_gkey = (gkey, "disk_gain")
                    if (
                        cache.get("disk_gain_key") != disk_gkey
                        or cache.get("disk_gain") is None
                    ):
                        cache["disk_gain"] = apply_display_gain(
                            cache["raw"], times, rel_offs, prep
                        )
                        cache["disk_gain_key"] = disk_gkey
                    gather = cache["disk_gain"]
                    tag = " [shots_mute+增益预览]"
                else:
                    gather = cache["raw"]
                    tag = " [shots_mute]"
            elif chain == "filter":
                # 预处理页：默认同道集范围，勾选显示 mute 时也可看速度 mute
                filter_on = bool(self.panel_prep.chk_auto_filter.isChecked())
                if want_mute_view and apply_prep:
                    _ensure_prep_muted()
                    gather = cache["prep_muted"]
                    tag = " [bp→mute→gain]"
                elif want_mute_view and use_gain:
                    _ensure_gain_muted()
                    gather = cache["gain_muted"]
                    tag = " [mute+增益]"
                elif want_mute_view:
                    _ensure_muted()
                    gather = cache["muted"]
                    tag = " [mute]"
                elif filter_on:
                    _ensure_prep_raw()
                    gather = cache["prep_raw"]
                    tag = " [bp+gain]"
                else:
                    gather = cache["raw"]
                    tag = ""
                apply_mute_preview = bool(canvas_poly and mute_on)
            elif apply_prep:
                if want_mute_view:
                    _ensure_prep_muted()
                    gather = cache["prep_muted"]
                    tag = " [bp→mute→gain]"
                    if canvas_poly:
                        tag += "+poly"
                else:
                    _ensure_prep_raw()
                    gather = cache["prep_raw"]
                    tag = " [bp+gain]"
                apply_mute_preview = bool(canvas_poly)
            elif want_mute_view:
                if use_gain:
                    _ensure_gain_muted()
                    gather = cache["gain_muted"]
                    tag = " [mute+增益]"
                else:
                    _ensure_muted()
                    gather = cache["muted"]
                    tag = " [mute]"
                if canvas_poly:
                    tag += "+poly"
                apply_mute_preview = bool(canvas_poly)
            else:
                gather = cache["raw"]
                tag = ""
                apply_mute_preview = False

            d1, o1 = cache["d1"], cache["o1"]
            used = cache["used"]
            hl = highlight_index_for_path(used, sel)
            if shot_filter is not None:
                if mont_src in ("shots_proc", "shots_mute"):
                    title_base = "定稿范围 %d 炮 ntr=%d" % (
                        len(used),
                        gather.shape[1],
                    )
                else:
                    title_base = "手选 %d 炮 ntr=%d" % (
                        len(used),
                        gather.shape[1],
                    )
            else:
                title_base = "全部炮 ntr=%d" % gather.shape[1]
            if self.panel_prep.montage_stride() > 1:
                title_base += " stride=%d" % self.panel_prep.montage_stride()
            title_base += tag
            cache["title_base"] = title_base
            title = title_base
            if sel:
                title += " · 选中 %s" % os.path.basename(sel)

            for i in range(self.panel_prep.cmb_poly_x.count()):
                if self.panel_prep.cmb_poly_x.itemData(i) == "offset":
                    self.panel_prep.cmb_poly_x.blockSignals(True)
                    self.panel_prep.cmb_poly_x.setCurrentIndex(i)
                    self.panel_prep.cmb_poly_x.blockSignals(False)
                    break
            if need_load:
                self._notify("正在绘制道集…", log=True)
            # 手选稀疏时邻道空隙大；定宽用全炮典型间距，避免波形横向拉满空隙
            stride_for_dx = (
                1
                if shot_filter is not None
                else max(1, int(self.panel_prep.montage_stride()))
            )
            try:
                nominal_dx = float(
                    typical_montage_dx_km(self.project, stride=stride_for_dx)
                )
            except Exception:
                nominal_dx = 0.0
            view_x = None
            if full_axes:
                try:
                    view_x = montage_offset_span_km(
                        self.project,
                        stride=max(1, int(self.panel_prep.montage_stride())),
                    )
                except Exception:
                    view_x = None
            self.panel_prep.canvas.show_gather(
                gather,
                d1=d1,
                o1=o1,
                title=title,
                pclip=self.panel_prep.pclip(),
                dscale=self.panel_prep.dscale(),
                display_mode=self.panel_prep.display_mode(),
                display_vred=self.panel_prep.display_vred(),
                x_coords=offs,
                x_label="Model x (km)",
                apply_mute_preview=apply_mute_preview,
                highlight_idx=hl,
                reset_view=bool(scope_switch),
                trace_labels=used,
                nominal_dx=nominal_dx if nominal_dx > 1e-12 else None,
                view_x_range=view_x,
                x_reduce_origin=obs_x,
            )
            self._on_prep_selection_changed()
            if need_load:
                self.append_log("绘制道集已完成")
            elif from_cache and scope_switch:
                self.append_log("绘制道集已完成（缓存）")
            self.statusBar().showMessage(
                "%s  hl=%s%s"
                % (
                    title,
                    str(hl) if hl is not None else "-",
                    " · 缓存" if from_cache else "",
                )
            )
            if switch_stage:
                self.append_log(
                    "montage: %d traces / %d shots —「应用选道」写出当前选道到 shots_proc/"
                    % (gather.shape[1], len(used))
                )
                self.stage_list.setCurrentIndex(2)
        except Exception as exc:
            self._invalidate_preview_cache()
            if switch_stage:
                QMessageBox.warning(self, "montage", str(exc))
            else:
                self.panel_prep.canvas.clear(str(exc))
            self.append_log("montage 失败: %s" % exc)
        finally:
            self._montage_busy = False
            if held_busy:
                self._set_busy(False, "道集预览就绪")
                self.append_log("道集预览流程已全部完成")
            # mute/增益等参数变更也可能挂起：缓存已在仍需重算处理层
            if (
                self._preview_pending
                and self._preview_suspend <= 0
                and not self._loading_panels
            ):
                self._preview_pending = False
                self._schedule_preview()

    def _on_prep_selection_changed(self) -> None:
        """手选红波形即时更新；RTM/多边形圈选合并到短定时器，避免闭合卡顿。"""
        try:
            ids = self.panel_prep.selected_shot_ids()
            self.panel_prep.canvas.set_selected_shot_ids(ids)
        except Exception:
            pass
        # 打开/同步面板时禁止启动：否则会在袋空时把 rtm.shot_list 清成全炮
        if (
            self._loading_panels
            or self._preview_suspend > 0
            or self._closing
            or self._montage_busy
        ):
            return
        # 多边形/手选 → RTM：防抖，不在此同步扫全炮
        self._rtm_scope_timer.start()
        # 手选拼图范围变了才可能缺缓存；多边形 mute 不改拼图范围
        if self.panel_prep.poly_mute_active():
            return
        load_key = self._montage_load_key()
        if load_key not in self._mont_caches:
            self._schedule_preview()

    def _flush_rtm_scope_from_prep(self) -> None:
        """防抖后：更新多边形圈选炮数标签 + 同步 RTM 黄星。"""
        try:
            if self.panel_prep.poly_mute_active():
                n_poly = len(self.panel_prep.poly_mute_shot_ids_for(self.project))
                inv = "(反)" if self.panel_prep.canvas.mute_invert() else ""
                self.panel_prep.lbl_mute.setText(
                    "Mute: ON%s · 多边形圈定 %d 炮 → RTM" % (inv, n_poly)
                )
        except Exception:
            pass
        self._refresh_rtm_shot_markers()

    def _mute_shot_paths(self):
        """选道范围：与 RTM 一致（多边形 > 手选 > 全炮）。"""
        scope = self.panel_prep.rtm_scope_shot_ids(self.project)
        if scope is None:
            paths = list_shot_rsf(self.project)
            if not paths:
                return None, None, "shots/ 下无 shot_*.rsf"
            return paths, None, None
        if not scope:
            if self.panel_prep.poly_mute_active():
                return (
                    None,
                    scope,
                    "多边形 mute 已启用，但未圈到任何炮（检查多边形横轴/model x）",
                )
            return None, scope, "当前选道为空"
        paths = list_shot_rsf_by_indices(self.project, scope)
        if not paths:
            return None, scope, "选道炮在 shots/ 下均未找到"
        return paths, scope, None

    def _apply_mute_selection(self) -> None:
        """应用选道：bp→mute→gain → shots_proc/；mute(raw)→shots_mute/。"""
        if self._busy:
            return
        self._panels_to_project()
        paths, scope, err = self._mute_shot_paths()
        if err:
            QMessageBox.information(self, "应用选道", err)
            return
        apply_prep = self.panel_prep.use_apply_preprocess()
        if scope is None:
            src = "全部"
        elif self.panel_prep.poly_mute_active():
            src = "多边形圈定"
        else:
            src = "手选"
        tip = (
            "对 %s %d 炮写出 %s/ 与 shots_mute/？\n"
            "流程：%s\n"
            "会删除两目录中非本次选道的旧 shot_*.rsf，保证盘上仅当前结果。"
            % (
                src,
                len(paths),
                self.project.shots_proc_dir,
                "原始 → 带通 → mute → 增益"
                if apply_prep
                else "原始 → mute（不做带通/增益）",
            )
        )
        reply = QMessageBox.question(self, "应用选道", tip)
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._set_busy(True)
        self.append_log(
            "--- 应用选道(%s) %d 炮 → %s/ (%s) ---"
            % (
                src,
                len(paths),
                self.project.shots_proc_dir,
                "bp→mute→gain" if apply_prep else "mute",
            )
        )

        def _done(n_ok: object) -> None:
            self._set_busy(False)
            self.append_log(
                "选道写出完成: %s / %d → %s/（已清理范围外旧文件）"
                % (n_ok, len(paths), self.project.shots_proc_dir)
            )
            # 写出后盘内容变了：只失效定稿相关槽，保留 shots/ 全炮缓存
            self._invalidate_mont_caches_for_sources("shots_proc", "shots_mute")
            self._sync_rtm_shots_from_prep()
            self._schedule_preview()

        def _fail(msg: str) -> None:
            self._set_busy(False)
            QMessageBox.warning(self, "应用选道失败", msg)

        start_worker(
            self,
            process_rtm_shots,
            self.project,
            shot_paths=paths,
            apply_prep=apply_prep,
            on_finished=_done,
            on_failed=_fail,
            on_log=self.append_log,
        )

    def _vel_source(self) -> str:
        self._panels_to_project()
        return str(
            getattr(self.project.velocity, "vel_source", "file") or "file"
        ).strip().lower()

    def _vel_rsf_matches_project_grid(self, path: str) -> bool:
        """盘上 vel.rsf 网格是否与当前工区一致。"""
        try:
            from .services.rsf_io import parse_rsf_header

            meta = parse_rsf_header(path)
            g = self.project.grid
            return (
                int(meta.get("n1", 0)) == int(g.nz)
                and int(meta.get("n2", 0)) == int(g.nx)
                and abs(float(meta.get("o1", 0)) - float(g.oz)) < 1e-6
                and abs(float(meta.get("d1", 0)) - float(g.dz)) < 1e-6
                and abs(float(meta.get("o2", 0)) - float(g.ox)) < 1e-6
                and abs(float(meta.get("d2", 0)) - float(g.dx)) < 1e-6
            )
        except Exception:
            return False

    def _keep_vel_preview_key_if_valid(self) -> None:
        """画布已有速度且参数键能对上时，保留缓存，避免切页重复加载。"""
        if getattr(self.panel_vel.canvas, "_vel", None) is None:
            return
        key = self._vel_preview_cache_key()
        if key is not None:
            self._vel_preview_key = key

    def _refresh_vel_canvas_geom(self) -> None:
        """几何/炮检变更后，热更新速度页与偏移页画布上的炮点·OBS（不重载速度体）。"""
        if not self.project.workdir:
            return
        shots = load_xz_txt(self.project.path(self.project.shots_xz))
        obs = load_xz_txt(self.project.path(self.project.obs_xz))
        for canvas in (self.panel_vel.canvas, self.panel_rtm.canvas):
            if getattr(canvas, "_vel", None) is None:
                continue
            try:
                canvas.set_shot_obs(shots or None, obs or None)
            except Exception:
                pass

    def _store_file_vel_preview_cache(
        self,
        *,
        key: tuple,
        vel: np.ndarray,
        meta: dict,
        kind: str,
        path: str,
        pdx: float,
        pdz: float,
        zelt_model=None,
    ) -> None:
        """保存用户模型预览体，供 GUI 内切换瞬间恢复。"""
        self._vel_file_preview_cache = {
            "key": key,
            "vel": np.ascontiguousarray(vel, dtype=np.float32),
            "meta": dict(meta),
            "kind": str(kind),
            "path": str(path),
            "pdx": float(pdx),
            "pdz": float(pdz),
            "zelt": zelt_model,
        }
        try:
            if self.project.workdir:
                _pdisk.save_vel_file(
                    self.project.workdir,
                    key=key,
                    vel=self._vel_file_preview_cache["vel"],
                    meta=meta,
                    kind=kind,
                    path=path,
                    pdx=float(pdx),
                    pdz=float(pdz),
                )
        except Exception:
            pass

    def _store_builtin_vel_preview_cache(
        self,
        *,
        key: tuple,
        vel: np.ndarray,
        bath: Optional[np.ndarray],
        title: str,
    ) -> None:
        """保存内置一维预览体。"""
        self._vel_builtin_preview_cache = {
            "key": key,
            "vel": np.ascontiguousarray(vel, dtype=np.float32),
            "bath": None
            if bath is None
            else np.ascontiguousarray(bath, dtype=np.float32),
            "title": str(title),
        }
        try:
            if self.project.workdir:
                _pdisk.save_vel_builtin(
                    self.project.workdir,
                    key=key,
                    vel=self._vel_builtin_preview_cache["vel"],
                    bath=self._vel_builtin_preview_cache["bath"],
                    title=title,
                )
        except Exception:
            pass

    def _load_file_vel_preview_from_disk(self, key: tuple) -> bool:
        """工区磁盘 → 内存用户模型缓存；不重绘。"""
        if not self.project.workdir:
            return False
        try:
            data = _pdisk.load_vel_file(self.project.workdir, key)
        except Exception:
            return False
        if not data:
            return False
        self._vel_file_preview_cache = data
        return True

    def _load_builtin_vel_preview_from_disk(self, key: tuple) -> bool:
        """工区磁盘 → 内存一维缓存；不重绘。"""
        if not self.project.workdir:
            return False
        try:
            data = _pdisk.load_vel_builtin(self.project.workdir, key)
        except Exception:
            return False
        if not data:
            return False
        self._vel_builtin_preview_cache = data
        return True

    def _zelt_from_file_cache_or_disk(self, path: str, cache: Optional[dict] = None):
        """优先用预览缓存 / overlay 缓存中的 Zelt，避免 GUI 切换反复解析 v.in。"""
        # 注意：磁盘缓存会带 zelt=None；勿因 key 存在就直接返回（否则 Interfaces 失效）
        if cache is not None and cache.get("zelt") is not None:
            zelt = cache.get("zelt")
            try:
                mtime = float(os.path.getmtime(path)) if path else 0.0
            except OSError:
                mtime = 0.0
            self._zelt_overlay_cache = (path or "", mtime, zelt)
            return zelt
        hit = self._zelt_overlay_cache
        try:
            mtime = float(os.path.getmtime(path)) if path else 0.0
        except OSError:
            mtime = 0.0
        if (
            path
            and hit
            and isinstance(hit, tuple)
            and len(hit) >= 3
            and os.path.normpath(os.path.abspath(str(hit[0])))
            == os.path.normpath(os.path.abspath(path))
            and abs(float(hit[1]) - mtime) < 1e-6
            and hit[2] is not None
        ):
            return hit[2]
        from .services.model_import import load_zelt_model_optional
        from .services.velocity import resolve_zelt_for_bath

        zelt = load_zelt_model_optional(path) if path else None
        zpath = path or ""
        if zelt is None:
            # tomo 已是 .rsf 时：回落到工程里保存的原始 v.in
            zelt, zpath = resolve_zelt_for_bath(self.project)
            if zpath:
                try:
                    mtime = float(os.path.getmtime(zpath))
                except OSError:
                    mtime = 0.0
        self._zelt_overlay_cache = (zpath or path or "", mtime, zelt)
        return zelt

    def _restore_file_vel_preview_cache(self, key: tuple) -> bool:
        """键命中则从内存缓存重绘用户模型；失败返回 False。"""
        cache = self._vel_file_preview_cache
        if not cache or cache.get("key") != key:
            return False
        if not key or key[0] == "builtin_1d":
            return False
        path = str(cache.get("path") or key[0])
        vel = cache.get("vel")
        meta = cache.get("meta") or {}
        if vel is None or not meta:
            return False
        # 速度页画布已是该键：只刷炮点/OBS + 同步偏移页
        if (
            getattr(self.panel_vel.canvas, "_vel", None) is not None
            and self._vel_preview_key == key
        ):
            self._refresh_vel_canvas_geom()
            self._sync_rtm_vel_display_from_vel_page(log=False)
            self.statusBar().showMessage("速度预览已就绪（文件缓存）")
            return True
        zelt = self._zelt_from_file_cache_or_disk(path, cache)
        shots = load_xz_txt(self.project.path(self.project.shots_xz))
        obs = load_xz_txt(self.project.path(self.project.obs_xz))
        kind = str(cache.get("kind") or "?")
        title = "%s [%s]" % (os.path.basename(path), kind)
        self.panel_vel.canvas.show_vel(
            vel,
            ox=float(meta["o2"]),
            dx=float(meta["d2"]),
            oz=float(meta["o1"]),
            dz=float(meta["d1"]),
            title=title,
            zelt_model=zelt,
            shots_xz=shots or None,
            obs_xz=obs or None,
        )
        self._apply_iface_to_canvas(self.panel_vel.canvas)
        self._sync_iface_selection_to_project()
        self._vel_preview_key = key
        self._sync_rtm_vel_display_from_vel_page(log=False)
        self.append_log(
            "速度预览←缓存 %s [%s]  nz×nx=%d×%d"
            % (os.path.basename(path), kind, vel.shape[0], vel.shape[1])
        )
        self.statusBar().showMessage("速度预览已就绪（文件缓存）")
        return True

    def _restore_builtin_vel_preview_cache(self, key: tuple) -> bool:
        """键命中则从内存缓存重绘内置一维；失败返回 False。"""
        cache = self._vel_builtin_preview_cache
        if not cache or cache.get("key") != key:
            return False
        if not key or key[0] != "builtin_1d":
            return False
        vel = cache.get("vel")
        if vel is None:
            return False
        if (
            getattr(self.panel_vel.canvas, "_vel", None) is not None
            and self._vel_preview_key == key
        ):
            self._refresh_vel_canvas_geom()
            self._sync_rtm_vel_display_from_vel_page(log=False)
            self.statusBar().showMessage("一维速度预览已就绪（缓存）")
            return True
        g = self.project.grid
        shots = load_xz_txt(self.project.path(self.project.shots_xz))
        obs = load_xz_txt(self.project.path(self.project.obs_xz))
        bath = cache.get("bath")
        self.panel_vel.canvas.show_vel(
            vel,
            ox=g.ox,
            dx=g.dx,
            oz=g.oz,
            dz=g.dz,
            title=str(cache.get("title") or "builtin_1d"),
            bath_1d=bath,
            shots_xz=shots or None,
            obs_xz=obs or None,
        )
        self._vel_preview_key = key
        self._sync_rtm_vel_display_from_vel_page(log=False)
        self.append_log(
            "一维预览←缓存  nz×nx=%d×%d" % (vel.shape[0], vel.shape[1])
        )
        self.statusBar().showMessage("一维速度预览已就绪（缓存）")
        return True

    def _sync_builtin_vel_to_grid(self, *, reason: str = "") -> None:
        """内置一维：按当前网格重写 vel.rsf（偏移作业读盘，须与速度预览一致）。"""
        if self._vel_source() != "builtin_1d":
            return
        if self._busy:
            self.append_log("内置速度待网格同步（忙，稍后进偏移页会补写）")
            return
        path = self._resolve_rtm_vel_path()
        if path and self._vel_rsf_matches_project_grid(path):
            # 盘上已对齐：只刷新缓存键，勿清掉导致切速度页再预览一遍
            self._keep_vel_preview_key_if_valid()
            return
        tag = reason or "网格变更"
        self.append_log("%s：重写内置一维 vel.rsf 以匹配网格…" % tag)
        # 重建过程中键暂空；_build_velocity 完成时会写入新键
        self._vel_preview_key = None
        self._build_velocity()

    def _vel_preview_cache_key(self) -> Optional[tuple]:
        self._panels_to_project()
        vp = self.project.velocity
        src = str(getattr(vp, "vel_source", "file") or "file").strip().lower()
        if src == "builtin_1d":
            g = self.project.grid
            bath = str(vp.bath_path or "")
            bath_mtime = 0.0
            if self.project.workdir and bath:
                bp = bath if os.path.isabs(bath) else self.project.path(bath)
                if os.path.isfile(bp):
                    try:
                        bath_mtime = float(os.path.getmtime(bp))
                    except OSError:
                        bath_mtime = 0.0
            return (
                "builtin_1d",
                str(getattr(vp, "v1d_preset", "linear_crust")),
                str(getattr(vp, "v1d_ref", "subbottom")),
                float(getattr(vp, "v1d_v0", 2.0)),
                float(getattr(vp, "v1d_grad", 0.5)),
                float(getattr(vp, "v1d_vmax", 8.0)),
                bool(getattr(vp, "v1d_iface_enable", False)),
                float(getattr(vp, "v1d_iface_v", 6.0)),
                float(getattr(vp, "v1d_iface_dv", 0.8)),
                float(vp.vwater),
                bool(vp.fill_water),
                float(vp.flat_bath_km),
                bath,
                float(bath_mtime),
                float(g.ox),
                float(g.dx),
                int(g.nx),
                float(g.oz),
                float(g.dz),
                int(g.nz),
            )
        path = (
            vp.tomo_path
            or self.project.tomo_vel
            or self.panel_vel.ed_tomo.text().strip()
        )
        if not path or not os.path.isfile(path):
            return None
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        return (
            os.path.normpath(os.path.abspath(path)),
            float(vp.vin_dx_km),
            float(vp.vin_dz_km),
            float(mtime),
        )

    def _ensure_vel_preview(self) -> None:
        """进入速度页：画布已有且键未变则跳过重载。"""
        key = self._vel_preview_cache_key()
        if key is None:
            return
        canvas = self.panel_vel.canvas
        if (
            getattr(canvas, "_vel", None) is not None
            and self._vel_preview_key == key
        ):
            # 速度体未变，仍同步最新炮点/OBS（几何页可能刚改过）
            self._refresh_vel_canvas_geom()
            self.statusBar().showMessage("速度预览已就绪（缓存，未重载）")
            return
        self._preview_tomo(force=False)

    def _preview_tomo(self, *, force: bool = True) -> None:
        """速度预览：用户文件或内置一维；绘图只在主线程。"""
        if self._busy:
            # 一维调参很快：排队，当前任务结束后再刷最新参数
            self._vel_preview_pending = bool(force)
            return
        self._vel_preview_pending = None
        # 确保 ed_tomo / vin_dx 已写回工程（选文件后立即预览）
        self._panels_to_project()
        key = self._vel_preview_cache_key()
        if key is None:
            if force and self.panel_vel._source() == "file":
                QMessageBox.information(
                    self,
                    "速度",
                    "请先选择速度模型（v.in / grd / rsf…），或将速度来源改为内置一维",
                )
            return
        # GUI 内切换：键未变一律走内存缓存；重开工区可走磁盘缓存
        if key[0] == "builtin_1d":
            if self._restore_builtin_vel_preview_cache(key):
                return
            if self._load_builtin_vel_preview_from_disk(key):
                if self._restore_builtin_vel_preview_cache(key):
                    self.append_log("一维速度预览：工区磁盘缓存命中")
                    return
            self._preview_builtin_1d(key=key, force=force)
            return
        if self._restore_file_vel_preview_cache(key):
            return
        if self._load_file_vel_preview_from_disk(key):
            if self._restore_file_vel_preview_cache(key):
                self.append_log("用户速度预览：工区磁盘缓存命中")
                return

        path = key[0]
        vp = self.project.velocity
        self._set_busy(True, "正在加载速度模型预览（后台粗网格）…")
        self.append_log("速度预览加载中（粗网格，界面可继续操作）…")

        def _job(log=None):
            # 工作线程只做 IO/数值，返回纯 numpy；Zelt 勿经信号传递（主线程再解析）
            from .services.model_import import (
                dataset_to_vel_meta,
                suggest_grid_from_model_file,
            )

            ds, _zelt, kind, pdx, pdz = load_velocity_for_preview(
                path, vin_dx_km=vp.vin_dx_km, vin_dz_km=vp.vin_dz_km
            )
            vel, meta = dataset_to_vel_meta(ds)
            vel = np.ascontiguousarray(vel, dtype=np.float32)
            if log and (pdx > vp.vin_dx_km or pdz > vp.vin_dz_km):
                log(
                    "预览栅格 dx=%.3g dz=%.3g km（成像仍用 %.3g / %.3g）"
                    % (pdx, pdz, vp.vin_dx_km, vp.vin_dz_km)
                )
            grid, gkind = suggest_grid_from_model_file(
                path, vin_dx_km=vp.vin_dx_km, vin_dz_km=vp.vin_dz_km
            )
            if log:
                log(
                    "自动网格←模型[%s] ox=%.3f dx=%.4f nx=%d oz=%.3f dz=%.4f nz=%d"
                    % (
                        gkind,
                        grid.ox,
                        grid.dx,
                        grid.nx,
                        grid.oz,
                        grid.dz,
                        grid.nz,
                    )
                )
            return {
                "vel": vel,
                "meta": dict(meta),
                "kind": kind,
                "pdx": float(pdx),
                "pdz": float(pdz),
                "key": key,
                "grid": grid,
                "grid_kind": gkind,
            }

        def _done(result: object) -> None:
            self._set_busy(False)
            data = result  # type: ignore
            vel = data["vel"]
            meta = data["meta"]
            kind = data["kind"]
            pdx, pdz = data["pdx"], data["pdz"]
            # 读用户模型后自动「用模型范围更新工区网格」
            grid = data.get("grid")
            if grid is not None:
                self.project.grid = grid
                self._with_panels_loading(
                    lambda: self.panel_geom.load_from_project(self.project)
                )
                g = self.project.grid
                self.append_log(
                    "已自动网格←模型[%s]: ox=%.3f dx=%.4f nx=%d oz=%.3f dz=%.4f nz=%d"
                    % (
                        data.get("grid_kind") or "?",
                        g.ox,
                        g.dx,
                        g.nx,
                        g.oz,
                        g.dz,
                        g.nz,
                    )
                )
                self._geom_preview_key = None
                self._preview_geometry()
            # Interfaces：主线程解析 v.in（避免跨线程对象失效）
            from .services.model_import import load_zelt_model_optional

            zelt = load_zelt_model_optional(path)
            try:
                mtime = float(os.path.getmtime(path))
            except OSError:
                mtime = 0.0
            self._zelt_overlay_cache = (path, mtime, zelt)
            if zelt is not None:
                self.project.velocity.zelt_vin_path = path
                # 默认用地形海底：S 未选 → Interface 2（第 2 个界面）
                if getattr(self.project.velocity, "iface_seafloor", None) is None:
                    from .services.velocity import suggest_seafloor_iface_idx

                    sug = suggest_seafloor_iface_idx(zelt)
                    if sug is not None:
                        self.project.velocity.iface_seafloor = int(sug)
                        self.append_log(
                            "Interfaces·S → Interface %d（默认第 2 界面=地形海底，bath/填水/压水柱）"
                            % (int(sug) + 1)
                        )
            shots = load_xz_txt(self.project.path(self.project.shots_xz))
            obs = load_xz_txt(self.project.path(self.project.obs_xz))
            self.panel_vel.canvas.show_vel(
                vel,
                ox=float(meta["o2"]),
                dx=float(meta["d2"]),
                oz=float(meta["o1"]),
                dz=float(meta["d1"]),
                title="%s [%s]" % (os.path.basename(path), kind),
                zelt_model=zelt,
                shots_xz=shots or None,
                obs_xz=obs or None,
            )
            # 下拉填好后套回工程/已选 B/S/M，并同步偏移页
            self._apply_iface_to_canvas(self.panel_vel.canvas)
            self._sync_iface_selection_to_project()
            self._apply_iface_to_canvas(self.panel_rtm.canvas)
            self._vel_preview_key = data.get("key") or key
            self._store_file_vel_preview_cache(
                key=self._vel_preview_key,
                vel=vel,
                meta=meta,
                kind=kind,
                path=path,
                pdx=float(pdx),
                pdz=float(pdz),
                zelt_model=zelt,
            )
            # 换模型后偏移页底图同步（预览用，不写 vel.rsf）
            self._sync_rtm_vel_display_from_vel_page(log=False)
            if zelt is not None:
                try:
                    n_iface = max(0, len(zelt.depth_nodes) - 1)
                except Exception:
                    n_iface = 0
                s_idx = getattr(self.project.velocity, "iface_seafloor", None)
                s_txt = (
                    "S=Interface %d" % (int(s_idx) + 1)
                    if s_idx is not None
                    else "S=Auto"
                )
                self.append_log(
                    "预览 %s [%s]  nz×nx=%d×%d  (prev dx/dz=%.3g/%.3g)  "
                    "Interfaces=%d  %s→bath"
                    % (
                        path,
                        kind,
                        vel.shape[0],
                        vel.shape[1],
                        pdx,
                        pdz,
                        n_iface,
                        s_txt,
                    )
                )
            else:
                self.append_log(
                    "预览 %s [%s]  nz×nx=%d×%d  (prev dx/dz=%.3g/%.3g)  "
                    "Interfaces=无（需 Zelt v.in）"
                    % (path, kind, vel.shape[0], vel.shape[1], pdx, pdz)
                )
            self.statusBar().showMessage("速度预览就绪")
            pending = self._vel_preview_pending
            if pending is not None:
                self._vel_preview_pending = None
                self._preview_tomo(force=bool(pending))

        def _fail(msg: str) -> None:
            self._set_busy(False)
            self._vel_preview_key = None
            QMessageBox.warning(self, "预览失败", msg)
            self.statusBar().showMessage("预览失败")
            self._vel_preview_pending = None

        start_worker(self, _job, on_finished=_done, on_failed=_fail, on_log=self.append_log)

    def _preview_builtin_1d(self, *, key: tuple, force: bool = True) -> None:
        """内置一维：按当前工区网格 + bath 生成预览（不写盘）。"""
        if self._busy:
            return
        canvas = self.panel_vel.canvas
        if (
            not force
            and getattr(canvas, "_vel", None) is not None
            and self._vel_preview_key == key
        ):
            self.statusBar().showMessage("一维速度预览已就绪（缓存）")
            return

        # 快照参数，避免工作线程读到面板半截状态
        project = self.project
        vp = project.velocity
        grid = project.grid
        preset = str(getattr(vp, "v1d_preset", "linear_crust"))
        ref = str(getattr(vp, "v1d_ref", "subbottom"))
        v0 = float(getattr(vp, "v1d_v0", 2.0))
        grad = float(getattr(vp, "v1d_grad", 0.5))
        vmax = float(getattr(vp, "v1d_vmax", 8.0))
        iface_on = bool(getattr(vp, "v1d_iface_enable", False))
        iface_v = float(getattr(vp, "v1d_iface_v", 6.0))
        iface_dv = float(getattr(vp, "v1d_iface_dv", 0.8))
        vwater = float(vp.vwater)
        fill_water = bool(vp.fill_water)
        flat_z = float(vp.flat_bath_km)
        bath_path = str(vp.bath_path or "")
        if bath_path and not os.path.isabs(bath_path) and project.workdir:
            bath_path = project.path(bath_path)
        obs_path = (
            project.path(project.obs_xz) if project.workdir else ""
        )

        self._set_busy(True, "正在生成内置一维速度预览…")

        def _job(log=None):
            bath = bath_on_grid(
                grid,
                bath_path=bath_path if bath_path and os.path.isfile(bath_path) else "",
                flat_z_km=flat_z,
                obs_xz_path=obs_path if obs_path and os.path.isfile(obs_path) else "",
            )
            vel = build_builtin_1d_velocity(
                grid,
                bath,
                vwater=vwater,
                fill_water=fill_water,
                ref=ref,
                preset=preset,
                v0=v0,
                grad=grad,
                vmax=vmax,
                iface_enable=iface_on,
                iface_v=iface_v,
                iface_dv=iface_dv,
                iface_log=log,
            )
            vel = np.ascontiguousarray(vel, dtype=np.float32)
            if log:
                extra = (
                    "  iface(v=%.3g ΔV=%+.3g)" % (iface_v, iface_dv)
                    if iface_on
                    else ""
                )
                log(
                    "一维预览 preset=%s ref=%s v0=%.3g grad=%.3g vmax=%.3g%s  "
                    "nz×nx=%d×%d"
                    % (
                        preset,
                        ref,
                        v0,
                        grad,
                        vmax,
                        extra,
                        vel.shape[0],
                        vel.shape[1],
                    )
                )
            title = "builtin_1d [%s]" % preset
            if iface_on:
                title += " · iface %.1f%+.2g" % (iface_v, iface_dv)
            return {
                "vel": vel,
                "bath": np.ascontiguousarray(bath, dtype=np.float32),
                "key": key,
                "title": title,
            }

        def _done(result: object) -> None:
            self._set_busy(False)
            data = result  # type: ignore
            vel = data["vel"]
            bath = data.get("bath")
            g = self.project.grid
            shots = load_xz_txt(self.project.path(self.project.shots_xz))
            obs = load_xz_txt(self.project.path(self.project.obs_xz))
            title = str(data.get("title") or "builtin_1d")
            self.panel_vel.canvas.show_vel(
                vel,
                ox=g.ox,
                dx=g.dx,
                oz=g.oz,
                dz=g.dz,
                title=title,
                bath_1d=bath,
                shots_xz=shots or None,
                obs_xz=obs or None,
            )
            self._vel_preview_key = data.get("key") or key
            self._store_builtin_vel_preview_cache(
                key=self._vel_preview_key,
                vel=vel,
                bath=bath,
                title=title,
            )
            self._sync_rtm_vel_display_from_vel_page(log=False)
            self.append_log(
                "一维预览就绪  vmin=%.3f vmax=%.3f  nz×nx=%d×%d"
                % (float(vel.min()), float(vel.max()), vel.shape[0], vel.shape[1])
            )
            self.statusBar().showMessage("一维速度预览就绪")
            pending = self._vel_preview_pending
            if pending is not None:
                self._vel_preview_pending = None
                self._preview_tomo(force=bool(pending))

        def _fail(msg: str) -> None:
            self._set_busy(False)
            self._vel_preview_key = None
            QMessageBox.warning(self, "一维预览失败", msg)
            self.statusBar().showMessage("预览失败")
            self._vel_preview_pending = None

        start_worker(self, _job, on_finished=_done, on_failed=_fail, on_log=self.append_log)

    def _convert_tomo_rsf(self) -> None:
        if self._busy:
            return
        self._panels_to_project()
        path = self.project.velocity.tomo_path or self.project.tomo_vel
        if not path or not os.path.isfile(path):
            QMessageBox.information(self, "转换", "请先选择速度模型")
            return
        try:
            self.project.ensure_workdir()
        except Exception as exc:
            QMessageBox.warning(self, "工区", str(exc))
            return
        from .services.workdir_layout import TOMO_VEL

        out = self.project.path(TOMO_VEL)
        vp = self.project.velocity
        self._set_busy(True)

        def _job(log=None):
            p, meta, kind = convert_model_to_rsf(
                path,
                out,
                vin_dx_km=vp.vin_dx_km,
                vin_dz_km=vp.vin_dz_km,
            )
            if log:
                log("converted [%s] → %s  n1=%d n2=%d" % (
                    kind, p, int(meta["n1"]), int(meta["n2"])))
            return p, meta, kind

        def _done(result: object) -> None:
            self._set_busy(False)
            p, meta, kind = result  # type: ignore
            self.project.tomo_vel = p
            self.project.velocity.tomo_path = p
            self.panel_vel.ed_tomo.setText(p)
            self.append_log("已转换 [%s] → %s" % (kind, p))
            self._preview_tomo()

        def _fail(msg: str) -> None:
            self._set_busy(False)
            QMessageBox.warning(self, "转换失败", msg)

        start_worker(self, _job, on_finished=_done, on_failed=_fail, on_log=self.append_log)

    def _sync_grid_from_model(self) -> None:
        if self._busy:
            return
        self._panels_to_project()
        path = self.project.velocity.tomo_path or self.project.tomo_vel
        if not path or not os.path.isfile(path):
            QMessageBox.information(self, "网格", "请先选择速度模型")
            return
        vp = self.project.velocity
        self._set_busy(True)
        self.append_log("根据模型范围更新网格…")

        def _job(log=None):
            grid, kind = suggest_grid_from_model_file(
                path, vin_dx_km=vp.vin_dx_km, vin_dz_km=vp.vin_dz_km
            )
            if log:
                log(
                    "grid←%s ox=%.3f dx=%.4f nx=%d"
                    % (kind, grid.ox, grid.dx, grid.nx)
                )
            return grid, kind

        def _done(result: object) -> None:
            self._set_busy(False)
            grid, kind = result  # type: ignore
            self.project.grid = grid
            self._with_panels_loading(
                lambda: self.panel_geom.load_from_project(self.project)
            )
            g = self.project.grid
            self.append_log(
                "网格←模型[%s]: ox=%.3f dx=%.4f nx=%d oz=%.3f dz=%.4f nz=%d"
                % (kind, g.ox, g.dx, g.nx, g.oz, g.dz, g.nz)
            )
            # 只改工区网格并刷新几何预览；速度图已在画布上，不必重载
            self._preview_geometry()

        def _fail(msg: str) -> None:
            self._set_busy(False)
            QMessageBox.warning(self, "网格", msg)

        start_worker(self, _job, on_finished=_done, on_failed=_fail, on_log=self.append_log)

    def _build_velocity(self) -> None:
        if self._busy:
            return
        self._panels_to_project()
        # 锁定本次生成的来源（完成回调勿读中途被改过的下拉框）
        src_snap = self._vel_source()
        try:
            self.project.ensure_workdir()
        except Exception as exc:
            QMessageBox.warning(self, "工区", str(exc))
            return
        self._set_busy(True)
        self.append_log(
            "--- 生成速度模型 [%s] ---"
            % ("内置一维" if src_snap == "builtin_1d" else "用户模型")
        )

        def _done(result: object) -> None:
            self._set_busy(False)
            vel, bath = result  # type: ignore
            g = self.project.grid
            shots = load_xz_txt(self.project.path(self.project.shots_xz))
            obs = load_xz_txt(self.project.path(self.project.obs_xz))
            # 叠层与标签一律按「开始生成时」的来源，避免 RTM 等待期间切来源搞混地形
            zelt = None
            if src_snap == "file":
                from .services.model_import import load_zelt_model_optional

                src = self._resolve_tomo_source_path()
                if src:
                    zelt = load_zelt_model_optional(src)
                    if zelt is not None:
                        try:
                            mtime = float(os.path.getmtime(src))
                        except OSError:
                            mtime = 0.0
                        self._zelt_overlay_cache = (src, mtime, zelt)
            self.panel_vel.canvas.show_vel(
                vel,
                ox=g.ox, dx=g.dx, oz=g.oz, dz=g.dz,
                title=self.project.velocity.out_vel,
                bath_1d=bath,
                zelt_model=zelt,
                shots_xz=shots or None,
                obs_xz=obs or None,
            )
            if src_snap == "file":
                self._apply_iface_to_canvas(self.panel_vel.canvas)
                self._apply_iface_to_canvas(self.panel_rtm.canvas)
            else:
                # 一维：清掉用户模型 Interfaces，只保留本次 bath 曲线
                for canvas in (self.panel_vel.canvas, self.panel_rtm.canvas):
                    try:
                        canvas._zelt_model = None
                        canvas._refresh_iface_combos(None)
                        canvas._refresh_overlays()
                    except Exception:
                        pass
            self._vel_preview_key = self._vel_preview_cache_key()
            self._vel_rsf_source = src_snap
            if src_snap == "file":
                self._vel_rsf_file_sig = self._current_file_vel_sig()
            else:
                self._vel_rsf_file_sig = None
                if self._vel_preview_key and self._vel_preview_key[0] == "builtin_1d":
                    self._store_builtin_vel_preview_cache(
                        key=self._vel_preview_key,
                        vel=vel,
                        bath=bath,
                        title=str(self.project.velocity.out_vel or "builtin_1d"),
                    )
            self._rtm_vel_display_key = None
            try:
                self.project.save()
            except Exception:
                pass
            self.append_log(
                "速度就绪: %s  (vmin=%.3f vmax=%.3f)  bath与来源=%s一致"
                % (
                    self.project.path(self.project.velocity.out_vel),
                    float(vel.min()),
                    float(vel.max()),
                    "一维" if src_snap == "builtin_1d" else "用户模型",
                )
            )
            # 写出 vel 后刷新偏移预览（跟速度页，不读旧盘混地形）
            self._preview_rtm_vel(silent=True)
            pending = self._rtm_pending_after_vel
            self._rtm_pending_after_vel = None
            if pending == "run":
                self._run_rtm()
            elif pending == "prepare":
                self._prepare_scons()

        def _fail(msg: str) -> None:
            self._set_busy(False)
            self._rtm_pending_after_vel = None
            QMessageBox.warning(self, "速度模型失败", msg)
            self.append_log("失败: %s" % msg)

        start_worker(
            self,
            build_velocity_model,
            self.project,
            on_finished=_done,
            on_failed=_fail,
            on_log=self.append_log,
        )

    @staticmethod
    def _rsf_data_dir(run: str) -> str:
        """从已有 .rsf 的 in= 推断 DATAPATH 目录（波场体常在此增长）。"""
        for name in ("wav.rsf", "vels.rsf", "den.rsf"):
            hdr = os.path.join(run, name)
            if not os.path.isfile(hdr):
                continue
            try:
                with open(hdr, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        s = line.strip()
                        if not s.startswith("in="):
                            continue
                        inp = s.split("=", 1)[1].strip().strip("\"'")
                        if inp.endswith("@"):
                            inp = inp[:-1]
                        d = os.path.dirname(inp)
                        if d:
                            return d
            except OSError:
                continue
        dp = (os.environ.get("DATAPATH") or "/var/tmp").rstrip("/")
        return dp

    @staticmethod
    def _bin_size_mb(data_dir: str, stem: str) -> Optional[float]:
        """stem 如 wfls_000 → 返回 .rsf@ 体积 MB（写入中/已完成均可）。"""
        if not data_dir or not stem:
            return None
        for cand in (
            os.path.join(data_dir, stem + ".rsf@"),
            os.path.join(data_dir, stem + ".rsf"),
        ):
            try:
                if os.path.isfile(cand):
                    return os.path.getsize(cand) / (1024.0 * 1024.0)
            except OSError:
                continue
        return None

    def _rtm_progress_files(self) -> str:
        """根据当前作业目录已生成文件判断阶段（勿读错正式/脉冲目录）。"""
        from .services.rtm_job import resolve_shot_indices, rtm_run_dir

        if not self.project.workdir:
            return "（无工区）"
        if str(self._rtm_mode or "") == "impulse":
            from .services.impulse_gather import impulse_run_dir

            run = impulse_run_dir(self.project)
            tag = "%03d" % int((self._impulse_meta or {}).get("iobs", 0))
            checks = [
                ("wav.rsf", "wav"),
                ("vels.rsf", "vel"),
                ("den.rsf", "den"),
                ("mut_t_%s.rsf" % tag, "mut"),
                ("wfls_%s.rsf" % tag, "正传"),
                ("wflr_%s.rsf" % tag, "反传"),
                ("img_impulse.rsf", "脉冲像"),
                ("img_impulse_lap.rsf", "脉冲lap"),
            ]
            done = [
                label
                for name, label in checks
                if os.path.isfile(os.path.join(run, name))
            ]
            base = " → ".join(done) if done else "尚无产物(impulse/)"
            data_dir = self._rsf_data_dir(run)
            if "正传" not in done:
                sz = self._bin_size_mb(data_dir, "wfls_%s" % tag)
                if sz is not None:
                    return base + " · 正传写入中(%.0fMB)" % sz
                # 缺 vels/mut 时还在准备，勿谎称已在 awefd2d
                if "vel" not in done or "mut" not in done:
                    miss = []
                    if "vel" not in done:
                        miss.append("vels")
                    if "den" not in done:
                        miss.append("den")
                    if "mut" not in done:
                        miss.append("mut")
                    return base + " · 准备中(待%s)" % ",".join(miss)
                return base + " · 正传awefd2d计算中"
            elif "反传" not in done:
                sz = self._bin_size_mb(data_dir, "wflr_%s" % tag)
                if sz is not None:
                    return base + " · 反传写入中(%.0fMB)" % sz
                return base + " · 反传awefd2d计算中"
            elif "脉冲lap" not in done:
                return base + " · 成像/lap中"
            return base

        run = rtm_run_dir(self.project)
        ids = resolve_shot_indices(self.project)
        if ids:
            tag = "%03d" % ids[0]
        else:
            tag = "%03d" % max(int(getattr(self.project.rtm, "first_shot", 0) or 0), 0)
        checks = [
            ("wav.rsf", "wav"),
            ("vels.rsf", "vel"),
            ("mut_t_%s.rsf" % tag, "mut"),
            ("wfls_%s.rsf" % tag, "正传波场"),
            ("wflr_%s.rsf" % tag, "反传波场"),
            ("img_obs_%s.rsf" % tag, "OBS像"),
            ("img_%s.rsf" % tag, "单炮像"),
            ("img_lap.rsf", "叠后lap"),
        ]
        done = [label for name, label in checks if os.path.isfile(os.path.join(run, name))]
        base = " → ".join(done) if done else "尚无产物(rtm_work/)"
        data_dir = self._rsf_data_dir(run)
        if "正传波场" not in done:
            sz = self._bin_size_mb(data_dir, "wfls_%s" % tag)
            if sz is not None:
                return base + " · 正传写入中(%.0fMB)" % sz
            if done:
                return base + " · 正传awefd2d计算中"
        elif "反传波场" not in done:
            sz = self._bin_size_mb(data_dir, "wflr_%s" % tag)
            if sz is not None:
                return base + " · 反传写入中(%.0fMB)" % sz
            return base + " · 反传awefd2d计算中"
        return base

    def _on_rtm_heartbeat(self) -> None:
        """长时间无输出时提醒：结合磁盘产物判断阶段，避免误以为卡在 wav。"""
        if self._closing or not self._qt_alive(self):
            return
        if self._rtm_proc is None:
            return
        try:
            if self._rtm_proc.state() == QProcess.ProcessState.NotRunning:
                return
        except RuntimeError:
            return
        import time

        now = time.time()
        elapsed = max(0.0, now - self._rtm_t0) if self._rtm_t0 is not None else 0.0
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        silent = 0.0
        if self._rtm_last_out_t is not None:
            silent = max(0.0, now - self._rtm_last_out_t)
        stage = self._rtm_progress_files()
        kind = "脉冲RTM" if str(self._rtm_mode or "") == "impulse" else "RTM"
        diag = self._rtm_child_diag()
        self.append_log(
            "… %s 仍在运行 %d:%02d（日志静默 %.0fs）。%s。%s可停止。"
            % (
                kind,
                mins,
                secs,
                silent,
                stage,
                (diag + "。") if diag else "",
            )
        )
        try:
            self.statusBar().showMessage(
                "%s %d:%02d · %s" % (kind, mins, secs, stage)
            )
        except RuntimeError:
            pass

    def _prep_current_shot_index(self) -> Optional[int]:
        path = self.panel_prep.current_shot_path()
        if not path:
            return None
        import re

        m = re.match(r"shot_(\d+)\.rsf$", os.path.basename(path), re.I)
        if not m:
            return None
        return int(m.group(1))

    def _sync_rtm_shots_from_prep(self, *, log: bool = True) -> None:
        """RTM 炮集：多边形 mute 优先；否则手选；再否则全炮。"""
        self._panels_to_project()
        scope = self.panel_prep.rtm_scope_shot_ids(self.project)
        self.panel_rtm.set_shots_from_hand_select(scope)
        self._panels_to_project()
        if not log:
            return
        if scope is not None and self.panel_prep.poly_mute_active():
            self.append_log(
                "RTM 炮集 ← 多边形 mute %d 炮: %s"
                % (
                    len(scope),
                    ",".join(str(i) for i in scope[:24])
                    + ("…" if len(scope) > 24 else ""),
                )
            )
        elif scope:
            self.append_log(
                "RTM 炮集 ← 手选 %d 炮: %s"
                % (
                    len(scope),
                    ",".join(str(i) for i in scope[:24])
                    + ("…" if len(scope) > 24 else ""),
                )
            )
        else:
            self.append_log("RTM 炮集 ← 全炮")

    def _refresh_rtm_shot_markers(self) -> None:
        """同步 RTM 炮集并刷新偏移页黄星（多边形或手选子集）。"""
        try:
            self._sync_rtm_shots_from_prep(log=False)
            scope = self.panel_prep.rtm_scope_shot_ids(self.project)
            self.panel_rtm.set_highlight_shot_idx(scope or None)
        except Exception:
            pass

    def _validate_rtm_shot_selection(self) -> Optional[str]:
        """返回错误信息；None 表示通过。"""
        from .services.rtm_job import describe_shot_selection, parse_shot_list

        r = self.project.rtm
        spec = str(getattr(r, "shot_list", "") or "").strip()
        if not spec:
            # 全炮：需 shots/ 非空
            if not list_shot_rsf(self.project):
                return "shots/ 下无炮，请先导入 SU 并手选或使用全炮"
            self.append_log("炮选择: 全炮")
            return None
        try:
            ids = parse_shot_list(spec)
        except ValueError as exc:
            return str(exc)
        if not ids:
            return "手选炮集为空"
        self.append_log("炮选择: %s" % describe_shot_selection(self.project))
        return None

    def _on_impulse_point_picked(self, pick: object) -> None:
        """脉冲模式左键点选后：确认 → 退出拾取 → 跑 RTM。"""
        if not isinstance(pick, dict):
            return
        if self._rtm_proc is not None and self._rtm_proc.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "脉冲成像", "已有 RTM 在运行，请先停止。")
            return
        self._panels_to_project()
        shot_idx = pick.get("shot_idx")
        if shot_idx is None:
            QMessageBox.warning(
                self,
                "脉冲成像",
                "当前道无法解析炮号（需拼图带 shot_NNN 标签）。请先「全部炮拼图」。",
            )
            return
        # 与拼图一致：每炮只取第 0 道（OBS0）；正式 RTM 的 OBS 列表不用于脉冲
        iobs = 0
        from .services.impulse_gather import (
            impulse_fm_hz,
            impulse_nt_jsnap,
            sample_impulse_from_raw_shot,
        )

        try:
            raw = sample_impulse_from_raw_shot(
                self.project,
                ishot=int(shot_idx),
                iobs=iobs,
                t_true=float(pick.get("t_true", 0.0)),
                shot_path_hint=str(pick.get("shot_path") or "") or None,
            )
        except Exception as exc:
            QMessageBox.warning(self, "脉冲成像", "读取原始道集失败：%s" % exc)
            return

        amp = float(raw["amp"])
        if abs(amp) < 1e-30:
            QMessageBox.warning(
                self,
                "脉冲成像",
                "原始道集样点振幅≈0，已拒绝（不再改成 amp=1）。\n"
                "请重新点选有效样点。\n来源：%s" % raw.get("source_note", "?"),
            )
            return

        dt = float(self.project.rtm.dt)
        nt = max(int(self.project.rtm.nt), 2)
        t_true = float(raw["t_true"])
        d1_shot = float(raw["d1"])
        # RTM 时间轴用工程 dt；与道集 d1 不一致时按 t 换算 it
        if dt > 1e-12:
            it = int(round(t_true / dt))
        else:
            it = int(raw["it"])
        it = int(max(0, min(it, nt - 1)))
        dt_note = ""
        if abs(d1_shot - dt) > 1e-6:
            dt_note = (
                "\n注意：道集 d1=%.6g s，RTM dt=%.6g s → it 按 RTM dt 换算"
                % (d1_shot, dt)
            )

        nt_imp, jsnap_imp, nsnap_imp = impulse_nt_jsnap(self.project, it)
        g = self.project.grid
        fm_imp = impulse_fm_hz(self.project)
        tip = (
            "已选脉冲点（振幅取自原始道集，非拼图显示链）\n\n"
            "OBS = 0（与拼图相同：各炮第 0 道；脉冲暂不跟「OBS/bin」列表）\n"
            "炮号 shot = %d\n"
            "时间 t = %.4f s（it=%d）\n"
            "峰值 amp = %.4g（%s）%s\n\n"
            "反传数据：与正传同频的带限 Ricker（fm=%.3g Hz），"
            "峰值对齐拾取时刻（非单点 δ）。\n"
            "将跑 2×awefd2d（正传+反传）→ rtm_work/impulse/\n"
            "截断含义：nt=%d（仅跑到拾取+子波右瓣垫，非整道 nt=%d）；"
            "jsnap=%d → 约 %d 帧波场动画（非每时间步）。\n"
            "网格仍为全工区 %d×%d。\n"
            "参数未变时点「开始」会复用已有结果；点「强制重算」清空 impulse/ 重跑。\n"
            "请关闭「打印进度」。"
            % (
                int(shot_idx),
                t_true,
                it,
                amp,
                raw.get("source_note", "shots/"),
                dt_note,
                fm_imp,
                nt_imp,
                nt,
                jsnap_imp,
                nsnap_imp,
                int(g.nx),
                int(g.nz),
            )
        )
        msg = QMessageBox(self)
        msg.setWindowTitle("脉冲成像")
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setText(tip)
        btn_ok = msg.addButton("开始", QMessageBox.ButtonRole.AcceptRole)
        btn_force = msg.addButton("强制重算", QMessageBox.ButtonRole.ActionRole)
        btn_cancel = msg.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(btn_ok)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked is None or clicked is btn_cancel:
            self.statusBar().showMessage("已取消；可重新左键点选，或 Esc 退出脉冲拾取")
            return
        force_clean = clicked is btn_force
        self.panel_prep.canvas.exit_impulse_mode(clear_marker=False)
        self._start_impulse_rtm(
            iobs=iobs,
            ishot=int(shot_idx),
            it=it,
            amp=amp,
            t_true=t_true,
            x=float(pick.get("x", 0.0)),
            force_clean=force_clean,
        )

    def _start_impulse_rtm(
        self,
        *,
        iobs: int,
        ishot: int,
        it: int,
        amp: float,
        t_true: float,
        x: float,
        force_clean: bool = False,
    ) -> None:
        """确认后启动脉冲 RTM 进程。"""
        if self._rtm_vel_needs_build():
            if self._busy:
                QMessageBox.information(
                    self, "脉冲成像", "正在生成成像速度，完成后请再试。"
                )
                return
            self.append_log("脉冲成像前：按当前速度来源自动生成 vel.rsf…")
            self._ensure_rtm_vel_ready()
            QMessageBox.information(
                self,
                "脉冲成像",
                "正在按当前速度来源生成 vel.rsf，完成后请再点一次脉冲成像。",
            )
            return
        vel = self._resolve_rtm_vel_path()
        if not vel:
            QMessageBox.warning(
                self,
                "脉冲成像",
                "未找到 vel.rsf。请点「运行 RTM」或「仅生成 SConstruct」以按速度来源自动生成。",
            )
            return
        try:
            self.project.ensure_workdir()
        except Exception as exc:
            QMessageBox.warning(self, "脉冲成像", str(exc))
            return
        self._impulse_meta = {
            "iobs": int(iobs),
            "ishot": int(ishot),
            "it": int(it),
            "amp": float(amp),
            "t": float(t_true),
            "x": float(x),
        }
        try:
            run, reused = prepare_impulse_workdir(
                self.project,
                iobs=int(iobs),
                ishot=int(ishot),
                it=int(it),
                amp=float(amp),
                force_clean=bool(force_clean),
                log=self.append_log,
            )
            cmd = build_impulse_scons_cmd(self.project)
        except Exception as exc:
            QMessageBox.warning(self, "脉冲成像", str(exc))
            return

        if reused:
            self.append_log(
                "--- 脉冲 RTM --- 跳过 scons（产物已是最新；需重跑请再点「强制重算」）"
            )
            self._rtm_mode = "impulse"
            self._preview_impulse_image()
            return

        self.append_log("--- 脉冲 RTM ---\n$ %s" % " ".join(cmd))
        if bool(getattr(self.project.rtm, "awefd_verb", False)):
            self.append_log(
                "警告: 已开「打印进度」(verb=y)。经 GUI 管道刷时间步会比终端慢很多；"
                "建议取消勾选后重跑。"
            )
        self._shutdown_bg_process(wait_ms=1000)
        self._rtm_mode = "impulse"
        self._rtm_proc = QProcess(self)
        self._rtm_proc.setWorkingDirectory(run)
        self.append_log("工作目录: %s" % run)
        self._rtm_proc.setProgram(cmd[0])
        self._rtm_proc.setArguments(cmd[1:])
        self._rtm_proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._rtm_proc.setProcessEnvironment(self._make_rtm_process_env())
        # 避免子进程卡在读 stdin；IDE jobserver 已在 env 里清掉
        try:
            self._rtm_proc.setStandardInputFile(QProcess.nullDevice())
        except Exception:
            pass
        self._rtm_log_flood_t = 0.0
        self._rtm_proc.readyReadStandardOutput.connect(self._on_rtm_stdout)
        self._rtm_proc.finished.connect(self._on_rtm_finished)
        self.panel_rtm.set_running(True)
        import time as _time

        self._rtm_t0 = _time.time()
        self._rtm_last_out_t = self._rtm_t0
        self.statusBar().showMessage("脉冲 RTM 运行中…（rtm_work/impulse/）")
        self._rtm_heartbeat.start()
        self._rtm_proc.start()
        if not self._rtm_proc.waitForStarted(5000):
            self._rtm_heartbeat.stop()
            self._rtm_t0 = None
            self.panel_rtm.set_running(False)
            self.statusBar().showMessage("脉冲 RTM 启动失败")
            QMessageBox.warning(
                self,
                "脉冲成像",
                "进程启动失败。请确认已安装 Madagascar / scons，且 PATH 含 sfawefd2d。",
            )
            self._shutdown_bg_process(wait_ms=500)
        else:
            try:
                self._rtm_proc.closeWriteChannel()
            except Exception:
                pass

    def _impulse_preview_keys(self) -> list:
        """扫描 impulse/ 中可预览的产品键。"""
        from .services.impulse_gather import impulse_run_dir

        run = impulse_run_dir(self.project)
        mapping = (
            ("impulse_raw", "img_impulse.rsf"),
            ("impulse_solid", "img_impulse_solid.rsf"),
            ("impulse_lap", "img_impulse_lap.rsf"),
        )
        out = []
        for key, name in mapping:
            if os.path.isfile(os.path.join(run, name)):
                out.append(key)
        return out

    def _preview_impulse_image(self, which: str = "impulse_lap") -> None:
        """预览 impulse 目录成像；which=impulse_raw|solid|lap。"""
        from .services.impulse_gather import impulse_run_dir

        self._stop_rtm_wfl()
        run = impulse_run_dir(self.project)
        name_map = {
            "impulse_raw": "img_impulse.rsf",
            "raw": "img_impulse.rsf",
            "img_impulse": "img_impulse.rsf",
            "impulse_solid": "img_impulse_solid.rsf",
            "solid": "img_impulse_solid.rsf",
            "impulse_lap": "img_impulse_lap.rsf",
            "lap": "img_impulse_lap.rsf",
        }
        prefer = name_map.get(str(which or "impulse_lap"), "img_impulse_lap.rsf")
        # 优先所选；缺则 raw → solid → lap
        order = [prefer, "img_impulse.rsf", "img_impulse_solid.rsf", "img_impulse_lap.rsf"]
        path = None
        seen = set()
        for name in order:
            if name in seen:
                continue
            seen.add(name)
            cand = os.path.join(run, name)
            if os.path.isfile(cand):
                path = cand
                break
        if not path:
            QMessageBox.information(
                self, "脉冲成像", "未找到 img_impulse*.rsf，请确认作业已成功结束。"
            )
            return
        # 同步下拉，便于再点「预览成像」切换 raw/lap
        try:
            self._refresh_rtm_img_source_combo()
            key_for = {
                "img_impulse.rsf": "impulse_raw",
                "img_impulse_solid.rsf": "impulse_solid",
                "img_impulse_lap.rsf": "impulse_lap",
            }.get(os.path.basename(path))
            if key_for:
                for i in range(self.panel_rtm.cmb_img_src.count()):
                    if self.panel_rtm.cmb_img_src.itemData(i) == key_for:
                        self.panel_rtm.cmb_img_src.setCurrentIndex(i)
                        break
        except Exception:
            pass
        try:
            img, meta = read_vel_rsf(path)
            ox = float(meta.get("o2", self.project.grid.ox))
            dx = float(meta.get("d2", self.project.grid.dx))
            oz = float(meta.get("o1", self.project.grid.oz))
            dz = float(meta.get("d1", self.project.grid.dz))
            shots = load_xz_txt(self.project.path(self.project.shots_xz))
            obs = load_xz_txt(self.project.path(self.project.obs_xz))
            meta_i = self._impulse_meta or {}
            hl = None
            if meta_i.get("ishot") is not None:
                hl = [int(meta_i["ishot"])]
            title = "脉冲成像 · shot=%s t=%.4fs amp=%.3g · %s" % (
                meta_i.get("ishot", "?"),
                float(meta_i.get("t", 0.0)),
                float(meta_i.get("amp", 0.0)),
                os.path.basename(path),
            )
            if not self._show_rtm_image_on_vel(
                img,
                title=title,
                shots_xz=shots or None,
                obs_xz=obs or None,
                highlight_shot_idx=hl,
            ):
                # 无速度底图时回退为纯振幅（与旧行为一致）
                vel_bg, vel_meta = self._rtm_vel_for_contours()
                self.panel_rtm.canvas.show_vel(
                    img,
                    ox=ox,
                    dx=dx,
                    oz=oz,
                    dz=dz,
                    title=title,
                    shots_xz=shots or None,
                    obs_xz=obs or None,
                    highlight_shot_idx=hl,
                    cbar_label="Amplitude",
                    zelt_model=self._zelt_for_rtm_overlay(),
                    vel_for_contours=vel_bg,
                    vel_for_contours_meta=vel_meta,
                )
                self._apply_iface_to_canvas(self.panel_rtm.canvas)
            self.append_log(
                "脉冲成像预览: %s  shape=%s  OBS=%s shot=%s  [vel 底图叠层]"
                % (
                    path,
                    img.shape,
                    meta_i.get("iobs", "?"),
                    meta_i.get("ishot", "?"),
                )
            )
            self.statusBar().showMessage(title)
        except Exception as exc:
            QMessageBox.warning(self, "脉冲成像", str(exc))

    def _run_rtm(self) -> None:
        if self._rtm_proc is not None and self._rtm_proc.state() != QProcess.ProcessState.NotRunning:
            return
        if self._rtm_vel_needs_build():
            if self._busy:
                QMessageBox.information(
                    self, "RTM", "正在生成成像速度，完成后请再运行（或稍候自动继续）。"
                )
                self._rtm_pending_after_vel = "run"
                return
            self._rtm_pending_after_vel = "run"
            self.append_log("运行 RTM 前：按当前速度来源自动生成 vel.rsf…")
            self._ensure_rtm_vel_ready()
            return
        self._sync_rtm_shots_from_prep()
        err = self._validate_rtm_shot_selection()
        if err:
            QMessageBox.warning(self, "炮选择", err)
            return
        try:
            self.project.ensure_workdir()
        except Exception as exc:
            QMessageBox.warning(self, "工区", str(exc))
            return
        os.makedirs(self.project.path(self.project.rtm.workdir), exist_ok=True)

        r = self.project.rtm
        self.append_log(
            "RTM 将使用界面时间: T=%.3f s, fs=%.3f Hz → nt=%d, dt=%g"
            % (
                float(getattr(r, "tmax", 0) or (int(r.nt) - 1) * float(r.dt)),
                float(getattr(r, "fs", 0) or (1.0 / float(r.dt) if r.dt else 0)),
                int(r.nt),
                float(r.dt),
            )
        )

        try:
            if (self.project.rtm.engine or "").lower() in ("madagascar", "scons", "awefd2d"):
                prepare_scons_workdir(self.project, log=self.append_log)
            mode, cmd = build_rtm_run_cmd(self.project)
        except Exception as exc:
            QMessageBox.warning(self, "RTM", str(exc))
            return

        from .services.rtm_job import describe_obs_source_plan, describe_shot_selection

        g = self.project.grid
        r = self.project.rtm
        jsnap = max(int(getattr(r, "jsnap", 80) or 80), 1)
        nsnap = max(int(r.nt) // jsnap, 1)
        self.append_log("--- RTM [%s] ---\n$ %s" % (mode, " ".join(cmd)))
        self.append_log(describe_obs_source_plan(self.project))
        self.append_log(
            "网格 nx=%d nz=%d nt=%d (T≈%.3fs) jsnap=%d → 约 %d 张波场快照/炮；%s。"
            "波场二进制一般在 DATAPATH（常为 /var/tmp），不是工区 /mnt/d。"
            "无「>>>」时看心跳进度；与终端同 SConstruct 应接近同速。"
            % (
                int(g.nx),
                int(g.nz),
                int(r.nt),
                (int(r.nt) - 1) * float(r.dt),
                jsnap,
                nsnap,
                describe_shot_selection(self.project),
            )
        )
        if bool(getattr(r, "awefd_verb", False)):
            self.append_log(
                "警告: 已开「打印进度」(verb=y)。这是 GUI 比终端慢的常见原因——"
                "请取消勾选后重跑（进度靠心跳扫文件即可）。"
            )
        # 若上次进程对象残留，先干净关掉再新建
        self._shutdown_bg_process(wait_ms=1000)
        self._rtm_mode = mode
        self._rtm_proc = QProcess(self)
        # Madagascar：在 rtm_work/ 下跑 scons（输出留在该目录）；自研循环仍用工区根 cwd
        if mode == "madagascar":
            from .services.rtm_job import rtm_run_dir

            cwd = rtm_run_dir(self.project)
        else:
            cwd = self.project.workdir
        self._rtm_proc.setWorkingDirectory(cwd)
        self.append_log("工作目录: %s" % cwd)
        self._rtm_proc.setProgram(cmd[0])
        self._rtm_proc.setArguments(cmd[1:])
        self._rtm_proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._rtm_proc.setProcessEnvironment(self._make_rtm_process_env())
        try:
            self._rtm_proc.setStandardInputFile(QProcess.nullDevice())
        except Exception:
            pass
        self._rtm_log_flood_t = 0.0

        self._rtm_proc.readyReadStandardOutput.connect(self._on_rtm_stdout)
        self._rtm_proc.finished.connect(self._on_rtm_finished)
        self.panel_rtm.set_running(True)
        import time as _time

        self._rtm_t0 = _time.time()
        self._rtm_last_out_t = self._rtm_t0
        self.statusBar().showMessage(
            "RTM 运行中…（产物在 rtm_work/；看状态栏「进度(文件)」）"
        )
        self._rtm_heartbeat.start()
        self._rtm_proc.start()
        if not self._rtm_proc.waitForStarted(5000):
            self._rtm_heartbeat.stop()
            self._rtm_t0 = None
            self.panel_rtm.set_running(False)
            self.statusBar().showMessage("RTM 启动失败")
            QMessageBox.warning(
                self,
                "RTM",
                "进程启动失败。请确认已安装 Madagascar / scons，且 PATH 含 sfawefd2d。",
            )
            self._shutdown_bg_process(wait_ms=500)
        else:
            try:
                self._rtm_proc.closeWriteChannel()
            except Exception:
                pass

    @Slot()
    def _on_rtm_stdout(self) -> None:
        if self._closing or not self._qt_alive(self) or self._rtm_proc is None:
            return
        try:
            data = bytes(self._rtm_proc.readAllStandardOutput()).decode(
                "utf-8", "replace"
            )
        except RuntimeError:
            return
        import time as _time

        now = _time.time()
        if data.strip():
            self._rtm_last_out_t = now
        # verb=y 时数千行时间步会卡死 GUI；里程碑行始终显示，其余最多约 2s 一条
        flood_t = float(getattr(self, "_rtm_log_flood_t", 0.0) or 0.0)
        for line in data.splitlines():
            s = line.strip()
            if not s:
                continue
            important = (
                s.startswith(">>>")
                or s.startswith("scons:")
                or s.startswith("Impulse")
                or s.startswith("OBS-as-source")
                or s.startswith("Building")
                or "error" in s.lower()
                or "wrote " in s.lower()
                or "FORWARD" in s
                or "ADJOINT" in s
            )
            if important or (now - flood_t) >= 2.0:
                self.append_log(line)
                flood_t = now
                self._rtm_log_flood_t = now

    @Slot(int, QProcess.ExitStatus)
    def _on_rtm_finished(self, code: int, _status) -> None:
        if self._closing or not self._qt_alive(self):
            self._rtm_proc = None
            return
        try:
            self._rtm_heartbeat.stop()
        except Exception:
            pass
        self._rtm_t0 = None
        mode = self._rtm_mode
        self._rtm_proc = None
        try:
            self.panel_rtm.set_running(False)
        except RuntimeError:
            pass
        try:
            self.statusBar().showMessage("RTM 结束（退出码 %s）" % code)
        except RuntimeError:
            pass
        self.append_log("RTM 退出码=%s" % code)
        if mode == "impulse":
            try:
                if int(code) == 0:
                    self.append_log("脉冲 RTM 完成，正在预览…")
                    self.stage_list.blockSignals(True)
                    self.stage_list.setCurrentIndex(4)
                    self.stack.setCurrentIndex(4)
                    self.stage_list.blockSignals(False)
                    self._preview_impulse_image()
                else:
                    self.append_log("脉冲 RTM 失败，跳过预览。")
            except Exception as exc:
                self.append_log("脉冲 RTM 收尾失败: %s" % exc)
        elif mode == "madagascar":
            try:
                if int(code) == 0:
                    self.append_log("scons 已结束，正在导出/预览成像…")
                    # 正式 RTM：离开脉冲成像/波场源与拾取点，预览叠后
                    self._clear_impulse_pick()
                    self._select_formal_img_source()
                    self._select_formal_wfl_source()
                    # 叠后=盘上全部 img_obs_*（scons 已按 disk∪本轮生成）；写清单
                    try:
                        sync_obs_stack_manifest(
                            self.project, log=self.append_log
                        )
                    except Exception as exc:
                        self.append_log("叠后清单更新跳过: %s" % exc)
                    try_export_img_lap_npy(self.project, log=self.append_log)
                    self._preview_rtm_image()
                    self.append_log(
                        "RTM 完成。单台像=img_obs_NNN；叠后=全部已有 OBS 像之和。"
                    )
                else:
                    self.append_log("scons 失败，跳过成像预览。")
            except Exception as exc:
                self.append_log("RTM 收尾失败: %s" % exc)
        else:
            sh = self.project.path(self.project.rtm.workdir, "run_shots.sh")
            if os.path.isfile(sh):
                self.append_log("脚本: %s" % sh)

    def _stop_rtm(self) -> None:
        if self._rtm_proc is None:
            return
        try:
            running = self._rtm_proc.state() != QProcess.ProcessState.NotRunning
        except RuntimeError:
            running = False
        if running:
            self.append_log("正在停止 RTM…")
        self._shutdown_bg_process(wait_ms=4000)
        try:
            self.panel_rtm.set_running(False)
        except RuntimeError:
            pass
        self.statusBar().showMessage("RTM 已停止")

    def _refresh_rtm_img_source_combo(self) -> None:
        """刷新成像源下拉（叠后文案含已有 OBS 台数 / 过期标记）。"""
        from .services.rtm_job import list_rtm_img_shots, rtm_run_dir

        avail = list_rtm_img_shots(self.project)
        run = rtm_run_dir(self.project)
        obs_mode = bool(avail) and any(
            os.path.isfile(os.path.join(run, "img_obs_%03d.rsf" % i))
            for i in avail
        )
        stale, why = obs_stack_staleness(self.project)
        tip = why if stale else None
        self.panel_rtm.refresh_img_source_list(
            avail,
            obs_mode=obs_mode,
            impulse_names=self._impulse_preview_keys(),
            stack_label=format_obs_stack_label(self.project),
            stack_tip=tip,
        )

    def _stack_rtm(self) -> None:
        self._panels_to_project()
        wd = self.project.path(self.project.rtm.workdir)

        def _job(log=None):
            # Madagascar 互易：扫盘叠全部 img_obs_*.rsf
            try:
                return rebuild_obs_image_stack(self.project, log=log)
            except Exception as obs_exc:
                if log:
                    log("叠 img_obs_* 未成功（%s），尝试旧 npy/导出…" % obs_exc)
            exported = try_export_img_lap_npy(self.project, log=log)
            if exported:
                return exported
            return stack_shot_images(wd, log=log)

        def _done(path: object) -> None:
            self.append_log("叠全部 OBS 像 / 导出完成: %s" % path)
            self._preview_rtm_image()

        def _fail(msg: str) -> None:
            QMessageBox.warning(self, "叠加 OBS 像", msg)

        start_worker(
            self,
            _job,
            on_finished=_done,
            on_failed=_fail,
            on_log=self.append_log,
        )

    def _resolve_rtm_vel_path(self) -> Optional[str]:
        """作业用速度文件：rtm.vel_rsf → velocity.out_vel → vel.rsf。"""
        self._panels_to_project()
        if not self.project.workdir:
            return None
        candidates = [
            self.project.rtm.vel_rsf,
            self.panel_rtm.ed_vel.text().strip(),
            self.project.velocity.out_vel,
            "rtm_in/vel.rsf",
            "vel.rsf",
        ]
        for name in candidates:
            if not name:
                continue
            path = name if os.path.isabs(name) else self.project.path(name)
            if os.path.isfile(path):
                return path
        return None

    def _rtm_vel_for_contours(self):
        """偏移预览 Contours 用的速度体 (vel, meta)；失败返回 (None, None)。"""
        vel, meta, _bath = self._rtm_vel_base_for_overlay()
        return vel, meta

    def _rtm_vel_base_for_overlay(self):
        """
        成像/波场叠层共用速度底图：(vel, meta, bath_1d)。

        优先速度页画布与预览缓存，最后才读盘 vel.rsf。
        """
        self._panels_to_project()
        g = self.project.grid
        bath_1d = None
        try:
            from .services.velocity import resolve_bath_1d

            bath_1d, _ = resolve_bath_1d(self.project, g)
        except Exception:
            bath_1d = None

        key = self._vel_preview_cache_key()
        c = self.panel_vel.canvas
        vel = getattr(c, "_vel", None)
        if (
            vel is not None
            and key is not None
            and self._vel_preview_key == key
        ):
            meta = {
                "o2": float(c._ox),
                "d2": float(c._dx),
                "o1": float(c._oz),
                "d1": float(c._dz),
            }
            bath = getattr(c, "_bath_1d", None)
            return (
                np.asarray(vel, dtype=np.float32),
                meta,
                bath if bath is not None else bath_1d,
            )

        src = self._vel_source()
        if src == "builtin_1d":
            cache = self._vel_builtin_preview_cache
            if key is not None and cache and cache.get("key") == key:
                vel = cache.get("vel")
                if vel is not None:
                    meta = {
                        "o2": float(g.ox),
                        "d2": float(g.dx),
                        "o1": float(g.oz),
                        "d1": float(g.dz),
                    }
                    bath = cache.get("bath")
                    return (
                        np.asarray(vel, dtype=np.float32),
                        meta,
                        bath if bath is not None else bath_1d,
                    )
        else:
            cache = self._vel_file_preview_cache
            if key is not None and cache and cache.get("key") == key:
                vel = cache.get("vel")
                meta = cache.get("meta") or {}
                if vel is not None and meta:
                    return (
                        np.asarray(vel, dtype=np.float32),
                        dict(meta),
                        bath_1d,
                    )

        path = self._resolve_rtm_vel_path()
        if not path:
            return None, None, bath_1d
        try:
            vel, meta = read_vel_rsf(path)
            return vel, meta, bath_1d
        except Exception:
            return None, None, bath_1d

    def _show_rtm_image_on_vel(
        self,
        img: np.ndarray,
        *,
        title: str,
        shots_xz=None,
        obs_xz=None,
        highlight_shot_idx=None,
        pclip: float = 98.0,
    ) -> bool:
        """
        成像振幅按波场策略叠在速度底图上。无速度底图时返回 False。
        """
        vel, vmeta, bath_1d = self._rtm_vel_base_for_overlay()
        if vel is None or not vmeta:
            return False
        g = self.project.grid
        ox = float(vmeta.get("o2", g.ox))
        dx = float(vmeta.get("d2", g.dx))
        oz = float(vmeta.get("o1", g.oz))
        dz = float(vmeta.get("d1", g.dz))
        ttl = title if "[vel" in title else (title + "  [vel 底图]")
        canvas = self.panel_rtm.canvas
        canvas.show_amp_on_vel(
            vel,
            img,
            ox=ox,
            dx=dx,
            oz=oz,
            dz=dz,
            title=ttl,
            shots_xz=shots_xz,
            obs_xz=obs_xz,
            highlight_shot_idx=highlight_shot_idx,
            pclip=pclip,
            zelt_model=self._zelt_for_rtm_overlay(),
            bath_1d=bath_1d,
            alpha_max=0.60,
            tone="gray",
        )
        self._apply_iface_to_canvas(canvas)
        try:
            canvas.pin_velocity_base_display()
        except Exception:
            pass
        return True

    def _clear_impulse_pick(self, *, reason: str = "") -> None:
        """取消脉冲拾取模式与道集上的脉冲标记。"""
        try:
            self.panel_prep.canvas.exit_impulse_mode(clear_marker=True)
        except Exception:
            pass
        if reason:
            try:
                self.statusBar().showMessage(reason)
            except RuntimeError:
                pass

    def _select_formal_img_source(self) -> None:
        """成像下拉切到叠后（离开脉冲 raw/lap）。"""
        try:
            cmb = self.panel_rtm.cmb_img_src
            cmb.blockSignals(True)
            for i in range(cmb.count()):
                if cmb.itemData(i) is None:
                    cmb.setCurrentIndex(i)
                    break
            cmb.blockSignals(False)
        except Exception:
            pass

    def _select_formal_wfl_source(self) -> None:
        """波场下拉若在「脉冲」上则切到正式 OBS_NNN。"""
        try:
            cmb = self.panel_rtm.cmb_wfl_src
            if self.panel_rtm.wfl_scope() != "impulse":
                return
            cmb.blockSignals(True)
            for i in range(cmb.count()):
                d = cmb.itemData(i)
                if isinstance(d, (tuple, list)) and len(d) >= 2 and d[0] == "formal":
                    cmb.setCurrentIndex(i)
                    break
            cmb.blockSignals(False)
        except Exception:
            pass

    def _resolve_tomo_source_path(self) -> Optional[str]:
        """用户速度源文件（v.in 等），供 Interfaces 叠层。"""
        self._panels_to_project()
        vp = self.project.velocity
        path = (
            str(vp.tomo_path or "").strip()
            or str(self.project.tomo_vel or "").strip()
            or self.panel_vel.ed_tomo.text().strip()
        )
        if not path:
            return None
        if not os.path.isabs(path) and self.project.workdir:
            path = self.project.path(path)
        return path if os.path.isfile(path) else None

    def _iface_selection_current(self) -> dict:
        """B/S/M：优先速度页画布，否则工程字段。"""
        try:
            live = self.panel_vel.canvas.interface_selection()
            if any(live.get(k) is not None for k in ("basement", "seafloor", "moho")):
                return live
        except Exception:
            pass
        v = self.project.velocity
        return {
            "basement": getattr(v, "iface_basement", None),
            "seafloor": getattr(v, "iface_seafloor", None),
            "moho": getattr(v, "iface_moho", None),
        }

    def _apply_iface_to_canvas(self, canvas, sel: Optional[dict] = None) -> None:
        if canvas is None:
            return
        try:
            canvas.apply_interface_selection(
                sel if sel is not None else self._iface_selection_current()
            )
        except Exception:
            pass

    def _sync_iface_selection_to_project(self, sel: Optional[dict] = None) -> None:
        sel = sel if sel is not None else self._iface_selection_current()
        v = self.project.velocity
        v.iface_basement = sel.get("basement")
        v.iface_seafloor = sel.get("seafloor")
        v.iface_moho = sel.get("moho")

    def _on_vel_iface_sync(self, sel: object) -> None:
        """速度页改 B/S/M → 工程 + 偏移页画布。"""
        d = sel if isinstance(sel, dict) else {}
        self._sync_iface_selection_to_project(d)
        self._apply_iface_to_canvas(self.panel_rtm.canvas, d)
        if d.get("seafloor") is not None:
            self.append_log(
                "Interfaces·S=Interface %d → 运行 RTM/生成 vel 时将用该层作 bath/填水"
                % (int(d["seafloor"]) + 1)
            )

    def _on_rtm_iface_changed(self, sel: object) -> None:
        """偏移页改 B/S/M → 工程 + 速度页画布。"""
        d = sel if isinstance(sel, dict) else {}
        self._sync_iface_selection_to_project(d)
        try:
            self.panel_vel.canvas.apply_interface_selection(d, emit=False)
        except Exception:
            pass

    def _zelt_for_rtm_overlay(self):
        """
        偏移页 Interfaces 勾选叠层用的 Zelt 模型。
        优先速度页已载入的模型 / zelt_vin_path；内置一维 → None。
        """
        if self._vel_source() != "file":
            return None
        # 速度页已有则直接复用
        try:
            zm = getattr(self.panel_vel.canvas, "_zelt_model", None)
            if zm is not None:
                return zm
        except Exception:
            pass
        path = self._resolve_tomo_source_path() or ""
        return self._zelt_from_file_cache_or_disk(
            path, self._vel_file_preview_cache
        )

    def _show_rtm_vel_array(
        self,
        vel: np.ndarray,
        meta: dict,
        *,
        title: str,
        zelt_model=None,
        shots=None,
        obs=None,
        log: bool = True,
    ) -> None:
        """在偏移画布显示速度数组（用户模型缓存或 vel.rsf）。"""
        ox = float(meta.get("o2", self.project.grid.ox))
        dx = float(meta.get("d2", self.project.grid.dx))
        oz = float(meta.get("o1", self.project.grid.oz))
        dz = float(meta.get("d1", self.project.grid.dz))
        if shots is None:
            shots = load_xz_txt(self.project.path(self.project.shots_xz))
        if obs is None:
            obs = load_xz_txt(self.project.path(self.project.obs_xz))
        self._sync_rtm_shots_from_prep(log=False)
        hl = self.panel_prep.rtm_scope_shot_ids(self.project)
        # 一维预览禁止回落加载用户 v.in（否则界面/海底线会「串」到一维上）
        if zelt_model is not None:
            zm = zelt_model
        elif self._vel_source() == "builtin_1d":
            zm = None
        else:
            zm = self._zelt_for_rtm_overlay()
        self.panel_rtm.canvas.show_vel(
            vel,
            ox=ox,
            dx=dx,
            oz=oz,
            dz=dz,
            title=title,
            shots_xz=shots or None,
            obs_xz=obs or None,
            highlight_shot_idx=hl or None,
            cbar_label="Velocity (km/s)",
            zelt_model=zm,
        )
        self._apply_iface_to_canvas(self.panel_rtm.canvas)
        if log:
            self.append_log(
                "偏移页速度预览: %s  ox=%.3f  nz×nx=%d×%d"
                % (title, ox, vel.shape[0], vel.shape[1])
            )
            self.statusBar().showMessage(title)

    def _sync_rtm_vel_display_from_vel_page(self, *, log: bool = True) -> bool:
        """
        偏移底图跟随速度页当前预览（只改显示，不写 vel.rsf）。
        优先速度页画布（含刚生成的成像 vel）；其次内存缓存。一维不同步 Zelt 界面。
        """
        self._panels_to_project()
        src = self._vel_source()
        key = self._vel_preview_cache_key()
        rtm_c = self.panel_rtm.canvas
        if (
            key is not None
            and self._rtm_vel_display_key == key
            and getattr(rtm_c, "_vel", None) is not None
            and "veloc" in str(getattr(rtm_c, "_cbar_label", "")).lower()
        ):
            try:
                rtm_c.set_shot_obs(
                    load_xz_txt(self.project.path(self.project.shots_xz)) or None,
                    load_xz_txt(self.project.path(self.project.obs_xz)) or None,
                )
            except Exception:
                pass
            if src != "builtin_1d":
                self._apply_iface_to_canvas(rtm_c)
            if log:
                self.statusBar().showMessage("偏移底图已与速度页同步（缓存）")
            return True

        # 1) 速度页画布（与当前键一致）——GUI 切换 / 生成后最准
        c = self.panel_vel.canvas
        vel = getattr(c, "_vel", None)
        if (
            vel is not None
            and key is not None
            and self._vel_preview_key == key
        ):
            meta = {
                "o2": float(c._ox),
                "d2": float(c._dx),
                "o1": float(c._oz),
                "d1": float(c._dz),
            }
            bath = getattr(c, "_bath_1d", None)
            if src == "builtin_1d":
                title = "builtin_1d [与速度页同步]"
                zelt = None
            else:
                title = "速度页预览 [与速度页同步]"
                zelt = getattr(c, "_zelt_model", None) or self._zelt_for_rtm_overlay()
            self._show_rtm_vel_array(
                vel,
                meta,
                title=title,
                zelt_model=zelt,
                shots=getattr(c, "_shots_xz", None),
                obs=getattr(c, "_obs_xz", None),
                log=log,
            )
            # 一维：清掉用户模型界面叠层；并带上 bath 曲线
            if src == "builtin_1d":
                try:
                    rtm_c._zelt_model = None
                    rtm_c._refresh_iface_combos(None)
                    rtm_c._refresh_overlays()
                except Exception:
                    pass
            if bath is not None:
                try:
                    rtm_c._bath_1d = np.asarray(bath, float)
                    nx = int(vel.shape[1])
                    if len(rtm_c._bath_1d) == nx:
                        x = float(c._ox) + np.arange(nx) * float(c._dx)
                        rtm_c._bath_curve.setData(x, rtm_c._bath_1d)
                except Exception:
                    pass
            self._rtm_vel_display_key = key
            return True

        # 2) 内存缓存
        if src != "builtin_1d":
            cache = self._vel_file_preview_cache
            if key is not None and cache and cache.get("key") == key:
                vel = cache.get("vel")
                meta = cache.get("meta") or {}
                if vel is not None and meta:
                    path = str(cache.get("path") or key[0])
                    kind = str(cache.get("kind") or "?")
                    zelt = cache.get("zelt")
                    if zelt is None:
                        zelt = self._zelt_from_file_cache_or_disk(path, cache)
                    self._show_rtm_vel_array(
                        vel,
                        meta,
                        title="%s [%s · 与速度页同步]"
                        % (os.path.basename(path), kind),
                        zelt_model=zelt,
                        log=log,
                    )
                    self._rtm_vel_display_key = key
                    return True
        else:
            bcache = self._vel_builtin_preview_cache
            if key is not None and bcache and bcache.get("key") == key:
                vel = bcache.get("vel")
                if vel is not None:
                    g = self.project.grid
                    meta = {
                        "o2": float(g.ox),
                        "d2": float(g.dx),
                        "o1": float(g.oz),
                        "d1": float(g.dz),
                    }
                    self._show_rtm_vel_array(
                        vel,
                        meta,
                        title="%s [与速度页同步]"
                        % str(bcache.get("title") or "builtin_1d"),
                        zelt_model=None,
                        log=log,
                    )
                    bath = bcache.get("bath")
                    if bath is not None:
                        try:
                            rtm_c._bath_1d = np.asarray(bath, float)
                            nx = int(vel.shape[1])
                            if len(rtm_c._bath_1d) == nx:
                                x = float(g.ox) + np.arange(nx) * float(g.dx)
                                rtm_c._bath_curve.setData(x, rtm_c._bath_1d)
                        except Exception:
                            pass
                    try:
                        rtm_c._zelt_model = None
                        rtm_c._refresh_iface_combos(None)
                        rtm_c._refresh_overlays()
                    except Exception:
                        pass
                    self._rtm_vel_display_key = key
                    return True
        return False

    def _current_file_vel_sig(self) -> Optional[tuple]:
        """当前用户模型签名：(abs_path, vin_dx, vin_dz)；无有效文件则 None。"""
        self._panels_to_project()
        vp = self.project.velocity
        tomo = str(
            getattr(vp, "tomo_path", "")
            or self.project.tomo_vel
            or self.panel_vel.ed_tomo.text().strip()
            or ""
        ).strip()
        if not tomo:
            return None
        if not os.path.isabs(tomo) and self.project.workdir:
            tomo = self.project.path(tomo)
        if not os.path.isfile(tomo):
            return None
        return (
            os.path.normpath(os.path.abspath(tomo)),
            float(getattr(vp, "vin_dx_km", 0.0) or 0.0),
            float(getattr(vp, "vin_dz_km", 0.0) or 0.0),
        )

    def _rtm_vel_needs_build(self) -> bool:
        """盘上 vel.rsf 是否与当前速度来源/模型/网格不一致（仅运行/准备 SConstruct 时用）。"""
        self._panels_to_project()
        src = self._vel_source()
        path = self._resolve_rtm_vel_path()
        tag = str(self._vel_rsf_source or "")
        if not path:
            return True
        if src == "builtin_1d":
            # 标签不是一维（含 None=来历不明的旧 vel）→ 必须按一维重写
            if tag != "builtin_1d":
                return True
            if not self._vel_rsf_matches_project_grid(path):
                return True
            return False
        # 用户模型：必须由 file 写出，且仍是「同一个模型路径 + 同一 vin 栅格」
        if tag != "file":
            return True
        cur = self._current_file_vel_sig()
        if cur is None:
            return True
        if cur != self._vel_rsf_file_sig:
            return True
        tomo = cur[0]
        try:
            if float(os.path.getmtime(tomo)) > float(os.path.getmtime(path)):
                return True
        except OSError:
            pass
        return False

    def _ensure_rtm_vel_ready(self) -> bool:
        """
        运行 RTM / 仅生成 SConstruct 前：按当前速度来源写出 vel.rsf。
        返回 True 表示已开始后台生成；False 表示盘上已可用或无法生成。
        """
        if not self._rtm_vel_needs_build():
            return False
        src = self._vel_source()
        if src == "file":
            key = self._vel_preview_cache_key()
            if key is None:
                self.append_log("无法生成 vel.rsf：尚未选择用户速度模型")
                self.statusBar().showMessage("请先在速度页选择用户模型")
                return False
        if self._busy:
            self.append_log("成像速度待生成（忙）：当前任务结束后将继续")
            return True
        self.append_log(
            "按速度来源「%s」自动生成成像速度 vel.rsf…"
            % ("内置一维" if src == "builtin_1d" else "用户模型")
        )
        self.statusBar().showMessage("正在按当前速度来源生成 vel.rsf…")
        self._build_velocity()
        return True

    def _ensure_rtm_vel_preview(self) -> None:
        """进入偏移页：只同步速度页底图预览，不写 vel.rsf。"""
        self._sync_rtm_shots_from_prep(log=False)
        self._preview_rtm_vel(silent=True)

    def _preview_rtm_vel(self, *, silent: bool = False) -> None:
        """偏移页速度预览：始终与速度页同步（GUI 缓存）；不拿旧 vel.rsf 冒充当前来源。"""
        self._stop_rtm_wfl()
        if self._sync_rtm_vel_display_from_vel_page(log=not silent):
            return
        if not silent:
            QMessageBox.information(
                self,
                "速度预览",
                "速度页尚无预览。\n"
                "请先在速度页加载/预览模型；运行 RTM 时会按来源自动生成 vel.rsf。",
            )
        else:
            self.statusBar().showMessage(
                "偏移页：请先在速度页预览模型（运行时再自动生成 vel.rsf）"
            )
            try:
                self.panel_rtm.canvas.clear(
                    "与速度页同步：请先在速度页预览模型"
                )
            except Exception:
                pass

    def _stop_rtm_wfl(self) -> None:
        self._wfl_anim_timer.stop()
        self._wfl_anim = None
        try:
            self.panel_rtm.set_wfl_animating(False)
        except Exception:
            pass

    def _impulse_shot_id(self) -> Optional[int]:
        """最近一次脉冲成像的炮号（内存或 impulse_stamp.json）。"""
        meta = self._impulse_meta or {}
        if meta.get("ishot") is not None:
            return int(meta["ishot"])
        try:
            import json

            from .services.impulse_gather import impulse_run_dir

            sp = os.path.join(impulse_run_dir(self.project), "impulse_stamp.json")
            if os.path.isfile(sp):
                with open(sp, "r", encoding="utf-8") as f:
                    st = json.load(f)
                if isinstance(st, dict) and st.get("ishot") is not None:
                    return int(st["ishot"])
        except (OSError, ValueError, TypeError):
            pass
        return None

    def _wfl_highlight_ids(self, wfl_path: Optional[str] = None) -> Optional[List[int]]:
        """
        波场图黄星：脉冲波场（路径含 impulse/）或刚跑完脉冲时，高亮脉冲炮；
        否则用手选/多边形 RTM 炮范围。
        """
        path_u = (wfl_path or "").replace("\\", "/")
        use_impulse = ("/impulse/" in path_u) or (
            str(self._rtm_mode or "") == "impulse"
        )
        if use_impulse:
            ishot = self._impulse_shot_id()
            if ishot is not None:
                return [int(ishot)]
        return self.panel_prep.rtm_scope_shot_ids(self.project)

    def _wfl_shared_base(self, *, wfl_path: Optional[str] = None):
        """速度 / 炮检公共部分；无速度则抛错。"""
        vel_path = self._resolve_rtm_vel_path()
        if not vel_path:
            raise FileNotFoundError("未找到 vel.rsf，请先生成成像速度")
        vel, vmeta = read_vel_rsf(vel_path)
        shots = load_xz_txt(self.project.path(self.project.shots_xz))
        obs = load_xz_txt(self.project.path(self.project.obs_xz))
        hl = self._wfl_highlight_ids(wfl_path)
        g = self.project.grid
        # 青色 bath 线：与成像速度同一套地形（默认 v.in·S）
        bath_1d = None
        try:
            from .services.velocity import resolve_bath_1d

            bath_1d, _ = resolve_bath_1d(self.project, g)
        except Exception:
            bath_1d = None
        return {
            "vel": vel,
            "vmeta": vmeta,
            "shots": shots,
            "obs": obs,
            "bath_1d": bath_1d,
            "highlight_ids": hl,
            "ox": float(vmeta.get("o2", g.ox)),
            "dx": float(vmeta.get("d2", g.dx)),
            "oz": float(vmeta.get("o1", g.oz)),
            "dz": float(vmeta.get("d1", g.dz)),
            "zelt_model": self._zelt_for_rtm_overlay(),
        }

    def _wfl_resolve_scope_tag(self, entries):
        """从下拉或 entries 默认项得到 (scope, tag)。"""
        scope = self.panel_rtm.wfl_scope()
        tag = self.panel_rtm.wfl_tag_id()
        if tag is not None:
            if scope is None:
                # 兼容：无 scope 时，mode=impulse 优先脉冲目录
                scope = (
                    "impulse"
                    if str(self._rtm_mode or "") == "impulse"
                    else "formal"
                )
            return str(scope), int(tag)
        if not entries:
            return None, None
        # 默认：脉冲模式选第一条 impulse；否则第一条 formal，再否则第一条
        if str(self._rtm_mode or "") == "impulse":
            for sc, tid in entries:
                if sc == "impulse":
                    return sc, int(tid)
        for sc, tid in entries:
            if sc == "formal":
                return sc, int(tid)
        sc, tid = entries[0]
        return str(sc), int(tid)

    def _wfl_preview_context(self, kind: Optional[str] = None):
        """解析波场文件；失败抛异常或返回 None 表示无波场。"""
        from .services.rtm_job import list_rtm_wfl_entries, resolve_wfl_rsf, wfl_n3

        kind = str(kind or self.panel_rtm.wfl_kind())
        entries = list_rtm_wfl_entries(self.project, kind)
        n3 = 0
        scope0, tag0 = self._wfl_resolve_scope_tag(entries)
        if tag0 is not None:
            path0 = resolve_wfl_rsf(
                self.project, kind, int(tag0), scope=scope0
            )
            if path0:
                n3 = wfl_n3(path0)
        if kind == self.panel_rtm.wfl_kind():
            self.panel_rtm.refresh_wfl_source_list(entries, n3=n3)
        if not entries:
            return None
        scope, tag = self._wfl_resolve_scope_tag(entries)
        if tag is None:
            return None
        path = resolve_wfl_rsf(self.project, kind, int(tag), scope=scope)
        if not path:
            # 回退：不限 scope
            path = resolve_wfl_rsf(
                self.project,
                kind,
                int(tag),
                prefer_impulse=(str(scope) == "impulse"),
            )
        if not path:
            raise FileNotFoundError(
                "无 %s_%03d.rsf（%s）" % (kind, int(tag), scope or "?")
            )
        n3 = wfl_n3(path)
        if kind == self.panel_rtm.wfl_kind():
            self.panel_rtm.refresh_wfl_source_list(entries, n3=n3)
        base = self._wfl_shared_base(wfl_path=path)
        return {
            "kind": kind,
            "tag": int(tag),
            "scope": str(scope or ""),
            "path": path,
            "n3": int(n3),
            "ids": [t for _s, t in entries],
            **base,
        }

    def _wfl_dual_contexts(self):
        """正传 wfls + 反传 wflr（同图叠层）；缺任一则抛错说明。"""
        from .services.rtm_job import list_rtm_wfl_entries, resolve_wfl_rsf, wfl_n3

        entries_s = list_rtm_wfl_entries(self.project, "wfls")
        entries_r = list_rtm_wfl_entries(self.project, "wflr")
        # 并集条目（同 scope+tag）
        seen = set()
        entries = []
        for e in list(entries_s) + list(entries_r):
            if e in seen:
                continue
            seen.add(e)
            entries.append(e)
        entries.sort(key=lambda x: (0 if x[0] == "impulse" else 1, x[1]))
        if not entries:
            return None
        scope, tag = self._wfl_resolve_scope_tag(entries)
        if tag is None:
            return None
        tag = int(tag)
        path_s = resolve_wfl_rsf(
            self.project, "wfls", tag, scope=scope
        )
        path_r = resolve_wfl_rsf(
            self.project, "wflr", tag, scope=scope
        )
        if not path_s or not path_r:
            miss = []
            if not path_s:
                miss.append("wfls_%03d" % tag)
            if not path_r:
                miss.append("wflr_%03d" % tag)
            raise FileNotFoundError(
                "正反同步需要同台 OBS 的正传与反传：缺少 %s（%s）"
                % ("、".join(miss), scope or "?")
            )
        n3 = min(wfl_n3(path_s), wfl_n3(path_r))
        self.panel_rtm.refresh_wfl_source_list(entries, n3=n3)
        # 高亮以反传路径为准（脉冲时含 impulse/ → 黄星=脉冲炮）
        base = self._wfl_shared_base(wfl_path=path_r or path_s)
        id_list = [t for _s, t in entries]
        ctx_s = {
            "kind": "wfls",
            "tag": tag,
            "scope": str(scope or ""),
            "path": path_s,
            "n3": int(n3),
            "ids": id_list,
            **base,
        }
        ctx_r = {
            "kind": "wflr",
            "tag": tag,
            "scope": str(scope or ""),
            "path": path_r,
            "n3": int(n3),
            "ids": id_list,
            **base,
        }
        return ctx_s, ctx_r

    def _show_wfl_frame(
        self,
        ctx: dict,
        frame: int,
        *,
        title_prefix: str = "",
    ) -> str:
        from .services.rsf_io import read_rsf_slice_n3

        canvas = self.panel_rtm.canvas
        wfl, meta = read_rsf_slice_n3(ctx["path"], int(frame))
        i3 = int(meta["i3"])
        n3 = int(meta["n3"])
        t = float(meta["t"])
        kind = ctx["kind"]
        prefix = "wflr" if str(kind).lower().startswith("wflr") else "wfls"
        kind_cn = "反传" if prefix == "wflr" else "正传"
        scope = str(ctx.get("scope") or "")
        tag_lbl = (
            "脉冲%03d" % int(ctx["tag"])
            if scope == "impulse"
            else "OBS_%03d" % int(ctx["tag"])
        )
        title = "%s%s波场 %s · %s · 帧 %d/%d · t≈%.3fs  [vel 底图]" % (
            title_prefix,
            kind_cn,
            prefix,
            tag_lbl,
            i3,
            n3,
            t,
        )
        ox = float(meta.get("o2", ctx["ox"]))
        dx = float(meta.get("d2", ctx["dx"]))
        oz = float(meta.get("o1", ctx["oz"]))
        dz = float(meta.get("d1", ctx["dz"]))
        canvas.show_wfl_on_vel(
            ctx["vel"],
            wfl,
            ox=ox,
            dx=dx,
            oz=oz,
            dz=dz,
            title=title,
            shots_xz=ctx["shots"] or None,
            obs_xz=ctx["obs"] or None,
            highlight_shot_idx=ctx["highlight_ids"] or None,
            zelt_model=ctx.get("zelt_model"),
            bath_1d=ctx.get("bath_1d"),
        )
        self._apply_iface_to_canvas(canvas)
        # Interfaces 刷新后再次钉速度色标（与动画逐帧一致）
        try:
            canvas.pin_velocity_base_display()
        except Exception:
            pass
        return title

    def _show_wfl_dual_frame(self, ctx_s: dict, ctx_r: dict, frame: int) -> str:
        """同一速度底图叠正传(暖)+反传(冷)。"""
        from .services.rsf_io import read_rsf_slice_n3

        wfl_s, meta_s = read_rsf_slice_n3(ctx_s["path"], int(frame))
        wfl_r, _meta_r = read_rsf_slice_n3(ctx_r["path"], int(frame))
        i3 = int(meta_s["i3"])
        n3 = int(meta_s["n3"])
        t = float(meta_s["t"])
        scope = str(ctx_s.get("scope") or "")
        tag_lbl = (
            "脉冲%03d" % int(ctx_s["tag"])
            if scope == "impulse"
            else "OBS_%03d" % int(ctx_s["tag"])
        )
        title = (
            "正反叠层 %s · 帧 %d/%d · t≈%.3fs  "
            "[暖=正传 / 冷=反传 · vel 底图]"
            % (tag_lbl, i3, n3, t)
        )
        ox = float(meta_s.get("o2", ctx_s["ox"]))
        dx = float(meta_s.get("d2", ctx_s["dx"]))
        oz = float(meta_s.get("o1", ctx_s["oz"]))
        dz = float(meta_s.get("d1", ctx_s["dz"]))
        self.panel_rtm.canvas.show_wfl_dual_on_vel(
            ctx_s["vel"],
            wfl_s,
            wfl_r,
            ox=ox,
            dx=dx,
            oz=oz,
            dz=dz,
            title=title,
            shots_xz=ctx_s["shots"] or None,
            obs_xz=ctx_s["obs"] or None,
            highlight_shot_idx=ctx_s["highlight_ids"] or None,
            zelt_model=ctx_s.get("zelt_model"),
            bath_1d=ctx_s.get("bath_1d"),
        )
        self._apply_iface_to_canvas(self.panel_rtm.canvas)
        try:
            self.panel_rtm.canvas.pin_velocity_base_display()
        except Exception:
            pass
        return title

    def _preview_rtm_wfl(self) -> None:
        """预览 awefd2d 波场单帧；可选正反同图叠层。"""
        self._stop_rtm_wfl()
        self._panels_to_project()
        try:
            dual = self.panel_rtm.wfl_dual()
            fr = self.panel_rtm.wfl_frame()
            if dual:
                pair = self._wfl_dual_contexts()
                if pair is None:
                    QMessageBox.information(
                        self,
                        "波场预览",
                        "rtm_work/ 中尚无 wfls/wflr。请先跑完正传与反传。",
                    )
                    return
                ctx_s, ctx_r = pair
                frame = ctx_s["n3"] // 2 if int(fr) < 0 else int(fr)
                frame = min(frame, ctx_s["n3"] - 1)
                title = self._show_wfl_dual_frame(ctx_s, ctx_r, frame)
                self.append_log(
                    "波场预览(正反叠层): %s" % title
                )
                self.statusBar().showMessage("正反同步 · 帧 %d" % frame)
                return
            ctx = self._wfl_preview_context()
            if ctx is None:
                QMessageBox.information(
                    self,
                    "波场预览",
                    "rtm_work/ 中尚无波场 RSF。\n请先跑完对应正传/反传（awefd2d snap）。",
                )
                return
            frame = ctx["n3"] // 2 if int(fr) < 0 else int(fr)
            title = self._show_wfl_frame(ctx, frame)
            kind_hint = (
                "正传：能量应从 OBS 往外"
                if str(ctx["kind"]).lower().startswith("wfls")
                else "反传：注入在检波炮"
            )
            self.append_log(
                "波场预览: %s  |  底图=速度色标 · 叠层=灰度波场 · %s"
                % (title, kind_hint)
            )
            self.statusBar().showMessage(title)
        except Exception as exc:
            QMessageBox.warning(self, "波场预览", str(exc))
            self.append_log("波场预览失败: %s" % exc)

    def _play_rtm_wfl(self) -> None:
        """逐帧播放；勾选正反同步则同图叠播正传+反传。"""
        self._stop_rtm_wfl()
        self._panels_to_project()
        try:
            dual = self.panel_rtm.wfl_dual()
            fr = self.panel_rtm.wfl_frame()
            fps = max(float(self.panel_rtm.wfl_fps()), 0.5)
            if dual:
                pair = self._wfl_dual_contexts()
                if pair is None:
                    QMessageBox.information(
                        self,
                        "波场动画",
                        "rtm_work/ 中尚无 wfls/wflr。请先跑完 awefd2d。",
                    )
                    return
                ctx_s, ctx_r = pair
                n3 = int(ctx_s["n3"])
                start = 0 if int(fr) < 0 else max(0, int(fr))
                start = min(start, max(n3 - 1, 0))
                title = self._show_wfl_dual_frame(ctx_s, ctx_r, start)
                self._wfl_anim = {
                    "dual": True,
                    "ctx_s": ctx_s,
                    "ctx_r": ctx_r,
                    "frame": start + 1,
                    "n3": n3,
                }
                self._wfl_anim_timer.start(int(round(1000.0 / fps)))
                self.panel_rtm.set_wfl_animating(True)
                self.panel_rtm.sp_wfl_frame.blockSignals(True)
                self.panel_rtm.sp_wfl_frame.setValue(start)
                self.panel_rtm.sp_wfl_frame.blockSignals(False)
                self.append_log(
                    "波场动画(正反叠层) @ %.1f fps: %s" % (fps, title)
                )
                self.statusBar().showMessage("正反同步播放 · 帧 %d/%d" % (start, n3))
                return

            ctx = self._wfl_preview_context()
            if ctx is None:
                QMessageBox.information(
                    self,
                    "波场动画",
                    "rtm_work/ 中尚无波场 RSF。请先跑完 awefd2d。",
                )
                return
            start = 0 if int(fr) < 0 else max(0, int(fr))
            start = min(start, max(ctx["n3"] - 1, 0))
            title = self._show_wfl_frame(ctx, start)
            self._wfl_anim = {
                "dual": False,
                "ctx": ctx,
                "frame": start + 1,
                "n3": int(ctx["n3"]),
            }
            self._wfl_anim_timer.start(int(round(1000.0 / fps)))
            self.panel_rtm.set_wfl_animating(True)
            self.panel_rtm.sp_wfl_frame.blockSignals(True)
            self.panel_rtm.sp_wfl_frame.setValue(start)
            self.panel_rtm.sp_wfl_frame.blockSignals(False)
            self.append_log("波场动画开始: %s  @ %.1f fps" % (title, fps))
            self.statusBar().showMessage(title)
        except Exception as exc:
            self._stop_rtm_wfl()
            QMessageBox.warning(self, "波场动画", str(exc))
            self.append_log("波场动画失败: %s" % exc)

    def _on_wfl_anim_tick(self) -> None:
        anim = self._wfl_anim
        if not anim:
            self._stop_rtm_wfl()
            return
        try:
            from .services.rsf_io import read_rsf_slice_n3

            frame = int(anim["frame"])
            n3 = int(anim["n3"])
            if frame >= n3:
                self._stop_rtm_wfl()
                self.statusBar().showMessage("波场动画结束")
                return
            if anim.get("dual"):
                ctx_s, ctx_r = anim["ctx_s"], anim["ctx_r"]
                wfl_s, meta_s = read_rsf_slice_n3(ctx_s["path"], frame)
                wfl_r, _meta_r = read_rsf_slice_n3(ctx_r["path"], frame)
                i3 = int(meta_s["i3"])
                t = float(meta_s["t"])
                scope = str(ctx_s.get("scope") or "")
                tag_lbl = (
                    "脉冲%03d" % int(ctx_s["tag"])
                    if scope == "impulse"
                    else "OBS_%03d" % int(ctx_s["tag"])
                )
                title = (
                    "正反叠层 %s · 帧 %d/%d · t≈%.3fs  "
                    "[暖=正传 / 冷=反传]"
                    % (tag_lbl, i3, n3, t)
                )
                self.panel_rtm.canvas.set_wfl_dual_overlay(
                    wfl_s, wfl_r, title=title
                )
                msg = "正反同步 · 帧 %d/%d · t≈%.3fs" % (i3, n3, t)
            else:
                ctx = anim["ctx"]
                wfl, meta = read_rsf_slice_n3(ctx["path"], frame)
                i3 = int(meta["i3"])
                t = float(meta["t"])
                kind = ctx["kind"]
                prefix = "wflr" if str(kind).lower().startswith("wflr") else "wfls"
                kind_cn = "反传" if prefix == "wflr" else "正传"
                scope = str(ctx.get("scope") or "")
                tag_lbl = (
                    "脉冲%03d" % int(ctx["tag"])
                    if scope == "impulse"
                    else "OBS_%03d" % int(ctx["tag"])
                )
                title = "%s波场 %s · %s · 帧 %d/%d · t≈%.3fs  [vel 底图]" % (
                    kind_cn,
                    prefix,
                    tag_lbl,
                    i3,
                    n3,
                    t,
                )
                self.panel_rtm.canvas.set_wfl_overlay(wfl, title=title)
                msg = title
            anim["frame"] = frame + 1
            self.panel_rtm.sp_wfl_frame.blockSignals(True)
            self.panel_rtm.sp_wfl_frame.setValue(i3)
            self.panel_rtm.sp_wfl_frame.blockSignals(False)
            self.statusBar().showMessage(msg)
        except Exception as exc:
            self._stop_rtm_wfl()
            self.append_log("波场动画中断: %s" % exc)

    def _preview_rtm_image(self) -> None:
        self._stop_rtm_wfl()
        self._panels_to_project()
        try:
            from .services.rtm_job import (
                describe_rtm_image_shots,
                list_rtm_img_shots,
                resolve_shot_indices,
                rtm_run_dir,
            )

            avail = list_rtm_img_shots(self.project)
            run = rtm_run_dir(self.project)
            obs_mode = bool(avail) and any(
                os.path.isfile(os.path.join(run, "img_obs_%03d.rsf" % i))
                for i in avail
            )
            # 「刷新列表」与预览前都同步下拉（保留当前选中）
            self._refresh_rtm_img_source_combo()
            shot_pick = self.panel_rtm.preview_shot_id()
            if isinstance(shot_pick, str) and shot_pick.startswith("impulse_"):
                self._preview_impulse_image(which=shot_pick)
                return
            if shot_pick is None and obs_mode:
                stale, why = obs_stack_staleness(self.project)
                if stale and why:
                    self.append_log("叠后可能过期：%s（可点「叠全部已有 OBS 像」）" % why)
            img, title = load_image_for_preview(
                self.project,
                prefer_stack=True,
                mute_water=self.project.rtm.mute_water_preview,
                shot_id=shot_pick if isinstance(shot_pick, int) else None,
            )
            if shot_pick is not None:
                if obs_mode:
                    shot_txt = "OBS %d（单台互易像预览）" % int(shot_pick)
                else:
                    shot_txt = "炮 %d（单炮预览）" % int(shot_pick)
            else:
                _, shot_txt = describe_rtm_image_shots(self.project)
            # 黄星高亮检波炮（手选/连续），勿用 OBS 像编号
            # max_shot=-1 时须传 n_shots，否则 resolve 返回 []、无黄星
            g = self.project.grid
            shots = load_xz_txt(self.project.path(self.project.shots_xz))
            obs = load_xz_txt(self.project.path(self.project.obs_xz))
            highlight_ids = resolve_shot_indices(
                self.project, n_shots=len(shots) if shots else None
            )
            if not self._show_rtm_image_on_vel(
                img,
                title=title,
                shots_xz=shots or None,
                obs_xz=obs or None,
                highlight_shot_idx=highlight_ids or None,
            ):
                # 无速度底图：纯振幅灰度（旧策略）
                nz, nx = img.shape
                dz = g.dz if nz == g.nz else (g.nz * g.dz / max(nz, 1))
                dx = g.dx if nx == g.nx else (g.nx * g.dx / max(nx, 1))
                vel_bg, vel_meta = self._rtm_vel_for_contours()
                self.panel_rtm.canvas.show_vel(
                    img,
                    ox=g.ox,
                    dx=dx,
                    oz=g.oz,
                    dz=dz,
                    title=title,
                    shots_xz=shots or None,
                    obs_xz=obs or None,
                    highlight_shot_idx=highlight_ids or None,
                    cbar_label="Amplitude",
                    zelt_model=self._zelt_for_rtm_overlay(),
                    vel_for_contours=vel_bg,
                    vel_for_contours_meta=vel_meta,
                )
                self._apply_iface_to_canvas(self.panel_rtm.canvas)
            self.append_log(
                "成像预览: %s  shape=%s  |  %s（黄星=检波炮 · vel 底图叠层）"
                % (title, img.shape, shot_txt)
            )
            if (
                not obs_mode
                and shot_pick is not None
                and shots
                and 0 <= int(shot_pick) < len(shots)
            ):
                xs, zs = shots[int(shot_pick)]
                self.append_log(
                    "  → 对应炮点 shot_%03d  x=%.3f km  z=%.3f km"
                    % (int(shot_pick), xs, zs)
                )
            self.statusBar().showMessage(title)
        except Exception as exc:
            # 尚无成像时回退到速度预览
            self.append_log("成像预览不可用，改显示 vel.rsf: %s" % exc)
            self._preview_rtm_vel(silent=False)

    def _sync_rtm_time_from_shot(self) -> None:
        """从炮集头文件 n1/d1 填入记录时长 T 与采样率 fs（用户仍可再改）。"""
        self._panels_to_project()
        paths = list_shot_rsf(self.project)
        if self.project.rtm.use_shots_proc:
            proc_dir = self.project.path(self.project.shots_proc_dir)
            if os.path.isdir(proc_dir):
                proc = sorted(
                    os.path.join(proc_dir, n)
                    for n in os.listdir(proc_dir)
                    if n.startswith("shot_")
                    and n.endswith(".rsf")
                    and not n.endswith(".rsf@")
                )
                if proc:
                    paths = proc
        if not paths:
            QMessageBox.information(
                self,
                "从炮集",
                "未找到 shot_*.rsf。请先在数据页完成 su_to_shots。",
            )
            return
        path = paths[0]
        try:
            meta = parse_rsf_header(path)
            nt = int(meta["n1"])
            dt = float(meta.get("d1", "0.004"))
            self.panel_rtm.set_time_from_nt_dt(nt, dt)
            self._panels_to_project()
            tmax = (nt - 1) * dt
            fs = 1.0 / dt if dt > 0 else 0.0
            self.append_log(
                "RTM 时间从炮集同步: %s  n1=%d d1=%g → T=%.3f s, fs=%.3f Hz"
                % (os.path.basename(path), nt, dt, tmax, fs)
            )
            self.statusBar().showMessage(
                "已从 %s 填入 T=%.3f s, fs=%.3f Hz（可再改）"
                % (os.path.basename(path), tmax, fs)
            )
        except Exception as exc:
            QMessageBox.warning(self, "从炮集", str(exc))

    def _prepare_scons(self) -> None:
        """清理中间文件并生成 SConstruct；全程状态栏 + 日志，避免长时间无反馈。"""
        if not self.project.workdir:
            QMessageBox.warning(self, "SConstruct", "请先打开或新建工区")
            return
        if self._rtm_vel_needs_build():
            if self._busy:
                QMessageBox.information(
                    self, "SConstruct", "正在生成成像速度，完成后将继续准备 SConstruct。"
                )
                self._rtm_pending_after_vel = "prepare"
                return
            self._rtm_pending_after_vel = "prepare"
            self.append_log("准备 SConstruct 前：按当前速度来源自动生成 vel.rsf…")
            self._ensure_rtm_vel_ready()
            return

        def _log(msg: str) -> None:
            self.append_log(msg)
            QApplication.processEvents()

        self._set_busy(True, "正在准备 SConstruct…")
        _log("--- 仅准备 SConstruct ---")
        try:
            self._sync_rtm_shots_from_prep()
            err = self._validate_rtm_shot_selection()
            if err:
                self._set_busy(False, "炮选择无效")
                QMessageBox.warning(self, "炮选择", err)
                return
            r = self.project.rtm
            _log(
                "准备 SConstruct 时间: T=%.3f s, fs=%.3f Hz → nt=%d, dt=%g"
                % (
                    float(getattr(r, "tmax", 0) or (int(r.nt) - 1) * float(r.dt)),
                    float(
                        getattr(r, "fs", 0)
                        or (1.0 / float(r.dt) if r.dt else 0)
                    ),
                    int(r.nt),
                    float(r.dt),
                )
            )
            from .services.rtm_job import (
                describe_obs_source_plan,
                describe_shot_selection,
            )

            run = prepare_scons_workdir(self.project, log=_log)
            sc = os.path.join(run, "SConstruct_obs_rtm")
            _log("SConstruct 已生成: %s" % sc)
            _log(describe_obs_source_plan(self.project))
            _log(
                "手跑(OBS为源): cd %s && scons -f SConstruct_obs_rtm img_lap.rsf"
                % run
            )
            self._set_busy(
                False,
                "SConstruct 已就绪 · %s" % os.path.basename(sc),
            )
            QMessageBox.information(
                self,
                "SConstruct",
                "已生成流程（nt=%d，T≈%.3f s）\n"
                "%s\n"
                "%s\n\n"
                "输入：工区根 vel/rr/shots\n"
                "输出：%s\n\n"
                "手跑（OBS 为源互易）：\n"
                "  cd rtm_work\n"
                "  scons -f SConstruct_obs_rtm img_lap.rsf\n\n"
                "或在本页点「运行 RTM」。"
                % (
                    int(r.nt),
                    (int(r.nt) - 1) * float(r.dt),
                    describe_shot_selection(self.project),
                    describe_obs_source_plan(self.project),
                    run,
                ),
            )
        except Exception as exc:
            self._set_busy(False, "准备 SConstruct 失败")
            self.append_log("准备 SConstruct 失败: %s" % exc)
            QMessageBox.warning(self, "SConstruct", str(exc))

    def _about(self) -> None:
        QMessageBox.information(
            self,
            "关于",
            "pyAOBS OBS RTM GUI\n\n"
            "1 数据 su_to_shots\n"
            "2 几何 offset / services.geometry\n"
            "3 预处理；应用选道→shots_proc（盘上仅当前选道）；RTM 用选道\n"
            "4 速度 tomo+bath → vel/ss/rr（工区根=输入）\n"
            "5 OBS 为源互易 RTM（Madagascar awefd2d：sou=OBS, rec=炮点）→ rtm_work/\n"
            "显示: PySide6 + pyqtgraph",
        )


def run_application(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv
    app = QApplication.instance() or QApplication(argv)
    apply_obs_rtm_font(app)
    win = ObsRtmMainWindow()
    try:
        from pyAOBS.utils.gui_logging import configure_gui_logging

        configure_gui_logging(win.append_log)
    except Exception:
        pass
    win.show()
    return int(app.exec())
