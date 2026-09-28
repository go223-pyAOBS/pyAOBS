"""tt_inverse 准实时监视窗：尾随 -L 曲线 + 最新写出 smesh（非模态）。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..dialog_utils import set_wrapping_caption, show_modeless_dialog, style_wrapping_caption
from ..plots.inv_monitor_model import MonitorModelWidget
from ..plots.inv_monitor_pg import MonitorCurveWidget
from ..services.inv_monitor import (
    InvMonitorSpec,
    InvMonitorSnapshot,
    collect_monitor_snapshot,
    format_monitor_status,
    progress_fraction,
)
from ..services.paths import resolve_work_dir
from ..services.smesh_plot_core import (
    load_smesh_plot_data,
    resolve_inv_start_smesh,
    resolve_plot_refl_for_smesh,
    resolve_plot_smesh_cmap,
)
from ..services.ui_prefs import restore_window_layout, save_window_layout
from ..state.form_state import FormState

_LAYOUT_KEY = "inv_monitor"
_POLL_MS = 2000
_STREAM_LABEL_MS = 120

_singleton: "InvMonitorDialog | None" = None


class InvMonitorDialog(QDialog):
    """轮询日志与 models/，展示过程指标；不自动选定最优模型。"""

    def __init__(self, state: FormState, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.setWindowTitle("反演监视（准实时）")
        self.resize(1100, 760)

        self._spec: InvMonitorSpec | None = None
        self._running = False
        self._last_smesh_key: tuple[str, float] | None = None
        self._last_ray_stamp = ""
        self._last_tres_stamp = ""
        self._last_dws_stamp = ""
        self._last_n_rows = -1
        self._mesh_path: Path | None = None
        self._showing_initial = False
        self._initial_key: tuple[str, float] | None = None
        self._start_panel_ready = False
        self._finish_announced = False
        self._pending_finish_banner = False
        self._closing = False
        self._prefer_rays = True
        self._stream_pending: str | None = None

        root = QVBoxLayout(self)
        tip = QLabel(
            "左：χ²/RMS。右上两行：折射 / 反射残差 vs 模型 X（与速度场对齐）；"
            "右下：先显示初始速度/界面（inv.mesh、-F），写出后切到最新 smesh（非「最优」）。"
            "残差需 out_level≥1 写 .tres；抽样射线需 ≥2。"
            "默认开 print_final_only 时中间模型/残差较少——属正常。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#475569;")
        root.addWidget(tip)

        self.lbl_status = QLabel(
            "未绑定运行目标 — 可在运行 tt_inverse 时自动打开，或点「绑定当前表单路径」"
        )
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("font-weight:600;")
        root.addWidget(self.lbl_status)

        prog_row = QHBoxLayout()
        prog_row.addWidget(QLabel("进度"))
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # 未知上限时忙碌条
        self.progress.setTextVisible(True)
        self.progress.setFormat("等待…")
        prog_row.addWidget(self.progress, stretch=1)
        root.addLayout(prog_row)

        self.lbl_finish = QLabel("")
        self.lbl_finish.setWordWrap(True)
        self.lbl_finish.setVisible(False)
        self.lbl_finish.setStyleSheet(
            "background:#ecfdf5;color:#065f46;border:1px solid #6ee7b7;"
            "border-radius:4px;padding:8px;font-weight:600;"
        )
        root.addWidget(self.lbl_finish)

        self.lbl_stream = QLabel("子进程流：—")
        self.lbl_stream.setWordWrap(True)
        self.lbl_stream.setStyleSheet(
            "color:#64748b;font-family:Consolas,monospace;font-size:12px;"
        )
        root.addWidget(self.lbl_stream)

        bar = QHBoxLayout()
        self.ck_auto = QCheckBox("自动刷新")
        self.ck_auto.setChecked(True)
        self.ck_rays = QCheckBox("抽样射线")
        self.ck_rays.setChecked(True)
        self.ck_rays.setToolTip(
            "默认开启：有 .ray 时按 OBS/炮着色叠加抽样射线"
            "（每炮约 30–100 条；细线半透明，少挡速度场）。"
            "需 out_level(-o)≥2。无射线文件时只画速度场。可随时关掉。"
        )
        self.ck_contours = QCheckBox("叠加等值线")
        from ..plots.velocity_contours import contours_enabled

        self.ck_contours.setChecked(contours_enabled(self.state))
        self.ck_contours.setToolTip(
            "勾选：按当前色标叠对应等值线（vp / vs / vpvs）。不勾选则只画色块与界面。"
        )
        self.ck_dws = QCheckBox("DWS 遮罩")
        from ..services.dws_plot import dws_mask_enabled
        from ..widgets.smesh_cmap_combo import SmeshCmapCombo

        self.cmap_combo = SmeshCmapCombo(
            self.state, on_changed=lambda *_a: self.refresh_once()
        )
        self.ck_dws.setChecked(dws_mask_enabled(self.state))
        self.ck_dws.setToolTip(
            "勾选：自动找该次运行 outputs/dws/（或与当前 smesh 同目录）。"
            "无覆盖（DWS≤0）留白；有覆盖按 log(DWS) 透明——越大越实、越小越淡。"
            "色标仍是速度。tt_inverse 在反演结束才写 -K。"
            "不勾选则整幅实色。"
        )
        self.btn_refresh = QPushButton("立即刷新")
        self.btn_bind = QPushButton("绑定当前表单路径")
        self.btn_analysis = QPushButton("打开反演分析…")
        self.btn_picker = QPushButton("模型挑选…")
        self.btn_picker.setToolTip("从 runs/ 选一次反演，再挑各轮 smesh 预览（右键写回表单）")
        self.btn_compare = QPushButton("模型对比…")
        self.btn_compare.setToolTip("选 A/B 两个 smesh 绘制差值 B−A（上 ΔV / 中 B / 下 A）")
        bar.addWidget(self.ck_auto)
        bar.addWidget(self.ck_rays)
        bar.addWidget(self.ck_contours)
        bar.addWidget(self.cmap_combo)
        bar.addWidget(self.ck_dws)
        bar.addWidget(self.btn_refresh)
        bar.addWidget(self.btn_bind)
        bar.addWidget(self.btn_analysis)
        bar.addWidget(self.btn_picker)
        bar.addWidget(self.btn_compare)
        bar.addStretch(1)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        bar.addWidget(btn_close)
        root.addLayout(bar)

        link_bar = QHBoxLayout()
        self.btn_open_models = QPushButton("打开 models/")
        self.btn_open_log = QPushButton("打开 -L 日志")
        self.btn_open_status = QPushButton("打开 status.jsonl")
        self.btn_open_out = QPushButton("打开输出目录")
        self.btn_save_curves = QPushButton("保存曲线…")
        self.btn_save_model = QPushButton("保存模型…")
        self.btn_save_curves.setToolTip("将左侧 χ²/RMS 保存为 PNG / JPEG / PDF / PS / SVG 等")
        self.btn_save_model.setToolTip("将右侧拟合+速度场保存为 PNG / JPEG / PDF / PS / SVG 等")
        for b in (
            self.btn_open_models,
            self.btn_open_log,
            self.btn_open_status,
            self.btn_open_out,
        ):
            link_bar.addWidget(b)
        link_bar.addStretch(1)
        link_bar.addWidget(self.btn_save_curves)
        link_bar.addWidget(self.btn_save_model)
        root.addLayout(link_bar)

        self._curve_pg = MonitorCurveWidget(self)
        self._model_pg = MonitorModelWidget(self)

        self._split = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("迭代指标（横轴=日志行序；末点高亮）"))
        ll.addWidget(self._curve_pg, stretch=1)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.lbl_mesh = QLabel("走时拟合（上） / 初始速度与界面（下）")
        style_wrapping_caption(self.lbl_mesh)
        rl.addWidget(self.lbl_mesh)
        rl.addWidget(self._model_pg, stretch=1)
        self._split.addWidget(left)
        self._split.addWidget(right)
        self._split.setStretchFactor(0, 1)
        self._split.setStretchFactor(1, 2)
        self._split.setSizes([320, 780])
        left.setMinimumWidth(200)
        right.setMinimumWidth(400)
        root.addWidget(self._split, stretch=1)

        self.btn_refresh.clicked.connect(self.refresh_once)
        self.btn_bind.clicked.connect(self.bind_from_form)
        self.btn_analysis.clicked.connect(self._open_analysis)
        self.btn_picker.clicked.connect(self._open_picker)
        self.btn_compare.clicked.connect(self._open_compare)
        self.ck_rays.toggled.connect(self._on_rays_toggled)
        self.ck_contours.toggled.connect(self._on_contours_toggled)
        self.ck_dws.toggled.connect(self._on_dws_toggled)
        self.btn_open_models.clicked.connect(self._open_models_dir)
        self.btn_open_log.clicked.connect(self._open_log)
        self.btn_open_status.clicked.connect(self._open_status)
        self.btn_open_out.clicked.connect(self._open_out_dir)
        self.btn_save_curves.clicked.connect(self._curve_pg.save_png)
        self.btn_save_model.clicked.connect(self._model_pg.save_png)

        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

        self._stream_flush = QTimer(self)
        self._stream_flush.setSingleShot(True)
        self._stream_flush.setInterval(_STREAM_LABEL_MS)
        self._stream_flush.timeout.connect(self._flush_stream_label)

        self._draw_empty_curves()
        self._ensure_initial_model()
        self._apply_save_dir()
        restore_window_layout(
            self, _LAYOUT_KEY, splitters={"main_mesh_v2": self._split}
        )

    def _apply_save_dir(self) -> None:
        d = ""
        if self._spec is not None:
            if self._spec.out_root is not None:
                d = str(Path(self._spec.out_root).parent)
            elif self._spec.run_dir is not None:
                d = str(self._spec.run_dir)
        if not d:
            try:
                d = str(resolve_work_dir(self.state.get_str("work_dir")))
            except Exception:
                d = ""
        self._curve_pg.set_save_dir(d)
        self._model_pg.set_save_dir(d)

    def attach_spec(self, spec: InvMonitorSpec, *, running: bool = True) -> None:
        self._spec = spec
        self._running = running
        self._apply_save_dir()
        self._last_smesh_key = None
        self._last_ray_stamp = ""
        self._last_tres_stamp = ""
        self._last_dws_stamp = ""
        self._last_n_rows = -1
        self._finish_announced = False
        self._pending_finish_banner = False
        self.lbl_finish.setVisible(False)
        self.lbl_finish.clear()
        self.ck_auto.setChecked(True)
        self._prefer_rays = True
        self.ck_rays.blockSignals(True)
        self.ck_rays.setChecked(True)
        self.ck_rays.blockSignals(False)
        self.refresh_once()

    def mark_run_finished(self) -> None:
        self._running = False
        self.lbl_stream.setText("子进程流：已结束")
        if not self._finish_announced:
            self._pending_finish_banner = True
        self.refresh_once()
        QTimer.singleShot(800, self.refresh_once)
        QTimer.singleShot(2000, self.refresh_once)

    def note_stream_line(self, stream: str, text: str) -> None:
        """接收 TomoAnd 流式回调（主线程，经 QueuedConnection）。"""
        tag = "stderr" if stream == "stderr" else "stdout"
        shown = (text or "").strip()
        if len(shown) > 160:
            shown = shown[:157] + "…"
        self._stream_pending = f"子进程流 [{tag}]: {shown or '…'}"
        low = shown.lower()
        if any(k in low for k in ("omp", "parallel", "ray tracing")):
            self._flush_stream_label()
            return
        if not self._stream_flush.isActive():
            self._stream_flush.start()

    def _flush_stream_label(self) -> None:
        if self._stream_pending is None:
            return
        self.lbl_stream.setText(self._stream_pending)
        self._stream_pending = None

    def bind_from_form(self) -> None:
        from ..services.inv_monitor import build_monitor_spec_from_paths

        work = resolve_work_dir(self.state.get_str("work_dir"))
        log = self.state.get_str("inv.log_file") or "tt_inverse.log"
        out = self.state.get_str("inv.out_root") or "out"
        niter = self.state.get_str("inv.niter") or None
        st = self.state.get_str("env.inv_status_jsonl_path") or "outputs/status.jsonl"
        spec = build_monitor_spec_from_paths(
            cwd=work,
            log_file=log
            if "/" in log.replace("\\", "/") or Path(log).is_absolute()
            else f"outputs/{Path(log).name}",
            out_root=out
            if "/" in out.replace("\\", "/") or Path(out).is_absolute()
            else f"outputs/{Path(out).name}",
            niter=niter,
            run_dir=None,
            status_jsonl=st,
        )
        spec.log_candidates.extend(
            [
                work / log,
                work / "outputs" / Path(log).name,
                work / "outputs" / "logs" / Path(log).name,
            ]
        )
        self.attach_spec(spec, running=False)

    def _want_overlay_rays(self) -> bool:
        """默认想叠加；真正扫盘/绘图仅当已有 .ray。手动取消勾选后不再自动打开。"""
        if self._spec is None or self._spec.out_root is None:
            return False
        try:
            from ..services.ray_sample import has_inverse_ray_files

            has = has_inverse_ray_files(self._spec.out_root)
        except Exception:
            has = False
        if self._prefer_rays:
            if has and not self.ck_rays.isChecked():
                self.ck_rays.blockSignals(True)
                self.ck_rays.setChecked(True)
                self.ck_rays.blockSignals(False)
            return has
        return bool(self.ck_rays.isChecked() and has)

    def refresh_once(self) -> None:
        if self._closing:
            return
        if self._spec is None:
            self.lbl_status.setText("未绑定目标")
            return
        include_rays = self._want_overlay_rays()
        try:
            snap = collect_monitor_snapshot(
                self._spec, include_rays=include_rays
            )
        except Exception as e:
            self.lbl_status.setText(f"刷新失败: {e}")
            return
        self._apply_snapshot(snap)

    def _on_tick(self) -> None:
        if not self.ck_auto.isChecked() or self._spec is None:
            return
        self.refresh_once()

    def _on_rays_toggled(self, _checked: bool) -> None:
        self._prefer_rays = False
        self._last_ray_stamp = ""
        if self._mesh_path is not None:
            self._update_smesh(
                self._mesh_path,
                ray_stamp="",
                is_initial=self._showing_initial,
            )

    def _on_contours_toggled(self, on: bool) -> None:
        from ..plots.velocity_contours import set_contours_enabled

        set_contours_enabled(self.state, bool(on))
        if self._mesh_path is not None:
            self._update_smesh(
                self._mesh_path,
                ray_stamp=self._last_ray_stamp,
                is_initial=self._showing_initial,
                reset_home=False,
            )

    def _on_dws_toggled(self, on: bool) -> None:
        from ..services.dws_plot import set_dws_mask_enabled

        set_dws_mask_enabled(self.state, bool(on))
        self._last_dws_stamp = ""
        if self._mesh_path is not None:
            self._update_smesh(
                self._mesh_path,
                ray_stamp=self._last_ray_stamp,
                is_initial=self._showing_initial,
                reset_home=False,
            )

    def _dws_stamp(self, smesh_path: Path | None = None) -> str:
        from ..services.dws_plot import dws_watch_stamp

        try:
            work = resolve_work_dir(self.state.get_str("work_dir"))
        except Exception:
            work = Path(".")
        path = smesh_path if smesh_path is not None else self._mesh_path
        out_root = self._spec.out_root if self._spec is not None else None
        run_dir = self._spec.run_dir if self._spec is not None else None
        return dws_watch_stamp(
            path, work, out_root=out_root, run_dir=run_dir, state=self.state
        )

    def _update_progress(self, snap: InvMonitorSnapshot) -> None:
        niter = self._spec.niter if self._spec else None
        cur, mx = progress_fraction(snap, niter=niter)
        if mx > 0:
            self.progress.setRange(0, mx)
            self.progress.setValue(cur)
            iset = snap.last_iset if snap.last_iset is not None else "?"
            self.progress.setFormat(f"iter %v/%m · iset {iset}")
        elif snap.last_iter is not None:
            self.progress.setRange(0, 0)
            self.progress.setFormat(
                f"iter {snap.last_iter}"
                + (f" · iset {snap.last_iset}" if snap.last_iset is not None else "")
            )
        else:
            self.progress.setRange(0, 0)
            self.progress.setFormat("等待…")

    def _apply_snapshot(self, snap: InvMonitorSnapshot) -> None:
        phase = "运行中" if self._running else "已结束/空闲"
        st = format_monitor_status(snap, niter=self._spec.niter if self._spec else None)
        self.lbl_status.setText(f"[{phase}] {st}")
        self._update_progress(snap)
        if self._pending_finish_banner and not self._finish_announced:
            self._finish_announced = True
            self._pending_finish_banner = False
            detail = format_monitor_status(
                snap, niter=self._spec.niter if self._spec else None
            )
            self.lbl_finish.setText(
                "反演进程已结束。"
                + (f" 末态：{detail}" if detail else "")
                + "  · 最新写出 ≠ 最优；可点「模型挑选…」。"
            )
            self.lbl_finish.setVisible(True)

        if snap.n_rows != self._last_n_rows and snap.n_rows > 0:
            self._last_n_rows = snap.n_rows
            self._update_curves(snap)

        mesh_changed = False
        if snap.smesh_path is not None and snap.smesh_mtime is not None:
            key = (str(snap.smesh_path.resolve()), float(snap.smesh_mtime))
            if key != self._last_smesh_key:
                self._last_smesh_key = key
                self._mesh_path = snap.smesh_path
                self._showing_initial = False
                self._start_panel_ready = True
                self._update_smesh(snap.smesh_path, ray_stamp=snap.ray_stamp)
                mesh_changed = True
        else:
            self._ensure_initial_model()

        if (
            self.ck_rays.isChecked()
            and not mesh_changed
            and snap.ray_stamp
            and snap.ray_stamp != self._last_ray_stamp
            and self._mesh_path is not None
        ):
            self._update_smesh(
                self._mesh_path,
                ray_stamp=snap.ray_stamp,
                is_initial=self._showing_initial,
            )
            mesh_changed = True

        if (
            self.ck_dws.isChecked()
            and not mesh_changed
            and self._mesh_path is not None
        ):
            stamp = self._dws_stamp(self._mesh_path)
            if stamp != self._last_dws_stamp:
                self._update_smesh(
                    self._mesh_path,
                    ray_stamp=self._last_ray_stamp,
                    is_initial=self._showing_initial,
                    reset_home=False,
                )
                mesh_changed = True

        if (
            not mesh_changed
            and snap.tres_stamp
            and snap.tres_stamp != self._last_tres_stamp
            and self._mesh_path is not None
        ):
            self._last_tres_stamp = snap.tres_stamp
            self._refresh_fit_axis()
        elif mesh_changed:
            self._last_tres_stamp = snap.tres_stamp or self._last_tres_stamp

    def _draw_empty_curves(self) -> None:
        self._curve_pg.show_waiting()

    def _update_curves(self, snap: InvMonitorSnapshot) -> None:
        self._curve_pg.update_snapshot(snap)

    def _draw_empty_right(self) -> None:
        self._model_pg.show_empty("未找到初始模型（请填 inv.mesh / -M）")

    def _resolve_start_smesh(self) -> Path | None:
        try:
            work = resolve_work_dir(self.state.get_str("work_dir"))
        except Exception:
            work = Path(".")
        return resolve_inv_start_smesh(self.state, work)

    def _ensure_initial_model(self) -> None:
        """尚无反演写出时，用表单初始速度 + 界面填右侧，避免空等 smesh。"""
        path = self._resolve_start_smesh()
        if path is None:
            if not self._start_panel_ready:
                self._draw_empty_right()
                set_wrapping_caption(
                    self.lbl_mesh, "未找到 inv.mesh（上：走时拟合，下：初始速度/界面）"
                )
                self._start_panel_ready = True
            return
        try:
            key = (str(path.resolve()), float(path.stat().st_mtime))
        except OSError:
            if not self._start_panel_ready:
                self._draw_empty_right()
                self._start_panel_ready = True
            return
        if self._showing_initial and self._initial_key == key:
            return
        self._initial_key = key
        self._mesh_path = path
        self._showing_initial = True
        self._start_panel_ready = True
        self._update_smesh(path, is_initial=True)

    def _draw_tres_fit(self, path: Path) -> str:
        from ..services.obs_stations import load_obs_context
        from ..services.smesh_ops import parse_inverse_smesh_name
        from ..services.tres_sample import load_outliers_for_monitor, load_tres_for_monitor

        if self._spec is None or self._spec.out_root is None:
            self._model_pg.set_residuals(None, None, note="走时拟合（残差 vs 接收点 X）")
            return ""
        work = resolve_work_dir(self.state.get_str("work_dir"))
        ctx = load_obs_context(self.state, work)
        key = parse_inverse_smesh_name(path)
        iter_prefer = key[0] if key else None
        iset_prefer = key[1] if key else None
        try:
            groups, note = load_tres_for_monitor(
                self._spec.out_root,
                iter_prefer=iter_prefer,
                run_dir=self._spec.run_dir,
            )
        except Exception as e:
            self._model_pg.set_residuals(None, ctx, note=f"残差读取失败: {e}")
            return str(e)
        outliers: list = []
        try:
            outliers, onote = load_outliers_for_monitor(
                self._spec.out_root,
                iter_prefer=iter_prefer,
                iset_prefer=iset_prefer,
            )
            if onote:
                note = f"{note} · {onote}" if note else onote
        except Exception:
            outliers = []
        if not groups:
            self._model_pg.set_residuals(
                None, ctx, note=note or "本轮残差写出后显示（需 out_level≥1）"
            )
            return note
        self._model_pg.set_residuals(
            groups, ctx, note=note, stations=ctx.stations, outliers=outliers or None
        )
        return note

    def _refresh_fit_axis(self) -> None:
        if self._mesh_path is None:
            return
        self._draw_tres_fit(self._mesh_path)

    def _update_smesh(
        self,
        path: Path,
        *,
        ray_stamp: str = "",
        is_initial: bool = False,
        reset_home: bool = True,
    ) -> None:
        try:
            from ..plots.velocity_contours import contour_specs_for_state
            from ..services.obs_stations import load_obs_context
            from ..services.ray_sample import load_sampled_rays_for_monitor
            from ..services.smesh_ops import parse_inverse_smesh_name

            work = resolve_work_dir(self.state.get_str("work_dir"))
            ctx = load_obs_context(self.state, work)
            refl = resolve_plot_refl_for_smesh(path, self.state, work)
            mesh, ds, extra = load_smesh_plot_data(path, refl)
            cmap = resolve_plot_smesh_cmap(self.state, work)
            title = (
                f"初始速度: {path.name}" if is_initial else f"最新写出: {path.name}"
            )
            from ..services.dws_plot import load_plot_dws

            out_root = self._spec.out_root if self._spec is not None else None
            run_dir = self._spec.run_dir if self._spec is not None else None
            dws_xyz, dws_path, dws_note = load_plot_dws(
                self.state,
                work,
                path,
                out_root=out_root,
                run_dir=run_dir,
                enabled=self.ck_dws.isChecked(),
            )
            self._last_dws_stamp = self._dws_stamp(path)
            if self.ck_dws.isChecked():
                if dws_xyz is not None and dws_path is not None:
                    title = f"{title}  ·  DWS {dws_path.name}"
                elif dws_note:
                    title = f"{title}  ·  {dws_note}"
            xmin, xmax = self._model_pg.set_velocity(
                ds,
                mesh,
                extra,
                cmap,
                title,
                reset_home=reset_home,
                contour_specs=contour_specs_for_state(self.state),
                dws_xyz=dws_xyz,
            )
            self._model_pg.set_model_source(path, self.state)
            ray_note = ""
            smesh_key = parse_inverse_smesh_name(path)
            ray_iter = smesh_key[0] if smesh_key else None
            if self.ck_rays.isChecked() and self._spec is not None and self._spec.out_root:
                groups, ray_note = load_sampled_rays_for_monitor(
                    self._spec.out_root,
                    iter_prefer=ray_iter,
                )
                self._model_pg.add_rays(groups, ctx)
                self._last_ray_stamp = ray_stamp
            else:
                self._last_ray_stamp = ray_stamp if not self.ck_rays.isChecked() else ""

            n_obs = self._model_pg.add_stations(
                ctx.stations,
                x_range=(xmin, xmax),
                label_ids=set(ctx.isrc_to_obs.values()) if ctx.isrc_to_obs else None,
            )
            obs_note = ""
            if n_obs:
                obs_note = f"OBS×{n_obs}"
                if ctx.station_path is not None:
                    obs_note = f"{obs_note} {ctx.station_path.name}"

            tres_note = self._draw_tres_fit(path)

            role = "初始速度/界面" if is_initial else "最新写出（非最优选定）"
            extras: list[str] = []
            if refl:
                extras.append(f"refl {Path(refl).name}")
            if obs_note:
                extras.append(obs_note)
            if tres_note:
                extras.append(tres_note)
            if ray_note:
                extras.append(ray_note)
            if self.ck_dws.isChecked() and dws_xyz is not None and dws_path is not None:
                extras.append(f"DWS {dws_path.name}")
            elif self.ck_dws.isChecked() and dws_note:
                extras.append(dws_note)
            set_wrapping_caption(
                self.lbl_mesh, f"{path.name} · {role}", " · ".join(extras)
            )
        except Exception as e:
            self._model_pg.show_empty(f"绘制失败: {e}")

    @staticmethod
    def _reveal(path: Path | None) -> None:
        if path is None:
            return
        target = path if path.is_dir() else path.parent
        if not target.exists() and path.is_file():
            target = path.parent
        if target.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))
        elif path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def _open_models_dir(self) -> None:
        if self._spec is None or self._spec.out_root is None:
            return
        parent = self._spec.out_root.parent
        models = parent / "models"
        self._reveal(models if models.is_dir() else parent)

    def _open_log(self) -> None:
        if self._spec is None:
            return
        log = self._spec.resolve_log()
        if log is not None and log.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(log.resolve())))
        else:
            self._reveal(self._spec.log_candidates[0] if self._spec.log_candidates else None)

    def _open_status(self) -> None:
        if self._spec is None:
            return
        st = self._spec.resolve_status()
        if st is not None and st.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(st.resolve())))
        elif self._spec.status_candidates:
            self._reveal(self._spec.status_candidates[0])

    def _open_out_dir(self) -> None:
        if self._spec is None:
            return
        if self._spec.run_dir is not None and self._spec.run_dir.is_dir():
            self._reveal(self._spec.run_dir / "outputs")
            return
        if self._spec.out_root is not None:
            self._reveal(self._spec.out_root.parent)

    def _open_analysis(self) -> None:
        from .inv_analysis_dialog import open_inv_analysis_dialog

        dlg = open_inv_analysis_dialog(self.state, None)
        if self._spec is not None:
            log = self._spec.resolve_log()
            if log is not None:
                try:
                    work = resolve_work_dir(self.state.get_str("work_dir"))
                    try:
                        rel = log.relative_to(work).as_posix()
                    except ValueError:
                        rel = str(log)
                    dlg.single_edit.setText(rel)
                except Exception:
                    pass

    def _open_picker(self) -> None:
        # 独立窗 setParent(None)，经顶层主窗入口写回以便 push 控件
        from PySide6.QtWidgets import QApplication

        for w in QApplication.topLevelWidgets():
            opener = getattr(w, "open_model_picker", None)
            if callable(opener):
                opener(self._spec)
                return
        from .model_picker_dialog import open_model_picker_dialog

        open_model_picker_dialog(self.state, None, spec=self._spec)

    def _open_compare(self) -> None:
        from PySide6.QtWidgets import QApplication

        for w in QApplication.topLevelWidgets():
            opener = getattr(w, "open_model_compare", None)
            if callable(opener):
                opener()
                return
        from .model_compare_dialog import open_model_compare_dialog

        open_model_compare_dialog(self.state, None)

    def closeEvent(self, event) -> None:  # noqa: N802
        global _singleton
        self._closing = True
        try:
            save_window_layout(
                self, _LAYOUT_KEY, splitters={"main_mesh_v2": self._split}
            )
        except Exception:
            pass
        try:
            self._timer.stop()
            self._stream_flush.stop()
        except Exception:
            pass
        if _singleton is self:
            _singleton = None
        super().closeEvent(event)


def open_inv_monitor_dialog(
    state: FormState,
    parent=None,
    *,
    spec: InvMonitorSpec | None = None,
    running: bool = False,
) -> InvMonitorDialog:
    """单例非模态监视窗；可复用已打开实例并改绑目标。"""
    global _singleton
    if _singleton is not None:
        try:
            _ = _singleton.isVisible()
            dlg = _singleton
        except RuntimeError:
            _singleton = None
            dlg = None
    else:
        dlg = None

    if dlg is None:
        dlg = InvMonitorDialog(state, parent)
        _singleton = dlg
        show_modeless_dialog(dlg, activate=True)
    else:
        dlg.state = state
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    if spec is not None:
        dlg.attach_spec(spec, running=running)
    return dlg
