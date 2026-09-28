# -*- coding: utf-8 -*-
"""非模态姿态校正对话框：参数 / 水深 / 预览 / 运行。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..orientation_correction import OrientationCorrectionResult, OrientationObservation
from ..services import (
    AttitudeRunner,
    AttitudeSolution,
    AttitudeUiParams,
    OrientationObservationBuilder,
    SessionState,
    WaveformSelectionStore,
)
from ..services.depth_sampler import make_depth_sampler, sample_initial_depth_km
from ..services.terrain_io import load_terrain_as_utm, xy_to_utm_guess
from .dialog_utils import show_modeless_dialog
from .orientation_result_plots import draw_terrain_utm_underlay


class AttitudeCorrectionDialog(QDialog):
    def __init__(
        self,
        *,
        loaded: dict,
        sel_store: WaveformSelectionStore,
        session: SessionState,
        apick: int = 1,
        on_solution: Optional[Callable[[AttitudeSolution, OrientationCorrectionResult], None]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("姿态校正参数")
        self.setModal(False)
        self.resize(1000, 680)

        self.loaded = loaded
        self.sel_store = sel_store
        self.session = session
        self.apick = int(apick)
        self.on_solution = on_solution

        ui = session.attitude_ui or AttitudeUiParams()
        lay = QVBoxLayout(self)
        form = QFormLayout()

        self.spin_wave_pre = QDoubleSpinBox(self)
        self.spin_wave_pre.setRange(0.05, 2.5)
        self.spin_wave_pre.setDecimals(3)
        self.spin_wave_pre.setSingleStep(0.05)
        self.spin_wave_pre.setValue(float(ui.wave_pre))
        self.spin_wave_post = QDoubleSpinBox(self)
        self.spin_wave_post.setRange(0.05, 3.5)
        self.spin_wave_post.setDecimals(3)
        self.spin_wave_post.setSingleStep(0.05)
        self.spin_wave_post.setValue(float(ui.wave_post))
        self.spin_iter = QSpinBox(self)
        self.spin_iter.setRange(1, 20)
        self.spin_iter.setValue(int(ui.att_iter))
        self.spin_prior_tt = QDoubleSpinBox(self)
        self.spin_prior_tt.setRange(-2.0, 2.0)
        self.spin_prior_tt.setDecimals(3)
        self.spin_prior_tt.setSingleStep(0.01)
        self.spin_prior_tt.setValue(float(ui.prior_tt_shift_sec))
        self.spin_prior_tt.setToolTip(
            "观测侧全局走时 shift：正=加走时(变晚)，负=减走时(变早)；"
            "最优值约等于残差(预测−观测)"
        )
        self.spin_wtt = QDoubleSpinBox(self)
        self.spin_wtt.setRange(0.0, 10.0)
        self.spin_wtt.setDecimals(2)
        self.spin_wtt.setSingleStep(0.05)
        self.spin_wtt.setValue(float(ui.att_wtt))
        self.spin_wtt.setToolTip("默认较低：走时项弱约束，校正主要看方位/位置与波形项")
        self.spin_wpol = QDoubleSpinBox(self)
        self.spin_wpol.setRange(0.0, 10.0)
        self.spin_wpol.setDecimals(2)
        self.spin_wpol.setSingleStep(0.1)
        self.spin_wpol.setValue(float(ui.att_wpol))
        self.spin_wpol.setToolTip(
            "ppol 侧：多炮 ORI 圆一致性 + T≈0；勾选校正倾角时含入射角残差"
        )
        self.spin_wsym = QDoubleSpinBox(self)
        self.spin_wsym.setRange(0.0, 10.0)
        self.spin_wsym.setDecimals(2)
        self.spin_wsym.setSingleStep(0.1)
        self.spin_wsym.setValue(float(ui.att_wsym))
        self.spin_wsym.setToolTip(
            "非 ppol 窗对称（默认 0）。主约束为多炮 ORI 一致性（w_pol）"
        )

        form.addRow("窗前 (s)", self.spin_wave_pre)
        form.addRow("窗后 (s)", self.spin_wave_post)
        form.addRow("迭代次数", self.spin_iter)
        form.addRow("预置走时 shift (s)", self.spin_prior_tt)
        form.addRow("走时权重 wtt", self.spin_wtt)
        form.addRow("极化权重 wpol", self.spin_wpol)
        form.addRow("对称权重 wsym", self.spin_wsym)

        self.chk_correct_tilt = QCheckBox("校正倾角 tilt", self)
        self.chk_correct_tilt.setChecked(bool(ui.correct_tilt))
        self.chk_correct_tilt.setToolTip(
            "仅直达水波（apick=1）参与 INC_th=atan(x/h)；"
            "仅有次生相时算法会强制关闭。默认关：tilt=0。"
        )
        form.addRow("倾角", self.chk_correct_tilt)

        self.lbl_phase_mode = QLabel(
            "震相：apick=1 直达（走时+姿态）；其它字次生相（默认仅姿态）。校正用全部 V 段。",
            self,
        )
        self.lbl_phase_mode.setWordWrap(True)
        self.lbl_phase_mode.setStyleSheet("color:#0f766e;")
        form.addRow("震相策略", self.lbl_phase_mode)

        self.chk_rmean = QCheckBox("rmean", self)
        self.chk_rmean.setChecked(bool(ui.use_rmean))
        self.chk_rtrend = QCheckBox("rtrend", self)
        self.chk_rtrend.setChecked(bool(ui.use_rtrend))
        self.chk_bandpass = QCheckBox("带通", self)
        self.chk_bandpass.setChecked(bool(ui.use_bandpass))
        self.chk_bandpass.setToolTip(
            f"带通 {ui.freqlo:g}–{ui.freqhi:g} Hz（与主图一致时可在工程参数中改）；不含增益"
        )
        prep_row = QHBoxLayout()
        prep_row.addWidget(self.chk_rmean)
        prep_row.addWidget(self.chk_rtrend)
        prep_row.addWidget(self.chk_bandpass)
        prep_row.addStretch(1)
        prep_wrap = QWidget(self)
        prep_wrap.setLayout(prep_row)
        form.addRow("校正波形预处理", prep_wrap)

        self.lbl_depth = QLabel("未计算", self)
        self.lbl_depth.setStyleSheet("color:#0f172a; font-weight:600;")
        form.addRow("当前采样水深(km)", self.lbl_depth)
        lay.addLayout(form)

        terrain_row = QHBoxLayout()
        btn_load = QPushButton("更换水深…", self)
        btn_load.setToolTip("工程输入页已指定地形时会自动共用；此处仅在需要更换时再选文件")
        btn_clear = QPushButton("清除水深文件", self)
        self.lbl_terrain = QLabel(self)
        self.lbl_terrain.setStyleSheet("color:#334155;")
        self.lbl_terrain.setWordWrap(True)
        terrain_row.addWidget(btn_load)
        terrain_row.addWidget(btn_clear)
        terrain_row.addWidget(self.lbl_terrain, stretch=1)
        lay.addLayout(terrain_row)

        self.plot = pg.PlotWidget(background="w")
        lay.addWidget(self.plot, stretch=1)
        self.lbl_click = QLabel("点击预览中的震源/接收点可查看水深与走时信息。", self)
        self.lbl_click.setWordWrap(True)
        self.lbl_click.setStyleSheet("color:#1f2937;")
        lay.addWidget(self.lbl_click)

        tip = QLabel(
            "说明：可预设全局走时 shift（正加负减）；默认低 wtt、不校正倾角。"
            "校正输入 = 原始截窗 + rmean/rtrend +（可选）带通，不含增益。"
            "走时优先用 V 段叠加基准；预览仅绘当前 apick 的 V 选波炮检点。",
            self,
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#4b5563;")
        lay.addWidget(tip)

        self.lbl_progress = QLabel("校正进度：未开始", self)
        self.bar_progress = QProgressBar(self)
        self.bar_progress.setRange(0, 1)
        self.bar_progress.setValue(0)
        self.bar_progress.setFormat("%v/%m")
        lay.addWidget(self.lbl_progress)
        lay.addWidget(self.bar_progress)

        row = QHBoxLayout()
        row.addStretch(1)
        btn_cancel = QPushButton("关闭", self)
        btn_export = QPushButton("导出解", self)
        btn_preview = QPushButton("刷新预览", self)
        self.btn_run = QPushButton("开始校正", self)
        btn_cancel.clicked.connect(self.close)
        btn_export.clicked.connect(self._export_solution)
        btn_preview.clicked.connect(self._refresh_all)
        self.btn_run.clicked.connect(self._run)
        row.addWidget(btn_cancel)
        row.addWidget(btn_export)
        row.addWidget(btn_preview)
        row.addWidget(self.btn_run)
        lay.addLayout(row)

        btn_load.clicked.connect(self._load_terrain)
        btn_clear.clicked.connect(self._clear_terrain)

        # 会话/工程输入已带地形路径时自动加载，无需再点选
        if self.session.terrain_meta_utm is None and self.session.terrain_path:
            self._ensure_terrain_from_path(str(self.session.terrain_path), silent=True)

        self._refresh_terrain_label()
        self._refresh_all()

    # ---- params / observations ----
    def _sync_ui_params(self) -> AttitudeUiParams:
        prev = self.session.attitude_ui or AttitudeUiParams()
        ui = AttitudeUiParams(
            wave_pre=float(self.spin_wave_pre.value()),
            wave_post=float(self.spin_wave_post.value()),
            att_iter=int(self.spin_iter.value()),
            att_wtt=float(self.spin_wtt.value()),
            att_wpol=float(self.spin_wpol.value()),
            att_wsym=float(self.spin_wsym.value()),
            prior_tt_shift_sec=float(self.spin_prior_tt.value()),
            correct_tilt=bool(self.chk_correct_tilt.isChecked()),
            use_rmean=bool(self.chk_rmean.isChecked()),
            use_rtrend=bool(self.chk_rtrend.isChecked()),
            use_bandpass=bool(self.chk_bandpass.isChecked()),
            freqlo=float(prev.freqlo),
            freqhi=float(prev.freqhi),
            npoles=int(prev.npoles),
            izerop=bool(prev.izerop),
        )
        self.session.attitude_ui = ui
        return ui

    def _build_observations(self) -> tuple[Optional[List[OrientationObservation]], str]:
        ui = self._sync_ui_params()
        builder = OrientationObservationBuilder(self.loaded, self.sel_store, ui)
        # 汇总全部 V 段；走时/倾角仅 apick=1 直达，方位可用次生相
        return builder.build(all_apicks=True)

    def _depth_sampler(self, observations: Optional[List[OrientationObservation]] = None):
        return make_depth_sampler(self.session.terrain_meta_utm, observations)

    # ---- terrain ----
    def _refresh_terrain_label(self) -> None:
        if self.session.terrain_meta_utm is not None:
            p = str(self.session.terrain_path or self.session.terrain_meta_utm.get("path", ""))
            self.lbl_terrain.setText(
                f"已共用水深: {Path(p).name if p else '(未知)'}（UTM，与工程输入共用）"
            )
        else:
            self.lbl_terrain.setText("未加载水深文件（可在工程输入页指定）")

    def _ensure_terrain_from_path(self, path: str, *, silent: bool = False) -> bool:
        p = str(path or "").strip()
        if not p or not Path(p).is_file():
            return False
        try:
            meta_utm = load_terrain_as_utm(p)
            self.session.terrain_path = p
            self.session.terrain_meta_utm = meta_utm
            return True
        except Exception as exc:
            if not silent:
                QMessageBox.warning(self, "加载水深失败", str(exc))
            return False

    def _load_terrain(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择水深/地形文件",
            "",
            "Terrain (*.grd *.nc *.xyz *.txt);;NetCDF (*.grd *.nc);;XYZ (*.xyz *.txt);;All (*)",
        )
        if not path:
            return
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QApplication.processEvents()
            if not self._ensure_terrain_from_path(path, silent=False):
                return
            self._refresh_terrain_label()
            self._refresh_all()
        finally:
            QApplication.restoreOverrideCursor()

    def _clear_terrain(self) -> None:
        self.session.terrain_path = None
        self.session.terrain_meta_utm = None
        self._refresh_terrain_label()
        self._refresh_all()

    # ---- preview ----
    def _refresh_all(self) -> None:
        self._sync_ui_params()
        observations, err = self._build_observations()
        self._render_preview(observations)
        if not observations:
            self.lbl_depth.setText(err or "无V段观测")
            return
        sampler = self._depth_sampler(observations)
        if sampler is None:
            self.lbl_depth.setText("未加载水深")
            return
        depth0 = sample_initial_depth_km(observations, sampler)
        if depth0 is None or not np.isfinite(float(depth0)) or float(depth0) <= 0:
            self.lbl_depth.setText("采样失败")
        else:
            self.lbl_depth.setText(f"{float(depth0):.4f}")

    def _render_preview(self, observations: Optional[List[OrientationObservation]]) -> None:
        pi = self.plot.getPlotItem()
        pi.clear()
        pi.showGrid(x=True, y=True, alpha=0.15)
        pi.setLabels(left="Y (m)", bottom="X (m)")

        meta = self.session.terrain_meta_utm
        if meta is not None:
            # 与位置 Map / OBS 漂移图共用「地形」色带
            draw_terrain_utm_underlay(self.plot, dict(meta), palette="terrain")

        if not observations:
            return

        sampler = self._depth_sampler(observations)
        spots = []
        for o in observations:
            src_xy = xy_to_utm_guess(float(o.source_xyz[0]), float(o.source_xyz[1]))
            rec_xy = xy_to_utm_guess(float(o.receiver_xyz[0]), float(o.receiver_xyz[1]))
            for role, xy in (("source", src_xy), ("receiver", rec_xy)):
                if not (np.isfinite(xy[0]) and np.isfinite(xy[1])):
                    continue
                data = {
                    "role": role,
                    "trace_idx": int(o.trace_idx),
                    "offset_km": float(abs(o.offset_km)),
                    "t_obs": float(o.t0),
                    "src": np.asarray(o.source_xyz, dtype=float),
                    "rec": np.asarray(o.receiver_xyz, dtype=float),
                }
                if sampler is not None:
                    d = sampler(float(xy[0]), float(xy[1]))
                    if d is not None and np.isfinite(float(d)) and float(d) > 0:
                        depth_km = float(d)
                        slant = float(np.sqrt(float(data["offset_km"]) ** 2 + depth_km ** 2))
                        t_pred = slant / 1.5
                        data["depth_km"] = depth_km
                        data["slant_km"] = slant
                        data["t_pred"] = t_pred
                        data["residual"] = float(t_pred - float(data["t_obs"]))
                    else:
                        data["depth_km"] = float("nan")
                else:
                    data["depth_km"] = float("nan")
                spots.append(
                    {
                        "pos": (float(xy[0]), float(xy[1])),
                        "size": 11.0 if role == "source" else 9.0,
                        "brush": pg.mkBrush("#ef4444" if role == "source" else "#94a3b8"),
                        "pen": pg.mkPen("#7f1d1d" if role == "source" else "#334155", width=1.0),
                        "symbol": "t" if role == "source" else "o",
                        "data": data,
                    }
                )
        if not spots:
            return
        pick_item = pg.ScatterPlotItem(pxMode=True)
        pick_item.setData(spots=spots)
        pick_item.setZValue(50)
        pi.addItem(pick_item)

        def _on_pick(_item, points):
            if not points:
                return
            data = points[0].data()
            if not isinstance(data, dict):
                return
            role = "震源" if str(data.get("role", "")) == "source" else "接收"
            tr = int(data.get("trace_idx", -1))
            off = float(data.get("offset_km", np.nan))
            t_obs = float(data.get("t_obs", np.nan))
            depth_km = float(data.get("depth_km", np.nan))
            if np.isfinite(depth_km) and depth_km > 0:
                self.lbl_click.setText(
                    f"{role}点 | 道{tr} | 水深={depth_km:.4f} km | 偏移={off:.4f} km | "
                    f"斜距={float(data.get('slant_km', np.nan)):.4f} km | "
                    f"预测={float(data.get('t_pred', np.nan)):.4f}s | 观测={t_obs:.4f}s | "
                    f"残差(预测−观测)={float(data.get('residual', np.nan)):.4f}s "
                    f"→ 建议观测校正≈{float(data.get('residual', np.nan)):.4f}s"
                )
            else:
                self.lbl_click.setText(
                    f"{role}点 | 道{tr} | 未采样到有效水深 | 偏移={off:.4f} km | 观测={t_obs:.4f}s"
                )

        pick_item.sigClicked.connect(_on_pick)

        xs = [float(s["pos"][0]) for s in spots]
        ys = [float(s["pos"][1]) for s in spots]
        if xs and ys:
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            dx = max(1.0, xmax - xmin)
            dy = max(1.0, ymax - ymin)
            self.plot.setXRange(xmin - 0.05 * dx, xmax + 0.05 * dx, padding=0.0)
            self.plot.setYRange(ymin - 0.05 * dy, ymax + 0.05 * dy, padding=0.0)

    # ---- run / export ----
    def _run(self) -> None:
        ui = self._sync_ui_params()
        observations, err = self._build_observations()
        if observations is None:
            QMessageBox.information(self, "姿态校正", err)
            return
        sampler = self._depth_sampler(observations)
        if sampler is None:
            QMessageBox.warning(self, "姿态校正", "请先加载水深文件")
            return

        self.btn_run.setEnabled(False)
        max_iters = max(1, int(ui.att_iter))
        self.bar_progress.setRange(0, max_iters)
        self.bar_progress.setValue(0)
        self.lbl_progress.setText(f"校正进度：0/{max_iters}（准备开始）")
        QApplication.processEvents()

        def _on_progress(cur: int, total: int, stage: str) -> None:
            total_safe = max(1, int(total))
            cur_safe = max(0, min(int(cur), total_safe))
            self.bar_progress.setMaximum(total_safe)
            self.bar_progress.setValue(cur_safe)
            self.lbl_progress.setText(f"校正进度：{cur_safe}/{total_safe}，{stage}")
            QApplication.processEvents()

        try:
            runner = AttitudeRunner(ui, self.session.attitude_solution)
            result, msg = runner.run(observations, sampler, progress_callback=_on_progress)
        finally:
            self.btn_run.setEnabled(True)
            QApplication.processEvents()

        if result is None or not result.success:
            self.lbl_progress.setText("校正进度：失败")
            QMessageBox.warning(self, "姿态校正失败", msg or "未知错误")
            return

        sol = AttitudeRunner.result_to_solution(result)
        self.session.attitude_solution = sol
        dx, dy, dz = result.position_correction
        t_prior = float(result.details.get("prior_time_shift_sec", 0.0))
        t_corr = float(result.details.get("tt_corr_sec", 0.0))
        t_final = float(result.details.get("time_shift_sec", 0.0))
        j_tt = float(result.details.get("J_tt", float("nan")))
        j_pol = float(result.details.get("J_pol", float("nan")))
        j_sym = float(result.details.get("J_sym", float("nan")))
        depth_info = "未使用地形采样"
        if result.source_depth_history:
            depth_info = (
                f"水深轮次={len(result.source_depth_history)}，"
                f"范围[{min(result.source_depth_history):.3f}, {max(result.source_depth_history):.3f}] km"
            )
        text = (
            f"方位修正: {result.azimuth_deg:.2f}°\n"
            f"倾斜修正: {result.tilt_deg:.2f}°\n"
            f"位置修正: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}\n"
            f"走时预置 prior: {t_prior:.3f} s\n"
            f"走时校正 corr:  {t_corr:.3f} s\n"
            f"走时最终 final: {t_final:.3f} s  (= prior + corr)\n"
            f"目标函数: J={result.objective:.4f} (Jtt={j_tt:.4f}, Jpol={j_pol:.4f}, Jsym={j_sym:.4f})\n"
            f"{depth_info}"
        )
        self.lbl_progress.setText(f"校正进度：完成（{max_iters}/{max_iters}）")
        QMessageBox.information(self, "姿态校正结果", text)
        # 统一页签结果窗（诊断/漂移/方位/ppol），不再多窗弹出
        from .orientation_result_plots import show_orientation_result_bundle

        show_orientation_result_bundle(observations, result, parent=None)
        if self.on_solution is not None:
            try:
                self.on_solution(sol, result)
            except Exception:
                pass

    def _export_solution(self) -> None:
        sol = self.session.attitude_solution
        if sol is None:
            QMessageBox.information(self, "导出解", "尚无姿态解")
            return
        from pyAOBS.utils.qt_file_dialog import get_save_file_name

        path, _ = get_save_file_name(
            self,
            "导出姿态解 JSON",
            "attitude_solution.json",
            "JSON (*.json);;All (*)",
            default_suffix=".json",
        )
        if not path:
            return
        payload = {
            "attitude_solution": sol.to_dict(),
            "attitude_ui": (self.session.attitude_ui or AttitudeUiParams()).to_dict(),
            "terrain_path": self.session.terrain_path,
        }
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        QMessageBox.information(self, "导出解", f"已保存：{Path(path).name}")


def open_attitude_dialog(
    *,
    loaded: dict,
    sel_store: WaveformSelectionStore,
    session: SessionState,
    apick: int = 1,
    on_solution: Optional[Callable[[AttitudeSolution, OrientationCorrectionResult], None]] = None,
    parent=None,
) -> AttitudeCorrectionDialog:
    dlg = AttitudeCorrectionDialog(
        loaded=loaded,
        sel_store=sel_store,
        session=session,
        apick=apick,
        on_solution=on_solution,
        parent=parent,
    )
    show_modeless_dialog(dlg, activate=True)
    return dlg
