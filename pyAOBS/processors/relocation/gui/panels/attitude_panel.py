# -*- coding: utf-8 -*-
"""阶段 3：姿态校正参数与运行。"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...project import RelocationProject
from ...services.models import AttitudeUiParams


class AttitudePanel(QWidget):
    project_changed = Signal()
    request_run = Signal()
    request_preview = Signal()
    request_open_workbench = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        tip = QLabel(
            "震相：apick=1=直达水波（走时/位置+姿态，倾角可选）；"
            "其它 apick=折射/反射（默认仅姿态/ORI）。可同时用多种 V 段，走时只信直达。"
            "可预设全局走时 shift；默认低 wtt、不校正倾角。"
            "校正输入：原始截窗 + rmean + rtrend +（可选）带通，不含增益。"
            "「保存姿态结果」仅 JSON；「接受为当前修正」写波形/道头。"
            "结果为统一页签窗（诊断/波形/极化/漂移/方位/ppol）。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#475569;")
        lay.addWidget(tip)

        box = QGroupBox("反演参数", self)
        form = QFormLayout(box)
        self.sp_pre = QDoubleSpinBox(box)
        self.sp_pre.setRange(0.05, 2.5)
        self.sp_pre.setDecimals(3)
        self.sp_pre.setSingleStep(0.05)
        self.sp_post = QDoubleSpinBox(box)
        self.sp_post.setRange(0.05, 3.5)
        self.sp_post.setDecimals(3)
        self.sp_post.setSingleStep(0.05)
        self.sp_iter = QSpinBox(box)
        self.sp_iter.setRange(1, 30)
        self.sp_prior_tt = QDoubleSpinBox(box)
        self.sp_prior_tt.setRange(-2.0, 2.0)
        self.sp_prior_tt.setDecimals(3)
        self.sp_prior_tt.setSingleStep(0.01)
        self.sp_prior_tt.setToolTip(
            "观测侧全局走时 shift：正=加走时(变晚)，负=减走时(变早)；最优≈残差(预测−观测)"
        )
        self.sp_wtt = QDoubleSpinBox(box)
        self.sp_wtt.setRange(0.0, 20.0)
        self.sp_wtt.setDecimals(2)
        self.sp_wtt.setSingleStep(0.05)
        self.sp_wtt.setToolTip("默认较低：走时项弱约束，校正主要看方位/位置")
        self.sp_wpol = QDoubleSpinBox(box)
        self.sp_wpol.setRange(0.0, 20.0)
        self.sp_wpol.setDecimals(2)
        self.sp_wpol.setToolTip(
            "ppol 侧：多炮 ORI 圆一致性 + T≈0；勾选校正倾角时含入射角残差"
        )
        self.sp_wsym = QDoubleSpinBox(box)
        self.sp_wsym.setRange(0.0, 20.0)
        self.sp_wsym.setDecimals(2)
        self.sp_wsym.setToolTip(
            "非 ppol 窗对称（默认 0）。主约束为多炮 ORI 一致性（w_pol）"
        )
        form.addRow("波窗前 (s)", self.sp_pre)
        form.addRow("波窗后 (s)", self.sp_post)
        form.addRow("迭代次数", self.sp_iter)
        form.addRow("预置走时 shift (s)", self.sp_prior_tt)
        form.addRow("权重 w_tt", self.sp_wtt)
        form.addRow("权重 w_pol", self.sp_wpol)
        form.addRow("权重 w_sym", self.sp_wsym)
        self.chk_correct_tilt = QCheckBox("校正倾角 tilt", box)
        self.chk_correct_tilt.setToolTip(
            "仅 apick=1 直达用水深几何 INC_th；仅次生相时强制关闭。"
            "默认关：tilt=0；勾选后搜 ±15°"
        )
        form.addRow("倾角", self.chk_correct_tilt)
        self.chk_rmean = QCheckBox("rmean", box)
        self.chk_rtrend = QCheckBox("rtrend", box)
        self.chk_bandpass = QCheckBox("带通(同主图频带)", box)
        prep = QHBoxLayout()
        prep.addWidget(self.chk_rmean)
        prep.addWidget(self.chk_rtrend)
        prep.addWidget(self.chk_bandpass)
        prep.addStretch(1)
        form.addRow("校正预处理", prep)
        lay.addWidget(box)

        sbox = QGroupBox("当前解（可作初值）", self)
        self.txt_sol = QPlainTextEdit(sbox)
        self.txt_sol.setReadOnly(True)
        self.txt_sol.setMaximumHeight(140)
        sl = QVBoxLayout(sbox)
        sl.addWidget(self.txt_sol)
        lay.addWidget(sbox)

        row = QHBoxLayout()
        self.btn_wb = QPushButton("打开波形工作台", self)
        self.btn_wb.clicked.connect(self.request_open_workbench.emit)
        self.btn_preview = QPushButton("预览当前解", self)
        self.btn_preview.setToolTip("打开校正结果图（三分量波形/极化等），不套整剖面主图")
        self.btn_preview.clicked.connect(self.request_preview.emit)
        self.btn_run = QPushButton("运行姿态校正", self)
        self.btn_run.clicked.connect(self.request_run.emit)
        row.addWidget(self.btn_wb)
        row.addWidget(self.btn_preview)
        row.addWidget(self.btn_run)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addStretch(1)

        for w in (self.sp_pre, self.sp_post, self.sp_wtt, self.sp_wpol, self.sp_wsym, self.sp_prior_tt):
            w.valueChanged.connect(lambda *_: self.project_changed.emit())
        self.sp_iter.valueChanged.connect(lambda *_: self.project_changed.emit())
        for c in (self.chk_correct_tilt, self.chk_rmean, self.chk_rtrend, self.chk_bandpass):
            c.toggled.connect(lambda *_: self.project_changed.emit())

        ui0 = AttitudeUiParams()
        self.sp_pre.setValue(ui0.wave_pre)
        self.sp_post.setValue(ui0.wave_post)
        self.sp_iter.setValue(ui0.att_iter)
        self.sp_prior_tt.setValue(ui0.prior_tt_shift_sec)
        self.sp_wtt.setValue(ui0.att_wtt)
        self.sp_wpol.setValue(ui0.att_wpol)
        self.sp_wsym.setValue(ui0.att_wsym)
        self.chk_correct_tilt.setChecked(bool(ui0.correct_tilt))
        self.chk_rmean.setChecked(bool(ui0.use_rmean))
        self.chk_rtrend.setChecked(bool(ui0.use_rtrend))
        self.chk_bandpass.setChecked(bool(ui0.use_bandpass))
        self._refresh_sol_text(RelocationProject().attitude_solution)

    def apply_to_project(self, project: RelocationProject) -> None:
        prev = project.attitude_ui or AttitudeUiParams()
        project.attitude_ui = AttitudeUiParams(
            wave_pre=float(self.sp_pre.value()),
            wave_post=float(self.sp_post.value()),
            att_iter=int(self.sp_iter.value()),
            att_wtt=float(self.sp_wtt.value()),
            att_wpol=float(self.sp_wpol.value()),
            att_wsym=float(self.sp_wsym.value()),
            prior_tt_shift_sec=float(self.sp_prior_tt.value()),
            correct_tilt=bool(self.chk_correct_tilt.isChecked()),
            use_rmean=bool(self.chk_rmean.isChecked()),
            use_rtrend=bool(self.chk_rtrend.isChecked()),
            use_bandpass=bool(self.chk_bandpass.isChecked()),
            freqlo=float(prev.freqlo),
            freqhi=float(prev.freqhi),
            npoles=int(prev.npoles),
            izerop=bool(prev.izerop),
        )

    def load_from_project(self, project: RelocationProject) -> None:
        ui = project.attitude_ui
        self.sp_pre.setValue(float(ui.wave_pre))
        self.sp_post.setValue(float(ui.wave_post))
        self.sp_iter.setValue(int(ui.att_iter))
        self.sp_prior_tt.setValue(float(ui.prior_tt_shift_sec))
        self.sp_wtt.setValue(float(ui.att_wtt))
        self.sp_wpol.setValue(float(ui.att_wpol))
        self.sp_wsym.setValue(float(ui.att_wsym))
        self.chk_correct_tilt.setChecked(bool(ui.correct_tilt))
        self.chk_rmean.setChecked(bool(ui.use_rmean))
        self.chk_rtrend.setChecked(bool(ui.use_rtrend))
        self.chk_bandpass.setChecked(bool(ui.use_bandpass))
        self._refresh_sol_text(project.attitude_solution)

    def _refresh_sol_text(self, sol) -> None:
        self.txt_sol.setPlainText(
            "\n".join(
                [
                    f"azimuth = {sol.azimuth_deg:.4f} °",
                    f"tilt    = {sol.tilt_deg:.4f} °",
                    f"dx,dy,dz = {sol.dx:.4f}, {sol.dy:.4f}, {sol.dz:.4f}",
                    f"走时预置 prior = {sol.prior_tt_shift_sec:.4f} s",
                    f"走时校正 corr  = {sol.tt_corr_sec:.4f} s",
                    f"走时最终 final = {sol.time_shift_sec:.4f} s  (= prior + corr)",
                ]
            )
        )

    def set_solution_from_project(self, project: RelocationProject) -> None:
        self._refresh_sol_text(project.attitude_solution)
