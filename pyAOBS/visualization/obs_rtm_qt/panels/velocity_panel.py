# -*- coding: utf-8 -*-
"""阶段 4：用户速度 / 内置一维 → vel.rsf。"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..dialog_utils import connect_combo_deferred
from ..project import ObsRtmProject
from ..styles import compact_form, hint_label, primary_button, side_panel_layout
from ..widgets.vel_canvas import VelCanvas


_FILTER = (
    "Velocity models ("
    "v.in *.vin *.grd *.nc *.rsf smesh *);;"
    "Zelt v.in (v.in *.vin);;"
    "Grid (*.grd *.nc);;"
    "RSF (*.rsf);;"
    "All (*.*)"
)


class VelocityPanel(QWidget):
    request_build = Signal()
    request_preview_tomo = Signal()
    request_convert = Signal()
    request_sync_grid_from_model = Signal()
    project_changed = Signal()
    iface_sync_requested = Signal(object)  # B/S/M dict → 偏移页同步

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self.request_preview_tomo.emit)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split)

        left = QWidget()
        left.setObjectName("ObsRtmSidePanel")
        left.setMinimumWidth(300)
        left.setMaximumWidth(520)
        left_l = side_panel_layout(left)

        box = QGroupBox("输入（自动转 .rsf）")
        form = compact_form(QFormLayout(box))

        self.cmb_source = QComboBox()
        self.cmb_source.addItem("用户速度文件（v.in / rsf…）", "file")
        self.cmb_source.addItem("内置一维速度（无需加载模型）", "builtin_1d")
        self.cmb_source.setToolTip(
            "无用户层析时可选内置一维：按工区网格生成 vel.rsf，"
            "海水由 bath/OBS 水深填入"
        )
        # UI 启用状态必须立刻同步（勿等 activated 延迟，否则一维参数会一直灰掉）
        self.cmb_source.currentIndexChanged.connect(self._sync_source_ui)
        connect_combo_deferred(self.cmb_source, self._after_source_changed)
        form.addRow("速度来源", self.cmb_source)

        row_t = QHBoxLayout()
        self.ed_tomo = QLineEdit()
        self.ed_tomo.setPlaceholderText("v.in / .grd /.nc / smesh / .rsf")
        self.ed_tomo.setToolTip("选择或输入后自动绘制速度图")
        btn_t = QPushButton("浏览…")
        btn_t.setToolTip("选择 v.in / .grd / .nc / smesh / .rsf；选定后自动出图")
        btn_t.clicked.connect(self._pick_tomo)
        self.btn_tomo = btn_t
        row_t.addWidget(self.ed_tomo)
        row_t.addWidget(btn_t)
        form.addRow("速度模型", row_t)

        self.sp_vin_dx = QDoubleSpinBox()
        self.sp_vin_dx.setRange(0.001, 10.0)
        self.sp_vin_dx.setDecimals(4)
        self.sp_vin_dx.setValue(0.5)
        self.sp_vin_dx.setSuffix(" km")
        self.sp_vin_dx.setToolTip(
            "v.in / smesh → tomo_vel 栅格化 dx；成像 vel 仍会再采样到工区网格。"
        )
        self.sp_vin_dz = QDoubleSpinBox()
        self.sp_vin_dz.setRange(0.001, 10.0)
        self.sp_vin_dz.setDecimals(4)
        self.sp_vin_dz.setValue(0.25)
        self.sp_vin_dz.setSuffix(" km")
        self.sp_vin_dz.setToolTip("v.in / smesh → tomo_vel 栅格化 dz")
        form.addRow("栅格 dx", self.sp_vin_dx)
        form.addRow("栅格 dz", self.sp_vin_dz)

        self.chk_auto = QCheckBox("生成成像速度时自动写 tomo_vel.rsf")
        self.chk_auto.setChecked(True)
        self.chk_auto.setToolTip(
            "输入非 .rsf 时，「生成成像速度」会先栅格化写出 tomo_vel.rsf"
        )
        form.addRow(self.chk_auto)

        # —— 内置一维 / 水深：两列×四行（控件可收缩，不撑宽侧栏）——
        def _shrink_combo(cmb: QComboBox) -> None:
            cmb.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            cmb.setMinimumContentsLength(5)
            cmb.setMinimumWidth(0)
            cmb.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )

        def _shrink_field(w) -> None:
            w.setMinimumWidth(0)
            w.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )

        self.cmb_v1d_preset = QComboBox()
        self.cmb_v1d_preset.addItem("线性梯度", "linear_crust")
        self.cmb_v1d_preset.addItem("分层表", "layered_crust")
        self.cmb_v1d_preset.setToolTip(
            "linear：v=v0+grad×深度；layered：内置分层结点插值"
        )
        _shrink_combo(self.cmb_v1d_preset)
        self.cmb_v1d_preset.currentIndexChanged.connect(self._sync_source_ui)
        connect_combo_deferred(self.cmb_v1d_preset, self._after_v1d_param_changed)

        self.cmb_v1d_ref = QComboBox()
        self.cmb_v1d_ref.addItem("海底以下", "subbottom")
        self.cmb_v1d_ref.addItem("绝对深度", "absolute")
        self.cmb_v1d_ref.setToolTip(
            "不同 OBS 水深时：\n"
            "· 海底以下：岩体速度按 z−bath(x)，沉积随海底起伏（推荐）\n"
            "· 绝对深度：同 z 同速，仅水柱厚度随 bath 变化"
        )
        _shrink_combo(self.cmb_v1d_ref)
        connect_combo_deferred(self.cmb_v1d_ref, self._after_v1d_param_changed)

        self.sp_v1d_v0 = QDoubleSpinBox()
        self.sp_v1d_v0.setRange(0.5, 9.0)
        self.sp_v1d_v0.setDecimals(3)
        self.sp_v1d_v0.setSingleStep(0.1)
        self.sp_v1d_v0.setKeyboardTracking(False)
        self.sp_v1d_v0.setValue(2.0)
        self.sp_v1d_v0.setSuffix(" km/s")
        self.sp_v1d_v0.setToolTip("线性预设：海底处岩体速度（可编辑）")
        self.sp_v1d_grad = QDoubleSpinBox()
        self.sp_v1d_grad.setRange(0.0, 5.0)
        self.sp_v1d_grad.setDecimals(3)
        self.sp_v1d_grad.setSingleStep(0.05)
        self.sp_v1d_grad.setKeyboardTracking(False)
        self.sp_v1d_grad.setValue(0.5)
        self.sp_v1d_grad.setSuffix(" /km")
        self.sp_v1d_grad.setToolTip("线性预设：垂向梯度 (km/s)/km（可编辑）")
        self.sp_v1d_vmax = QDoubleSpinBox()
        self.sp_v1d_vmax.setRange(1.5, 9.5)
        self.sp_v1d_vmax.setDecimals(3)
        self.sp_v1d_vmax.setSingleStep(0.1)
        self.sp_v1d_vmax.setKeyboardTracking(False)
        self.sp_v1d_vmax.setValue(8.0)
        self.sp_v1d_vmax.setSuffix(" km/s")
        self.sp_v1d_vmax.setToolTip("线性预设：速度上限（可编辑）")

        self.chk_v1d_iface = QCheckBox("等值线→界面")
        self.chk_v1d_iface.setChecked(False)
        self.chk_v1d_iface.setToolTip(
            "诊断：把选定速度等值线做成有限跳变界面（仅海底以下）。\n"
            "默认关；与 Contours 的 5/6/7/8 km/s 档位对应，便于脉冲 RTM 对比。"
        )
        self.cmb_v1d_iface_v = QComboBox()
        for vv in (5.0, 6.0, 7.0, 8.0):
            self.cmb_v1d_iface_v.addItem("%.1f km/s" % vv, float(vv))
        self.cmb_v1d_iface_v.setCurrentIndex(1)  # 6.0
        self.cmb_v1d_iface_v.setToolTip("作为界面的速度等值线（km/s）")
        _shrink_combo(self.cmb_v1d_iface_v)
        self.sp_v1d_iface_dv = QDoubleSpinBox()
        self.sp_v1d_iface_dv.setRange(-3.0, 3.0)
        self.sp_v1d_iface_dv.setDecimals(2)
        self.sp_v1d_iface_dv.setSingleStep(0.1)
        self.sp_v1d_iface_dv.setKeyboardTracking(False)
        self.sp_v1d_iface_dv.setValue(0.8)
        self.sp_v1d_iface_dv.setSuffix(" km/s")
        self.sp_v1d_iface_dv.setToolTip(
            "界面下侧速度跳变 ΔV（可负）。过大会产生强假像；建议 0.5–1.0"
        )
        _shrink_field(self.sp_v1d_iface_dv)
        row_iface = QHBoxLayout()
        row_iface.setContentsMargins(0, 0, 0, 0)
        row_iface.setSpacing(4)
        row_iface.addWidget(self.chk_v1d_iface, 0)
        row_iface.addWidget(self.cmb_v1d_iface_v, 1)
        row_iface.addWidget(self.sp_v1d_iface_dv, 1)
        self._row_iface = row_iface

        row_b = QHBoxLayout()
        row_b.setContentsMargins(0, 0, 0, 0)
        row_b.setSpacing(4)
        self.ed_bath = QLineEdit()
        self.ed_bath.setText("prep/geom/bath_x.txt")
        self.ed_bath.setToolTip(
            "可选水深剖面 prep/geom/bath_x.txt（x z，km）。\n"
            "用户 v.in：默认优先 Interfaces·S（地形海底）；\n"
            "无 S/无 v.in 时：bath 文件 → OBS z → 「平海底」。"
        )
        _shrink_field(self.ed_bath)
        btn_b = QPushButton("…")
        btn_b.setFixedWidth(28)
        btn_b.setToolTip("选择 bath_x.txt 水深文件")
        btn_b.clicked.connect(self._pick_bath)
        row_b.addWidget(self.ed_bath, 1)
        row_b.addWidget(btn_b)

        self.sp_flat = QDoubleSpinBox()
        self.sp_flat.setRange(0.0, 50.0)
        self.sp_flat.setDecimals(3)
        self.sp_flat.setValue(1.901)
        self.sp_flat.setSuffix(" km")
        self.sp_flat.setToolTip(
            "无 bath 文件且无 OBS 时的常数海底深度（km）"
        )

        self.sp_vw = QDoubleSpinBox()
        self.sp_vw.setRange(1.0, 2.0)
        self.sp_vw.setDecimals(3)
        self.sp_vw.setValue(1.50)
        self.sp_vw.setSuffix(" km/s")
        self.sp_vw.setToolTip("海水速度（km/s），用于 z < bath(x) 填水")

        for w in (
            self.sp_v1d_v0,
            self.sp_v1d_grad,
            self.sp_v1d_vmax,
            self.sp_flat,
            self.sp_vw,
        ):
            _shrink_field(w)

        col_l = QFormLayout()
        col_l.setContentsMargins(0, 0, 4, 0)
        col_l.setHorizontalSpacing(4)
        col_l.setVerticalSpacing(2)
        col_l.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        col_l.addRow("预设", self.cmb_v1d_preset)
        col_l.addRow("参照", self.cmb_v1d_ref)
        col_l.addRow("v0", self.sp_v1d_v0)
        col_l.addRow("梯度", self.sp_v1d_grad)
        col_r = QFormLayout()
        col_r.setContentsMargins(4, 0, 0, 0)
        col_r.setHorizontalSpacing(4)
        col_r.setVerticalSpacing(2)
        col_r.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        col_r.addRow("vmax", self.sp_v1d_vmax)
        col_r.addRow("bath", row_b)
        col_r.addRow("平海底", self.sp_flat)
        col_r.addRow("vwater", self.sp_vw)
        row_1d = QHBoxLayout()
        row_1d.setContentsMargins(0, 0, 0, 0)
        row_1d.setSpacing(4)
        row_1d.addLayout(col_l, 1)
        row_1d.addLayout(col_r, 1)
        form.addRow(row_1d)
        form.addRow("界面", self._row_iface)

        self.chk_fill = QCheckBox("z < bath 填海水")
        self.chk_fill.setChecked(True)
        self.chk_fill.setToolTip(
            "按 bath(x) 填水柱：各 OBS 水深不同则水柱厚度不同（RTM 推荐开）"
        )
        form.addRow(self.chk_fill)

        self.sp_smooth = QSpinBox()
        self.sp_smooth.setRange(0, 80)
        self.sp_smooth.setValue(0)
        self.sp_smooth.setToolTip("盒式光滑半宽 rect（网格点）；0=不光滑")
        form.addRow("光滑 rect", self.sp_smooth)

        self.ed_out = QLineEdit("rtm_in/vel.rsf")
        self.ed_out.setToolTip("成像速度输出文件名（相对工区），供 RTM 使用")
        form.addRow("输出成像速度", self.ed_out)
        left_l.addWidget(box)

        left_l.addWidget(
            hint_label(
                "无用户速度时：选「内置一维」→ 自动预览；改 v0/梯度/vmax 会刷新图。\n"
                "起伏水深：bath 优先，否则用 obs_xz 的 z 插值；"
                "一维参照选「海底以下」时岩体随海底起伏。\n"
                "「等值线→界面」：把 5/6/7/8 km/s 做成有限跳变（诊断用，默认关）。\n"
                "用户文件：tomo_vel=栅格中间体；vel=工区网格+填水，供 RTM。"
            )
        )

        btn_conv = QPushButton("转换模型 → tomo_vel.rsf")
        btn_conv.setToolTip(
            "仅栅格化写出中间体 tomo_vel.rsf（不填海水、不强制工区网格）"
        )
        btn_conv.clicked.connect(self.request_convert.emit)
        self.btn_conv = btn_conv
        btn_grid_m = QPushButton("用模型范围更新工区网格")
        btn_grid_m.setToolTip(
            "按速度模型 x/z 范围写工区网格（选择/预览用户模型时会自动执行；"
            "此按钮可手动重做）。\n"
            "按炮/OBS 建议 ox/nx 请到「工区几何」页使用「由炮点建议网格」。"
        )
        btn_grid_m.clicked.connect(self.request_sync_grid_from_model.emit)
        self.btn_grid_m = btn_grid_m
        btn_build = primary_button("生成成像速度 → rtm_in/vel.rsf")
        btn_build.setToolTip(
            "写出 RTM 用 rtm_in/vel.rsf（用户模型或内置一维 + bath 填水）。\n"
            "亦可在偏移页点「运行 RTM」或「仅生成 SConstruct」时按当前来源自动生成。"
        )
        btn_build.clicked.connect(self.request_build.emit)
        left_l.addWidget(btn_conv)
        left_l.addWidget(btn_grid_m)
        left_l.addWidget(btn_build)
        left_l.addStretch(1)
        split.addWidget(left)

        self.canvas = VelCanvas()
        self.canvas.setObjectName("ObsRtmPlotPanel")
        self.canvas.iface_changed.connect(self._on_iface_changed)
        split.addWidget(self.canvas)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        split.setSizes([420, 680])

        for w in (self.ed_tomo, self.ed_bath, self.ed_out):
            w.textChanged.connect(lambda *_: self.project_changed.emit())
        self.ed_tomo.editingFinished.connect(self._on_tomo_path_ready)
        self.ed_bath.editingFinished.connect(self._schedule_1d_preview)
        for w in (self.sp_vin_dx, self.sp_vin_dz):
            w.valueChanged.connect(lambda *_: self.project_changed.emit())
        self.sp_vin_dx.editingFinished.connect(self._on_tomo_path_ready)
        self.sp_vin_dz.editingFinished.connect(self._on_tomo_path_ready)
        for w in (
            self.sp_flat,
            self.sp_vw,
            self.sp_v1d_v0,
            self.sp_v1d_grad,
            self.sp_v1d_vmax,
            self.sp_v1d_iface_dv,
        ):
            w.valueChanged.connect(self._on_1d_spin_changed)
        self.sp_smooth.valueChanged.connect(lambda *_: self.project_changed.emit())
        self.chk_fill.toggled.connect(self._on_1d_flag_changed)
        self.chk_v1d_iface.toggled.connect(self._on_1d_flag_changed)
        connect_combo_deferred(self.cmb_v1d_iface_v, self._after_v1d_param_changed)
        self.chk_auto.toggled.connect(lambda *_: self.project_changed.emit())
        self._sync_source_ui()

    def _source(self) -> str:
        return str(self.cmb_source.currentData() or "file")

    def _sync_source_ui(self, *_args) -> None:
        """立刻启用/禁用控件（不重绘，避免下拉卡住）。"""
        is_file = self._source() == "file"
        is_1d = not is_file
        for w in (
            self.ed_tomo,
            self.btn_tomo,
            self.sp_vin_dx,
            self.sp_vin_dz,
            self.chk_auto,
            self.btn_conv,
            self.btn_grid_m,
        ):
            w.setEnabled(is_file)
        for w in (self.cmb_v1d_preset, self.cmb_v1d_ref, self.chk_v1d_iface):
            w.setEnabled(is_1d)
        linear = is_1d and str(self.cmb_v1d_preset.currentData()) == "linear_crust"
        for w in (self.sp_v1d_v0, self.sp_v1d_grad, self.sp_v1d_vmax):
            w.setEnabled(linear)
            w.setReadOnly(not linear)
        iface_on = is_1d and self.chk_v1d_iface.isChecked()
        self.cmb_v1d_iface_v.setEnabled(iface_on)
        self.sp_v1d_iface_dv.setEnabled(iface_on)

    def _after_source_changed(self, *_args) -> None:
        self._sync_source_ui()
        self.project_changed.emit()
        if self._source() == "builtin_1d":
            # 一维不用用户 v.in 地形；走一维预览/缓存
            self._schedule_1d_preview()
        else:
            # 切回用户模型：主窗口优先用文件预览缓存重绘
            path = self.ed_tomo.text().strip()
            if path and os.path.isfile(path):
                self.request_preview_tomo.emit()

    def _after_v1d_param_changed(self, *_args) -> None:
        self._sync_source_ui()
        self.project_changed.emit()
        self._schedule_1d_preview()

    def _on_1d_spin_changed(self, *_args) -> None:
        self.project_changed.emit()
        self._schedule_1d_preview()

    def _on_1d_flag_changed(self, *_args) -> None:
        self._sync_source_ui()
        self.project_changed.emit()
        self._schedule_1d_preview()

    def _schedule_1d_preview(self, *_args) -> None:
        if self._source() != "builtin_1d":
            return
        self._preview_timer.start()

    def _on_tomo_path_ready(self) -> None:
        if self._source() != "file":
            return
        path = self.ed_tomo.text().strip()
        if path and os.path.isfile(path):
            self.request_preview_tomo.emit()

    def _pick_tomo(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "速度模型", "", _FILTER)
        if path:
            self.ed_tomo.setText(path)
            self.request_preview_tomo.emit()

    def _pick_bath(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "水深 bath_x.txt", "", "Text (*.txt);;All (*.*)"
        )
        if path:
            self.ed_bath.setText(path)
            self._schedule_1d_preview()

    def _on_iface_changed(self, sel: object) -> None:
        """B/S/M 变更 → 写回工程并通知主窗口同步偏移页。"""
        self.project_changed.emit()
        try:
            self.iface_sync_requested.emit(sel if isinstance(sel, dict) else {})
        except Exception:
            pass

    def apply_to_project(self, project: ObsRtmProject) -> None:
        v = project.velocity
        v.vel_source = self._source()
        v.tomo_path = self.ed_tomo.text().strip()
        project.tomo_vel = v.tomo_path
        # 若当前仍是 v.in，记住路径（转成 tomo_vel.rsf 后仍能取地形 S）
        tp = v.tomo_path
        if tp:
            abs_tp = tp if os.path.isabs(tp) else (
                project.path(tp) if project.workdir else tp
            )
            try:
                from ..services.model_import import load_zelt_model_optional

                if load_zelt_model_optional(abs_tp) is not None:
                    v.zelt_vin_path = abs_tp
            except Exception:
                pass
        v.bath_path = self.ed_bath.text().strip()
        v.flat_bath_km = float(self.sp_flat.value())
        v.vwater = float(self.sp_vw.value())
        v.fill_water = self.chk_fill.isChecked()
        v.smooth_rect = int(self.sp_smooth.value())
        v.out_vel = self.ed_out.text().strip() or "rtm_in/vel.rsf"
        v.vin_dx_km = float(self.sp_vin_dx.value())
        v.vin_dz_km = float(self.sp_vin_dz.value())
        v.auto_convert_rsf = self.chk_auto.isChecked()
        v.v1d_preset = str(self.cmb_v1d_preset.currentData() or "linear_crust")
        v.v1d_ref = str(self.cmb_v1d_ref.currentData() or "subbottom")
        v.v1d_v0 = float(self.sp_v1d_v0.value())
        v.v1d_grad = float(self.sp_v1d_grad.value())
        v.v1d_vmax = float(self.sp_v1d_vmax.value())
        v.v1d_iface_enable = bool(self.chk_v1d_iface.isChecked())
        v.v1d_iface_v = float(self.cmb_v1d_iface_v.currentData() or 6.0)
        v.v1d_iface_dv = float(self.sp_v1d_iface_dv.value())
        sel = self.canvas.interface_selection()
        v.iface_basement = sel.get("basement")
        v.iface_seafloor = sel.get("seafloor")
        v.iface_moho = sel.get("moho")

    def load_from_project(self, project: ObsRtmProject) -> None:
        v = project.velocity
        src = str(getattr(v, "vel_source", "file") or "file")
        idx = self.cmb_source.findData(src)
        if idx >= 0:
            self.cmb_source.blockSignals(True)
            self.cmb_source.setCurrentIndex(idx)
            self.cmb_source.blockSignals(False)
        path = v.tomo_path or project.tomo_vel
        self.ed_tomo.setText(path or "")
        self.ed_bath.setText(v.bath_path or "prep/geom/bath_x.txt")
        self.sp_flat.setValue(v.flat_bath_km)
        self.sp_vw.setValue(v.vwater)
        self.chk_fill.setChecked(v.fill_water)
        self.sp_smooth.setValue(v.smooth_rect)
        self.ed_out.setText(v.out_vel or "rtm_in/vel.rsf")
        self.sp_vin_dx.setValue(getattr(v, "vin_dx_km", 0.5))
        self.sp_vin_dz.setValue(getattr(v, "vin_dz_km", 0.25))
        self.chk_auto.setChecked(getattr(v, "auto_convert_rsf", True))
        for cmb, key, default in (
            (self.cmb_v1d_preset, "v1d_preset", "linear_crust"),
            (self.cmb_v1d_ref, "v1d_ref", "subbottom"),
        ):
            i = cmb.findData(str(getattr(v, key, default) or default))
            if i >= 0:
                cmb.setCurrentIndex(i)
        self.sp_v1d_v0.blockSignals(True)
        self.sp_v1d_grad.blockSignals(True)
        self.sp_v1d_vmax.blockSignals(True)
        self.sp_v1d_v0.setValue(float(getattr(v, "v1d_v0", 2.0)))
        self.sp_v1d_grad.setValue(float(getattr(v, "v1d_grad", 0.5)))
        self.sp_v1d_vmax.setValue(float(getattr(v, "v1d_vmax", 8.0)))
        self.sp_v1d_v0.blockSignals(False)
        self.sp_v1d_grad.blockSignals(False)
        self.sp_v1d_vmax.blockSignals(False)
        self.chk_v1d_iface.blockSignals(True)
        self.chk_v1d_iface.setChecked(bool(getattr(v, "v1d_iface_enable", False)))
        self.chk_v1d_iface.blockSignals(False)
        iv = float(getattr(v, "v1d_iface_v", 6.0))
        i = self.cmb_v1d_iface_v.findData(iv)
        if i < 0:
            # 兼容旧工程非标档：就近选
            best, best_d = 0, 1e9
            for j in range(self.cmb_v1d_iface_v.count()):
                d = abs(float(self.cmb_v1d_iface_v.itemData(j)) - iv)
                if d < best_d:
                    best, best_d = j, d
            i = best
        self.cmb_v1d_iface_v.blockSignals(True)
        self.cmb_v1d_iface_v.setCurrentIndex(i)
        self.cmb_v1d_iface_v.blockSignals(False)
        self.sp_v1d_iface_dv.blockSignals(True)
        self.sp_v1d_iface_dv.setValue(float(getattr(v, "v1d_iface_dv", 0.8)))
        self.sp_v1d_iface_dv.blockSignals(False)
        # B/S/M：有 zelt 时由预览 show_vel 再套一次；此处先写入画布状态
        self.canvas.apply_interface_selection(
            {
                "basement": getattr(v, "iface_basement", None),
                "seafloor": getattr(v, "iface_seafloor", None),
                "moho": getattr(v, "iface_moho", None),
            }
        )
        self._sync_source_ui()
