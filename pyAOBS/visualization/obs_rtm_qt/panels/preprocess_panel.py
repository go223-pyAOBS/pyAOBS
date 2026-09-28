# -*- coding: utf-8 -*-
"""阶段 3：预处理（带通/增益浏览）+ 道集范围（应用选道 → shots_proc/）。"""

from __future__ import annotations

import os
from typing import List, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..dialog_utils import connect_combo_deferred
from ..project import ObsRtmProject
from ..styles import compact_form as _compact_form, hint_label, primary_button
from ..widgets.gather_canvas import GatherCanvas


def _spin(
    lo: float,
    hi: float,
    val: float,
    *,
    decimals: int = 3,
    step: float = 0.1,
    suffix: str = "",
) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(val)
    if suffix:
        w.setSuffix(suffix)
    w.setMaximumWidth(110)
    return w


def _scroll_wrap(inner: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setObjectName("ObsRtmSidePanel")
    scroll.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(inner)
    return scroll


class PreprocessPanel(QWidget):
    request_preview = Signal()
    request_apply_mute = Signal()  # 应用选道 → shots_proc/（盘上仅当前选道）
    request_preview_proc = Signal()  # 直接读 shots_proc 拼图目视检查
    request_refresh_shots = Signal()
    request_montage = Signal()
    request_style_only = Signal()
    request_impulse_mode = Signal()  # 进入/切换脉冲拾取模式
    impulse_point_picked = Signal(object)  # 左键点选后的取样 dict
    project_changed = Signal()
    shot_changed = Signal()
    selection_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._shot_paths: List[str] = []
        self._current_shot_path: Optional[str] = None
        self._bag_ids: List[int] = []
        self._tabs: Optional[QTabWidget] = None
        self._gain_timer = QTimer(self)
        self._gain_timer.setSingleShot(True)
        self._gain_timer.setInterval(100)
        self._gain_timer.timeout.connect(self.request_preview.emit)
        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(180)
        self._filter_timer.timeout.connect(self.request_preview.emit)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        # 预处理在前（浏览滤波/增益）；道集范围在后（应用选道）
        tabs.addTab(self._build_process_tab(), "预处理")
        tabs.addTab(self._build_select_tab(), "道集范围")
        tabs.setMinimumWidth(360)
        tabs.setMaximumWidth(520)
        tabs.currentChanged.connect(self._on_tab_changed)
        self._tabs = tabs
        split.addWidget(tabs)

        self.canvas = GatherCanvas()
        self.canvas.setObjectName("ObsRtmPlotPanel")
        self.btn_mute_m.clicked.connect(self.canvas.toggle_mute_mode)
        self.btn_mute_inv.clicked.connect(self.canvas.toggle_invert)
        self.btn_mute_clr.clicked.connect(self.canvas.clear_mute)
        self.canvas.status.connect(self._on_mute_status)
        self.canvas.mute_changed.connect(self._on_mute_changed)
        self.canvas.trace_picked.connect(self._on_trace_picked)
        self.canvas.navigate_shot.connect(self._on_navigate_shot)
        self.canvas.impulse_point_picked.connect(self.impulse_point_picked.emit)
        self.canvas.impulse_mode_changed.connect(self._on_impulse_mode_changed)
        self.canvas.set_hand_pick_enabled(self.chk_hand.isChecked())
        # 默认在预处理页：关闭手选叠画
        self.canvas.set_overlays_enabled(False)
        split.addWidget(self.canvas)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 2)
        split.setSizes([440, 700])

    def _build_select_tab(self) -> QWidget:
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # —— 全局：作用于整个道集范围页（预览/写出）——
        prep_box = QGroupBox("全局处理")
        pform = QFormLayout(prep_box)
        _compact_form(pform)
        self.chk_apply_prep = QCheckBox("应用预处理")
        self.chk_apply_prep.setChecked(True)
        self.chk_apply_prep.setToolTip(
            "全局开关。勾选：预览/定稿 = 带通→mute→增益（各一次）；"
            "不勾选：仅 mute。不对已定稿波形重复滤波/增益。"
        )
        self.chk_apply_prep.toggled.connect(self._on_apply_prep_toggled)
        self.chk_show_proc = QCheckBox("显示 mute 效果")
        self.chk_show_proc.setChecked(True)
        self.chk_show_proc.setToolTip(
            "本页全局预览开关，不写盘。"
            "开：叠速度 mute（及多边形）到显示；启用速度 mute 时会自动勾选。"
            "若勾选「应用预处理」，显示 = mute(原始) 后再带通+增益一次。"
            "关：不显示 mute 效果。"
        )
        self.chk_show_proc.toggled.connect(lambda: self.request_preview.emit())
        self.chk_show_all = QCheckBox("全部炮拼图")
        self.chk_show_all.setChecked(True)
        self.chk_show_all.setToolTip(
            "全局拼图范围。开：拼全部炮（每炮一道）。"
            "关：只拼已追加手选炮（多道各一，仍按工区几何偏移距排布）；"
            "无手选则仅当前浏览炮。"
        )
        self.chk_show_all.toggled.connect(self._on_show_all_toggled)
        self.sp_vred = _spin(0.0, 20.0, 8.0, decimals=3, step=0.1, suffix=" km/s")
        self.sp_vred.setToolTip(
            "全局显示折合：t'=t-|x-xobs|/vred（x=model x）；0=关闭"
        )
        self.sp_vred.valueChanged.connect(self._on_style_ui)
        self.cmb_mont_src = QComboBox()
        self.cmb_mont_src.addItem("shots/ 原始", "shots")
        self.cmb_mont_src.addItem("shots_proc/ 定稿", "shots_proc")
        self.cmb_mont_src.addItem("shots_mute/ 仅mute", "shots_mute")
        self.cmb_mont_src.setToolTip(
            "拼图读盘来源。\n"
            "shots_proc/：定稿原样显示，禁止再 mute/带通/增益（仅显示用 pclip/模式/vred）。\n"
            "shots_mute/：盘上 mute 结果，可叠显示增益，勿再带通。\n"
            "各源分槽缓存；「应用选道」后刷新 proc/mute 缓存。"
        )
        connect_combo_deferred(
            self.cmb_mont_src, lambda *_: self.request_montage.emit()
        )
        row_global = QHBoxLayout()
        row_global.addWidget(self.chk_apply_prep)
        row_global.addWidget(self.chk_show_proc)
        row_global.addWidget(self.chk_show_all)
        row_global.addStretch(1)
        pform.addRow(row_global)
        row_vred = QHBoxLayout()
        row_vred.addWidget(self.sp_vred, 1)
        row_vred.addStretch(1)
        pform.addRow("折合 vred", row_vred)
        pform.addRow("拼图数据源", self.cmb_mont_src)
        lay.addWidget(prep_box)

        # —— 手选道集（分组名保留）——
        hand_box = QGroupBox("手选道集")
        hform = QFormLayout(hand_box)
        _compact_form(hform)
        self.chk_hand = QCheckBox("启用手选")
        self.chk_hand.setChecked(True)
        self.chk_hand.setToolTip(
            "开：左键/Shift 琥珀选道；右键仅追加已选（不按落点）；中键取消当前。"
            "仅用于浏览/拼图范围；未开多边形时手选也驱动「应用选道」。"
            "未启用多边形 mute 时，手选可单独限定 RTM 炮集；启用多边形后 RTM 改由多边形圈定。"
        )
        self.chk_hand.toggled.connect(self._on_hand_toggled)
        btn_ref = QPushButton("↻ 刷新炮")
        btn_ref.setToolTip("重新扫描 shots/")
        btn_ref.clicked.connect(self.request_refresh_shots.emit)
        self.btn_clear_sel = QPushButton("一键清空")
        self.btn_clear_sel.setToolTip("清空琥珀临时选与已追加红高亮")
        self.btn_clear_sel.clicked.connect(self._bag_clear)
        row_hand = QHBoxLayout()
        row_hand.addWidget(self.chk_hand)
        row_hand.addWidget(btn_ref)
        row_hand.addWidget(self.btn_clear_sel)
        row_hand.addStretch(1)
        hform.addRow(row_hand)

        self.cmb_disp = QComboBox()
        self.cmb_disp.addItem("Density（多炮推荐）", "density")
        self.cmb_disp.addItem("Wiggle（单炮推荐）", "wiggle")
        self.cmb_disp.addItem("正填", "fill+")
        self.cmb_disp.addItem("负填", "fill-")
        self.cmb_disp.setToolTip("显示模式；画布上按 I 查看道信息")
        connect_combo_deferred(self.cmb_disp, lambda *_: self.request_style_only.emit())
        self.sp_pclip = _spin(50.0, 100.0, 98.0, decimals=1, step=0.5)
        self.sp_pclip.setToolTip("density 色标百分位")
        self.sp_pclip.valueChanged.connect(lambda: self.request_style_only.emit())
        row_d = QHBoxLayout()
        row_d.addWidget(self.cmb_disp, 1)
        row_d.addWidget(QLabel("pclip"))
        row_d.addWidget(self.sp_pclip)
        hform.addRow("显示", row_d)

        row_mt = QHBoxLayout()
        self.sp_stride = QSpinBox()
        self.sp_stride.setRange(1, 500)
        self.sp_stride.setValue(1)
        self.sp_stride.setMaximumWidth(56)
        self.sp_stride.setToolTip("拼图炮抽样步长")
        self.sp_maxtr = QSpinBox()
        self.sp_maxtr.setRange(10, 5000)
        self.sp_maxtr.setValue(2000)
        self.sp_maxtr.setMaximumWidth(64)
        self.sp_maxtr.setToolTip("拼图最大道数")
        btn_mt = QPushButton("刷新拼图")
        btn_mt.setToolTip("强制重载拼图缓存")
        btn_mt.clicked.connect(self.request_montage.emit)
        row_mt.addWidget(QLabel("st"))
        row_mt.addWidget(self.sp_stride)
        row_mt.addWidget(QLabel("max"))
        row_mt.addWidget(self.sp_maxtr)
        row_mt.addWidget(btn_mt)
        hform.addRow(row_mt)

        self.lbl_bag = hint_label(
            "已追加 0 炮 · 左键选→右键追加 · 中键取消 · 一键清空"
        )
        hform.addRow(self.lbl_bag)
        lay.addWidget(hand_box)

        # —— 速度 mute（默认关；作用在原始道集，与全局折合无关）——
        vbox = QGroupBox("速度 mute")
        vform = QFormLayout(vbox)
        _compact_form(vform)
        self.chk_mute = QCheckBox("启用")
        self.chk_mute.setChecked(False)
        self.chk_mute.setToolTip(
            "默认关。对原始道集做速度 mute（与手选无关），并打开「显示 mute 效果」。\n"
            "线：t = tm + |offset|/vm（左右分支）。\n"
            "默认切深；勾选「反选」则切浅。tp 为边缘余弦过渡（对齐 mutter）。\n"
            "绘制时用全局「折合 vred」，不参与 mute 计算。"
        )
        self.chk_mute.toggled.connect(self._on_vel_mute_toggled)
        self.chk_vel_mute_inv = QCheckBox("反选（切浅）")
        self.chk_vel_mute_inv.setChecked(False)
        self.chk_vel_mute_inv.setToolTip("关=切深（保留初至段）；开=切浅（保留线以下）")
        self.chk_vel_mute_inv.toggled.connect(self._on_mute_ui)
        row_en = QHBoxLayout()
        row_en.addWidget(self.chk_mute)
        row_en.addWidget(self.chk_vel_mute_inv)
        row_en.addStretch(1)
        vform.addRow(row_en)
        self.sp_tmute = _spin(-10.0, 100.0, 0.0, decimals=3, step=0.05, suffix=" s")
        self.sp_vmute = _spin(0.1, 20.0, 6.0, decimals=2, step=0.1, suffix=" km/s")
        self.sp_mute_tp = _spin(0.0, 2.0, 0.15, decimals=3, step=0.01, suffix=" s")
        self.sp_tmute.setToolTip("零偏时刻 tm（s）")
        self.sp_vmute.setToolTip("mute 速度 vm（km/s），左右分支斜率 1/vm")
        self.sp_mute_tp.setToolTip(
            "边缘余弦过渡带宽 tp（s），速度 mute 与多边形 mute 共用。\n"
            "对齐 Madagascar mutter tp=0.15。0=硬切（假震相更重）。\n"
            "定稿顺序为 带通→mute→增益，避免 mute 后再滤波振铃。"
        )
        for w in (self.sp_tmute, self.sp_vmute, self.sp_mute_tp):
            w.valueChanged.connect(self._on_mute_ui)
        row_vm = QHBoxLayout()
        row_vm.addWidget(QLabel("tm"))
        row_vm.addWidget(self.sp_tmute)
        row_vm.addWidget(QLabel("vm"))
        row_vm.addWidget(self.sp_vmute)
        row_vm.addWidget(QLabel("tp"))
        row_vm.addWidget(self.sp_mute_tp)
        vform.addRow(row_vm)
        lay.addWidget(vbox)

        # —— 多边形 mute（默认关；不自动改动手选勾选）——
        pbox = QGroupBox("多边形 mute")
        pform = QFormLayout(pbox)
        _compact_form(pform)
        self.chk_poly = QCheckBox("启用多边形 mute")
        self.chk_poly.setChecked(False)
        self.chk_poly.setToolTip(
            "默认关。闭合后按多边形圈选炮（offset 落入包围盒）并 mute 波形；"
            "与手选完全独立。定稿/RTM 走多边形路线。"
        )
        self.chk_poly.toggled.connect(self._on_poly_toggled)
        self.cmb_poly_x = QComboBox()
        self.cmb_poly_x.addItem("model x", "offset")  # 与拼图横轴一致（模型测线坐标）
        self.cmb_poly_x.addItem("trace", "trace")
        self.cmb_poly_x.setMaximumWidth(90)
        self.cmb_poly_x.setToolTip("多边形横轴：model x = OBS x + 相对 offset")
        connect_combo_deferred(self.cmb_poly_x, lambda *_: self.project_changed.emit())
        row_px = QHBoxLayout()
        row_px.addWidget(self.chk_poly)
        row_px.addWidget(QLabel("x"))
        row_px.addWidget(self.cmb_poly_x)
        pform.addRow(row_px)

        row_btn = QHBoxLayout()
        self.btn_mute_m = QPushButton("M")
        self.btn_mute_m.setFixedWidth(36)
        self.btn_mute_m.setToolTip("绘制多边形（滚轮缩放；右键闭合）")
        self.btn_mute_inv = QPushButton("反选")
        self.btn_mute_inv.setToolTip("Shift+M：保留多边形外部")
        self.btn_mute_clr = QPushButton("清空顶点")
        self.btn_mute_clr.setToolTip("清除多边形顶点（不影响手选炮袋）")
        row_btn.addWidget(self.btn_mute_m)
        row_btn.addWidget(self.btn_mute_inv)
        row_btn.addWidget(self.btn_mute_clr)
        pform.addRow(row_btn)
        self.lbl_mute = hint_label("Mute: OFF")
        pform.addRow(self.lbl_mute)
        lay.addWidget(pbox)

        self.btn_apply_mute = primary_button("应用选道 → shots_proc/")
        self.btn_apply_mute.setToolTip(
            "按当前选道（多边形 > 手选 > 全炮）写出："
            "带通→mute→增益 → shots_proc/；mute(raw)→shots_mute/。"
            "清理非本次选道旧文件。预览定稿只看 shots_proc 原样。"
        )
        self.btn_apply_mute.clicked.connect(self.request_apply_mute.emit)
        self.btn_preview_proc = primary_button("预览定稿 shots_proc/")
        self.btn_preview_proc.setToolTip(
            "预览定稿：数据源=shots_proc/，原样显示盘上波形，不做 mute/带通/增益。"
            "视窗=全炮 offset × 全时程；仅显示用 pclip/模式/折合 vred。"
        )
        self.btn_preview_proc.clicked.connect(self._on_preview_proc_clicked)
        self.btn_impulse = QPushButton("脉冲成像")
        self.btn_impulse.setCheckable(True)
        self.btn_impulse.setToolTip(
            "进入脉冲拾取：左键在道集上点一点（显示标记）→ 确认后跑单脉冲 RTM。\n"
            "Shift+P 同上；Esc / 再点按钮取消。结果在 rtm_work/impulse/。"
        )
        self.btn_impulse.clicked.connect(self._on_impulse_btn)
        row_mute_act = QHBoxLayout()
        row_mute_act.addWidget(self.btn_apply_mute, 1)
        row_mute_act.addWidget(self.btn_preview_proc)
        row_mute_act.addWidget(self.btn_impulse)
        lay.addLayout(row_mute_act)
        lay.addWidget(
            hint_label(
                "预览定稿 = shots_proc 原样（无二次处理）。"
                "「应用选道」写出后盘上仅保留当前选道。"
            )
        )
        lay.addStretch(1)
        return _scroll_wrap(inner)

    def _build_process_tab(self) -> QWidget:
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        box = QGroupBox("滤波")
        form = QFormLayout(box)
        _compact_form(form)
        self.chk_auto_filter = QCheckBox("自动预览")
        self.chk_auto_filter.setChecked(True)
        self.chk_auto_filter.setToolTip(
            "本页浏览：对原始道集套带通+增益（不写盘、不 mute）"
        )
        self.chk_auto_filter.toggled.connect(lambda: self.request_preview.emit())

        self.chk_bp = QCheckBox("带通")
        self.chk_bp.setChecked(True)
        self.chk_bp.setToolTip("预处理参数；手选勾选「应用预处理」时用于定稿写出")
        self.chk_bp.toggled.connect(self._on_filter_ui)
        row_f = QHBoxLayout()
        self.sp_flo = _spin(0.1, 200.0, 3.0, decimals=1, step=0.5, suffix=" Hz")
        self.sp_fhi = _spin(0.1, 200.0, 15.0, decimals=1, step=0.5, suffix=" Hz")
        self.sp_flo.setToolTip("低频截止 fL（Hz）")
        self.sp_fhi.setToolTip("高频截止 fH（Hz）")
        self.sp_flo.valueChanged.connect(self._on_filter_ui)
        self.sp_fhi.valueChanged.connect(self._on_filter_ui)
        row_f.addWidget(self.chk_auto_filter)
        row_f.addWidget(self.chk_bp)
        row_f.addWidget(QLabel("fL"))
        row_f.addWidget(self.sp_flo)
        row_f.addWidget(QLabel("fH"))
        row_f.addWidget(self.sp_fhi)
        form.addRow(row_f)
        lay.addWidget(box)

        gbox = QGroupBox("增益")
        ggrid = QGridLayout(gbox)
        ggrid.setContentsMargins(4, 4, 4, 4)
        ggrid.setHorizontalSpacing(4)
        ggrid.setVerticalSpacing(2)
        self.chk_gain = QCheckBox("开")
        self.chk_gain.setChecked(True)
        self.chk_gain.setToolTip(
            "增益参数。在 shots/ 上预览时可实时试效果；"
            "shots_mute/ 可叠显示增益；shots_proc/ 定稿预览绝不叠增益（看盘上原样）。"
            "「应用选道」时按当前参数写入 shots_proc/。"
        )
        self.chk_gain.toggled.connect(self._on_gain_ui)
        ggrid.addWidget(self.chk_gain, 0, 0)

        self.sp_amp = _spin(0.01, 1000.0, 1.2, decimals=3, step=0.05)
        self.cmb_iscale = QComboBox()
        self.cmb_iscale.addItem("0自动", 0)
        self.cmb_iscale.addItem("1固定", 1)
        self.cmb_iscale.addItem("2变增", 2)
        self.cmb_iscale.setMaximumWidth(72)
        connect_combo_deferred(self.cmb_iscale, lambda *_: self._on_gain_ui())
        ggrid.addWidget(QLabel("amp"), 0, 1)
        ggrid.addWidget(self.sp_amp, 0, 2)
        ggrid.addWidget(QLabel("isc"), 0, 3)
        ggrid.addWidget(self.cmb_iscale, 0, 4)

        self.sp_rcor = _spin(-5.0, 5.0, 0.3, decimals=3, step=0.05)
        self.sp_sf = _spin(0.0, 100.0, 0.0, decimals=4, step=0.001)
        self.sp_tvg = _spin(0.0, 20.0, 1.0, decimals=3, step=0.05)
        self.sp_pvg = _spin(-4.0, 4.0, 1.0, decimals=3, step=0.05)
        self.sp_clip = _spin(0.0, 20.0, 0.0, decimals=3, step=0.1)
        self.sp_dscale = _spin(0.2, 5.0, 1.0, decimals=3, step=0.05)
        ggrid.addWidget(QLabel("rcor"), 1, 0)
        ggrid.addWidget(self.sp_rcor, 1, 1)
        ggrid.addWidget(QLabel("sf"), 1, 2)
        ggrid.addWidget(self.sp_sf, 1, 3, 1, 2)
        ggrid.addWidget(QLabel("tvg"), 2, 0)
        ggrid.addWidget(self.sp_tvg, 2, 1)
        ggrid.addWidget(QLabel("pvg"), 2, 2)
        ggrid.addWidget(self.sp_pvg, 2, 3, 1, 2)
        ggrid.addWidget(QLabel("clip"), 3, 0)
        ggrid.addWidget(self.sp_clip, 3, 1)
        ggrid.addWidget(QLabel("dsc"), 3, 2)
        ggrid.addWidget(self.sp_dscale, 3, 3, 1, 2)

        for w in (
            self.sp_amp,
            self.sp_rcor,
            self.sp_sf,
            self.sp_tvg,
            self.sp_pvg,
            self.sp_clip,
        ):
            w.valueChanged.connect(self._on_gain_ui)
        self.sp_dscale.valueChanged.connect(self._on_style_ui)

        prow = QHBoxLayout()
        for text, slot, tip in (
            ("平衡", self._preset_balanced, "iscale=0 自动平衡"),
            ("远偏", self._preset_far, "iscale=1 + 远偏增强"),
            ("强", self._preset_strong, "iscale=1 强振幅"),
        ):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setMaximumHeight(24)
            b.clicked.connect(slot)
            prow.addWidget(b)
        ggrid.addLayout(prow, 4, 0, 1, 5)
        lay.addWidget(gbox)

        lay.addWidget(
            hint_label(
                "本页只浏览带通+增益是否合适（不写盘、不 mute）。"
                "定稿在「道集范围」点「应用选道」。"
            )
        )
        btn_prev = QPushButton("预览")
        btn_prev.setToolTip("预览原始道集 + 带通 + 增益（不写盘）")
        btn_prev.clicked.connect(self.request_preview.emit)
        lay.addWidget(btn_prev)
        lay.addStretch(1)
        return _scroll_wrap(inner)

    def _on_tab_changed(self, idx: int) -> None:
        """0=预处理（无手选叠画）；1=手选。"""
        if idx <= 0:
            self.canvas.set_overlays_enabled(False)
            if self.chk_auto_filter.isChecked():
                self.request_preview.emit()
        else:
            self.canvas.set_overlays_enabled(True)
            self.request_preview.emit()

    def preview_chain_mode(self) -> str:
        """filter=预处理页浏览；hand=手选页。"""
        if self._tabs is not None and self._tabs.currentIndex() <= 0:
            return "filter"
        return "hand"

    def use_apply_preprocess(self) -> bool:
        return bool(self.chk_apply_prep.isChecked())

    def _on_apply_prep_toggled(self, *_args) -> None:
        self.project_changed.emit()
        if self.preview_chain_mode() == "hand":
            self.request_preview.emit()

    # ----- bag (手选炮集) -----
    def selected_shot_ids(self) -> List[int]:
        return list(self._bag_ids)

    def proc_shot_list_text(self) -> str:
        return ",".join(str(i) for i in self._bag_ids)

    def poly_mute_active(self) -> bool:
        """多边形已闭合启用（与手选无关）。"""
        return bool(
            self.chk_poly.isChecked()
            and self.canvas.mute_enabled()
            and len(self.canvas.mute_points()) >= 3
        )

    def poly_mute_shot_ids_for(self, project: ObsRtmProject) -> List[int]:
        """多边形圈定的炮号（独立路线；手选不参与）。"""
        if not self.poly_mute_active():
            return []
        from ..services.polygon_mute import shot_ids_from_polygon

        return shot_ids_from_polygon(
            project,
            self.canvas.mute_points(),
            invert=bool(self.canvas.mute_invert()),
        )

    def mute_scope_shot_ids(self, project: ObsRtmProject) -> Optional[List[int]]:
        """
        mute 炮范围（仅多边形，不含手选；预览定稿过滤可用）：
          多边形启用 → 多边形圈定炮；
          否则 None=全炮。
        「应用选道」请用 rtm_scope_shot_ids（多边形 > 手选 > 全炮）。
        """
        if not self.poly_mute_active():
            return None
        return self.poly_mute_shot_ids_for(project)

    def rtm_scope_shot_ids(self, project: ObsRtmProject) -> Optional[List[int]]:
        """
        RTM 炮集：
          多边形 mute 启用 → 多边形圈定（优先，与手选无关）；
          否则手选袋非空 → 手选（不要求「启用手选」勾选，便于重开工程恢复）；
          否则 None=全炮。
        """
        if self.poly_mute_active():
            return self.poly_mute_shot_ids_for(project)
        ids = self.selected_shot_ids()
        if ids:
            return ids
        return None

    def montage_shot_filter(self) -> Optional[List[int]]:
        """
        拼图炮范围（仅显示，与 mute 定稿无关）：
          None = 全炮；
          list = 手选/当前炮。
        多边形 mute 不改变拼图范围（始终可在全炮上画门）。
        """
        if self.chk_show_all.isChecked():
            return None
        if self.chk_hand.isChecked():
            ids = self.selected_shot_ids()
            if ids:
                return ids
        idx = self._shot_index_from_path(self.current_shot_path())
        if idx is not None:
            return [int(idx)]
        return None

    def hand_select_enabled(self) -> bool:
        return bool(self.chk_hand.isChecked())

    def _set_bag_ids(self, ids: List[int], *, emit: bool = True) -> None:
        seen = set()
        out: List[int] = []
        for i in ids:
            v = int(i)
            if v in seen:
                continue
            seen.add(v)
            out.append(v)
        self._bag_ids = out
        self._refresh_bag_label()
        if emit:
            self.project_changed.emit()
            self.selection_changed.emit()

    def _refresh_bag_label(self) -> None:
        ids = self.selected_shot_ids()
        if not self.chk_hand.isChecked():
            self.lbl_bag.setText("手选已关 · 仅浏览；RTM 见多边形或全炮")
            return
        if not ids:
            self.lbl_bag.setText("已选 0 炮 · 未开多边形时用于选道写出")
            return
        shown = ",".join(str(i) for i in ids[:12])
        if len(ids) > 12:
            shown += "…"
        self.lbl_bag.setText("已选 %d 炮: %s · 未开多边形时用于选道写出" % (len(ids), shown))

    def _on_hand_toggled(self, checked: bool) -> None:
        self.canvas.set_hand_pick_enabled(bool(checked))
        if not checked:
            self._set_bag_ids([])
            self.lbl_bag.setText("手选已关 · 已清空（不影响多边形 mute）")
        else:
            self._refresh_bag_label()
            self.selection_changed.emit()
        self.project_changed.emit()
        # 仅影响拼图显示范围；选道写出范围见 rtm_scope_shot_ids
        if self.preview_chain_mode() == "hand":
            self.request_preview.emit()

    def _on_impulse_btn(self, checked: bool = False) -> None:
        """按钮：进入/退出脉冲拾取模式。"""
        if self.canvas.impulse_mode():
            self.canvas.exit_impulse_mode(clear_marker=True)
        else:
            self.canvas.enter_impulse_mode()
        # 状态由 impulse_mode_changed 回写 checked
        self.btn_impulse.blockSignals(True)
        self.btn_impulse.setChecked(self.canvas.impulse_mode())
        self.btn_impulse.blockSignals(False)
        self.request_impulse_mode.emit()

    def _on_impulse_mode_changed(self, on: bool) -> None:
        self.btn_impulse.blockSignals(True)
        self.btn_impulse.setChecked(bool(on))
        self.btn_impulse.setText("取消脉冲拾取" if on else "脉冲成像")
        self.btn_impulse.blockSignals(False)

    def _bag_append_many(self, add: List[int]) -> None:
        if not self.chk_hand.isChecked():
            self.lbl_bag.setText("手选已关：请先勾选「启用手选」")
            return
        ids = self.selected_shot_ids()
        n0 = len(ids)
        for i in add:
            if int(i) not in ids:
                ids.append(int(i))
        self._set_bag_ids(ids)
        self.canvas.clear_pending_shot_ids()
        self.lbl_bag.setText(
            "追加 %d 炮（共 %d，红波形保留）" % (len(ids) - n0, len(ids))
        )

    def _bag_clear(self) -> None:
        self._set_bag_ids([])
        self.canvas.clear_pending_shot_ids()
        self.canvas.set_highlight_idx(None)
        self.lbl_bag.setText("已一键清空（临时选 + 已追加）")

    # ----- helpers / slots -----
    def _on_show_all_toggled(self, checked: bool) -> None:
        # 只切换拼图范围，不改 Density/Wiggle，避免视窗被显示模式打乱
        self.request_preview.emit()

    def _set_display_mode(self, mode: str) -> None:
        self.cmb_disp.blockSignals(True)
        for i in range(self.cmb_disp.count()):
            if self.cmb_disp.itemData(i) == mode:
                self.cmb_disp.setCurrentIndex(i)
                break
        self.cmb_disp.blockSignals(False)

    def _on_vel_mute_toggled(self, checked: bool) -> None:
        """启用速度 mute 时打开「显示 mute 效果」。"""
        if checked and not self.chk_show_proc.isChecked():
            self.chk_show_proc.blockSignals(True)
            self.chk_show_proc.setChecked(True)
            self.chk_show_proc.blockSignals(False)
        self._on_mute_ui()

    def _on_mute_ui(self, *_args) -> None:
        """速度 mute 参数改动 → 刷新道集预览（含 mute 层）。"""
        self.canvas.set_mute_tp(float(self.sp_mute_tp.value()))
        self.project_changed.emit()
        # 预处理页也可预览速度 mute（与「显示 mute 效果」联动）
        self._filter_timer.start()
        # 多边形叠层也用同一 tp：改 tp 时重绘当前图
        if self.canvas.mute_enabled() and self.canvas._data is not None:
            self.canvas._redraw_image(apply_mute=True)

    def _on_gain_ui(self, *_args) -> None:
        """增益变更：刷新预览。shots_proc 定稿原样显示，改增益不触发重算。"""
        self.project_changed.emit()
        if self.montage_source() == "shots_proc":
            return
        mode = self.preview_chain_mode()
        if (
            mode == "filter"
            or self.use_apply_preprocess()
            or self.chk_show_proc.isChecked()
            or self.montage_source() == "shots_mute"
        ):
            self._gain_timer.start()

    def _on_filter_ui(self, *_args) -> None:
        """预处理页，或手选且勾选「应用预处理」时刷新。"""
        self.project_changed.emit()
        mode = self.preview_chain_mode()
        if mode == "filter" or (mode == "hand" and self.use_apply_preprocess()):
            self._filter_timer.start()

    def _on_style_ui(self, *_args) -> None:
        self.project_changed.emit()
        self.request_style_only.emit()

    def _block_gain(self, block: bool) -> None:
        for w in (
            self.chk_gain,
            self.sp_amp,
            self.cmb_iscale,
            self.sp_rcor,
            self.sp_sf,
            self.sp_tvg,
            self.sp_pvg,
            self.sp_clip,
            self.sp_dscale,
        ):
            w.blockSignals(block)

    def _preset_balanced(self) -> None:
        self._block_gain(True)
        self.chk_gain.setChecked(True)
        self.cmb_iscale.setCurrentIndex(0)
        self.sp_rcor.setValue(0.3)
        self.sp_amp.setValue(1.2)
        self.sp_tvg.setValue(1.0)
        self.sp_pvg.setValue(1.0)
        self.sp_clip.setValue(0.0)
        self.sp_dscale.setValue(1.0)
        self._block_gain(False)
        self._on_gain_ui()

    def _preset_far(self) -> None:
        self._block_gain(True)
        self.chk_gain.setChecked(True)
        self.cmb_iscale.setCurrentIndex(1)
        self.sp_rcor.setValue(0.8)
        self.sp_amp.setValue(1.6)
        self.sp_tvg.setValue(0.8)
        self.sp_pvg.setValue(1.2)
        self.sp_clip.setValue(2.5)
        self.sp_dscale.setValue(1.25)
        self._block_gain(False)
        self._on_gain_ui()

    def _preset_strong(self) -> None:
        self._block_gain(True)
        self.chk_gain.setChecked(True)
        self.cmb_iscale.setCurrentIndex(1)
        self.sp_rcor.setValue(1.2)
        self.sp_amp.setValue(2.2)
        self.sp_tvg.setValue(0.6)
        self.sp_pvg.setValue(1.5)
        self.sp_clip.setValue(2.5)
        self.sp_dscale.setValue(1.6)
        self._block_gain(False)
        self._on_gain_ui()

    def _on_mute_status(self, msg: str) -> None:
        self.lbl_mute.setText(msg)

    def _on_poly_toggled(self, checked: bool) -> None:
        """勾选↔画布 mute 同步；取消：恢复剖面且不画顶点；再勾选：重绘顶点+mute。"""
        if checked:
            if len(self.canvas.mute_points()) >= 3:
                self.canvas.set_mute_enabled(True)
            else:
                self.lbl_mute.setText("Mute: 请先画多边形并右键闭合")
        else:
            if self.canvas.mute_enabled() or self.canvas.mute_points():
                self.canvas.set_mute_enabled(False)
            self.lbl_mute.setText(
                "Mute: OFF · %d 点已隐藏（勾选再绘制）"
                % len(self.canvas.mute_points())
            )
        self.project_changed.emit()
        self.selection_changed.emit()
        self.request_preview.emit()

    def _on_mute_changed(self) -> None:
        c = self.canvas
        n = len(c.mute_points())
        editing = bool(getattr(c, "_mute_edit", False))
        # 绘制过程：只改标签，禁止工程同步 / 扫炮 / 整页预览
        if editing:
            self.lbl_mute.setText("Mute: DRAW(%d) · 右键闭合后才 mute" % n)
            return
        # 闭合/M 启用 → 勾选；M 关闭 → 取消勾选（与 checkbox 互相同步）
        if c.mute_enabled() and not self.chk_poly.isChecked():
            self.chk_poly.blockSignals(True)
            self.chk_poly.setChecked(True)
            self.chk_poly.blockSignals(False)
        elif (not c.mute_enabled()) and self.chk_poly.isChecked() and n >= 3:
            self.chk_poly.blockSignals(True)
            self.chk_poly.setChecked(False)
            self.chk_poly.blockSignals(False)
        if c.mute_enabled():
            txt = "Mute: ON%s · %d 顶点 · 独立圈选炮→RTM" % (
                "(反)" if c.mute_invert() else "",
                n,
            )
        else:
            txt = "Mute: OFF · %d" % n
        self.lbl_mute.setText(txt)
        c._refresh_highlight()
        self.project_changed.emit()
        # 闭合/开关后同步 RTM（主窗口内已防抖）；不再整页重拼图
        self.selection_changed.emit()

    def set_shot_list(self, paths: List[str]) -> None:
        self._shot_paths = list(paths)
        cur = self._current_shot_path
        if cur and cur in self._shot_paths:
            self._current_shot_path = cur
        elif cur and self._shot_paths:
            # 尽量保留同名；找不到则清空，勿默认第 0 道（否则未选也会画琥珀高亮）
            base = os.path.basename(cur)
            hit = next(
                (p for p in self._shot_paths if os.path.basename(p) == base),
                None,
            )
            self._current_shot_path = hit
        else:
            self._current_shot_path = None

    def current_shot_path(self) -> Optional[str]:
        return self._current_shot_path

    @staticmethod
    def _shot_index_from_path(path: Optional[str]) -> Optional[int]:
        if not path:
            return None
        import re

        m = re.match(r"shot_(\d+)\.rsf$", os.path.basename(path), re.I)
        return int(m.group(1)) if m else None

    def current_shot_index(self) -> Optional[int]:
        idx = self.canvas.shot_index_at_trace()
        if idx is not None:
            return idx
        return self._shot_index_from_path(self.current_shot_path())

    def _set_current_shot_path(self, path: str, *, emit_shot: bool = True) -> bool:
        if not path:
            return False
        if path not in self._shot_paths:
            base = os.path.basename(path)
            for p in self._shot_paths:
                if os.path.basename(p) == base:
                    path = p
                    break
            else:
                return False
        if self._current_shot_path == path:
            return True
        self._current_shot_path = path
        if emit_shot:
            self.shot_changed.emit()
        return True

    def _on_trace_picked(self, j: int, mode: str = "browse") -> None:
        path = self.canvas.shot_path_at_trace(int(j))
        if path:
            self._set_current_shot_path(path, emit_shot=True)
        idx = self.canvas.shot_index_at_trace(int(j))
        if idx is None:
            self.lbl_bag.setText("已点道 %d（无法解析炮号）" % int(j))
            return
        idx = int(idx)
        mode = str(mode or "browse")
        if mode == "browse":
            # 左键：琥珀临时单选，不清空已追加红波形
            self.canvas.set_pending_shot_ids([idx])
            self.lbl_bag.setText(
                "浏览炮 %d（琥珀）· 已追加 %d · 右键追加"
                % (idx, len(self.selected_shot_ids()))
            )
            return
        if mode == "pending_add":
            self.canvas.add_pending_shot_id(idx)
            n = len(self.canvas.pending_shot_ids())
            self.lbl_bag.setText(
                "临时多选 %d 炮（琥珀）· 右键追加 · 已追加 %d"
                % (n, len(self.selected_shot_ids()))
            )
            return
        if mode == "append":
            # 只追加左键/Shift 临时选，忽略右键落点道
            add = list(self.canvas.pending_shot_ids())
            if not add:
                self.lbl_bag.setText("请先左键/Shift 选道，再右键追加")
                return
            self._bag_append_many(add)
            return
        if mode == "cancel":
            # 清临时选；若该炮已追加则移出（已追加的其它道保留）
            pending = self.canvas.pending_shot_ids()
            self.canvas.clear_pending_shot_ids()
            self.canvas.set_highlight_idx(None)
            ids = self.selected_shot_ids()
            drop = set(pending or [])
            drop.add(idx)
            keep = [i for i in ids if i not in drop]
            if keep != ids:
                self._set_bag_ids(keep)
            self.lbl_bag.setText(
                "已取消选择 %s · 仍追加 %d 炮"
                % (
                    ",".join(str(i) for i in sorted(drop)[:8])
                    + ("…" if len(drop) > 8 else ""),
                    len(keep),
                )
            )

    def _on_navigate_shot(self, delta: int, shift_multi: bool) -> None:
        """←/→ 浏览；Shift+←/→ 加入临时多选；均不追加，右键才追加。"""
        n = len(self._shot_paths)
        if n <= 0:
            return
        try:
            i = self._shot_paths.index(self._current_shot_path)  # type: ignore[arg-type]
        except ValueError:
            i = 0
        i = max(0, min(n - 1, i + int(delta)))
        self._set_current_shot_path(self._shot_paths[i], emit_shot=True)
        idx = self._shot_index_from_path(self._current_shot_path)
        if idx is None:
            return
        if shift_multi:
            self.canvas.add_pending_shot_id(int(idx))
            self.lbl_bag.setText(
                "临时多选 %d 炮（琥珀）· 右键追加"
                % len(self.canvas.pending_shot_ids())
            )
        else:
            self.canvas.set_pending_shot_ids([int(idx)])
            self.lbl_bag.setText(
                "浏览炮 %d（琥珀）· 已追加 %d"
                % (int(idx), len(self.selected_shot_ids()))
            )

    def pclip(self) -> float:
        return float(self.sp_pclip.value())

    def dscale(self) -> float:
        return float(self.sp_dscale.value())

    def display_vred(self) -> float:
        return float(self.sp_vred.value())

    def display_mode(self) -> str:
        return str(self.cmb_disp.currentData() or "density")

    def preview_processed(self) -> bool:
        """当前子页是否开启自动处理预览。"""
        if self.preview_chain_mode() == "filter":
            return bool(self.chk_auto_filter.isChecked())
        return bool(self.chk_show_proc.isChecked())

    def preview_all_shots(self) -> bool:
        return self.chk_show_all.isChecked()

    def poly_x_mode(self) -> str:
        return str(self.cmb_poly_x.currentData())

    def montage_source(self) -> str:
        """拼图读盘：shots | shots_proc | shots_mute。"""
        return str(self.cmb_mont_src.currentData() or "shots")

    def set_montage_source(self, source: str) -> None:
        src = str(source or "shots")
        for i in range(self.cmb_mont_src.count()):
            if self.cmb_mont_src.itemData(i) == src:
                self.cmb_mont_src.blockSignals(True)
                self.cmb_mont_src.setCurrentIndex(i)
                self.cmb_mont_src.blockSignals(False)
                return

    def _on_preview_proc_clicked(self) -> None:
        self.set_montage_source("shots_proc")
        self.request_preview_proc.emit()

    def montage_stride(self) -> int:
        return int(self.sp_stride.value())

    def montage_max(self) -> int:
        return int(self.sp_maxtr.value())

    def sync_mute_to_project(self, project: ObsRtmProject) -> None:
        p = project.preprocess
        pts = self.canvas.mute_points()
        p.poly_points = [[a, b] for a, b in pts]
        # 仅闭合启用后才写入 use_poly_mute，避免画顶点过程中就开始 mute
        p.use_poly_mute = (
            self.chk_poly.isChecked()
            and self.canvas.mute_enabled()
            and len(pts) >= 3
        )
        p.poly_invert = self.canvas.mute_invert()
        p.poly_x_mode = self.poly_x_mode()

    def sync_mute_from_project(self, project: ObsRtmProject) -> None:
        p = project.preprocess
        self.chk_poly.setChecked(bool(p.use_poly_mute))
        for i in range(self.cmb_poly_x.count()):
            if self.cmb_poly_x.itemData(i) == p.poly_x_mode:
                self.cmb_poly_x.setCurrentIndex(i)
                break
        pts = [(float(a[0]), float(a[1])) for a in (p.poly_points or []) if len(a) >= 2]
        self.canvas.set_mute_points(
            pts, enabled=bool(p.use_poly_mute) and len(pts) >= 3, invert=bool(p.poly_invert)
        )

    def sync_gain_to_project(self, project: ObsRtmProject) -> None:
        p = project.preprocess
        p.use_gain = self.chk_gain.isChecked()
        p.iscale = int(self.cmb_iscale.currentData())
        p.amp = float(self.sp_amp.value())
        p.rcor = float(self.sp_rcor.value())
        p.sf = float(self.sp_sf.value())
        p.tvg = float(self.sp_tvg.value())
        p.pvg = float(self.sp_pvg.value())
        p.clip = float(self.sp_clip.value())
        p.dscale = float(self.sp_dscale.value())
        p.display_mode = self.display_mode()
        p.pclip = float(self.sp_pclip.value())
        p.display_vred = float(self.sp_vred.value())

    def sync_gain_from_project(self, project: ObsRtmProject) -> None:
        p = project.preprocess
        self._block_gain(True)
        self.chk_gain.setChecked(bool(getattr(p, "use_gain", True)))
        isc = int(getattr(p, "iscale", 0))
        for i in range(self.cmb_iscale.count()):
            if int(self.cmb_iscale.itemData(i)) == isc:
                self.cmb_iscale.setCurrentIndex(i)
                break
        self.sp_amp.setValue(float(getattr(p, "amp", 1.2)))
        self.sp_rcor.setValue(float(getattr(p, "rcor", 0.3)))
        self.sp_sf.setValue(float(getattr(p, "sf", 0.0)))
        self.sp_tvg.setValue(float(getattr(p, "tvg", 1.0)))
        self.sp_pvg.setValue(float(getattr(p, "pvg", 1.0)))
        self.sp_clip.setValue(float(getattr(p, "clip", 0.0)))
        self.sp_dscale.setValue(float(getattr(p, "dscale", 1.0)))
        self._block_gain(False)
        mode = str(getattr(p, "display_mode", "density"))
        self.cmb_disp.blockSignals(True)
        for i in range(self.cmb_disp.count()):
            if self.cmb_disp.itemData(i) == mode:
                self.cmb_disp.setCurrentIndex(i)
                break
        self.cmb_disp.blockSignals(False)
        self.sp_pclip.blockSignals(True)
        self.sp_pclip.setValue(float(getattr(p, "pclip", 98.0)))
        self.sp_pclip.blockSignals(False)
        self.sp_vred.blockSignals(True)
        self.sp_vred.setValue(float(getattr(p, "display_vred", 8.0)))
        self.sp_vred.blockSignals(False)

    def apply_to_project(self, project: ObsRtmProject) -> None:
        p = project.preprocess
        p.use_bandpass = self.chk_bp.isChecked()
        p.freqlo = float(self.sp_flo.value())
        p.freqhi = float(self.sp_fhi.value())
        p.use_mute = self.chk_mute.isChecked()
        p.tmute = float(self.sp_tmute.value())
        p.vmute = float(self.sp_vmute.value())
        p.mute_tp = float(self.sp_mute_tp.value())
        p.vel_mute_invert = self.chk_vel_mute_inv.isChecked()
        p.hand_select = self.chk_hand.isChecked()
        p.apply_preprocess = self.chk_apply_prep.isChecked()
        p.proc_shot_list = self.proc_shot_list_text()
        self.canvas.set_mute_tp(p.mute_tp)
        self.sync_mute_to_project(project)
        self.sync_gain_to_project(project)

    def load_from_project(self, project: ObsRtmProject) -> None:
        from ..services.rtm_job import parse_shot_list

        p = project.preprocess
        self.chk_bp.setChecked(p.use_bandpass)
        self.sp_flo.setValue(p.freqlo)
        self.sp_fhi.setValue(p.freqhi)
        self.chk_mute.setChecked(bool(p.use_mute))
        self.sp_tmute.setValue(p.tmute)
        self.sp_vmute.setValue(float(p.vmute))
        self.sp_mute_tp.setValue(float(getattr(p, "mute_tp", 0.15) or 0.0))
        self.chk_vel_mute_inv.blockSignals(True)
        self.chk_vel_mute_inv.setChecked(bool(getattr(p, "vel_mute_invert", False)))
        self.chk_vel_mute_inv.blockSignals(False)
        self.chk_hand.blockSignals(True)
        self.chk_hand.setChecked(bool(getattr(p, "hand_select", True)))
        self.chk_hand.blockSignals(False)
        self.canvas.set_hand_pick_enabled(self.chk_hand.isChecked())
        self.chk_apply_prep.blockSignals(True)
        self.chk_apply_prep.setChecked(bool(getattr(p, "apply_preprocess", True)))
        self.chk_apply_prep.blockSignals(False)
        try:
            ids = parse_shot_list(str(getattr(p, "proc_shot_list", "") or ""))
        except ValueError:
            ids = []
        # 手选袋空时回退 RTM 已存炮表（避免重开后高亮/作业炮集丢失）
        if not ids:
            try:
                ids = parse_shot_list(
                    str(getattr(project.rtm, "shot_list", "") or "")
                )
            except ValueError:
                ids = []
        self._set_bag_ids(ids, emit=False)
        self.sync_mute_from_project(project)
        self.sync_gain_from_project(project)
        try:
            self.canvas.set_selected_shot_ids(self._bag_ids)
        except Exception:
            pass
        # 不在此处 emit selection_changed：打开流程结束再统一刷 RTM 黄星，
        # 避免加载中定时器把 rtm.shot_list 同步成「全炮」清空。
