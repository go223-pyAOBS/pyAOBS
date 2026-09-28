# -*- coding: utf-8 -*-
"""阶段 5：Madagascar awefd2d RTM / rtm_shot_loop + 叠炮预览。"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..dialog_utils import connect_combo_deferred
from ..project import ObsRtmProject
from ..styles import compact_form, primary_button, side_panel_layout
from ..widgets.vel_canvas import VelCanvas


class RtmPanel(QWidget):
    request_run = Signal()
    request_stop = Signal()
    request_stack = Signal()
    request_preview = Signal()
    request_preview_vel = Signal()
    request_preview_wfl = Signal()
    request_play_wfl = Signal()
    request_stop_wfl = Signal()
    request_prepare_scons = Signal()
    request_sync_time_from_shot = Signal()
    project_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split)

        left = QWidget()
        left.setObjectName("ObsRtmSidePanel")
        left.setMinimumWidth(300)
        left.setMaximumWidth(480)
        left_l = side_panel_layout(left)

        box = QGroupBox("RTM（Madagascar）")
        form = compact_form(QFormLayout(box))

        self.cmb_engine = QComboBox()
        self.cmb_engine.addItem("Madagascar awefd2d（OBS 为源）", "madagascar")
        self.cmb_engine.addItem("自定义 rtm-bin（rtm_shot_loop）", "custom_bin")
        self.cmb_engine.addItem("仅写脚本 dry-run", "dry_run")
        self.cmb_engine.setToolTip(
            "madagascar=互易 OBS 为源 scons（推荐）；"
            "custom_bin=仍为逐炮脚本（未改互易）；dry_run=只写脚本不跑"
        )
        form.addRow("引擎", self.cmb_engine)

        self.ed_workdir = QLineEdit("rtm_work")
        self.ed_workdir.setToolTip(
            "偏移输出目录（相对工区，默认 rtm_work/）："
            "SConstruct、波场、单炮像、img_lap 等全部写于此；"
            "输入在 rtm_in/、prep/、inputs/（见 WORKDIR_LAYOUT.txt）"
        )
        self.ed_vel = QLineEdit("rtm_in/vel.rsf")
        self.ed_vel.setToolTip(
            "成像速度 RSF，通常为速度页生成的 rtm_in/vel.rsf；进入本页会自动预览"
        )
        self.btn_prev_vel = QPushButton("预览")
        self.btn_prev_vel.setToolTip("在右侧预览当前速度 RSF（rtm_in/vel.rsf）")
        self.btn_prev_vel.clicked.connect(self.request_preview_vel.emit)
        row_dir = QHBoxLayout()
        row_dir.setContentsMargins(0, 0, 0, 0)
        row_dir.setSpacing(4)
        row_dir.addWidget(self.ed_workdir, 1)
        row_dir.addWidget(self.ed_vel, 1)
        row_dir.addWidget(self.btn_prev_vel, 0)
        form.addRow("输出/速度", row_dir)

        self.chk_proc = QCheckBox("使用 shots_proc/")
        self.chk_proc.setChecked(True)
        self.chk_proc.setToolTip(
            "优先用预处理炮集 shots_proc/（若存在）；"
            "勾选后 SConstruct 内不再重复 bandpass / sfmutter"
            "（应已在预处理页完成）。未勾选时：速度 mute 的 tmute→t0、vmute→v0"
        )
        form.addRow("炮集", self.chk_proc)

        self.sp_tmax = QDoubleSpinBox()
        self.sp_tmax.setRange(0.1, 200.0)
        self.sp_tmax.setDecimals(3)
        self.sp_tmax.setSingleStep(1.0)
        self.sp_tmax.setKeyboardTracking(False)
        self.sp_tmax.setValue(40.0)
        self.sp_tmax.setSuffix(" s")
        self.sp_tmax.setToolTip(
            "记录时长 T（秒）。由您决定；nt ≈ T×采样率 + 1（含 t=0）。"
            "改完请点回车或点别处再点「运行 RTM」"
        )
        self.sp_fs = QDoubleSpinBox()
        self.sp_fs.setRange(1.0, 10000.0)
        self.sp_fs.setDecimals(3)
        self.sp_fs.setSingleStep(1.0)
        self.sp_fs.setKeyboardTracking(False)
        self.sp_fs.setValue(250.0)
        self.sp_fs.setSuffix(" Hz")
        self.sp_fs.setToolTip(
            "采样率 fs（Hz）。dt=1/fs；须与炮集 d1 一致（或刻意截断/重采样）"
        )
        btn_from_shot = QPushButton("从炮集")
        btn_from_shot.setToolTip("从当前 shots/ 第一炮头文件读取 n1、d1 → 填入 T 与 fs")
        btn_from_shot.clicked.connect(self.request_sync_time_from_shot.emit)
        row_t = QHBoxLayout()
        row_t.setContentsMargins(0, 0, 0, 0)
        row_t.setSpacing(4)
        row_t.addWidget(self.sp_tmax, 1)
        row_t.addWidget(self.sp_fs, 1)
        row_t.addWidget(btn_from_shot, 0)
        form.addRow("T / fs", row_t)
        self.lbl_nt_dt = QLabel("→ nt=10001, dt=0.004000 s")
        self.lbl_nt_dt.setToolTip("由 T、fs 自动计算，写入 SConstruct 的 nt/dt")
        form.addRow("导出", self.lbl_nt_dt)
        self.sp_tmax.valueChanged.connect(self._on_time_param_changed)
        self.sp_fs.valueChanged.connect(self._on_time_param_changed)
        self._sync_nt_dt_label()

        self.sp_fmin = QDoubleSpinBox()
        self.sp_fmin.setRange(0.1, 100.0)
        self.sp_fmin.setValue(3.0)
        self.sp_fmin.setSuffix(" Hz")
        self.sp_fmin.setToolTip("震源/滤波低频（Hz）")
        self.sp_fmax = QDoubleSpinBox()
        self.sp_fmax.setRange(0.1, 100.0)
        self.sp_fmax.setValue(8.0)
        self.sp_fmax.setSuffix(" Hz")
        self.sp_fmax.setToolTip("震源/滤波高频（Hz）")
        row_f = QHBoxLayout()
        row_f.setContentsMargins(0, 0, 0, 0)
        row_f.setSpacing(4)
        row_f.addWidget(self.sp_fmin, 1)
        row_f.addWidget(self.sp_fmax, 1)
        form.addRow("fmin/fmax", row_f)

        self._shot_list = ""
        self._first_shot = 0
        self._max_shot = -1

        self.ed_obs_list = QLineEdit()
        self.ed_obs_list.setPlaceholderText("空=全部；分批如 0 再 1,2")
        self.ed_obs_list.setToolTip(
            "互易震源 OBS 下标（与 obs_xz 行号一致）。\n"
            "空=全部；也可分批填写（如先 0，再 1,2）——只更新对应 img_obs_NNN，\n"
            "其它台已有像保留。全局叠后点「叠全部已有 OBS 像」。\n"
            "脉冲成像固定 OBS0，不受本框影响。"
        )
        self.ed_bin = QLineEdit()
        self.ed_bin.setPlaceholderText("仅 custom_bin")
        self.ed_bin.setToolTip("自定义 RTM 可执行文件路径（仅 custom_bin 引擎）")
        btn_b = QPushButton("…")
        btn_b.setFixedWidth(28)
        btn_b.setToolTip("选择 rtm-bin 可执行文件")
        btn_b.clicked.connect(self._pick_bin)
        row_obs_bin = QHBoxLayout()
        row_obs_bin.setContentsMargins(0, 0, 0, 0)
        row_obs_bin.setSpacing(4)
        row_obs_bin.addWidget(self.ed_obs_list, 1)
        row_obs_bin.addWidget(self.ed_bin, 1)
        row_obs_bin.addWidget(btn_b, 0)
        form.addRow("OBS/bin", row_obs_bin)

        self.sp_jsnap = QSpinBox()
        self.sp_jsnap.setRange(1, 5000)
        self.sp_jsnap.setValue(80)
        self.sp_jsnap.setToolTip(
            "awefd2d 波场时间抽样；越大越省内存、成像越糙；正式作业首跑建议 ≥80。"
            "脉冲诊断会自动取更密快照（约 32–40 帧）；本值更小时脉冲动画更密。"
        )
        self.sp_nb = QSpinBox()
        self.sp_nb.setRange(10, 200)
        self.sp_nb.setValue(40)
        self.sp_nb.setToolTip("吸收边界 ABC 厚度（网格点）")
        row_jn = QHBoxLayout()
        row_jn.setContentsMargins(0, 0, 0, 0)
        row_jn.setSpacing(4)
        row_jn.addWidget(self.sp_jsnap, 1)
        row_jn.addWidget(self.sp_nb, 1)
        form.addRow("jsnap/nb", row_jn)

        self.chk_mute_w = QCheckBox("预览压水柱")
        self.chk_mute_w.setChecked(True)
        self.chk_mute_w.setToolTip(
            "仅作用于「成像」预览：将 z<bath 置零，突出海底以下构造。\n"
            "不影响波场预览（波场始终显示全深度；OBS 源在海底，"
            "正传能量从海底向水柱/岩体辐射属正常）。"
        )
        self.chk_verb = QCheckBox("打印进度")
        self.chk_verb.setChecked(False)
        self.chk_verb.setToolTip(
            "awefd2d verb=y：每时间步刷一行。终端里往往还快，"
            "经 GUI 管道进日志会严重拖慢——排查时再开；平时看「>>>」与心跳进度即可"
        )
        row_opt = QHBoxLayout()
        row_opt.setContentsMargins(0, 0, 0, 0)
        row_opt.setSpacing(8)
        row_opt.addWidget(self.chk_mute_w)
        row_opt.addWidget(self.chk_verb)
        row_opt.addStretch(1)
        form.addRow("选项", row_opt)
        left_l.addWidget(box)

        self.btn_run = primary_button("运行 RTM")
        self.btn_run.setToolTip(
            "清理中间文件、按 OBS 为源生成 SConstruct 并启动"
            "（madagascar / rtm-bin / dry-run）"
        )
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setToolTip("终止当前 RTM 子进程")
        self.btn_stop.setEnabled(False)
        self.btn_stack = QPushButton("叠全部已有 OBS 像")
        self.btn_stack.setToolTip(
            "扫 rtm_work/ 全部 img_obs_NNN.rsf 求和 → img_stack / img_solid / img_lap，\n"
            "并写 img_stack_manifest.json（与本轮 obs_list 无关，支持分批跑）"
        )
        self.btn_scons = QPushButton("仅准备 SConstruct")
        self.btn_scons.setToolTip(
            "清理 rtm_work/、预拼 OBS 道集并生成互易 SConstruct；\n"
            "手跑：cd rtm_work && scons -f SConstruct_obs_rtm img_lap.rsf\n"
            "（叠后会并入盘上已有 img_obs_*）"
        )

        box_run = QGroupBox("作业")
        form_run = compact_form(QFormLayout(box_run))
        row_run = QHBoxLayout()
        row_run.setContentsMargins(0, 0, 0, 0)
        row_run.setSpacing(4)
        row_run.addWidget(self.btn_run, 1)
        row_run.addWidget(self.btn_stop, 1)
        form_run.addRow("运行", row_run)
        form_run.addRow("叠OBS", self.btn_stack)
        form_run.addRow("准备", self.btn_scons)
        left_l.addWidget(box_run)

        box_img = QGroupBox("成像预览")
        form_img = compact_form(QFormLayout(box_img))
        self.cmb_img_src = QComboBox()
        self.cmb_img_src.setToolTip(
            "预览哪一幅像：叠后(全部已有 OBS 像之和)，或单台 "
            "img_obs_NNN / 旧 img_NNN"
        )
        self.cmb_img_src.addItem("叠后 (img_lap / stack)", None)
        self.btn_refresh_img = QPushButton("刷新")
        self.btn_refresh_img.setToolTip(
            "重新扫描 rtm_work/ 中的 img_obs_*.rsf / img_*.rsf"
        )
        self.btn_prev = QPushButton("预览成像")
        self.btn_prev.setToolTip(
            "按「成像源」加载叠后或指定 img_obs_NNN / img_NNN；\n"
            "绘图与波场快照相同：速度色标底图 + 振幅半透明叠层"
        )
        row_img = QHBoxLayout()
        row_img.setContentsMargins(0, 0, 0, 0)
        row_img.setSpacing(4)
        row_img.addWidget(self.cmb_img_src, 1)
        row_img.addWidget(self.btn_refresh_img, 0)
        row_img.addWidget(self.btn_prev, 0)
        form_img.addRow("成像源", row_img)
        left_l.addWidget(box_img)

        box_wfl = QGroupBox("波场快照")
        form_wfl = compact_form(QFormLayout(box_wfl))
        self.cmb_wfl_kind = QComboBox()
        self.cmb_wfl_kind.addItem("反传", "wflr")
        self.cmb_wfl_kind.addItem("正传", "wfls")
        self.cmb_wfl_kind.setMaximumWidth(72)
        self.cmb_wfl_kind.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.cmb_wfl_kind.setMinimumContentsLength(2)
        self.cmb_wfl_kind.setToolTip(
            "正传 wfls：OBS 为源，能量从 OBS 往外发散；\n"
            "反传 wflr：数据注入在检波炮点，能量从炮点一侧传回（勿与正传混淆）"
        )
        self.cmb_wfl_src = QComboBox()
        self.cmb_wfl_src.setMaximumWidth(120)
        self.cmb_wfl_src.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.cmb_wfl_src.setMinimumContentsLength(8)
        self.cmb_wfl_src.setToolTip(
            "选择波场：脉冲NNN = rtm_work/impulse/；"
            "OBS_NNN = 正式 rtm_work/（同编号分列）"
        )
        self.btn_refresh_wfl = QPushButton("刷新")
        self.btn_refresh_wfl.setFixedWidth(40)
        self.btn_refresh_wfl.setToolTip(
            "扫描 rtm_work/ 与 impulse/ 中的 wflr_*/wfls_*"
        )
        row_wfl_1 = QHBoxLayout()
        row_wfl_1.setContentsMargins(0, 0, 0, 0)
        row_wfl_1.setSpacing(4)
        row_wfl_1.addWidget(self.cmb_wfl_kind, 0)
        row_wfl_1.addWidget(self.cmb_wfl_src, 1)
        row_wfl_1.addWidget(self.btn_refresh_wfl, 0)
        form_wfl.addRow("种类", row_wfl_1)

        self.sp_wfl_frame = QSpinBox()
        self.sp_wfl_frame.setRange(-1, 99999)
        self.sp_wfl_frame.setSpecialValueText("中间帧")
        self.sp_wfl_frame.setValue(-1)
        self.sp_wfl_frame.setToolTip(
            "快照帧号（0…n3-1）；选「中间帧」= n3//2，便于检查孔径内传播"
        )
        self.lbl_wfl_n3 = QLabel("n3=?")
        self.lbl_wfl_n3.setToolTip("当前波场文件的快照张数")
        self.sp_wfl_fps = QDoubleSpinBox()
        self.sp_wfl_fps.setRange(1.0, 30.0)
        self.sp_wfl_fps.setDecimals(0)
        self.sp_wfl_fps.setValue(8.0)
        self.sp_wfl_fps.setSuffix("fps")
        self.sp_wfl_fps.setMaximumWidth(72)
        self.sp_wfl_fps.setToolTip("动画帧率")
        self.btn_wfl_frame0 = QPushButton("清零")
        self.btn_wfl_frame0.setMaximumWidth(48)
        self.btn_wfl_frame0.setToolTip("帧号回到 0 并预览该帧")
        self.btn_wfl_frame0.clicked.connect(self._on_wfl_frame_zero)
        row_wfl_2 = QHBoxLayout()
        row_wfl_2.setContentsMargins(0, 0, 0, 0)
        row_wfl_2.setSpacing(4)
        row_wfl_2.addWidget(self.sp_wfl_frame, 1)
        row_wfl_2.addWidget(self.lbl_wfl_n3, 0)
        row_wfl_2.addWidget(self.sp_wfl_fps, 0)
        row_wfl_2.addWidget(self.btn_wfl_frame0, 0)
        form_wfl.addRow("帧/fps", row_wfl_2)

        self.chk_wfl_dual = QCheckBox("正反同步")
        self.chk_wfl_dual.setToolTip(
            "勾选后在同一速度底图上叠加正传 wfls（暖色）与反传 wflr（冷色），同帧推进"
        )
        self.btn_prev_wfl = QPushButton("预览")
        self.btn_prev_wfl.setToolTip(
            "速度色标底图 + 半透明波场（单帧）；勾选「正反同步」则同图叠正传+反传"
        )
        self.btn_play_wfl = QPushButton("播放")
        self.btn_play_wfl.setToolTip(
            "按 n3 逐帧播放；勾选「正反同步」则同图叠播正传/反传"
        )
        self.btn_stop_wfl = QPushButton("停止")
        self.btn_stop_wfl.setToolTip("停止波场动画")
        self.btn_stop_wfl.setEnabled(False)
        row_wfl_anim = QHBoxLayout()
        row_wfl_anim.setContentsMargins(0, 0, 0, 0)
        row_wfl_anim.setSpacing(4)
        row_wfl_anim.addWidget(self.chk_wfl_dual, 0)
        row_wfl_anim.addWidget(self.btn_prev_wfl, 1)
        row_wfl_anim.addWidget(self.btn_play_wfl, 1)
        row_wfl_anim.addWidget(self.btn_stop_wfl, 1)
        form_wfl.addRow("动画", row_wfl_anim)
        left_l.addWidget(box_wfl)

        self.btn_run.clicked.connect(self.request_run.emit)
        self.btn_stop.clicked.connect(self.request_stop.emit)
        self.btn_stack.clicked.connect(self.request_stack.emit)
        self.btn_prev.clicked.connect(self.request_preview.emit)
        self.btn_refresh_img.clicked.connect(self.request_preview.emit)
        self.btn_prev_wfl.clicked.connect(self.request_preview_wfl.emit)
        self.btn_play_wfl.clicked.connect(self.request_play_wfl.emit)
        self.btn_stop_wfl.clicked.connect(self.request_stop_wfl.emit)
        self.btn_refresh_wfl.clicked.connect(self._on_refresh_wfl_clicked)
        self.btn_scons.clicked.connect(self.request_prepare_scons.emit)
        left_l.addStretch(1)
        split.addWidget(left)

        self.canvas = VelCanvas()
        self.canvas.setObjectName("ObsRtmPlotPanel")
        split.addWidget(self.canvas)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        # 侧栏默认更窄，给成像预览留宽；下限由 left.minimumWidth 保证不裁切
        split.setSizes([340, 760])

        connect_combo_deferred(
            self.cmb_engine, lambda *_: self.project_changed.emit()
        )
        for w in (self.ed_workdir, self.ed_vel, self.ed_bin, self.ed_obs_list):
            w.textChanged.connect(lambda: self.project_changed.emit())
        for w in (self.sp_jsnap, self.sp_nb):
            w.valueChanged.connect(lambda: self.project_changed.emit())
        for w in (self.sp_fmin, self.sp_fmax):
            w.valueChanged.connect(lambda: self.project_changed.emit())
        self.chk_proc.toggled.connect(lambda *_: self.project_changed.emit())
        self.chk_verb.toggled.connect(lambda *_: self.project_changed.emit())
        # 压水柱：写回工程并立即重载成像预览
        self.chk_mute_w.toggled.connect(self._on_mute_water_toggled)
        connect_combo_deferred(
            self.cmb_img_src, lambda *_: self.request_preview.emit()
        )
        connect_combo_deferred(
            self.cmb_wfl_kind, lambda *_: self._on_wfl_kind_changed()
        )
        connect_combo_deferred(
            self.cmb_wfl_src, lambda *_: self.request_preview_wfl.emit()
        )
        self.sp_wfl_frame.editingFinished.connect(self.request_preview_wfl.emit)

    def _on_mute_water_toggled(self, *_args) -> None:
        """预览压水柱：立即同步工程并重绘成像。"""
        self.project_changed.emit()
        self.request_preview.emit()

    def preview_shot_id(self):
        """None=叠后；int=OBS/炮像编号；str 如 impulse_raw / impulse_lap。"""
        return self.cmb_img_src.currentData()

    def wfl_kind(self) -> str:
        return str(self.cmb_wfl_kind.currentData() or "wflr")

    def wfl_tag_id(self):
        """当前波场 OBS 编号；无列表时为 None。兼容旧 int 与 (scope, tag)。"""
        data = self.cmb_wfl_src.currentData()
        if data is None:
            return None
        if isinstance(data, (tuple, list)) and len(data) >= 2:
            return int(data[1])
        return int(data)

    def wfl_scope(self) -> Optional[str]:
        """``impulse`` | ``formal`` | None（未选/旧数据）。"""
        data = self.cmb_wfl_src.currentData()
        if isinstance(data, (tuple, list)) and len(data) >= 2:
            return str(data[0])
        return None

    def wfl_frame(self) -> int:
        """-1 = 中间帧。"""
        return int(self.sp_wfl_frame.value())

    def wfl_fps(self) -> float:
        return float(self.sp_wfl_fps.value())

    def wfl_dual(self) -> bool:
        return bool(self.chk_wfl_dual.isChecked())

    def set_highlight_shot_idx(self, ids) -> None:
        """主画布黄星。"""
        try:
            self.canvas.set_highlight_shot_idx(ids)
        except Exception:
            pass

    def set_wfl_animating(self, playing: bool) -> None:
        try:
            self.btn_play_wfl.setEnabled(not playing)
            self.btn_stop_wfl.setEnabled(playing)
            self.btn_prev_wfl.setEnabled(not playing)
            self.chk_wfl_dual.setEnabled(not playing)
        except RuntimeError:
            pass

    def _on_refresh_wfl_clicked(self) -> None:
        self.request_preview_wfl.emit()

    def _on_wfl_frame_zero(self) -> None:
        """帧号回到 0 并触发预览。"""
        self.sp_wfl_frame.setValue(0)
        self.request_preview_wfl.emit()

    def _on_wfl_kind_changed(self) -> None:
        # 种类切换后由主窗口刷新列表并预览
        self.request_preview_wfl.emit()

    def refresh_wfl_source_list(self, tag_ids, *, n3: int = 0) -> None:
        """
        刷新波场下拉。

        ``tag_ids`` 可为 ``[0,1,…]`` 或 ``[('impulse',0), ('formal',0), …]``。
        """
        cur = self.cmb_wfl_src.currentData()
        self.cmb_wfl_src.blockSignals(True)
        self.cmb_wfl_src.clear()
        for item in tag_ids or []:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                scope, tid = str(item[0]), int(item[1])
                label = (
                    "脉冲%03d" % tid
                    if scope == "impulse"
                    else "OBS_%03d" % tid
                )
                tip = (
                    "脉冲作业 rtm_work/impulse/"
                    if scope == "impulse"
                    else "正式作业 rtm_work/（OBS 为源）"
                )
                self.cmb_wfl_src.addItem(label, (scope, tid))
                i = self.cmb_wfl_src.count() - 1
                self.cmb_wfl_src.setItemData(i, tip, Qt.ItemDataRole.ToolTipRole)
            else:
                tid = int(item)
                self.cmb_wfl_src.addItem("OBS_%03d" % tid, ("formal", tid))
        idx = 0
        if cur is not None:
            for i in range(self.cmb_wfl_src.count()):
                if self.cmb_wfl_src.itemData(i) == cur:
                    idx = i
                    break
            else:
                # 旧 int 选中 → 匹配同编号（优先 impulse）
                try:
                    cur_tid = (
                        int(cur[1])
                        if isinstance(cur, (tuple, list))
                        else int(cur)
                    )
                except (TypeError, ValueError):
                    cur_tid = None
                if cur_tid is not None:
                    for i in range(self.cmb_wfl_src.count()):
                        d = self.cmb_wfl_src.itemData(i)
                        if (
                            isinstance(d, (tuple, list))
                            and int(d[1]) == cur_tid
                        ):
                            idx = i
                            break
        if self.cmb_wfl_src.count() > 0:
            self.cmb_wfl_src.setCurrentIndex(idx)
        self.cmb_wfl_src.blockSignals(False)
        n3 = max(int(n3 or 0), 0)
        if n3 > 0:
            self.sp_wfl_frame.blockSignals(True)
            self.sp_wfl_frame.setRange(-1, n3 - 1)
            self.sp_wfl_frame.blockSignals(False)
            self.lbl_wfl_n3.setText("n3=%d" % n3)
        else:
            self.lbl_wfl_n3.setText("n3=?")

    def refresh_img_source_list(
        self,
        shot_ids,
        *,
        obs_mode: bool = False,
        impulse_names: Optional[list] = None,
        stack_label: Optional[str] = None,
        stack_tip: Optional[str] = None,
    ) -> None:
        """用 rtm_work 中已有单像刷新下拉；保留当前选中。"""
        cur = self.cmb_img_src.currentData()
        self.cmb_img_src.blockSignals(True)
        self.cmb_img_src.clear()
        self.cmb_img_src.addItem(
            str(stack_label or "叠后 (img_lap / stack)"), None
        )
        # 脉冲：raw 无 lap，更易看等时线弧形
        for key, label in (
            ("impulse_raw", "脉冲 raw (img_impulse)"),
            ("impulse_solid", "脉冲 solid (压水柱)"),
            ("impulse_lap", "脉冲 lap (拉普拉斯)"),
        ):
            if impulse_names and key in impulse_names:
                self.cmb_img_src.addItem(label, key)
        for sid in shot_ids or []:
            if obs_mode:
                self.cmb_img_src.addItem("OBS_%03d" % int(sid), int(sid))
            else:
                self.cmb_img_src.addItem("单炮 img_%03d" % int(sid), int(sid))
        tip = (
            "预览：叠后(全部已有 OBS) / 脉冲(raw|lap) / "
            + ("OBS 互易像" if obs_mode else "单炮像")
        )
        self.cmb_img_src.setToolTip(tip)
        self.cmb_img_src.setItemData(
            0,
            stack_tip
            or "叠后：盘上全部 img_obs_* 之和 → img_lap（见 img_stack_manifest.json）",
            Qt.ItemDataRole.ToolTipRole,
        )
        # 恢复选中（支持 int 与 str）
        idx = 0
        if cur is not None:
            for i in range(self.cmb_img_src.count()):
                if self.cmb_img_src.itemData(i) == cur:
                    idx = i
                    break
        self.cmb_img_src.setCurrentIndex(idx)
        self.cmb_img_src.blockSignals(False)

    def set_shots_from_hand_select(self, ids) -> None:
        """ids 非空 → 只跑这些炮；None/空 → 全炮 (first=0, max=-1)。无列表 UI。"""
        if ids:
            self._shot_list = ",".join(str(int(i)) for i in ids)
            self._first_shot = int(ids[0])
            self._max_shot = len(ids)
        else:
            self._shot_list = ""
            self._first_shot = 0
            self._max_shot = -1
        self.project_changed.emit()

    def _pick_bin(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "RTM 可执行文件", "", "All (*.*)")
        if path:
            self.ed_bin.setText(path)

    @staticmethod
    def nt_dt_from_tmax_fs(tmax_s: float, fs_hz: float) -> tuple:
        """T(s)、fs(Hz) → (nt, dt)。nt = round(T*fs)+1（含 t=0）。"""
        fs = max(float(fs_hz), 1e-6)
        dt = 1.0 / fs
        tmax = max(float(tmax_s), dt)
        nt = int(round(tmax * fs)) + 1
        nt = max(nt, 2)
        return nt, dt

    def _on_time_param_changed(self, *_args) -> None:
        self._sync_nt_dt_label()
        self.project_changed.emit()

    def _sync_nt_dt_label(self) -> None:
        nt, dt = self.nt_dt_from_tmax_fs(self.sp_tmax.value(), self.sp_fs.value())
        self.lbl_nt_dt.setText("→ nt=%d, dt=%.6g s" % (nt, dt))

    def set_time_from_nt_dt(self, nt: int, dt: float, *, emit: bool = True) -> None:
        """由 nt/dt 回填 T、fs（加载工程 / 从炮集同步）。"""
        dt = max(float(dt), 1e-9)
        nt = max(int(nt), 2)
        fs = 1.0 / dt
        tmax = (nt - 1) * dt
        self.sp_tmax.blockSignals(True)
        self.sp_fs.blockSignals(True)
        self.sp_tmax.setValue(tmax)
        self.sp_fs.setValue(fs)
        self.sp_tmax.blockSignals(False)
        self.sp_fs.blockSignals(False)
        self._sync_nt_dt_label()
        if emit:
            self.project_changed.emit()

    def set_running(self, running: bool) -> None:
        """窗口关闭后 C++ 控件可能已删，必须吞掉访问错误。"""
        try:
            from shiboken6 import isValid

            if not isValid(self):
                return
        except Exception:
            pass
        try:
            self.btn_run.setEnabled(not running)
            self.btn_stop.setEnabled(running)
        except RuntimeError:
            pass

    def apply_to_project(self, project: ObsRtmProject) -> None:
        r = project.rtm
        r.engine = str(self.cmb_engine.currentData() or "madagascar")
        r.workdir = self.ed_workdir.text().strip() or "rtm_work"
        r.vel_rsf = self.ed_vel.text().strip() or "rtm_in/vel.rsf"
        r.use_shots_proc = self.chk_proc.isChecked()
        tmax = float(self.sp_tmax.value())
        fs = float(self.sp_fs.value())
        nt, dt = self.nt_dt_from_tmax_fs(tmax, fs)
        r.tmax = tmax
        r.fs = fs
        r.nt = int(nt)
        r.dt = float(dt)
        r.fmin = float(self.sp_fmin.value())
        r.fmax = float(self.sp_fmax.value())
        r.first_shot = int(self._first_shot)
        r.max_shot = int(self._max_shot)
        r.shot_list = str(self._shot_list or "")
        r.obs_list = self.ed_obs_list.text().strip()
        r.jsnap = int(self.sp_jsnap.value())
        r.nb = int(self.sp_nb.value())
        r.rtm_bin = self.ed_bin.text().strip()
        r.dry_run = r.engine == "dry_run"
        r.mute_water_preview = self.chk_mute_w.isChecked()
        r.awefd_verb = self.chk_verb.isChecked()

    def load_from_project(self, project: ObsRtmProject) -> None:
        r = project.rtm
        eng = getattr(r, "engine", "madagascar") or "madagascar"
        idx = self.cmb_engine.findData(eng)
        if idx < 0 and r.dry_run:
            idx = self.cmb_engine.findData("dry_run")
        if idx >= 0:
            self.cmb_engine.setCurrentIndex(idx)
        self.ed_workdir.setText(r.workdir)
        self.ed_vel.setText(r.vel_rsf)
        self.chk_proc.setChecked(r.use_shots_proc)
        # nt/dt 是作业真值；tmax/fs 仅当与 nt/dt 一致时才用来还原旋钮
        # （避免旧 JSON 无 tmax 时 dataclass 默认 40s 盖掉 nt=7501）
        nt = max(int(r.nt), 2)
        dt = max(float(r.dt), 1e-9)
        tmax_s = float(getattr(r, "tmax", 0) or 0)
        fs_s = float(getattr(r, "fs", 0) or 0)
        use_tf = False
        if tmax_s > 0 and fs_s > 0:
            nt2, dt2 = self.nt_dt_from_tmax_fs(tmax_s, fs_s)
            if abs(nt2 - nt) <= 1 and abs(dt2 - dt) <= 1e-12:
                use_tf = True
        if use_tf:
            self.sp_tmax.blockSignals(True)
            self.sp_fs.blockSignals(True)
            self.sp_tmax.setValue(tmax_s)
            self.sp_fs.setValue(fs_s)
            self.sp_tmax.blockSignals(False)
            self.sp_fs.blockSignals(False)
            self._sync_nt_dt_label()
        else:
            self.set_time_from_nt_dt(nt, dt, emit=False)
        self.sp_fmin.setValue(r.fmin)
        self.sp_fmax.setValue(r.fmax)
        sl = str(getattr(r, "shot_list", "") or "").strip()
        if sl:
            from ..services.rtm_job import parse_shot_list

            try:
                ids = parse_shot_list(sl)
            except ValueError:
                ids = []
            # 加载时勿 emit：避免打开工程过程中触发回写/清空
            if ids:
                self._shot_list = ",".join(str(int(i)) for i in ids)
                self._first_shot = int(ids[0])
                self._max_shot = len(ids)
            else:
                self._shot_list = ""
                self._first_shot = 0
                self._max_shot = -1
        else:
            self._shot_list = ""
            self._first_shot = 0
            self._max_shot = -1
        self.ed_obs_list.setText(str(getattr(r, "obs_list", "") or ""))
        self.sp_jsnap.setValue(int(getattr(r, "jsnap", 40)))
        self.sp_nb.setValue(int(getattr(r, "nb", 40)))
        self.ed_bin.setText(r.rtm_bin)
        self.chk_mute_w.setChecked(r.mute_water_preview)
        self.chk_verb.setChecked(bool(getattr(r, "awefd_verb", True)))
