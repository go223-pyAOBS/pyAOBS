# -*- coding: utf-8 -*-
"""阶段 2：工区几何 + 炮检/网格预览。"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
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
from ..widgets.geom_canvas import GeomCanvas


class GeometryPanel(QWidget):
    request_check = Signal()
    request_check_offset_sign = Signal()
    request_suggest_grid = Signal()
    request_preview = Signal()
    request_apply_obs_x = Signal()  # 将 shots/OBS 平移到当前 OBS x（模型坐标）
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

        def _shrink_combo(cmb: QComboBox) -> None:
            cmb.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            cmb.setMinimumContentsLength(5)
            cmb.setMinimumWidth(0)
            cmb.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )

        box = QGroupBox("炮检几何（su_to_shots）")
        form = compact_form(QFormLayout(box))

        self.cmb_geom = QComboBox()
        self.cmb_geom.addItem("offset", "offset")
        self.cmb_geom.addItem("obs", "obs")
        self.cmb_geom.addItem("segy", "segy")
        self.cmb_geom.setToolTip(
            "offset：仅信任道头 offset；shot_x = obs_x + sign*(offset_m/1000)\n"
            "obs：本工区炮=gx、OBS=sx\n"
            "segy：字面 Source/Group"
        )
        _shrink_combo(self.cmb_geom)
        connect_combo_deferred(self.cmb_geom, self._on_geom_changed)
        form.addRow("模式", self.cmb_geom)

        self.sp_obs_x = QDoubleSpinBox()
        self.sp_obs_x.setRange(-1e6, 1e6)
        self.sp_obs_x.setDecimals(3)
        self.sp_obs_x.setSuffix(" km")
        self.sp_obs_x.setToolTip(
            "仅 offset 模式可编辑。OBS 在速度模型测线上的 x（km），须与层析同坐标系。\n"
            "改后点「应用」重建 shots_xz（相对 offset 不变，model x 随 OBS 平移）；"
            "拼图横轴=model x。重新导入 SU 时也会用此值。obs/segy 模式 OBS 来自道头。"
        )
        self.sp_obs_x.setMinimumWidth(0)
        self.sp_obs_x.valueChanged.connect(lambda: self.project_changed.emit())
        row_obs = QHBoxLayout()
        row_obs.addWidget(self.sp_obs_x, 1)
        self.btn_apply_obs_x = QPushButton("应用并更新网格")
        self.btn_apply_obs_x.setToolTip(
            "仅 offset 模式。按 offsets.txt 以新 OBS x 重建 shots_xz\n"
            "（shot = OBS x + sign×offset），并把成像网格 ox 同步平移 ΔOBS\n"
            "（nx/dx 不变；例：ox=-100 且 OBS 0→100 → ox=0）。\n"
            "内置一维还会再按炮点范围微调 ox/nx。\n"
            "拼图横轴随 OBS 热更新（不重读炮集）。"
        )
        self.btn_apply_obs_x.clicked.connect(self.request_apply_obs_x.emit)
        row_obs.addWidget(self.btn_apply_obs_x)
        form.addRow("OBS x", row_obs)

        self.cmb_sign = QComboBox()
        self.cmb_sign.addItem("+1", 1.0)
        self.cmb_sign.addItem("-1", -1.0)
        self.cmb_sign.setToolTip(
            "offset 模式：+1 → shot = obs + offset；-1 → 左右翻转"
        )
        _shrink_combo(self.cmb_sign)
        connect_combo_deferred(
            self.cmb_sign, lambda *_: self.project_changed.emit()
        )
        form.addRow("offset 符号", self.cmb_sign)

        self.sp_zshot = QDoubleSpinBox()
        self.sp_zshot.setRange(0.0, 50.0)
        self.sp_zshot.setDecimals(4)
        self.sp_zshot.setSuffix(" km")
        self.sp_zshot.setValue(0.01)
        self.sp_zshot.setToolTip("炮点深度（km），写入 shots_xz / 后续 RTM 源深度")
        self.sp_zshot.valueChanged.connect(lambda: self.project_changed.emit())
        form.addRow("炮深 zshot", self.sp_zshot)

        self.cmb_zobs = QComboBox()
        for t in ("const", "auto", "selev", "sdepth", "gwdep"):
            self.cmb_zobs.addItem(t, t)
        self.cmb_zobs.setToolTip(
            "OBS 深度来源：const=下方常数；auto/道头字 selev·sdepth·gwdep"
        )
        _shrink_combo(self.cmb_zobs)
        connect_combo_deferred(
            self.cmb_zobs, lambda *_: self.project_changed.emit()
        )
        form.addRow("深度模式", self.cmb_zobs)

        self.sp_zobs = QDoubleSpinBox()
        self.sp_zobs.setRange(0.0, 50.0)
        self.sp_zobs.setDecimals(3)
        self.sp_zobs.setSuffix(" km")
        self.sp_zobs.setValue(1.901)
        self.sp_zobs.setToolTip("深度模式为 const 时的固定 OBS 深度（km）")
        self.sp_zobs.valueChanged.connect(lambda: self.project_changed.emit())
        form.addRow("深度 const", self.sp_zobs)
        left_l.addWidget(box)

        gbox = QGroupBox("成像网格（km，n1=z n2=x）")
        gform = compact_form(QFormLayout(gbox))
        self.sp_ox = QDoubleSpinBox()
        self.sp_ox.setRange(-1e6, 1e6)
        self.sp_ox.setDecimals(3)
        self.sp_ox.setValue(-400.0)
        self.sp_dx = QDoubleSpinBox()
        self.sp_dx.setRange(1e-4, 10.0)
        self.sp_dx.setDecimals(4)
        self.sp_dx.setValue(0.5)
        self.sp_nx = QSpinBox()
        self.sp_nx.setRange(2, 10_000_000)
        self.sp_nx.setValue(1001)  # 勿用控件最小值 2（曾导致全线 OUT）
        self.sp_oz = QDoubleSpinBox()
        self.sp_oz.setRange(0.0, 100.0)
        self.sp_oz.setDecimals(3)
        self.sp_oz.setValue(0.0)
        self.sp_dz = QDoubleSpinBox()
        self.sp_dz.setRange(1e-4, 10.0)
        self.sp_dz.setDecimals(4)
        self.sp_dz.setValue(0.25)
        self.sp_nz = QSpinBox()
        self.sp_nz.setRange(2, 10_000_000)
        self.sp_nz.setValue(161)
        for w in (self.sp_ox, self.sp_dx, self.sp_oz, self.sp_dz):
            w.valueChanged.connect(lambda: self.project_changed.emit())
        self.sp_nx.valueChanged.connect(lambda: self.project_changed.emit())
        self.sp_nz.valueChanged.connect(lambda: self.project_changed.emit())
        self.sp_ox.setToolTip("水平网格原点 ox（km，n2=x）")
        self.sp_dx.setToolTip(
            "成像网格水平步长 dx（km）。「建议网格」只改 ox/nx，不改 dx。"
            "默认 0.5，与速度页栅格一致；精细 RTM 可再改小（如 0.025）"
        )
        self.sp_nx.setToolTip("水平网格点数 nx；勿过小，否则炮点易全落在网格外（OUT）")
        self.sp_oz.setToolTip("深度网格原点 oz（km，通常 0）")
        self.sp_dz.setToolTip(
            "成像网格深度步长 dz（km）。「建议网格」一般保留当前 dz。"
            "默认 0.25；精细 RTM 可再改小（如 0.025）"
        )
        self.sp_nz.setToolTip("深度网格点数 nz；zmax ≈ oz+(nz-1)*dz")
        gform.addRow("ox", self.sp_ox)
        gform.addRow("dx", self.sp_dx)
        gform.addRow("nx", self.sp_nx)
        gform.addRow("oz", self.sp_oz)
        gform.addRow("dz", self.sp_dz)
        gform.addRow("nz", self.sp_nz)
        left_l.addWidget(gbox)

        left_l.addWidget(
            hint_label(
                "红三角=炮 · 绿圆=OBS · 灰框=网格 · 蓝虚线=OBS x\n"
                "offset：改 OBS x 后点「应用并更新网格」或「预览几何」\n"
                "（会平移炮/OBS 并更新 ox/nx）；导入后自动填入 obs_xz。"
            )
        )

        btn_prev = primary_button("预览几何")
        btn_prev.setToolTip("绘制炮（红）、OBS（绿）、网格框与 obs_x 标记")
        btn_prev.clicked.connect(self.request_preview.emit)
        btn_sug = QPushButton("由炮点建议网格")
        btn_sug.setToolTip("按 shots/OBS 范围写 ox/nx，保留当前 dx/dz")
        btn_sug.clicked.connect(self.request_suggest_grid.emit)
        btn_chk = QPushButton("检查落点")
        btn_chk.setToolTip("检查炮/OBS 是否落在网格内；结果见日志与 diag/geom_check.txt")
        btn_chk.clicked.connect(self.request_check.emit)
        btn_chk_sign = QPushButton("检查 offset 符号")
        btn_chk_sign.setToolTip(
            "核对 shots_xz−OBS 与 offsets.txt×offset_sign 是否一致，并评估翻转符号"
        )
        btn_chk_sign.clicked.connect(self.request_check_offset_sign.emit)
        left_l.addWidget(btn_prev)
        left_l.addWidget(btn_sug)
        left_l.addWidget(btn_chk)
        left_l.addWidget(btn_chk_sign)
        left_l.addStretch(1)
        split.addWidget(left)

        self.canvas = GeomCanvas()
        self.canvas.setObjectName("ObsRtmPlotPanel")
        split.addWidget(self.canvas)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        # 侧栏默认更窄；下限保证标签/控件不裁切
        split.setSizes([340, 760])
        self._on_geom_changed()

    def _on_geom_changed(self, *_args) -> None:
        is_off = self.cmb_geom.currentData() == "offset"
        self.sp_obs_x.setEnabled(is_off)
        self.btn_apply_obs_x.setEnabled(is_off)
        self.cmb_sign.setEnabled(is_off)
        self.project_changed.emit()

    def sync_obs_x_from_file(self, project: ObsRtmProject) -> None:
        """从 obs_xz.txt 读入第一台 OBS x 到旋钮（导入后调用）。"""
        from ..services.geometry import current_obs_x_km

        x = current_obs_x_km(project)
        self.sp_obs_x.blockSignals(True)
        self.sp_obs_x.setValue(float(x))
        self.sp_obs_x.blockSignals(False)
        project.geometry.obs_x_km = float(x)

    def apply_to_project(self, project: ObsRtmProject) -> None:
        g = project.geometry
        g.geom = str(self.cmb_geom.currentData())
        g.obs_x_km = float(self.sp_obs_x.value())
        g.offset_sign = float(self.cmb_sign.currentData())
        g.zshot_km = float(self.sp_zshot.value())
        g.zobs_mode = str(self.cmb_zobs.currentData())
        g.zobs_const_km = float(self.sp_zobs.value())
        grid = project.grid
        grid.ox = float(self.sp_ox.value())
        grid.dx = float(self.sp_dx.value())
        grid.nx = int(self.sp_nx.value())
        grid.oz = float(self.sp_oz.value())
        grid.dz = float(self.sp_dz.value())
        grid.nz = int(self.sp_nz.value())

    def load_from_project(self, project: ObsRtmProject) -> None:
        g = project.geometry
        for i in range(self.cmb_geom.count()):
            if self.cmb_geom.itemData(i) == g.geom:
                self.cmb_geom.setCurrentIndex(i)
                break
        # 有 obs_xz 时以文件为准，否则用工程参数
        try:
            self.sync_obs_x_from_file(project)
        except Exception:
            self.sp_obs_x.setValue(g.obs_x_km)
        for i in range(self.cmb_sign.count()):
            if float(self.cmb_sign.itemData(i)) == float(g.offset_sign):
                self.cmb_sign.setCurrentIndex(i)
                break
        self.sp_zshot.setValue(g.zshot_km)
        for i in range(self.cmb_zobs.count()):
            if self.cmb_zobs.itemData(i) == g.zobs_mode:
                self.cmb_zobs.setCurrentIndex(i)
                break
        self.sp_zobs.setValue(g.zobs_const_km)
        grid = project.grid
        self.sp_ox.setValue(grid.ox)
        self.sp_dx.setValue(grid.dx)
        self.sp_nx.setValue(grid.nx)
        self.sp_oz.setValue(grid.oz)
        self.sp_dz.setValue(grid.dz)
        self.sp_nz.setValue(grid.nz)
        self._on_geom_changed()
