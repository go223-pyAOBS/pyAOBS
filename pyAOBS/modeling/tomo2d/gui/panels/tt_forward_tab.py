"""tt_forward 参数页（常用展开，其余折叠）。"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QWidget

from ..state.form_state import FormState
from ..widgets.field_form import CollapsibleSection, FieldFormTab

_CORE = [
    ("fwd.smesh", "smesh (-M)", "", "open"),
    ("fwd.geom", "geom (-G)", "", "open"),
    ("fwd.out_ttime", "out_ttime (-T)", "", "save"),
    ("fwd.out_ray", "out_ray (-R，默认空=不写)", "", "save"),
]

_RAY = [
    ("fwd.xorder", "xorder (-N)", "4", "text"),
    ("fwd.zorder", "zorder (-N)", "4", "text"),
    ("fwd.clen", "clen (-N)", "0.8", "text"),
    ("fwd.nintp", "nintp (-N)", "8", "text"),
    ("fwd.bend_cg_tol", "bend_cg_tol (-N)", "1e-4", "text"),
    ("fwd.bend_br_tol", "bend_br_tol (-N)", "1e-5", "text"),
]

_OUT = [
    ("fwd.out_elements", "out_elements (-E)", "", "save"),
    ("fwd.out_obs_ttime", "out_obs_ttime (-O)", "", "save"),
    ("fwd.out_source", "out_source (-S)", "", "save"),
    ("fwd.out_vgrid", "out_vgrid (-I)", "", "save"),
    ("fwd.out_diff", "out_diff (-D)", "", "save"),
]

_IFACE = [
    ("fwd.refl_file", "refl_file (-F 反射/莫霍)", "", "open"),
    ("fwd.seafloor_file", "seafloor_file (-B 海底，2/3/4/5)", "", "open"),
    ("fwd.conv_file", "conv_file (-X 转换面，6/7/8)", "", "open"),
    ("fwd.vsmesh", "vsmesh (-U 独立 Vs)", "", "open"),
    ("fwd.kappa", "kappa (-k，无 -U 时 Vp/Vs)", "", "text"),
    ("fwd.do_full_refl", "贴面反射 (-A，改路径)", False, "check"),
]

_ADV = [
    ("fwd.vred", "vred (-r)", "", "text"),
    ("fwd.clock_file", "clock_file (-C)", "", "open"),
    ("fwd.sub_west", "west (-i)", "", "text"),
    ("fwd.sub_east", "east (-i)", "", "text"),
    ("fwd.sub_south", "south (-i)", "", "text"),
    ("fwd.sub_north", "north (-i)", "", "text"),
    ("fwd.sub_dx", "dx (-i)", "", "text"),
    ("fwd.sub_dz", "dz (-i)", "", "text"),
    ("fwd.verbose_level", "verbose_level (-V，>0 启用)", "", "text"),
    ("fwd.graph_only", "graph_only (-g)", False, "check"),
    ("fwd.omit_air_water", "omit_air_water (-n)", False, "check"),
]

_SECTIONS = [
    ("常用", _CORE, True),
    ("界面（-F 反射 / -B 海底）", _IFACE, True),
    ("射线弯曲 (-N)", _RAY, False),
    ("更多输出", _OUT, False),
    ("子网格 / 开关", _ADV, False),
]


class TtForwardTab(FieldFormTab):
    bridge_to_inv_requested = Signal()

    def __init__(self, state: FormState, parent=None) -> None:
        super().__init__(
            state,
            sections=_SECTIONS,
            preview_text="预览 tt_forward",
            run_text="运行 tt_forward",
            parent=parent,
        )

        # 合成数据衔接：默认折叠，不抢主流程
        syn = CollapsibleSection("合成数据（可选）", expanded=False)
        syn_tip = QLabel(
            "仅当要用已知模型造合成走时、再交给反演拟合时使用。"
            "反演每轮本身会做正演，二者不是「必须串起来」的两步。"
            "需要时：填 out_ttime（stdout 走时，不是原生 -T 折合图）→ 运行 → 将空位写到 inv.mesh / inv.data。"
        )
        syn_tip.setWordWrap(True)
        syn_tip.setStyleSheet("color:#64748b;")
        syn.body_layout.addWidget(syn_tip)

        row = QWidget()
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        if not state.has("gui.auto_fwd_to_inv"):
            state.set("gui.auto_fwd_to_inv", False)
        self.ck_auto = QCheckBox("合成流程：完成后写入反演空位")
        self.ck_auto.setToolTip(
            "仅合成数据时建议勾选。成功后把空的 inv.mesh / inv.data（及 -N/-F/-Y）从本页填入。"
        )
        self.binder.bind_check("gui.auto_fwd_to_inv", self.ck_auto)
        btn = QPushButton("写入 inv.mesh / inv.data…")
        btn.setToolTip("把当前 fwd.smesh、out_ttime、海底(-B→-Y)、转换面(-X→-B)、Vs(-U) 及 -N/-F 填到反演页空字段")
        btn.clicked.connect(self.bridge_to_inv_requested.emit)
        hl.addWidget(self.ck_auto)
        hl.addWidget(btn)
        hl.addStretch(1)
        syn.body_layout.addWidget(row)
        self.insert_widget_before_actions(syn)
        sf = self._path_rows.get("fwd.seafloor_file")
        if sf is not None:
            sf.edit.setPlaceholderText("可空：只做 0/1 时不填")
