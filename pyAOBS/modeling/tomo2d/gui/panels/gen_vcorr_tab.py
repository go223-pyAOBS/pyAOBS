"""gen_vcorr 参数页：简单 2×2 顶/底，或调用 gen_vcorr 二进制。"""

from __future__ import annotations

from pyAOBS.utils.qt_combo import connect_combo_deferred

from ..state.form_state import FormState
from ..widgets.field_form import FieldFormTab
from .vel_grid_sync import sync_vel_grid_options

_CORE = [
    ("vcorr.mode", "mode", "simple_2x2", "combo", ["simple_2x2", "program"]),
    (
        "_row",
        [
            ("vcorr.Lht", "Lht 水平顶", "2.0", "text"),
            ("vcorr.Lhb", "Lhb 水平底", "4.0", "text"),
        ],
    ),
    (
        "_row",
        [
            ("vcorr.Lvt", "Lvt 垂直顶", "1.0", "text"),
            ("vcorr.Lvb", "Lvb 垂直底", "4.0", "text"),
        ],
    ),
    (
        "_row",
        [
            ("vcorr.xmin", "xmin", "0", "text"),
            ("vcorr.xmax", "xmax", "", "text"),
        ],
    ),
    (
        "_row",
        [
            ("vcorr.zmin", "zmin", "0", "text"),
            ("vcorr.zmax", "zmax", "", "text"),
        ],
    ),
    ("vcorr.out_file", "vcorr 输出文件", "", "save"),
]

_PROGRAM = [
    ("vcorr.vel_opt", "vel_opt", "uniform", "combo", ["uniform", "zelt"]),
    ("vcorr.grid_opt", "grid_opt", "uniform", "combo", ["uniform", "variable", "zelt"]),
    ("vcorr.abnormal_h", "abnormal_h (-A 1/4)", "", "text"),
    ("vcorr.abnormal_v", "abnormal_v (-A 2/4)", "", "text"),
    ("vcorr.normal_h", "normal_h (-A 3/4)", "", "text"),
    ("vcorr.normal_v", "normal_v (-A 4/4)", "", "text"),
    ("vcorr.nx", "nx (-N)", "", "text"),
    ("vcorr.nz", "nz (-N)", "", "text"),
]

_ZELT = [
    ("vcorr.v_in", "v_in (-C)", "", "open"),
    ("vcorr.ilayer", "ilayer (-C)", "", "text"),
    ("vcorr.top_layer", "top_layer (-F)", "", "text"),
    ("vcorr.bot_layer", "bot_layer (-F)", "", "text"),
]

_GRID_ADV = [
    ("vcorr.x_file", "x_file (-X)", "", "open"),
    ("vcorr.z_file", "z_file (-Z)", "", "open"),
    ("vcorr.topo_file", "topo_file (-T)", "", "open"),
    ("vcorr.dx", "dx (-E)", "", "text"),
]

_SECTIONS = [
    ("常用：2×2 顶/底相关长度（或选 program）", _CORE, True),
    ("program：gen_vcorr 二进制（-A 划区）", _PROGRAM, False),
    ("Zelt 速度输入", _ZELT, False),
    ("非均匀网格（variable / zelt）", _GRID_ADV, False),
]

_SIMPLE_KEYS = [
    "vcorr.Lht",
    "vcorr.Lhb",
    "vcorr.Lvt",
    "vcorr.Lvb",
    "vcorr.xmin",
    "vcorr.xmax",
    "vcorr.zmin",
    "vcorr.zmax",
]
_PROGRAM_A_KEYS = [
    "vcorr.abnormal_h",
    "vcorr.abnormal_v",
    "vcorr.normal_h",
    "vcorr.normal_v",
]


class GenVcorrTab(FieldFormTab):
    def __init__(self, state: FormState, parent=None) -> None:
        super().__init__(
            state,
            sections=_SECTIONS,
            preview_text="预览 gen_vcorr",
            run_text="运行 gen_vcorr",
            parent=parent,
        )
        connect_combo_deferred(self._combo_keys["vcorr.mode"], self._sync)  # type: ignore[arg-type]
        connect_combo_deferred(self._combo_keys["vcorr.vel_opt"], self._sync)  # type: ignore[arg-type]
        connect_combo_deferred(self._combo_keys["vcorr.grid_opt"], self._sync)  # type: ignore[arg-type]
        self._sync()

    def _sync(self, *_args) -> None:
        mode_combo = self._combo_keys.get("vcorr.mode")
        if mode_combo is not None:
            mode = str(mode_combo.currentText() or "").strip()
            self.state.set("vcorr.mode", mode)
        else:
            mode = self.state.get_str("vcorr.mode") or "simple_2x2"
        simple = mode == "simple_2x2"
        self.set_enabled_keys(_SIMPLE_KEYS, simple)
        self.set_enabled_keys(["vcorr.out_file"], True)
        self.set_enabled_keys(["vcorr.vel_opt", "vcorr.grid_opt"], not simple)
        self.set_enabled_keys(_PROGRAM_A_KEYS, not simple)
        self.set_section_expanded(1, not simple)
        if simple:
            self.set_enabled_keys(
                [
                    "vcorr.nx",
                    "vcorr.nz",
                    "vcorr.v_in",
                    "vcorr.ilayer",
                    "vcorr.top_layer",
                    "vcorr.bot_layer",
                    "vcorr.x_file",
                    "vcorr.z_file",
                    "vcorr.topo_file",
                    "vcorr.dx",
                ],
                False,
            )
            self.set_section_expanded(2, False)
            self.set_section_expanded(3, False)
            return
        # program：xmax/zmax 在常用区，由 vel_grid_sync 按 uniform 开关
        sync_vel_grid_options(
            self,
            vel_key="vcorr.vel_opt",
            grid_key="vcorr.grid_opt",
            uniform_vel_keys=[],
            zelt_vel_keys=[
                "vcorr.v_in",
                "vcorr.ilayer",
                "vcorr.top_layer",
                "vcorr.bot_layer",
            ],
            uniform_grid_keys=["vcorr.nx", "vcorr.nz", "vcorr.xmax", "vcorr.zmax"],
            variable_grid_keys=["vcorr.x_file", "vcorr.z_file", "vcorr.topo_file"],
            zelt_grid_keys=["vcorr.z_file", "vcorr.dx"],
            zelt_section=2,
            grid_section=3,
        )
        self.set_enabled_keys(_PROGRAM_A_KEYS, True)
