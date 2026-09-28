"""gen_damp 参数页（常用展开，其余折叠）。"""

from __future__ import annotations

from pyAOBS.utils.qt_combo import connect_combo_deferred

from ..state.form_state import FormState
from ..widgets.field_form import FieldFormTab
from .vel_grid_sync import sync_vel_grid_options

_CORE = [
    ("damp.vel_opt", "vel_opt", "uniform", "combo", ["uniform", "zelt"]),
    ("damp.grid_opt", "grid_opt", "uniform", "combo", ["uniform", "variable", "zelt"]),
    ("damp.abnormal_damp", "abnormal_damp (-A 前半)", "", "text"),
    ("damp.normal_damp", "normal_damp (-A 后半)", "", "text"),
    ("damp.nx", "nx (-N)", "", "text"),
    ("damp.nz", "nz (-N)", "", "text"),
    ("damp.xmax", "xmax (-D)", "", "text"),
    ("damp.zmax", "zmax (-D)", "", "text"),
]

_ZELT = [
    ("damp.v_in", "v_in (-C)", "", "open"),
    ("damp.ilayer", "ilayer (-C)", "", "text"),
    ("damp.top_layer", "top_layer (-F)", "", "text"),
    ("damp.bot_layer", "bot_layer (-F)", "", "text"),
]

_GRID_ADV = [
    ("damp.x_file", "x_file (-X)", "", "open"),
    ("damp.z_file", "z_file (-Z)", "", "open"),
    ("damp.topo_file", "topo_file (-T)", "", "open"),
    ("damp.dx", "dx (-E)", "", "text"),
]

_SECTIONS = [
    ("常用（均匀阻尼 + 均匀网格）", _CORE, True),
    ("Zelt 速度输入", _ZELT, False),
    ("非均匀网格（variable / zelt）", _GRID_ADV, False),
]


class GenDampTab(FieldFormTab):
    def __init__(self, state: FormState, parent=None) -> None:
        super().__init__(
            state,
            sections=_SECTIONS,
            preview_text="预览 gen_damp",
            run_text="运行 gen_damp",
            parent=parent,
        )
        connect_combo_deferred(self._combo_keys["damp.vel_opt"], self._sync)  # type: ignore[arg-type]
        connect_combo_deferred(self._combo_keys["damp.grid_opt"], self._sync)  # type: ignore[arg-type]
        self._sync()

    def _sync(self, *_args) -> None:
        # -A 阻尼值始终可填；zelt 时 -C/-F 只划异常区几何（见 gen_damp.cc）
        sync_vel_grid_options(
            self,
            vel_key="damp.vel_opt",
            grid_key="damp.grid_opt",
            uniform_vel_keys=[],
            zelt_vel_keys=[
                "damp.v_in",
                "damp.ilayer",
                "damp.top_layer",
                "damp.bot_layer",
            ],
            uniform_grid_keys=["damp.nx", "damp.nz", "damp.xmax", "damp.zmax"],
            variable_grid_keys=["damp.x_file", "damp.z_file", "damp.topo_file"],
            zelt_grid_keys=["damp.z_file", "damp.dx"],
            zelt_section=1,
            grid_section=2,
        )
        self.set_enabled_keys(
            ["damp.abnormal_damp", "damp.normal_damp"], True
        )
