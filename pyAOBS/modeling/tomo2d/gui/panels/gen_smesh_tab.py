"""gen_smesh 参数页（常用展开，其余折叠）。"""

from __future__ import annotations

from PySide6.QtCore import QTimer

from pyAOBS.utils.qt_combo import connect_combo_deferred

from ..state.form_state import FormState
from ..widgets.field_form import FieldFormTab
from .vel_grid_sync import sync_vel_grid_options

_CORE = [
    ("gen.vel_opt", "vel_opt", "uniform", "combo", ["uniform", "zelt", "help"]),
    ("gen.grid_opt", "grid_opt", "uniform", "combo", ["uniform", "variable", "zelt"]),
    ("gen.v0", "v0", "1.5", "text"),
    ("gen.gradient", "gradient", "0.1", "text"),
    ("gen.nx", "nx (-N)", "101", "text"),
    ("gen.nz", "nz (-N)", "51", "text"),
    ("gen.xmax", "xmax (-D)", "100", "text"),
    ("gen.zmax", "zmax (-D)", "30", "text"),
    ("gen.smesh_out", "smesh 输出文件", "", "save"),
]

_ZELT = [
    ("gen.v_in", "v_in (-C)", "", "open"),
    ("gen.ilayer", "ilayer", "", "text"),
    ("gen.refl_layer", "refl_layer", "", "text"),
    ("gen.refl_file", "refl_file (-F 输出)", "", "save"),
    ("gen.hang_sea_surface", "挂海面 topo=0 (-S)", False, "check"),
    ("gen.seafloor_out", "seafloor_out (-G 海底输出)", "", "save"),
    ("gen.zelt_dump", "zelt_dump (-d 输出)", "", "save"),
]

_GRID_ADV = [
    ("gen.x_file", "x_file (-X)", "", "open"),
    ("gen.z_file", "z_file (-Z)", "", "open"),
    ("gen.topo_file", "topo_file (-T)", "", "open"),
    ("gen.dx", "dx (-E)", "", "text"),
]

# 与 gen_smesh.cc 默认一致：wcol=0、v_water=1.5、v_air=0.33
_WATER_DEFAULTS = {
    "gen.water_col": "0",
    "gen.v_water": "1.5",
    "gen.v_air": "0.33",
}
_WATER = [
    ("gen.water_col", "water_col (-W)", _WATER_DEFAULTS["gen.water_col"], "text"),
    ("gen.v_water", "v_water (-Q)", _WATER_DEFAULTS["gen.v_water"], "text"),
    ("gen.v_air", "v_air (-R)", _WATER_DEFAULTS["gen.v_air"], "text"),
]

_SECTIONS = [
    ("常用（均匀速度 + 均匀网格）", _CORE, True),
    ("Zelt 速度输入", _ZELT, False),
    ("非均匀网格（variable / zelt）", _GRID_ADV, False),
    ("水柱 / 空气速度", _WATER, False),
]


class GenSmeshTab(FieldFormTab):
    def __init__(self, state: FormState, parent=None) -> None:
        super().__init__(
            state,
            sections=_SECTIONS,
            preview_text="预览 gen_smesh",
            run_text="运行 gen_smesh",
            parent=parent,
        )
        # activated：用户点选；currentTextChanged：恢复配置/程序改值后也同步启用状态
        connect_combo_deferred(self._combo_keys["gen.vel_opt"], self._sync)  # type: ignore[arg-type]
        connect_combo_deferred(self._combo_keys["gen.grid_opt"], self._sync)  # type: ignore[arg-type]
        connect_combo_deferred(  # type: ignore[arg-type]
            self._combo_keys["gen.vel_opt"],
            self._sync,
            signal="currentTextChanged",
        )
        connect_combo_deferred(  # type: ignore[arg-type]
            self._combo_keys["gen.grid_opt"],
            self._sync,
            signal="currentTextChanged",
        )
        self._sync()
        # 恢复工区在 ~150ms push；再刷启用状态（并补空水速默认）
        QTimer.singleShot(200, self._sync)
        prow = self._path_rows.get("gen.seafloor_out")
        if prow is not None:
            prow.edit.setPlaceholderText("可空：勾选 -S 后写出给 -Y/-B")
            prow.edit.textChanged.connect(self._on_seafloor_out_changed)

    def on_state_pushed(self) -> None:
        self._on_seafloor_out_changed()
        self._sync()

    def _on_seafloor_out_changed(self, *_args) -> None:
        prow = self._path_rows.get("gen.seafloor_out")
        if prow is None or not str(prow.edit.text() or "").strip():
            return
        box = self._check_keys.get("gen.hang_sea_surface")
        if box is not None and not box.isChecked():
            box.setChecked(True)

    def _ensure_water_defaults(self) -> None:
        """旧 profile 里常为空串；补成与 C++ 一致的默认，便于直接跑。"""
        for k, v in _WATER_DEFAULTS.items():
            if self.state.get_str(k):
                continue
            self.state.set(k, v)
            le = self._line_edits.get(k)
            if le is not None:
                le.setText(v)  # type: ignore[union-attr]

    def _sync(self, *_args) -> None:
        self._ensure_water_defaults()
        sync_vel_grid_options(
            self,
            vel_key="gen.vel_opt",
            grid_key="gen.grid_opt",
            uniform_vel_keys=["gen.v0", "gen.gradient"],
            zelt_vel_keys=[
                "gen.v_in",
                "gen.ilayer",
                "gen.refl_layer",
                "gen.refl_file",
                "gen.hang_sea_surface",
                "gen.seafloor_out",
                "gen.zelt_dump",
            ],
            uniform_grid_keys=["gen.nx", "gen.nz", "gen.xmax", "gen.zmax"],
            variable_grid_keys=["gen.x_file", "gen.z_file", "gen.topo_file"],
            zelt_grid_keys=["gen.z_file", "gen.dx"],
            zelt_section=1,
            grid_section=2,
        )
