"""stat_smesh 参数页（常用展开，其余折叠）。"""

from __future__ import annotations

from ..state.form_state import FormState
from ..widgets.field_form import FieldFormTab

_CORE = [
    ("stat.mode", "mode (-L | -M)", "mesh", "combo", ["mesh", "list"]),
    ("stat.cmd_type", "cmd_type", "a", "combo", ["a", "b", "r"]),
    ("stat.list_file", "list_file (-L)", "", "open"),
    ("stat.mesh_file", "mesh_file (-M)", "", "open"),
    ("stat.ave_file", "ave_file (-Cr)", "", "open"),
    ("stat.refl_nnodes", "refl_nnodes (-R)", "", "text"),
    ("stat.ave_x", "ave_x (-Da)", "", "text"),
    ("stat.xmin", "xmin (-Db)", "", "text"),
    ("stat.xmax", "xmax (-Db)", "", "text"),
    ("stat.dx", "dx (-Db)", "", "text"),
    ("stat.window_len", "window_len", "", "text"),
]

_BOUNDS = [
    ("stat.top_bound", "top_bound (-T)", "", "open"),
    ("stat.bot_bound", "bot_bound (-B)", "", "open"),
    ("stat.mid_bound", "mid_bound (-m)", "", "open"),
]

_RANGE = [
    ("stat.pt_corr", "pt_corr (-P)", "", "text"),
    ("stat.vrepl", "vrepl (-U)", "", "text"),
    ("stat.abs_xmin", "abs_xmin (-X)", "", "text"),
    ("stat.abs_xmax", "abs_xmax (-X)", "", "text"),
    ("stat.exclude_cxmin", "exclude_cxmin (-x)", "", "text"),
    ("stat.exclude_cxmax", "exclude_cxmax (-x)", "", "text"),
    ("stat.exclude_top_bound", "exclude_top (-t)", "", "open"),
    ("stat.exclude_bot_bound", "exclude_bot (-b)", "", "open"),
    ("stat.verbose", "verbose (-V)", False, "check"),
]

_SECTIONS = [
    ("常用（模式 / 输入 / 平均）", _CORE, True),
    ("边界文件", _BOUNDS, False),
    ("范围 / 排除 / 其它", _RANGE, False),
]


class StatSmeshTab(FieldFormTab):
    def __init__(self, state: FormState, parent=None) -> None:
        super().__init__(
            state,
            sections=_SECTIONS,
            preview_text="预览 stat_smesh",
            run_text="运行 stat_smesh",
            parent=parent,
        )
        self._combo_keys["stat.mode"].currentTextChanged.connect(self._sync)  # type: ignore[attr-defined]
        self._combo_keys["stat.cmd_type"].currentTextChanged.connect(self._sync)  # type: ignore[attr-defined]
        self._sync()

    def _sync(self) -> None:
        self.binder.pull_from_widgets()
        mode = self.state.get_str("stat.mode")
        cmd = self.state.get_str("stat.cmd_type")
        list_mode = mode == "list"
        mesh_mode = mode == "mesh"
        self.set_enabled_keys(["stat.list_file"], list_mode)
        self.set_enabled_keys(
            [
                "stat.mesh_file",
                "stat.top_bound",
                "stat.bot_bound",
                "stat.mid_bound",
                "stat.exclude_top_bound",
                "stat.exclude_bot_bound",
                "stat.pt_corr",
                "stat.vrepl",
                "stat.abs_xmin",
                "stat.abs_xmax",
                "stat.exclude_cxmin",
                "stat.exclude_cxmax",
            ],
            mesh_mode,
        )
        self.set_enabled_keys(["stat.ave_file"], list_mode and cmd == "r")
        self.set_enabled_keys(
            ["stat.ave_x", "stat.window_len"], cmd == "a" or cmd == "b"
        )
        self.set_enabled_keys(["stat.xmin", "stat.xmax", "stat.dx"], cmd == "b")
