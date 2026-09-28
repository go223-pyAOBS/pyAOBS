"""edit_smesh_HHB 参数页（常用展开，其余折叠）。"""

from __future__ import annotations

from ..state.form_state import FormState
from ..widgets.field_form import FieldFormTab

_CMDS = ["a", "p", "P", "B", "s", "rm", "c", "d", "g", "l", "R", "S", "G", "m", "b"]

_CORE = [
    ("edit.cmd_type", "cmd_type (-C)", "a", "combo", _CMDS),
    ("edit.smesh_file", "smesh_file", "", "open"),
]

# 随 cmd_type 启用；默认展开以便看到当前命令所需项
_CMD_PARAMS = [
    ("edit.paste_file", "paste_file", "", "open"),
    ("edit.prof_file", "prof_file", "", "open"),
    ("edit.remove_bg_file", "remove_bg_prof", "", "open"),
    ("edit.h_len", "h_len", "", "text"),
    ("edit.v_len", "v_len", "", "text"),
    ("edit.mx", "mx", "", "text"),
    ("edit.mz", "mz", "", "text"),
    ("edit.amp", "amp", "", "text"),
    ("edit.xmin", "xmin", "", "text"),
    ("edit.xmax", "xmax", "", "text"),
    ("edit.zmin", "zmin", "", "text"),
    ("edit.zmax", "zmax", "", "text"),
    ("edit.x0", "x0", "", "text"),
    ("edit.z0", "z0", "", "text"),
    ("edit.Lh", "Lh", "", "text"),
    ("edit.Lv", "Lv", "", "text"),
    ("edit.seed", "seed", "", "text"),
    ("edit.nrand", "nrand", "", "text"),
    ("edit.N", "N", "", "text"),
    ("edit.dx", "dx", "", "text"),
    ("edit.dz", "dz", "", "text"),
    ("edit.vel", "vel", "", "text"),
    ("edit.moho_file", "moho_file", "", "open"),
    ("edit.k", "k", "", "text"),
    ("edit.base_file", "base_file", "", "open"),
]

_BOUNDS = [
    ("edit.corr_file", "corr_file (-L)", "", "open"),
    ("edit.upper_bound", "upper_bound (-U)", "", "open"),
]

_MAP: dict[str, list[str]] = {
    "a": [],
    "p": ["edit.paste_file"],
    "P": ["edit.prof_file"],
    "B": ["edit.remove_bg_file"],
    "s": ["edit.h_len", "edit.v_len"],
    "rm": ["edit.mx", "edit.mz"],
    "c": ["edit.amp", "edit.h_len", "edit.v_len"],
    "d": ["edit.amp", "edit.xmin", "edit.xmax", "edit.zmin", "edit.zmax"],
    "g": ["edit.amp", "edit.x0", "edit.z0", "edit.Lh", "edit.Lv"],
    "l": [],
    "R": ["edit.seed", "edit.amp", "edit.nrand"],
    "S": [
        "edit.seed",
        "edit.amp",
        "edit.xmin",
        "edit.xmax",
        "edit.dx",
        "edit.zmin",
        "edit.zmax",
        "edit.dz",
    ],
    "G": [
        "edit.seed",
        "edit.amp",
        "edit.N",
        "edit.xmin",
        "edit.xmax",
        "edit.zmin",
        "edit.zmax",
    ],
    "m": ["edit.vel", "edit.moho_file"],
    "b": ["edit.k", "edit.base_file"],
}

_ALL_OPTIONAL = [spec[0] for spec in _CMD_PARAMS]

_SECTIONS = [
    ("常用", _CORE, True),
    ("命令参数（随 cmd_type 启用）", _CMD_PARAMS, True),
    ("约束文件（-L / -U）", _BOUNDS, False),
]


class EditSmeshTab(FieldFormTab):
    def __init__(self, state: FormState, parent=None) -> None:
        super().__init__(
            state,
            sections=_SECTIONS,
            preview_text="预览 edit_smesh_HHB",
            run_text="运行 edit_smesh_HHB",
            parent=parent,
        )
        self._combo_keys["edit.cmd_type"].currentTextChanged.connect(self._sync)  # type: ignore[attr-defined]
        self._sync()

    def _sync(self) -> None:
        self.binder.pull_from_widgets()
        cmd = self.state.get_str("edit.cmd_type")
        need = set(_MAP.get(cmd, []))
        for key in _ALL_OPTIONAL:
            self.set_enabled_keys([key], key in need)
        self.set_enabled_keys(
            ["edit.smesh_file", "edit.corr_file", "edit.upper_bound"], True
        )
