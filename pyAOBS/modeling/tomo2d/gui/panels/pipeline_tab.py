"""pipeline 参数页（常用展开，其余折叠）。"""

from __future__ import annotations

from ..state.form_state import FormState
from ..widgets.field_form import FieldFormTab

_RECIPES = [
    "gen_smesh -> tt_forward",
    "gen_smesh -> tt_forward -> tt_inverse",
    "gen_smesh -> tt_inverse",
    "gen_smesh -> gen_damp -> tt_inverse",
    "gen_smesh -> gen_vcorr -> tt_inverse",
    "gen_smesh -> gen_dcorr -> tt_inverse",
]

_CORE = [
    ("pipe.recipe", "recipe", _RECIPES[0], "combo", _RECIPES),
    ("pipe.auto_wire", "自动衔接到下游参数（目标为空时）", True, "check"),
]

_LINKS = [
    ("pipe.link_smesh", "link_smesh (桥接)", "", "open"),
    ("pipe.link_damp", "link_damp (桥接)", "", "open"),
    ("pipe.link_vcorr_v", "link_vcorr_v (桥接)", "", "open"),
    ("pipe.link_vcorr_d", "link_vcorr_d (桥接)", "", "open"),
    ("pipe.link_dcorr", "link_dcorr (桥接)", "", "open"),
]

_SECTIONS = [
    ("常用", _CORE, True),
    ("桥接路径（一般可留空，由自动衔接填写）", _LINKS, False),
]


class PipelineTab(FieldFormTab):
    def __init__(self, state: FormState, parent=None) -> None:
        super().__init__(
            state,
            sections=_SECTIONS,
            preview_text="预览 pipeline",
            run_text="运行 pipeline",
            parent=parent,
        )
