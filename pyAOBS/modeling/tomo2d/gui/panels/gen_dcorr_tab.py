"""gen_dcorr 参数页：写出 tt_inverse -CD 的 1D 相关长度。"""

from __future__ import annotations

from pyAOBS.utils.qt_combo import connect_combo_deferred

from ..state.form_state import FormState
from ..widgets.field_form import FieldFormTab

_CORE = [
    ("dcorr.mode", "mode", "uniform", "combo", ["uniform", "zelt", "from_vcorr"]),
    ("dcorr.lh", "Lh (-A 均匀)", "", "text"),
    ("dcorr.xmin", "xmin (-D)", "", "text"),
    ("dcorr.xmax", "xmax (-D)", "", "text"),
    ("dcorr.nx", "nx (-N，空=两端点)", "", "text"),
    ("dcorr.out_file", "dcorr 输出文件", "", "save"),
]

_ZELT = [
    ("dcorr.abnormal_d", "abnormal_d (-A 前半)", "", "text"),
    ("dcorr.normal_d", "normal_d (-A 后半)", "", "text"),
    ("dcorr.v_in", "v_in (-C)", "", "open"),
    ("dcorr.ilayer", "ilayer (-C)", "", "text"),
    ("dcorr.top_layer", "top_layer (-F)", "", "text"),
    ("dcorr.bot_layer", "bot_layer (-F)", "", "text"),
    ("dcorr.dx", "dx (-E)", "", "text"),
]

_FROM = [
    ("dcorr.vcorr_file", "vcorr_file (-V)", "", "open"),
    ("dcorr.refl_file", "refl_file (-R)", "", "open"),
]

_SECTIONS = [
    ("常用（均匀 Lh + 输出）", _CORE, True),
    ("Zelt 划区", _ZELT, False),
    ("从二维 vcorr 取样", _FROM, False),
]


class GenDcorrTab(FieldFormTab):
    def __init__(self, state: FormState, parent=None) -> None:
        super().__init__(
            state,
            sections=_SECTIONS,
            preview_text="预览 gen_dcorr",
            run_text="运行 gen_dcorr",
            parent=parent,
        )
        connect_combo_deferred(self._combo_keys["dcorr.mode"], self._sync)  # type: ignore[arg-type]
        self._sync()

    def _sync(self, *_args) -> None:
        mode = self.state.get_str("dcorr.mode") or "uniform"
        uni = mode == "uniform"
        zelt = mode == "zelt"
        fv = mode == "from_vcorr"
        self.set_enabled_keys(
            ["dcorr.lh", "dcorr.xmin", "dcorr.xmax", "dcorr.nx"], uni
        )
        self.set_enabled_keys(
            [
                "dcorr.abnormal_d",
                "dcorr.normal_d",
                "dcorr.v_in",
                "dcorr.ilayer",
                "dcorr.top_layer",
                "dcorr.bot_layer",
                "dcorr.dx",
            ],
            zelt,
        )
        self.set_enabled_keys(["dcorr.vcorr_file"], fv)
        self.set_enabled_keys(["dcorr.refl_file"], fv or zelt)
        self.set_section_expanded(1, zelt)
        self.set_section_expanded(2, fv)
