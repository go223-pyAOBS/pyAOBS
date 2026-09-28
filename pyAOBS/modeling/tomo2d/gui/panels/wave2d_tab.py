# -*- coding: utf-8 -*-
"""wave2d：OBS 为源的弹性道集（接入 run_obs_gather）。"""

from __future__ import annotations

from ..state.form_state import FormState
from ..widgets.field_form import FieldFormTab

_MODEL = [
    ("wave.vp_smesh", "Vp smesh", "true_vp.smesh", "open"),
    ("wave.vs_smesh", "Vs smesh", "true_vs.smesh", "open"),
    ("wave.seafloor", "海底 seafloor", "seafloor.refl", "open"),
    ("wave.out", "输出目录（相对工作目录）", "wave_fwd", "text"),
    ("wave.syn", "射线走时 syn（可空，有则叠点）", "", "open"),
]

_GEOM = [
    ("_grid2", [
        ("wave.obs", "OBS x (km)", "50", "text"),
        ("wave.obs_z", "OBS z (km，源)", "2", "text"),
        ("wave.offset", "偏移 ±km", "80", "text"),
        ("wave.drec", "道距 km", "0.2", "text"),
        ("wave.src_z", "浅水检波参考 z", "0.01", "text"),
        ("wave.water_h", "水深 H (km)", "2", "text"),
        ("wave.water_v", "水速 v (km/s)", "1.5", "text"),
    ]),
]

_WAVE = [
    ("_grid2", [
        ("wave.dx", "网格 dx=dz km", "0.1", "text"),
        ("wave.tmax", "记录时长 s", "22", "text"),
        ("wave.f0", "Ricker f0 Hz", "3", "text"),
        ("wave.vred", "折合速度 km/s", "8", "text"),
        ("wave.tred_max", "折合显示到 s（起点 0）", "12", "text"),
        ("wave.pclip", "pclip", "98", "text"),
    ]),
    ("wave.layout", "几何", "reciprocal", "combo", ["reciprocal", "water-obs"]),
    ("wave.absorb", "吸收边界", "pml", "combo", ["pml", "cerjan"]),
    ("wave.src_kind", "water-obs 源型", "expl", "combo", ["expl", "vz", "vx"]),
    ("wave.skip_ray", "不重跑 tt_forward（有 syn 仍叠点）", True, "check"),
    ("wave.quick", "快速（dx=0.16, tmax=12）", False, "check"),
]


class Wave2dTab(FieldFormTab):
    def __init__(self, state: FormState, parent=None) -> None:
        super().__init__(
            state,
            sections=[
                ("模型", _MODEL, True),
                ("OBS / 水柱", _GEOM, True),
                ("正演与显示", _WAVE, True),
            ],
            preview_text="预览 wave2d",
            run_text="运行 wave2d",
            parent=parent,
        )
