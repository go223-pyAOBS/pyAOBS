"""Reproduction / Modern track definitions for H–Vp GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from petrology.reference.kkhs02_workflow import (
    TRACK_FIG12_LINEAR_ID,
    TRACK_FIG12_LINEAR_LABEL,
    TRACK_MODERN_ID,
    TRACK_MODERN_LABEL,
)

TrackId = Literal["modern", "fig12_linear"]

HVP_DISPLAY_MODERN = "modern"
HVP_DISPLAY_HVP = "fig12"  # computed H–Vp (eq.1); key kept for compatibility
HVP_DISPLAY_FIG12 = HVP_DISPLAY_HVP  # alias
HVP_DISPLAY_BOTH = "both"

# Default GUI mode: computed H–Vp (KKHS02 Fig.12-style eq.(1) track)
HVP_DISPLAY_DEFAULT = HVP_DISPLAY_HVP

HVP_DISPLAY_CHOICES: tuple[tuple[str, str], ...] = (
    (HVP_DISPLAY_HVP, TRACK_FIG12_LINEAR_LABEL),
    (HVP_DISPLAY_MODERN, TRACK_MODERN_LABEL),
    (HVP_DISPLAY_BOTH, "双轨：H–Vp + Modern"),
)


@dataclass(frozen=True)
class TrackSpec:
    id: TrackId
    label: str
    melting: str
    vp_model: str
    h_geometry: str
    reference: str
    notes: str = ""


TRACK_REGISTRY: dict[str, TrackSpec] = {
    TRACK_MODERN_ID: TrackSpec(
        id="modern",
        label=TRACK_MODERN_LABEL,
        melting="Katz / pymelt REEBOX + isentropic decompression（Step 4 Modern 轨）",
        vp_model="快速扫描：方程 (1)；精修：CIPW+BurnMan norm Vp（Step 1）",
        h_geometry="triangular_crust_thickness_km (χ, b)",
        reference="README Modern M1–M3",
        notes=(
            "程序 **现代正演轨**；Vp 整体常高于默认 H–Vp 计算轨属预期。"
        ),
    ),
    TRACK_FIG12_LINEAR_ID: TrackSpec(
        id="fig12_linear",
        label=TRACK_FIG12_LINEAR_LABEL,
        melting="Step 4：线性 F(P)，(∂F/∂P)_S = 12%/GPa（§4/Fig.11 标准）",
        vp_model="Step 1 方程 (1) V_bulk(P̄, F̄)（生产 b1=-0.55）",
        h_geometry="active_upwelling sweep_hvp（reproduce_fig12.py）",
        reference="KKHS02 Fig.12 方法；reproduce_fig12.py",
        notes=(
            "**默认 H–Vp 图**：公式 (1) 计算轨。"
            "印刷数字化 Fig.12a 仅保留在 validation/ 对照，不进入 GUI。"
        ),
    ),
}


def track_spec(track_id: str) -> TrackSpec | None:
    return TRACK_REGISTRY.get(track_id)
