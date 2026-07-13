"""Backward-compatible wrapper — use hvp_overlay_preview_qt."""

from __future__ import annotations

from petrology.imodel_bridge.export_contract import CrustObservation

from .hvp_overlay_preview_qt import show_hvp_overlay_preview


def show_fig12a_preview(
    parent,
    observation: CrustObservation,
    *,
    delta_vp_max_km_s: float = 0.15,
    thick_crust_h_min_km: float = 15.0,
) -> None:
    show_hvp_overlay_preview(
        parent,
        observation=observation,
        delta_vp_max_km_s=delta_vp_max_km_s,
        thick_crust_h_min_km=thick_crust_h_min_km,
    )
