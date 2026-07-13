"""Backward-compatible wrapper — use hvp_overlay_preview_qt."""

from __future__ import annotations

from petrology.seismic.transect import TransectWindow

from .hvp_overlay_preview_qt import show_hvp_overlay_preview


def show_fig15c_preview(
    parent,
    windows: list[TransectWindow],
    *,
    h_min_km: float = 15.0,
    delta_vp_max_km_s: float = 0.15,
) -> None:
    show_hvp_overlay_preview(
        parent,
        windows=windows,
        delta_vp_max_km_s=delta_vp_max_km_s,
        thick_crust_h_min_km=h_min_km,
    )
