"""imodel → Fig.12a overlay helpers."""

from __future__ import annotations

from pathlib import Path

from petrology.hvp.fig12a_background import (
    DEFAULT_HVP_DIGITIZED,
    H_LIM,
    VP_LIM,
    draw_fig12a_background,
    parse_hvp_digitized,
)
from petrology.hvp.observation_overlay import ObservationPoint, plot_observation_on_hvp
from petrology.imodel_bridge.export_contract import CrustObservation, Fig12aOverlayRequest


def crust_observation_to_point(
    obs: CrustObservation,
    *,
    h_min_km: float = 15.0,
) -> ObservationPoint:
    thick = obs.h_whole_km > h_min_km
    label = f"imodel ({obs.source})"
    if obs.x_km is not None:
        label += f" x={obs.x_km:.0f} km"
    return ObservationPoint(
        h_km=obs.h_whole_km,
        v_lc_km_s=obs.v_lc_km_s,
        v_lc_sigma_km_s=obs.v_lc_sigma_km_s,
        thick_crust=thick,
        label=label,
    )


def draw_fig12a_with_observation(
    ax,
    request: Fig12aOverlayRequest,
    *,
    digitized_path: Path | None = None,
) -> None:
    """Draw Fig.12a background + single imodel observation.

    Kept independent of ``fig15c_plot`` to avoid a circular import
    (``fig15c_plot`` already imports ``crust_observation_to_point`` from here).
    """
    path = digitized_path or DEFAULT_HVP_DIGITIZED
    series = parse_hvp_digitized(path)
    draw_fig12a_background(ax, series, h_lim=H_LIM, vp_lim=VP_LIM, show_legend=True)
    pt = crust_observation_to_point(
        request.observation,
        h_min_km=request.thick_crust_h_min_km,
    )
    plot_observation_on_hvp(
        ax,
        pt,
        delta_vp_max_km_s=request.delta_vp_max_km_s,
        h_min_km=request.thick_crust_h_min_km,
        show_read_band=request.read_tp and (pt.thick_crust or False),
    )
