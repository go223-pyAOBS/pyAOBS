"""Public imodel bridge API."""

from petrology.imodel_bridge.crust_metrics import (
    metrics_from_depth_samples,
    metrics_from_lower_crust_polygon,
    thickness_from_interfaces,
)
from petrology.imodel_bridge.export_contract import (
    CrustObservation,
    Fig12aOverlayRequest,
    load_observation_json,
    save_observation_json,
)
from petrology.imodel_bridge.fig12a_overlay import draw_fig12a_with_observation
from petrology.imodel_bridge.imodel_adapter import (
    crust_from_interfaces_at_x,
    crust_from_polygon_selection,
    crust_from_vertical_profile,
    crust_thickness_at_x,
    format_crust_observation_summary,
)

__all__ = [
    "CrustObservation",
    "Fig12aOverlayRequest",
    "crust_from_interfaces_at_x",
    "crust_from_polygon_selection",
    "crust_from_vertical_profile",
    "crust_thickness_at_x",
    "draw_fig12a_with_observation",
    "format_crust_observation_summary",
    "load_observation_json",
    "metrics_from_depth_samples",
    "metrics_from_lower_crust_polygon",
    "save_observation_json",
    "thickness_from_interfaces",
]
