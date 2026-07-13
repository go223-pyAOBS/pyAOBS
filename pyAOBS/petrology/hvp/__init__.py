"""H–Vp diagram utilities (Fig.12a digitized background, track registry)."""

from petrology.hvp.fig12a_background import (
    CHI1_B_DASHED,
    CHI_SOLID,
    DEFAULT_HVP_DIGITIZED,
    H_LIM,
    TP_CONTOURS,
    VP_LIM,
    CurveSeries,
    draw_fig12a_background,
    find_series,
    parse_hvp_digitized,
    smooth_spline,
)

__all__ = [
    "CHI1_B_DASHED",
    "CHI_SOLID",
    "DEFAULT_HVP_DIGITIZED",
    "H_LIM",
    "TP_CONTOURS",
    "VP_LIM",
    "CurveSeries",
    "draw_fig12a_background",
    "find_series",
    "parse_hvp_digitized",
    "smooth_spline",
]
