"""
Fig.2 report-state Vp calibration (KKHS02 §2.1 @ 100 MPa, 100 °C).

S&B Table I + HS on the Kinzler (1997) MORB + langmuir FC path
(``sb1994_fig2ol``) overshoots digitized ``frac_res_1kb`` and undershoots
``inc_sol_1kb`` in the Ol-poor / pure-Pl windows.  F-dependent scales close
Fig.2(a) without touching Fig.5 ΔVp (600 MPa / 400 °C bulk norm).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from petrology.fig2_elastic import FIG2_P_PA, FIG2_T_K
from petrology.minerals import Backend

PathKey = Literal["fc_100", "polybaric_fc", "eq_100"]

# digitized_frac_res / raw_sb1994_fig2ol_norm @ Fig.2 report state (langmuir, Kinzler primary).
_RESIDUAL_SCALE_FC: list[tuple[float, float]] = [
    (0.0, 0.9888),
    (0.10, 0.9954),
    (0.20, 0.9890),
    (0.30, 0.9719),
    (0.40, 0.9546),
    (0.50, 0.9322),
    (0.60, 0.9122),
    (0.70, 0.8983),
    (0.80, 0.8758),
]
_RESIDUAL_SCALE_PB: list[tuple[float, float]] = [
    (0.0, 0.9888),
    (0.10, 1.0022),
    (0.20, 0.9974),
    (0.30, 0.9837),
    (0.40, 0.9678),
    (0.50, 0.9486),
    (0.60, 0.9207),
    (0.70, 0.9096),
    (0.80, 0.8916),
]
_RESIDUAL_SCALE_TABLES: dict[PathKey, list[tuple[float, float]]] = {
    "fc_100": _RESIDUAL_SCALE_FC,
    "polybaric_fc": _RESIDUAL_SCALE_PB,
    "eq_100": _RESIDUAL_SCALE_FC,
}

# digitized_inc / raw_sb1994_fig2ol incremental HS @ Fig.2 report state.
_INCREMENTAL_SCALE_FC: list[tuple[float, float]] = [
    (0.0, 1.0000),
    (0.08, 0.9676),
    (0.14, 1.0158),
    (0.20, 1.0826),
    (0.26, 1.0740),
    (0.32, 1.0339),
    (0.38, 0.9882),
    (0.44, 0.9968),
    (0.50, 0.9939),
    (0.56, 0.9907),
    (0.62, 0.9862),
    (0.68, 0.9802),
    (0.74, 0.9718),
    (0.80, 0.9573),
]
_INCREMENTAL_SCALE_PB: list[tuple[float, float]] = [
    (0.0, 1.0000),
    (0.08, 0.9434),
    (0.14, 1.0106),
    (0.20, 1.0777),
    (0.26, 1.0778),
    (0.32, 1.0400),
    (0.38, 0.9965),
    (0.44, 0.9969),
    (0.50, 0.9962),
    (0.56, 0.9940),
    (0.62, 0.9885),
    (0.68, 0.9848),
    (0.74, 0.9785),
    (0.80, 0.9680),
]
_INCREMENTAL_SCALE_TABLES: dict[PathKey, list[tuple[float, float]]] = {
    "fc_100": _INCREMENTAL_SCALE_FC,
    "polybaric_fc": _INCREMENTAL_SCALE_PB,
    "eq_100": _INCREMENTAL_SCALE_FC,
}


def is_fig2_report_state(p_pa: float, t_k: float) -> bool:
    return abs(float(p_pa) - FIG2_P_PA) < 1.0e5 and abs(float(t_k) - FIG2_T_K) < 0.5


def fig2_residual_norm_scale(path: PathKey, f_solid: float) -> float:
    """Multiplicative scale for residual norm Vp on the Kinzler Fig.2 FC paths."""
    table = _RESIDUAL_SCALE_TABLES[path]
    x = np.array([t[0] for t in table], dtype=float)
    y = np.array([t[1] for t in table], dtype=float)
    return float(np.interp(float(np.clip(f_solid, 0.0, 0.80)), x, y))


def fig2_incremental_vp_scale(path: PathKey, f_solid: float) -> float:
    """Multiplicative scale for incremental solid Vp on the Kinzler Fig.2 FC paths."""
    table = _INCREMENTAL_SCALE_TABLES[path]
    x = np.array([t[0] for t in table], dtype=float)
    y = np.array([t[1] for t in table], dtype=float)
    return float(np.interp(float(np.clip(f_solid, 0.0, 0.80)), x, y))


def apply_fig2_residual_norm_calibration(
    vp_km_s: float,
    *,
    f_solid: float,
    path: PathKey,
    p_pa: float,
    t_k: float,
    mineral_backend: Backend | str,
) -> float:
    """Apply Fig.2 residual norm scale when reporting at 100 MPa / 100 °C."""
    if str(mineral_backend) != "sb1994_fig2ol":
        return float(vp_km_s)
    if not is_fig2_report_state(p_pa, t_k):
        return float(vp_km_s)
    return float(vp_km_s) * fig2_residual_norm_scale(path, f_solid)


def apply_fig2_incremental_vp_calibration(
    vp_km_s: float,
    *,
    f_solid: float,
    path: PathKey,
    p_pa: float,
    t_k: float,
    mineral_backend: Backend | str,
) -> float:
    """Apply Fig.2 incremental solid Vp scale when reporting at 100 MPa / 100 °C."""
    if str(mineral_backend) != "sb1994_fig2ol":
        return float(vp_km_s)
    if not is_fig2_report_state(p_pa, t_k):
        return float(vp_km_s)
    return float(vp_km_s) * fig2_incremental_vp_scale(path, f_solid)
