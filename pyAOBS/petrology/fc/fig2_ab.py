"""
KKHS02 Figure 2 (a,b) calibration anchors.

Digitized cumulative phase proportions and mineral compositions from the paper
(Fig. 2c–h), used to drive incremental FC modes so cumulate Vp/rho match panels a–b.

Panel (a,b) Vp: default ``fig2_ab_calibrate`` uses paper phases (c-h) + **computed** HS/CIPW
(``fig2_ab_anchor_vp=False``).  Set ``fig2_ab_anchor_vp=True`` to interpolate sparse anchors
for figure layout only.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

PathKey = Literal["fc_100", "polybaric_fc", "eq_100"]

# Panel (a): cumulative / equilibrium solid Vp (km/s) @ 100 MPa, 100 °C — sparse calibration anchors.
_VP_CUM_FC: list[tuple[float, float]] = [
    (0.0, 7.57), (0.08, 7.56), (0.25, 7.55), (0.50, 7.42), (0.70, 7.34), (0.80, 7.31),
]
_VP_CUM_PB: list[tuple[float, float]] = [
    (0.0, 7.55), (0.10, 7.53), (0.25, 7.51), (0.50, 7.38), (0.70, 7.30), (0.80, 7.27),
]
_VP_CUM_EQ: list[tuple[float, float]] = [
    (0.0, 7.50), (0.20, 7.43), (0.40, 7.35), (0.60, 7.29), (0.80, 7.24),
]

# Panel (a): incremental solid Vp.
_VP_INC_FC: list[tuple[float, float]] = [
    (0.0, 7.22), (0.22, 7.22), (0.32, 7.20), (0.40, 7.28), (0.55, 7.25), (0.72, 7.10), (0.80, 6.98),
]
_VP_INC_PB: list[tuple[float, float]] = [
    (0.0, 7.20), (0.22, 7.20), (0.35, 7.18), (0.55, 7.12), (0.72, 7.02), (0.80, 6.95),
]
_VP_INC_EQ: list[tuple[float, float]] = [
    (0.0, 7.22), (0.22, 7.22), (0.40, 7.28), (0.55, 7.25), (0.72, 7.10), (0.80, 6.98),
]

# Residual liquid norm Vp (fractional crystallization overlays).
_VP_RES_FC: list[tuple[float, float]] = [
    (0.0, 7.15), (0.20, 7.08), (0.40, 7.04), (0.60, 6.98), (0.80, 6.90),
]
_VP_RES_PB: list[tuple[float, float]] = [
    (0.0, 7.15), (0.20, 7.00), (0.40, 6.88), (0.60, 6.80), (0.80, 6.70),
]

# Panel (b): cumulative solid density (g/cm³).
_RHO_CUM_FC: list[tuple[float, float]] = [
    (0.0, 3.33), (0.05, 3.34), (0.12, 3.10), (0.30, 2.99), (0.55, 3.00), (0.80, 3.02),
]
_RHO_CUM_PB: list[tuple[float, float]] = [
    (0.0, 3.31), (0.05, 3.33), (0.12, 3.05), (0.30, 2.97), (0.55, 2.98), (0.80, 3.00),
]
_RHO_CUM_EQ: list[tuple[float, float]] = [
    (0.0, 3.33), (0.12, 3.10), (0.30, 2.99), (0.55, 3.00), (0.80, 3.02),
]

# Panel (b): incremental solid density.
_RHO_INC_FC: list[tuple[float, float]] = [
    (0.0, 2.72), (0.40, 2.72), (0.60, 2.74), (0.80, 2.80),
]
_RHO_INC_PB: list[tuple[float, float]] = [
    (0.0, 2.71), (0.40, 2.70), (0.60, 2.73), (0.80, 2.78),
]

# Residual liquid density for panel (b).
_RHO_RES_FC: list[tuple[float, float]] = [
    (0.0, 2.98), (0.20, 2.95), (0.40, 2.97), (0.60, 2.98), (0.80, 3.00),
]
_RHO_RES_PB: list[tuple[float, float]] = [
    (0.0, 2.98), (0.20, 2.94), (0.40, 2.96), (0.60, 2.97), (0.80, 2.99),
]

_RHO_RES_TABLES: dict[PathKey, list[tuple[float, float]]] = {
    "fc_100": _RHO_RES_FC,
    "polybaric_fc": _RHO_RES_PB,
    "eq_100": _RHO_RES_FC,
}

_VP_CUM_TABLES: dict[PathKey, list[tuple[float, float]]] = {
    "fc_100": _VP_CUM_FC,
    "polybaric_fc": _VP_CUM_PB,
    "eq_100": _VP_CUM_EQ,
}
_VP_INC_TABLES: dict[PathKey, list[tuple[float, float]]] = {
    "fc_100": _VP_INC_FC,
    "polybaric_fc": _VP_INC_PB,
    "eq_100": _VP_INC_EQ,
}
_VP_RES_TABLES: dict[PathKey, list[tuple[float, float]]] = {
    "fc_100": _VP_RES_FC,
    "polybaric_fc": _VP_RES_PB,
    "eq_100": _VP_RES_FC,
}

_RHO_CUM_TABLES: dict[PathKey, list[tuple[float, float]]] = {
    "fc_100": _RHO_CUM_FC,
    "polybaric_fc": _RHO_CUM_PB,
    "eq_100": _RHO_CUM_EQ,
}
_RHO_INC_TABLES: dict[PathKey, list[tuple[float, float]]] = {
    "fc_100": _RHO_INC_FC,
    "polybaric_fc": _RHO_INC_PB,
    "eq_100": _RHO_INC_FC,
}

_LEGACY_VP_CUM_TABLES = _VP_CUM_TABLES
_LEGACY_VP_INC_TABLES = _VP_INC_TABLES
_LEGACY_VP_RES_TABLES = _VP_RES_TABLES

# Cumulative solid phase wt% (Ol, Cpx, Pl) — corrected F=0 → 100% Ol for FC.
_CUM_FC: list[tuple[float, float, float, float]] = [
    (0.00, 100.0, 0.0, 0.0),
    (0.08, 82.0, 0.0, 18.0),
    (0.22, 35.0, 0.0, 65.0),
    (0.40, 12.0, 20.0, 68.0),
    (0.55, 12.0, 28.0, 60.0),
    (0.70, 13.0, 34.0, 53.0),
    (0.80, 13.0, 36.0, 51.0),
]

_CUM_PB: list[tuple[float, float, float, float]] = [
    (0.00, 100.0, 0.0, 0.0),
    (0.15, 32.0, 0.0, 68.0),
    (0.28, 24.0, 10.0, 66.0),
    (0.45, 15.0, 28.0, 57.0),
    (0.60, 15.0, 36.0, 49.0),
    (0.80, 15.0, 38.0, 47.0),
]

_CUM_EQ: list[tuple[float, float, float, float]] = [
    (0.00, 100.0, 0.0, 0.0),
    (0.08, 82.0, 0.0, 18.0),
    (0.15, 68.0, 1.0, 31.0),
    (0.25, 55.0, 3.0, 42.0),
    (0.40, 47.0, 8.0, 45.0),
    (0.60, 40.0, 15.0, 45.0),
    (0.80, 33.0, 20.0, 47.0),
]

# Incremental mineral Fo, An, Di (wt%) — panels f/g/h.
_INC_FO_FC = [(0.0, 90), (0.15, 88), (0.30, 85), (0.45, 81), (0.60, 72), (0.72, 63), (0.80, 58)]
_INC_AN_FC = [(0.0, 80), (0.15, 77), (0.30, 73), (0.45, 69), (0.60, 62), (0.72, 57), (0.80, 54)]
_INC_DI_FC = [(0.0, 90), (0.20, 89), (0.35, 87), (0.45, 86), (0.55, 82), (0.65, 72), (0.80, 58)]

_INC_FO_PB = [(0.0, 90), (0.15, 88), (0.30, 86), (0.45, 82), (0.60, 74), (0.72, 65), (0.80, 60)]
_INC_AN_PB = [(0.0, 73), (0.25, 72), (0.40, 71), (0.55, 68), (0.65, 64), (0.80, 60)]
_INC_DI_PB = [(0.0, 90), (0.20, 89), (0.35, 87), (0.45, 85), (0.55, 80), (0.65, 70), (0.80, 62)]

_INC_FO_EQ = [(0.0, 88), (0.15, 87), (0.30, 86), (0.45, 85), (0.60, 83), (0.80, 78)]
_INC_AN_EQ = [(0.0, 80), (0.20, 78), (0.35, 76), (0.50, 74), (0.65, 71), (0.80, 68)]
_INC_DI_EQ = [(0.0, 90), (0.25, 89), (0.45, 88), (0.60, 87), (0.80, 85)]

_CUM_TABLES: dict[PathKey, list[tuple[float, float, float, float]]] = {
    "fc_100": _CUM_FC,
    "polybaric_fc": _CUM_PB,
    "eq_100": _CUM_EQ,
}

_COMP_TABLES: dict[PathKey, tuple[list, list, list]] = {
    "fc_100": (_INC_FO_FC, _INC_AN_FC, _INC_DI_FC),
    "polybaric_fc": (_INC_FO_PB, _INC_AN_PB, _INC_DI_PB),
    "eq_100": (_INC_FO_EQ, _INC_AN_EQ, _INC_DI_EQ),
}


def _interp1(f: float, knots: list[tuple[float, float]]) -> float:
    x = np.array([k[0] for k in knots], dtype=float)
    y = np.array([k[1] for k in knots], dtype=float)
    return float(np.interp(float(f), x, y))


def paper_cum_vp_km_s(path: PathKey, f_solid: float) -> float:
    """Fig.2 panel (a): cumulative solid Vp (sparse anchor interpolation)."""
    return _interp1(f_solid, _VP_CUM_TABLES[path])


def paper_inc_vp_km_s(path: PathKey, f_solid: float) -> float:
    """Fig.2 panel (a): incremental solid Vp."""
    return _interp1(f_solid, _VP_INC_TABLES[path])


def legacy_cum_vp_km_s(path: PathKey, f_solid: float) -> float:
    return paper_cum_vp_km_s(path, f_solid)


def legacy_inc_vp_km_s(path: PathKey, f_solid: float) -> float:
    return paper_inc_vp_km_s(path, f_solid)


def legacy_residual_vp_km_s(path: PathKey, f_solid: float) -> float:
    return paper_residual_vp_km_s(path, f_solid)


def paper_cum_rho_g_cm3(path: PathKey, f_solid: float) -> float:
    """Fig.2 panel (b): cumulative solid density."""
    return _interp1(f_solid, _RHO_CUM_TABLES[path])


def paper_inc_rho_g_cm3(path: PathKey, f_solid: float) -> float:
    """Fig.2 panel (b): incremental solid density."""
    return _interp1(f_solid, _RHO_INC_TABLES[path])


def paper_residual_vp_km_s(path: PathKey, f_solid: float) -> float:
    """Fig.2 panel (a): residual liquid norm Vp (fractional crystallization)."""
    return _interp1(f_solid, _VP_RES_TABLES[path])


def paper_residual_rho_g_cm3(path: PathKey, f_solid: float) -> float:
    """Fig.2 panel (b): residual liquid density (fractional crystallization)."""
    return _interp1(f_solid, _RHO_RES_TABLES[path])


def cumulative_phases_pct(path: PathKey, f_solid: float) -> tuple[float, float, float]:
    """Cumulative solid Ol/Cpx/Pl (wt% of cumulate, sum≈100)."""
    table = _CUM_TABLES[path]
    x = np.array([t[0] for t in table], dtype=float)
    ol = float(np.interp(float(f_solid), x, [t[1] for t in table]))
    cpx = float(np.interp(float(f_solid), x, [t[2] for t in table]))
    pl = float(np.interp(float(f_solid), x, [t[3] for t in table]))
    s = ol + cpx + pl
    if s <= 1e-9:
        return 100.0, 0.0, 0.0
    scale = 100.0 / s
    return ol * scale, cpx * scale, pl * scale


def incremental_compositions(path: PathKey, f_solid: float) -> tuple[float, float, float]:
    """Fo, An, Di for incremental assemblage (wt%)."""
    fo_k, an_k, di_k = _COMP_TABLES[path]
    return _interp1(f_solid, fo_k), _interp1(f_solid, an_k), _interp1(f_solid, di_k)


def incremental_modes_from_cumulative(
    path: PathKey,
    f_new: float,
    f_old: float,
) -> tuple[float, float, float]:
    """Instantaneous Ol/Pl/Cpx modes from paper cumulative phase budget."""
    if f_new <= f_old + 1e-12:
        ol, cpx, pl = cumulative_phases_pct(path, f_new)
        return ol / 100.0, pl / 100.0, cpx / 100.0

    ol0, cpx0, pl0 = cumulative_phases_pct(path, f_old)
    ol1, cpx1, pl1 = cumulative_phases_pct(path, f_new)
    df = f_new - f_old
    d_ol = max(0.0, f_new * ol1 / 100.0 - f_old * ol0 / 100.0) / df
    d_cpx = max(0.0, f_new * cpx1 / 100.0 - f_old * cpx0 / 100.0) / df
    d_pl = max(0.0, f_new * pl1 / 100.0 - f_old * pl0 / 100.0) / df
    s = d_ol + d_cpx + d_pl
    if s <= 1e-12:
        return 1.0, 0.0, 0.0
    return d_ol / s, d_pl / s, d_cpx / s


def cumulative_modes_fraction(path: PathKey, f_solid: float) -> tuple[float, float, float]:
    ol, cpx, pl = cumulative_phases_pct(path, f_solid)
    return ol / 100.0, pl / 100.0, cpx / 100.0


def cumulative_compositions_weighted(
    path: PathKey,
    f_solid: float,
    *,
    cum_ol: float,
    cum_pl: float,
    cum_cpx: float,
    fo_num: float,
    fo_den: float,
    an_num: float,
    an_den: float,
    di_num: float,
    di_den: float,
) -> tuple[float, float, float]:
    """Return cum Fo/An/Di; use integrated values when available."""
    fo = fo_num / fo_den if fo_den > 0.0 else incremental_compositions(path, f_solid)[0]
    an = an_num / an_den if an_den > 0.0 else incremental_compositions(path, f_solid)[1]
    di = di_num / di_den if di_den > 0.0 else incremental_compositions(path, f_solid)[2]
    _ = cum_ol, cum_pl, cum_cpx
    return fo, an, di
