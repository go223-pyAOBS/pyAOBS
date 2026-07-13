"""
REEBOX-core column geometry (no KKHS02 linear F(P)).

- P0: adiabat ∩ Katz solidus (heterogeneous: deepest lithology solidus).
- H: triangular melt-flux integration (pyMelt spreadingCentre-style) with
  optional active weight ∝ (χ − 1).
- Pf–H self-consistency (default): P_f = (H + b) / 30 (KKHS02 eq. 7);
  isentropic path and pooled melts are truncated at that P_f.

pyMelt v2 uses pressure in GPa and bulk density in g/cm³ (≈3.3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from petrology.active_upwelling import adiabat_mb88_c, pf_from_geometry
from petrology.constants import KM_PER_GPA

from .isentropic import IsentropicPath, integrate_isentropic_path
from .lithology import Lithology

# pyMelt bulkProperties convention (g/cm³), not kg/m³
RHO_GCM3 = 3.3
G_M_S2 = 9.81
P_FLOOR_GPA = 0.05
F_EPS = 1e-4
# pyMelt ``_integrate_tri`` accumulates thickness in Mm-scale units; ``self.tc = tc_found * 1e6`` → km.
PYMELT_TC_TO_KM = 1e6


def km_to_lithosphere_pressure_gpa(b_km: float, *, rho_kg_m3: float = 3300.0) -> float:
    return float(max(b_km, 0.0)) * 1000.0 * rho_kg_m3 * G_M_S2 / 1e9


def adiabat_t_c(tp_c: float, p_gpa: float) -> float:
    return adiabat_mb88_c(tp_c, p_gpa)


def p0_gpa_at_solidus(
    tp_c: float,
    lithologies: list[Lithology],
    *,
    p_hi: float = 5.0,
    p_lo: float = 0.05,
) -> float:
    """Pressure where linear adiabat first intersects the bulk solidus."""
    if not lithologies:
        raise ValueError("lithologies required")

    active = [L for L in lithologies if L.u0 > 0.0]
    if not active:
        raise ValueError("lithologies required")

    def res(p: float) -> float:
        t_ad = adiabat_t_c(tp_c, p)
        tsol = min(L.solidus_gpa(p) for L in active)
        return t_ad - tsol

    if res(p_lo) < 0:
        return float(p_lo)
    if res(p_hi) > 0:
        return float(p_hi)

    a, b = float(p_lo), float(p_hi)
    fa, fb = res(a), res(b)
    for _ in range(80):
        mid = 0.5 * (a + b)
        fm = res(mid)
        if abs(fm) < 1e-6:
            return float(mid)
        if fa * fm <= 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return float(0.5 * (a + b))


def _f_bulk(path: IsentropicPath, step) -> float:
    """Bulk melt fraction Σ φ_i F_i (pyMelt MeltingColumn.F)."""
    return float(path.f_bulk_at(step))


def pf_gpa_from_path(path: IsentropicPath, *, p_floor_gpa: float) -> float:
    """Shallowest pressure on the path with bulk F > F_EPS."""
    pf = float(p_floor_gpa)
    for st in reversed(path.steps):
        if _f_bulk(path, st) > F_EPS:
            pf = float(st.p_gpa)
            break
    return pf


def _active_weight(p: float, p_max: float, p_min: float, chi: float) -> float:
    if chi <= 1.0:
        return 0.0
    lam = float(chi) - 1.0
    span = max(p_max - p_min, 1e-9)
    w_norm = (p_max - float(p)) / span
    return lam * float(np.exp(-w_norm / 0.5))


def _integrate_tri_h_pf(
    path: IsentropicPath,
    *,
    b_km: float,
    chi: float = 1.0,
    rho_gcm3: float = RHO_GCM3,
    n_interp: int = 1001,
) -> tuple[float, float | None]:
    """
    Triangular crust thickness and pyMelt stop pressure.

    Returns (H_km, Pf_stop_gpa). Pf_stop is where crust load equals melting P;
    None if the stop condition never triggers (use full-column H).
    """
    p_lith = km_to_lithosphere_pressure_gpa(b_km)
    steps = path.steps
    if len(steps) < 2:
        return 0.0, None

    p_vals = np.array([s.p_gpa for s in steps], dtype=float)
    f_vals = np.clip(
        np.array([_f_bulk(path, s) for s in steps], dtype=float),
        0.0,
        0.999,
    )

    scale = 1.0 / (rho_gcm3 * G_M_S2 * 1e3)
    tc = scale * f_vals / np.maximum(1.0 - f_vals, 1e-9)

    p_max = float(np.max(p_vals))
    p_min = float(np.min(p_vals))
    p_grid = np.linspace(p_max, p_min, int(n_interp))
    order = np.argsort(p_vals)
    p_asc = p_vals[order]
    tc_asc = tc[order]
    tc_grid = np.interp(p_grid, p_asc, tc_asc)

    weights = np.array([_active_weight(p, p_max, p_min, chi) for p in p_grid])

    tc_int = 0.0
    tc_found: float | None = None
    pf_stop: float | None = None
    for i in range(1, len(p_grid)):
        dp = abs(p_grid[i] - p_grid[i - 1])
        w = weights[i]
        tc_int += 0.5 * (tc_grid[i] + tc_grid[i - 1]) * dp * (1.0 + w)
        tc_int_p = tc_int * rho_gcm3 * G_M_S2 * 1e3
        if tc_found is None and tc_int_p + p_lith > p_grid[i]:
            tc_found = tc_int
            pf_stop = float(p_grid[i])

    if tc_found is not None:
        return float(tc_found * PYMELT_TC_TO_KM), pf_stop
    return float(max(tc_int, 0.0) * PYMELT_TC_TO_KM), None


def triangular_crust_thickness_km(
    path: IsentropicPath,
    *,
    b_km: float,
    chi: float = 1.0,
    rho_gcm3: float = RHO_GCM3,
    n_interp: int = 1001,
) -> float:
    """
    Igneous crust thickness (km) from triangular melting-zone integration.

    Matches pyMelt ``spreadingCentre._integrate_tri`` (P in GPa, ρ in g/cm³).

    Uses bulk F = Σ φ_i F_i (not Σ F_i); otherwise heterogeneous sources inflate H.
    """
    h_km, _pf = _integrate_tri_h_pf(
        path, b_km=b_km, chi=chi, rho_gcm3=rho_gcm3, n_interp=n_interp
    )
    return h_km


def _clamp_pf(pf: float, *, p0_gpa: float, p_min_gpa: float) -> float:
    """Keep Pf in (p_min, p0) with a small gap for a non-empty column."""
    hi = float(p0_gpa) - 1e-3
    lo = float(p_min_gpa)
    if hi <= lo:
        return float(p0_gpa) * 0.5
    return float(min(max(pf, lo), hi))


@dataclass(frozen=True)
class ReeboxColumnGeometry:
    p0_gpa: float
    pf_gpa: float
    p_floor_gpa: float
    h_km: float
    path: IsentropicPath
    pf_h_consistent: bool = True


def build_reebox_column(
    lithologies: list[Lithology],
    *,
    tp_c: float,
    b_km: float = 0.0,
    chi: float = 1.0,
    n_isentropic_steps: int = 120,
    p_floor_gpa: float = P_FLOOR_GPA,
    pf_h_consistent: bool = True,
) -> ReeboxColumnGeometry:
    """
    P0, isentropic path, triangular H, and Pf.

    With ``pf_h_consistent=True`` (default): triangular stop gives H, then
    ``P_f = (H + b) / 30`` (KKHS02 eq. 7); the isentropic / RMC path is
    re-integrated only down to that Pf (no free-surface chemistry).
    """
    p0 = p0_gpa_at_solidus(tp_c, lithologies)
    p_lith = km_to_lithosphere_pressure_gpa(b_km)
    p_min = max(float(p_floor_gpa), p_lith, P_FLOOR_GPA)

    if not pf_h_consistent:
        path = integrate_isentropic_path(
            lithologies,
            tp_c=tp_c,
            p0_gpa=p0,
            pf_gpa=p_min,
            n_steps=n_isentropic_steps,
        )
        pf = pf_gpa_from_path(path, p_floor_gpa=p_min)
        h_km = triangular_crust_thickness_km(path, b_km=b_km, chi=chi)
        return ReeboxColumnGeometry(
            p0_gpa=float(p0),
            pf_gpa=float(pf),
            p_floor_gpa=float(p_min),
            h_km=float(h_km),
            path=path,
            pf_h_consistent=False,
        )

    # Deep scout → triangular stop (pyMelt): crust load = melting P gives (H, Pf).
    # That stop is already ≈ KKHS02 eq.(7); we report Pf = (H+b)/30 and truncate
    # the isentropic / RMC path at Pf so F̄ and chemistry exclude sub-crust melts.
    scout = integrate_isentropic_path(
        lithologies,
        tp_c=tp_c,
        p0_gpa=p0,
        pf_gpa=p_min,
        n_steps=max(int(n_isentropic_steps), 80),
    )
    h_km, pf_stop = _integrate_tri_h_pf(scout, b_km=b_km, chi=chi)
    if h_km > 1e-6:
        pf = pf_from_geometry(h_km, b_km)
    elif pf_stop is not None:
        pf = float(pf_stop)
        h_km = KM_PER_GPA * pf - float(b_km)
    else:
        pf = p_min
    pf = _clamp_pf(float(pf), p0_gpa=p0, p_min_gpa=p_min)

    path = integrate_isentropic_path(
        lithologies,
        tp_c=tp_c,
        p0_gpa=p0,
        pf_gpa=pf,
        n_steps=n_isentropic_steps,
    )
    # Keep H from full-column triangular stop (not re-integrated on the truncated
    # path: active-weight span would change and break the stop / eq.7 match).

    return ReeboxColumnGeometry(
        p0_gpa=float(p0),
        pf_gpa=float(pf),
        p_floor_gpa=float(p_min),
        h_km=float(h_km),
        path=path,
        pf_h_consistent=True,
    )
