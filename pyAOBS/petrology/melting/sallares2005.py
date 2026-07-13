"""Sallarès et al. (2005 GJI) 2-D wet melting model → H, F̄, Z̄ → KKHS02 eq.(1) Vp.

Based on §6 Mantle melting model (eqs 2–10): triangular corner-flow melting region
with dry + damp productivity and depth-dependent active upwelling beneath the dry
solidus.  Vp uses Korenaga et al. (2002) windowed regression (Step 1 eq.1).

Reference: Sallarès, Charvis, Flueh et al. (2005), GJI, doi:10.1111/j.1365-246X.2005.02592.x
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from petrology.constants import KM_PER_GPA
from petrology.vp_regression import predict_vp_km_s
RHO_M_KG_M3 = 3300.0
RHO_C_KG_M3 = 2900.0


@dataclass(frozen=True)
class SallaresMeltingParams:
    """Fig. 10 panel parameters (defaults = reference model 10a)."""

    d_percent_per_gpa: float = 15.0  # dry melt productivity Γ_d
    w_percent_per_gpa: float = 1.0  # damp melt productivity Γ_w
    damp_zone_km: float = 50.0  # Δz
    alpha_upwelling_decay: float = 0.25  # α
    b_lid_km: float = 0.0  # pre-existing lithospheric lid
    rho_m: float = RHO_M_KG_M3
    rho_c: float = RHO_C_KG_M3


@dataclass(frozen=True)
class SallaresMeltingResult:
    tp_c: float
    upwelling_x: float
    h_km: float
    f_bar: float
    z_bar_km: float
    p_bar_gpa: float
    z0_km: float
    zf_km: float
    vp_bulk_km_s: float
    params: SallaresMeltingParams


def productivity_per_km(percent_per_gpa: float) -> float:
    """Convert %/GPa to melt fraction per km (P ≈ z/30 GPa)."""
    return (float(percent_per_gpa) / 100.0) / KM_PER_GPA


def solidus_temperature_mb88_c(t0_c: float) -> float:
    """MB88 dry peridotite solidus temperature at melting onset (eq. 2)."""
    t = float(t0_c)
    return (t - 1100.0) / 136.0 + 4.968e-4 * np.exp(0.012 * (t - 1100.0))


def p0_gpa_from_tp(tp_c: float) -> float:
    """
    Initial melting pressure P₀ (GPa) from potential temperature Tp (eq. 3 + 2).

    Adiabat: T₀ = Tp + 20·P₀; solidus gives P₀(T₀).
    """
    tp = float(tp_c)

    def residual(p0: float) -> float:
        t0 = tp + 20.0 * p0
        return p0 - solidus_temperature_mb88_c(t0)

    lo, hi = 0.05, 4.5
    r_lo, r_hi = residual(lo), residual(hi)
    if r_lo * r_hi > 0:
        raise ValueError(f"No solidus crossing for Tp={tp_c}°C")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if residual(lo) * residual(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def z0_km_from_tp(tp_c: float) -> float:
    return p0_gpa_from_tp(tp_c) * KM_PER_GPA


def _denom(z0: float, zf: float, d: float, w: float, x_up: float, dz: float, alpha: float) -> float:
    return 6.0 * d * (z0**2 - zf**2) + 2.0 * w * x_up * dz * (3.0 * z0 * (1.0 + alpha) + dz * (2.0 + alpha))


def _h_residual(
    h_km: float,
    *,
    z0: float,
    b_km: float,
    d: float,
    w: float,
    x_up: float,
    dz: float,
    alpha: float,
    rho_m: float,
    rho_c: float,
) -> float:
    zf = h_km + b_km
    if zf >= z0:
        return h_km - 1e-6
    num = d * (z0**2 - zf**2) + w * x_up * dz * (3.0 * z0 * (1.0 + alpha) + dz * (2.0 + alpha))
    h_pred = (rho_m / (3.0 * rho_c)) * num
    return h_km - h_pred


def solve_crust_thickness_km(
    *,
    z0_km: float,
    b_km: float,
    d: float,
    w: float,
    x_up: float,
    dz: float,
    alpha: float,
    rho_m: float,
    rho_c: float,
) -> float:
    """Solve eq. (8) with eq. (6): zf = H + b."""
    zf_max = z0_km - 0.5
    if b_km >= zf_max:
        return 0.0
    lo = 0.0
    hi = max(zf_max - b_km, 0.1)
    r_lo = _h_residual(
        lo, z0=z0_km, b_km=b_km, d=d, w=w, x_up=x_up, dz=dz, alpha=alpha, rho_m=rho_m, rho_c=rho_c
    )
    r_hi = _h_residual(
        hi, z0=z0_km, b_km=b_km, d=d, w=w, x_up=x_up, dz=dz, alpha=alpha, rho_m=rho_m, rho_c=rho_c
    )
    if r_lo * r_hi > 0:
        return max(float(hi if abs(r_hi) < abs(r_lo) else lo), 0.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        r_mid = _h_residual(
            mid,
            z0=z0_km,
            b_km=b_km,
            d=d,
            w=w,
            x_up=x_up,
            dz=dz,
            alpha=alpha,
            rho_m=rho_m,
            rho_c=rho_c,
        )
        if r_lo * r_mid <= 0:
            hi = mid
            r_hi = r_mid
        else:
            lo = mid
            r_lo = r_mid
    return 0.5 * (lo + hi)


def mean_melt_fraction(z0: float, zf: float, d: float, w: float, x_up: float, dz: float, alpha: float) -> float:
    """Eq. (9): pooled-melt average degree of melting F̄ (fraction 0–1)."""
    den = _denom(z0, zf, d, w, x_up, dz, alpha)
    if abs(den) < 1e-15:
        return 0.0
    num = 2.0 * d * (z0**3 - 3.0 * z0 * zf**2 + 2.0 * zf**3)
    num += 2.0 * w * x_up * dz**2 * (2.0 * z0 * (1.0 + 2.0 * alpha) + dz * (1.0 + alpha))
    # Paper eq.(9) returns mean melt fraction in **percent**; convert to fraction for eq.(1).
    return float(num / den / 100.0)


def mean_melt_depth_km(z0: float, zf: float, d: float, w: float, x_up: float, dz: float, alpha: float) -> float:
    """Eq. (10): mean melting depth Z̄ (km below surface)."""
    den = _denom(z0, zf, d, w, x_up, dz, alpha)
    if abs(den) < 1e-15:
        return float(zf)
    num = 4.0 * d * (z0**3 - zf**3)
    num += w * x_up * dz * (
        6.0 * z0**2 * (1.0 + alpha) + dz**2 * (3.0 + alpha) + 4.0 * z0 * dz * (2.0 + alpha)
    )
    return float(num / den)


def solve_sallares_melting(
    *,
    tp_c: float,
    upwelling_x: float,
    params: SallaresMeltingParams | None = None,
    vp_bias_km_s: float = 0.0,
) -> SallaresMeltingResult:
    """
    One (Tp, X) state: H from eq.(8), F̄/Z̄ from eq.(9–10), Vp from KKHS02 eq.(1).

    Mean pressure for eq.(1): P̄ = Z̄ / 30 GPa (lithostatic, eq. 5).
    """
    par = params or SallaresMeltingParams()
    d = productivity_per_km(par.d_percent_per_gpa)
    w = productivity_per_km(par.w_percent_per_gpa)
    z0 = z0_km_from_tp(tp_c)
    h = solve_crust_thickness_km(
        z0_km=z0,
        b_km=par.b_lid_km,
        d=d,
        w=w,
        x_up=float(upwelling_x),
        dz=par.damp_zone_km,
        alpha=par.alpha_upwelling_decay,
        rho_m=par.rho_m,
        rho_c=par.rho_c,
    )
    zf = h + par.b_lid_km
    f_bar = mean_melt_fraction(z0, zf, d, w, float(upwelling_x), par.damp_zone_km, par.alpha_upwelling_decay)
    z_bar = mean_melt_depth_km(z0, zf, d, w, float(upwelling_x), par.damp_zone_km, par.alpha_upwelling_decay)
    p_bar = z_bar / KM_PER_GPA
    vp = predict_vp_km_s(p_bar, f_bar) + float(vp_bias_km_s)
    return SallaresMeltingResult(
        tp_c=float(tp_c),
        upwelling_x=float(upwelling_x),
        h_km=float(h),
        f_bar=float(f_bar),
        z_bar_km=float(z_bar),
        p_bar_gpa=float(p_bar),
        z0_km=float(z0),
        zf_km=float(zf),
        vp_bulk_km_s=float(vp),
        params=par,
    )


def sweep_sallares_hvp(
    *,
    tp_values_c: list[float] | np.ndarray,
    x_values: list[float] | np.ndarray,
    params: SallaresMeltingParams | None = None,
    vp_bias_km_s: float = 0.0,
) -> list[SallaresMeltingResult]:
    out: list[SallaresMeltingResult] = []
    for tp in tp_values_c:
        for x in x_values:
            try:
                out.append(
                    solve_sallares_melting(
                        tp_c=float(tp),
                        upwelling_x=float(x),
                        params=params,
                        vp_bias_km_s=vp_bias_km_s,
                    )
                )
            except (ValueError, ZeroDivisionError):
                continue
    return out


# Fig. 10 panel presets (Sallarès et al. 2005).
FIG10_PANELS: dict[str, SallaresMeltingParams] = {
    "10a_ref": SallaresMeltingParams(d_percent_per_gpa=15, w_percent_per_gpa=1, damp_zone_km=50, alpha_upwelling_decay=0.25),
    "10b_uniform_chi": SallaresMeltingParams(d_percent_per_gpa=15, w_percent_per_gpa=1, damp_zone_km=50, alpha_upwelling_decay=1.0),
    "10c_thick_damp": SallaresMeltingParams(d_percent_per_gpa=15, w_percent_per_gpa=1, damp_zone_km=75, alpha_upwelling_decay=0.25),
    "10d_high_w": SallaresMeltingParams(d_percent_per_gpa=15, w_percent_per_gpa=2, damp_zone_km=50, alpha_upwelling_decay=0.25),
    "10e_high_d": SallaresMeltingParams(d_percent_per_gpa=20, w_percent_per_gpa=1, damp_zone_km=50, alpha_upwelling_decay=0.25),
}
