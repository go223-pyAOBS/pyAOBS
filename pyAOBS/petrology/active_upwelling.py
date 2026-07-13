"""KKHS02 §4 active-upwelling mantle melting model (eq. 6–11).

Linear melting (Fig.11): F(P) = (∂F/∂P)_S·(P₀−P),  F̄ = ½F(P_f),  P̄ = (P₀+P_f)/2,
H = 30·χ·(P₀−P_f)·F̄,  P_f = (H+b)/30.  For linear F, P_f has closed form eq.(11).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from petrology.constants import ADIABAT_GRAD_C_PER_GPA, KM_PER_GPA
from petrology.melting_laws import FIG11_MELTING, LinearMeltingLaw, MeltingLaw
from petrology.vp_regression import predict_vp_km_s

# KKHS02 §4 dry peridotite solidus (Takahashi & Kushiro, 1983): T = 1150 + 120 P (°C, GPa)
TK83_SOLIDUS_T0_C = 1150.0
TK83_SOLIDUS_DTDp_C_PER_GPA = 120.0
# Eq.(7): depth–pressure conversion — alias for backward compatibility
DPDZ_KM_PER_GPA = KM_PER_GPA
# Eq.(6): Fig.11 standard (∂F/∂P)_S = 12%/GPa; sensitivity scans use 10–20%/GPa
DFDP_DEFAULT_PER_GPA = 0.12


@dataclass(frozen=True)
class ActiveUpwellingResult:
    tp_c: float
    b_km: float
    chi: float
    p0_gpa: float
    pf_gpa: float
    pbar_gpa: float
    fbar: float
    f_total: float
    h_km: float
    vp_bulk_km_s: float
    melting_law: str = "linear"


def solidus_tk83_c(p_gpa: float) -> float:
    """Dry peridotite solidus (Takahashi & Kushiro, 1983): T = 1150 + 120 P (°C, GPa)."""
    return TK83_SOLIDUS_T0_C + TK83_SOLIDUS_DTDp_C_PER_GPA * float(p_gpa)


def adiabat_mb88_c(tp_c: float, p_gpa: float | np.ndarray) -> float | np.ndarray:
    """Linear adiabat (McKenzie & Bickle, 1988): T(P) = T_p + 20 P."""
    p = np.asarray(p_gpa, dtype=float)
    tp = np.asarray(tp_c, dtype=float)
    out = tp + ADIABAT_GRAD_C_PER_GPA * p
    if np.ndim(p_gpa) == 0 and np.ndim(tp_c) == 0:
        return float(out)
    return out


def solidus_temperature_at_p0(p0_gpa: float) -> float:
    """Solidus temperature T_0 at initial melting pressure P_0."""
    return solidus_tk83_c(p0_gpa)


def tp_from_solidus_at_p0(t0_c: float, p0_gpa: float) -> float:
    """Invert MB88: T_p = T_0 - 20 P_0."""
    return float(t0_c) - ADIABAT_GRAD_C_PER_GPA * float(p0_gpa)


def p0_from_tp(tp_c: float) -> float:
    """Start-of-melting pressure: adiabat T_p + 20 P = solidus 1150 + 120 P."""
    denom = TK83_SOLIDUS_DTDp_C_PER_GPA - ADIABAT_GRAD_C_PER_GPA
    p0 = (float(tp_c) - TK83_SOLIDUS_T0_C) / denom
    if p0 <= 0.0:
        raise ValueError(f"No melting for Tp={tp_c}°C (solidus intercept {TK83_SOLIDUS_T0_C}°C)")
    return float(p0)


def melt_fraction_linear(p_gpa: float, p0_gpa: float, *, dfdp_per_gpa: float) -> float:
    """Eq.(6): F(P) = (∂F/∂P)_S · (P₀ − P)."""
    return max(float(dfdp_per_gpa) * (float(p0_gpa) - float(p_gpa)), 0.0)


def pf_from_geometry(h_km: float, b_km: float) -> float:
    """Eq.(7): P_f = (H + b) / 30."""
    return (float(h_km) + float(b_km)) / DPDZ_KM_PER_GPA


def fbar_linear_mean(delta_p_gpa: float, *, dfdp_per_gpa: float) -> float:
    """Eq.(8): F̄ = ½ F(P_f) for linear F(P)."""
    dp = max(float(delta_p_gpa), 0.0)
    return 0.5 * float(dfdp_per_gpa) * dp


def pbar_mean(p0_gpa: float, pf_gpa: float) -> float:
    """Eq.(9): P̄ = (P₀ + P_f) / 2."""
    return 0.5 * (float(p0_gpa) + float(pf_gpa))


def h_from_eq10(*, chi: float, delta_p_gpa: float, fbar: float) -> float:
    """Eq.(10): H = 30 · χ · (P₀ − P_f) · F̄."""
    return float(chi) * DPDZ_KM_PER_GPA * float(delta_p_gpa) * float(fbar)


def delta_p_closed_form_linear(
    *,
    p0_gpa: float,
    b_km: float,
    chi: float,
    dfdp_per_gpa: float,
) -> float | None:
    """
    Eq.(11): ΔP = P₀ − P_f for linear melting (equating eq. 7 and eq. 10).

    P_f = P₀ + (1 / (χ·α)) · {1 − [1 + 2·χ·α·(P₀ − b/30)]^½},  α = (∂F/∂P)_S.
    """
    alpha = float(dfdp_per_gpa)
    chi = float(chi)
    p0 = float(p0_gpa)
    a = p0 - float(b_km) / DPDZ_KM_PER_GPA
    if a <= 0.0 or alpha <= 0.0 or chi < 1.0:
        return None
    disc = 1.0 + 2.0 * chi * alpha * a
    if disc < 0.0:
        return None
    delta_p = (float(np.sqrt(disc)) - 1.0) / (chi * alpha)
    if delta_p <= 0.0 or delta_p >= p0 or fbar_linear_mean(delta_p, dfdp_per_gpa=alpha) >= 0.99:
        return None
    return float(delta_p)


def pf_closed_form_linear(
    *,
    p0_gpa: float,
    b_km: float,
    chi: float,
    dfdp_per_gpa: float,
) -> float | None:
    """Eq.(11): P_f from T_p column inputs (requires P₀ already known)."""
    dp = delta_p_closed_form_linear(
        p0_gpa=p0_gpa,
        b_km=b_km,
        chi=chi,
        dfdp_per_gpa=dfdp_per_gpa,
    )
    if dp is None:
        return None
    return float(p0_gpa) - dp


def _max_delta_p(melting_law: MeltingLaw, *, p0_gpa: float) -> float:
    """Upper bracket for ΔP = P₀ − P_f with F̄ < 1."""
    dmax = min(p0_gpa * 0.999, 4.5)
    grid = np.linspace(1e-6, dmax, 512)
    last_ok = float(grid[0])
    for g in grid:
        if melting_law.fbar_mean(float(g)) >= 0.99:
            break
        last_ok = float(g)
    return max(last_ok, 1e-6)


def _column_residual(
    delta_p: float,
    *,
    p0_gpa: float,
    b_km: float,
    chi: float,
    melting_law: MeltingLaw,
) -> float:
    fbar = melting_law.fbar_mean(delta_p)
    if fbar >= 0.999:
        return 1e9
    h_from_mass = h_from_eq10(chi=chi, delta_p_gpa=delta_p, fbar=fbar)
    h_from_geom = DPDZ_KM_PER_GPA * (p0_gpa - delta_p) - b_km
    return h_from_mass - h_from_geom


def _result_from_delta_p(
    *,
    tp_c: float,
    b_km: float,
    chi: float,
    p0: float,
    delta_p: float,
    law: MeltingLaw,
    vp_bias_km_s: float,
) -> ActiveUpwellingResult:
    pf = p0 - delta_p
    fbar = law.fbar_mean(delta_p)
    f_total = law.f_total(delta_p)
    h = DPDZ_KM_PER_GPA * pf - b_km
    pbar = pbar_mean(p0, pf)
    vp = predict_vp_km_s(pbar, fbar) + float(vp_bias_km_s)
    return ActiveUpwellingResult(
        tp_c=float(tp_c),
        b_km=float(b_km),
        chi=float(chi),
        p0_gpa=float(p0),
        pf_gpa=float(pf),
        pbar_gpa=float(pbar),
        fbar=float(fbar),
        f_total=float(f_total),
        h_km=float(h),
        vp_bulk_km_s=float(vp),
        melting_law=str(getattr(law, "name", "linear")),
    )


def solve_active_upwelling(
    *,
    tp_c: float,
    b_km: float,
    chi: float,
    dfdp_per_gpa: float = DFDP_DEFAULT_PER_GPA,
    melting_law: MeltingLaw | None = None,
    vp_bias_km_s: float = 0.0,
) -> ActiveUpwellingResult:
    """
    Solve KKHS02-style active upwelling column and return H–Vp state.

    Default melting is linear F(ΔP) with ``dfdp_per_gpa``.
    Pass ``melting_law`` for Fig.13 variants (16%/GPa linear or Asimow 3-stage).
    """
    if chi < 1.0:
        raise ValueError("chi must be >= 1")
    law: MeltingLaw = melting_law or (
        FIG11_MELTING if dfdp_per_gpa == FIG11_MELTING.dfdp_per_gpa else LinearMeltingLaw(dfdp_per_gpa=dfdp_per_gpa)
    )
    p0 = p0_from_tp(tp_c)
    # Eq.(7)/(11): need P₀ > b/30 so a melting interval exists above the lid.
    b_over_30 = float(b_km) / DPDZ_KM_PER_GPA
    if p0 <= b_over_30:
        tp_min = TK83_SOLIDUS_T0_C + (
            TK83_SOLIDUS_DTDp_C_PER_GPA - ADIABAT_GRAD_C_PER_GPA
        ) * b_over_30
        raise ValueError(
            f"No melting column for Tp={tp_c:g}°C, b={b_km:g} km "
            f"(P0={p0:.3f} GPa ≤ b/30={b_over_30:.3f} GPa). "
            f"Raise Tp above ~{tp_min:.0f}°C or reduce b."
        )

    if isinstance(law, LinearMeltingLaw):
        dp_cf = delta_p_closed_form_linear(
            p0_gpa=p0,
            b_km=b_km,
            chi=chi,
            dfdp_per_gpa=law.dfdp_per_gpa,
        )
        if dp_cf is not None:
            return _result_from_delta_p(
                tp_c=tp_c,
                b_km=b_km,
                chi=chi,
                p0=p0,
                delta_p=dp_cf,
                law=law,
                vp_bias_km_s=vp_bias_km_s,
            )

    dmax = _max_delta_p(law, p0_gpa=p0)
    lo = 1e-6
    hi = max(lo * 10.0, dmax)
    f_lo = _column_residual(lo, p0_gpa=p0, b_km=b_km, chi=chi, melting_law=law)
    f_hi = _column_residual(hi, p0_gpa=p0, b_km=b_km, chi=chi, melting_law=law)

    if f_lo * f_hi > 0:
        grid = np.linspace(lo, hi, 256)
        vals = [
            _column_residual(g, p0_gpa=p0, b_km=b_km, chi=chi, melting_law=law)
            for g in grid
        ]
        idx = None
        for i in range(len(vals) - 1):
            if vals[i] == 0 or vals[i] * vals[i + 1] <= 0:
                idx = i
                break
        if idx is None:
            raise ValueError(
                f"Failed to bracket Pf root for Tp={tp_c:g}°C, b={b_km:g} km, "
                f"χ={chi:g} (P0={p0:.3f} GPa); residual does not change sign"
            )
        lo, hi = float(grid[idx]), float(grid[idx + 1])
        f_lo, f_hi = float(vals[idx]), float(vals[idx + 1])

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = _column_residual(mid, p0_gpa=p0, b_km=b_km, chi=chi, melting_law=law)
        if abs(fm) < 1e-8:
            lo = hi = mid
            break
        if f_lo * fm <= 0:
            hi = mid
            f_hi = fm
        else:
            lo = mid
            f_lo = fm
    delta_p = 0.5 * (lo + hi)
    return _result_from_delta_p(
        tp_c=tp_c,
        b_km=b_km,
        chi=chi,
        p0=p0,
        delta_p=delta_p,
        law=law,
        vp_bias_km_s=vp_bias_km_s,
    )


def sweep_hvp(
    *,
    tp_values_c: list[float] | np.ndarray,
    chi_values: list[float] | np.ndarray,
    b_km: float = 0.0,
    dfdp_per_gpa: float = DFDP_DEFAULT_PER_GPA,
    melting_law: MeltingLaw | None = None,
    vp_bias_km_s: float = 0.0,
) -> list[ActiveUpwellingResult]:
    return sweep_hvp_law(
        tp_values_c=tp_values_c,
        chi_values=chi_values,
        b_km=b_km,
        melting_law=melting_law
        or (
            FIG11_MELTING
            if dfdp_per_gpa == FIG11_MELTING.dfdp_per_gpa
            else LinearMeltingLaw(dfdp_per_gpa=dfdp_per_gpa)
        ),
        vp_bias_km_s=vp_bias_km_s,
    )


def sweep_hvp_law(
    *,
    tp_values_c: list[float] | np.ndarray,
    chi_values: list[float] | np.ndarray,
    b_km: float = 0.0,
    melting_law: MeltingLaw,
    vp_bias_km_s: float = 0.0,
) -> list[ActiveUpwellingResult]:
    out: list[ActiveUpwellingResult] = []
    for tp in tp_values_c:
        for chi in chi_values:
            try:
                out.append(
                    solve_active_upwelling(
                        tp_c=float(tp),
                        b_km=b_km,
                        chi=float(chi),
                        melting_law=melting_law,
                        vp_bias_km_s=vp_bias_km_s,
                    )
                )
            except Exception:
                continue
    return out
