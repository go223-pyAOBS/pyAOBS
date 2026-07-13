"""Hashin–Shtrikman isotropic mixing (KKHS02 §2.1)."""

from __future__ import annotations

import numpy as np


def vp_from_kgrho(k_gpa: float, g_gpa: float, rho_gcm3: float) -> float:
    """P-wave velocity in km/s."""
    rho = rho_gcm3 * 1000.0
    k = k_gpa * 1e9
    g = g_gpa * 1e9
    return float(np.sqrt((k + 4.0 / 3.0 * g) / rho) / 1000.0)


def _hs_two_phase(k1, g1, k2, g2, f2, upper: bool) -> tuple[float, float]:
    f1 = 1.0 - f2
    nu1 = (3 * k1 - 2 * g1) / (6 * k1 + 2 * g1)
    if upper:
        k = k1 + f2 / (1 / (k2 - k1) + f1 / (k1 + 4 * g1 / 3))
        g = g1 + f2 / (1 / (g2 - g1) + f1 / (5 * g1 + 4 * k1 + 6 * k1 * nu1))
    else:
        k = k1 + f2 / (1 / (k2 - k1) - f1 / (k1 + 4 * g1 / 3))
        g = g1 + f2 / (1 / (g2 - g1) - f1 / (5 * g1 + 4 * k1 + 6 * k1 * nu1))
    return float(k), float(g)


def _hs_multiphase(k: np.ndarray, g: np.ndarray, f: np.ndarray, upper: bool) -> tuple[float, float]:
    """Recursive HS mixing sorted by K (upper) or descending K (lower)."""
    idx = np.argsort(k if upper else -k)
    k, g, f = k[idx], g[idx], f[idx]
    k_eff, g_eff = float(k[0]), float(g[0])
    f_done = float(f[0])
    for i in range(1, len(k)):
        fi = float(f[i])
        if fi <= 0:
            continue
        # mix (k_eff, g_eff) at fraction f_done with (k[i], g[i]) at fi
        f2 = fi / (f_done + fi)
        k_new, g_new = _hs_two_phase(k_eff, g_eff, float(k[i]), float(g[i]), f2, upper=upper)
        k_eff, g_eff = k_new, g_new
        f_done += fi
    return k_eff, g_eff


def hashin_shtrikman_vp(
    volume_fractions: np.ndarray,
    k_gpa: np.ndarray,
    g_gpa: np.ndarray,
    rho_gcm3: np.ndarray,
) -> dict[str, float]:
    f = np.asarray(volume_fractions, dtype=float)
    f = f / f.sum()
    k_lo, g_lo = _hs_multiphase(k_gpa, g_gpa, f, upper=False)
    k_hi, g_hi = _hs_multiphase(k_gpa, g_gpa, f, upper=True)
    k_avg = 0.5 * (k_lo + k_hi)
    g_avg = 0.5 * (g_lo + g_hi)
    rho = float(np.sum(f * rho_gcm3))
    vp_lo = vp_from_kgrho(k_lo, g_lo, rho)
    vp_hi = vp_from_kgrho(k_hi, g_hi, rho)
    vp_avg = vp_from_kgrho(k_avg, g_avg, rho)
    return {
        "vp_km_s": vp_avg,
        "vp_hs_lower_km_s": vp_lo,
        "vp_hs_upper_km_s": vp_hi,
        "vp_hs_spread_km_s": vp_hi - vp_lo,
        "rho_g_cm3": rho,
    }
