"""
Iwamori, McKenzie & Takahashi (1995) dry peridotite batch melting — isobaric F(T).

Reference
---------
Iwamori, H., McKenzie, D., & Takahashi, E. (1995). Melt generation by isentropic
mantle upwelling. *Earth and Planetary Science Letters*, 134(3–4), 253–266.
doi:10.1016/0012-821X(95)00122-S

The original paper formulates **isentropic decompression** melting (McKenzie, 1984
thermodynamics). Katz et al. (2003) Fig. 9 compares isobaric F(T) at 1 and 3 GPa
by evaluating the same batch thermodynamic relation at fixed *P*.

Implementation
--------------
- Solidus: Takahashi (1986) / IMT95 fertile peridotite (same form as Katz eq. 4).
- Batch isobaric melting (McKenzie, 1984; IMT95): integrate dF/dT from solidus with
  constant Cp and entropy of fusion ΔS (J kg⁻¹ K⁻¹).
- Analytic integral (constant Cp, ΔS): F = (Cp/ΔS) ln(T/T_s), capped at F = 1 at
  the model liquidus T_l(P).
"""

from __future__ import annotations

import numpy as np

# Takahashi (1986) / IMT95 dry peridotite solidus (°C, P in GPa)
_SOLIDUS = (1088.0, 132.9, -5.1)

# High-T liquidus cap (experimental peridotite suite, IMT95 / Takahashi melting)
_LIQUIDUS = (1780.0, 45.0, -2.0)

# McKenzie & Bickle (1988) / IMT95 thermodynamic defaults (J kg⁻¹ K⁻¹)
_CP_J_KG = 1200.0
_DELTA_S_J_KG = 407.0  # McKenzie (1984) peridotite; IMT95 Table 1 range ~300–407


def _quad(p_gpa: float, c0: float, c1: float, c2: float) -> float:
    p = float(p_gpa)
    return c0 + c1 * p + c2 * p * p


def iwamori1995_solidus(p_gpa: float) -> float:
    """Dry peridotite solidus T_s(P), IMT95 / Takahashi (1986) form."""
    return _quad(p_gpa, *_SOLIDUS)


def iwamori1995_liquidus(p_gpa: float) -> float:
    """Model bulk liquidus T_l(P) for F → 1 cap."""
    return _quad(p_gpa, *_LIQUIDUS)


def iwamori1995_melt_fraction_dry(
    p_gpa: float,
    t_c: float,
    *,
    cp_j_kg: float = _CP_J_KG,
    delta_s_j_kg: float = _DELTA_S_J_KG,
) -> float:
    """
    Iwamori et al. (1995) batch isobaric F(P, T) from McKenzie (1984) thermodynamics.

    F(T) = (Cp/ΔS) ln(T/T_s) for T_s < T < T_l; 0 below solidus; 1 above liquidus.
    """
    p = float(p_gpa)
    t = float(t_c)
    ts = iwamori1995_solidus(p)
    tl = iwamori1995_liquidus(p)
    if t <= ts:
        return 0.0
    if t >= tl:
        return 1.0
    ds = max(float(delta_s_j_kg), 1e-6)
    cp = float(cp_j_kg)
    f = (cp / ds) * np.log(t / max(ts, 1e-6))
    return float(np.clip(f, 0.0, 1.0))


def iwamori1995_melt_fraction_dry_profile(
    p_gpa: float,
    t_c: np.ndarray | list[float],
    *,
    cp_j_kg: float = _CP_J_KG,
    delta_s_j_kg: float = _DELTA_S_J_KG,
) -> np.ndarray:
    return np.array(
        [
            iwamori1995_melt_fraction_dry(
                p_gpa, float(ti), cp_j_kg=cp_j_kg, delta_s_j_kg=delta_s_j_kg
            )
            for ti in t_c
        ],
        dtype=float,
    )
