"""
Langmuir et al. (1992) dry peridotite melting — batch isobaric F(T).

Reference
---------
Langmuir, C. H., Klein, E. M., & Plank, T. (1992). How deep do common basaltic
magmas form and differentiate? *Journal of Geophysical Research*, 97(B4), 3227–3252.

Katz et al. (2003) Fig. 9 compares this parameterization at fixed pressure. The
Langmuir forward model uses incremental / aggregated **batch** melting along
decompression paths; for an isobaric F(T) slice Katz evaluates the same batch
melting function at constant *P* (see Katz et al. 2003 §5).

Implementation notes
--------------------
- Solidus: Hirschmann (2000) dry peridotite regression, widely used in the
  Langmuir–Klein–Plank MORB melting lineage (cf. Klein & Langmuir, 1987).
- Lherzolite / bulk liquidus: experimental brackets compiled in Langmuir (1992)
  and Kinzler & Grove (1992) experiments (chemistry not required for *F* only).
- Melt fraction: **linear batch** segments (β = 1) between solidus, cpx-out,
  and liquidus — the batch limit used in Langmuir-type aggregated-melt models
  (Langmuir et al., 1977; McKenzie, 1985).
"""

from __future__ import annotations

import numpy as np

# Hirschmann (2000) G³ dry peridotite solidus (°C, P in GPa)
_HIRSCHMANN2000_SOLIDUS = (1120.661, 132.899, -5.104)

# Lherzolite / liquidus brackets (Langmuir 1992 / experimental peridotite suite)
_LHERZOLITE_LIQUIDUS = (1450.0, 85.0, -3.0)
_LIQUIDUS = (1760.0, 52.0, -2.2)

# Cpx reaction coefficient (Langmuir/Katz-style harzburgite transition)
_R_CPX = (0.50, 0.08)
_DEFAULT_CPX_MASS = 0.15


def _quad(p_gpa: float, c0: float, c1: float, c2: float) -> float:
    p = float(p_gpa)
    return c0 + c1 * p + c2 * p * p


def langmuir1992_solidus(p_gpa: float) -> float:
    """Dry peridotite solidus T_s(P), Hirschmann (2000) via Langmuir lineage."""
    return _quad(p_gpa, *_HIRSCHMANN2000_SOLIDUS)


def langmuir1992_lherzolite_liquidus(p_gpa: float) -> float:
    """Lherzolite multisaturated liquidus T_lliq(P)."""
    return _quad(p_gpa, *_LHERZOLITE_LIQUIDUS)


def langmuir1992_liquidus(p_gpa: float) -> float:
    """Bulk peridotite liquidus T_liq(P)."""
    return _quad(p_gpa, *_LIQUIDUS)


def langmuir1992_fcpx_out(*, p_gpa: float, cpx_mass: float = _DEFAULT_CPX_MASS) -> float:
    """Melt fraction at cpx exhaustion (harzburgite transition)."""
    rc = _R_CPX[0] + _R_CPX[1] * float(p_gpa)
    return float(cpx_mass) / max(rc, 1e-9)


def langmuir1992_melt_fraction_dry(
    p_gpa: float,
    t_c: float,
    *,
    cpx_mass: float = _DEFAULT_CPX_MASS,
) -> float:
    """
    Langmuir (1992) batch isobaric melt fraction F(P, T).

    Piecewise **linear** batch melting (β = 1) between solidus, cpx-out, and
    liquidus — distinct from Katz (2003) power-law β = 1.5 curves.
    """
    p = float(p_gpa)
    t = float(t_c)
    ts = langmuir1992_solidus(p)
    if t <= ts:
        return 0.0
    tlh = langmuir1992_lherzolite_liquidus(p)
    tl = langmuir1992_liquidus(p)
    if t >= tl:
        return 1.0

    fc = langmuir1992_fcpx_out(p_gpa=p, cpx_mass=cpx_mass)
    td = (t - ts) / max(tlh - ts, 1e-6)
    f_lin = float(np.clip(td, 0.0, 1.0))  # β = 1
    if f_lin <= fc:
        return f_lin

    tc = ts + fc * (tlh - ts)
    td2 = (t - tc) / max(tl - tc, 1e-6)
    return float(np.clip(fc + (1.0 - fc) * td2, 0.0, 1.0))


def langmuir1992_melt_fraction_dry_profile(
    p_gpa: float,
    t_c: np.ndarray | list[float],
    *,
    cpx_mass: float = _DEFAULT_CPX_MASS,
) -> np.ndarray:
    return np.array(
        [langmuir1992_melt_fraction_dry(p_gpa, float(ti), cpx_mass=cpx_mass) for ti in t_c],
        dtype=float,
    )
