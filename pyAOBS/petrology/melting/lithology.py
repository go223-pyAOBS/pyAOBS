"""
Lithology melting curves for REEBOX-style isentropic decompression.

Peridotite: Katz et al. (2003) dry solidus; polynomial coefficients
follow Brown & Lesher (2016) REEBOX PRO Appendix A (G³, compiled MATLAB app).
``katz2003_peridotite_liquidus`` is the **full liquidus** (eq. 10, REEBOX-refined),
not the lherzolite liquidus (eq. 5) — see ``melting.katz2003`` for paper Table 2.
G2 pyroxenite: Pertermann & Hirschmann (2003); native solidus offset ~150°C
below Katz peridotite (REEBOX PRO Fig. 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from .kinzler1997_batch import HZ_DEP1_WT
from .melt_chemistry import ChemistryBackend, melt_oxides

# REEBOX PRO Table 1 (Brown & Lesher 2016, G³ 17, 3929–3968).
CP_J_KG_K = 1200.0
ALPHA_K_INV = 3.0e-5
RHO_KG_M3 = 3300.0
DELTA_S_PERIDOTITE = 300.0  # J/kg/K
DELTA_S_PYROXENITE = 240.0

# G2 pyroxenite bulk (Pertermann & Hirschmann 2003) wt%.
G2_BULK_WT: dict[str, float] = {
    "SiO2": 51.87,
    "TiO2": 1.97,
    "Al2O3": 5.09,
    "Cr2O3": 0.03,
    "FeO": 11.58,
    "MgO": 11.08,
    "CaO": 16.44,
    "Na2O": 0.58,
    "K2O": 0.0,
    "P2O5": 0.04,
    "H2O": 0.0,
}


@dataclass(frozen=True)
class Lithology:
    """Single source lithology with T(P, F) and thermodynamic constants."""

    name: str
    u0: float
    delta_s_fusion: float
    source_wt: Mapping[str, float]
    solidus_c: Callable[[float], float]
    liquidus_c: Callable[[float], float]
    melt_beta: float = 1.25
    rho_kg_m3: float = RHO_KG_M3
    chemistry_backend: ChemistryBackend = "kinzler1997"
    _pymelt: Any = field(default=None, compare=False, repr=False)
    katz_cpx_mass: float | None = None

    def solidus_gpa(self, p_gpa: float) -> float:
        return float(self.solidus_c(float(p_gpa)))

    def liquidus_gpa(self, p_gpa: float) -> float:
        return float(self.liquidus_c(float(p_gpa)))

    def f_from_t(self, p_gpa: float, t_c: float) -> float:
        if self._pymelt is not None:
            return float(np.clip(self._pymelt.F(float(p_gpa), float(t_c)), 0.0, 0.999))
        if self.katz_cpx_mass is not None:
            from .katz2003 import katz2003_melt_fraction_dry

            return float(
                np.clip(
                    katz2003_melt_fraction_dry(
                        float(p_gpa),
                        float(t_c),
                        cpx_mass=float(self.katz_cpx_mass),
                    ),
                    0.0,
                    0.999,
                )
            )
        ts = self.solidus_gpa(p_gpa)
        tl = self.liquidus_gpa(p_gpa)
        if t_c <= ts:
            return 0.0
        if t_c >= tl:
            return 1.0
        x = (t_c - ts) / max(tl - ts, 1e-6)
        return float(np.clip(x**self.melt_beta, 0.0, 1.0))

    def t_from_f(self, p_gpa: float, f: float) -> float:
        f = float(np.clip(f, 0.0, 1.0))
        if self._pymelt is not None:
            from .pymelt_lithology_adapter import t_from_f_pymelt

            return t_from_f_pymelt(self._pymelt, p_gpa, f)
        if self.katz_cpx_mass is not None:
            from .katz2003 import katz2003_t_from_f

            return katz2003_t_from_f(
                float(p_gpa),
                f,
                cpx_mass=float(self.katz_cpx_mass),
            )
        ts = self.solidus_gpa(p_gpa)
        tl = self.liquidus_gpa(p_gpa)
        if f <= 0.0:
            return ts
        if f >= 1.0:
            return tl
        return ts + (tl - ts) * f ** (1.0 / self.melt_beta)

    def is_melting(self, p_gpa: float, t_c: float, f: float) -> bool:
        return f < 0.999 and t_c >= self.solidus_gpa(p_gpa) - 1e-6

    def dT_dP_at_F(self, p_gpa: float, f: float, *, dp: float = 1e-3) -> float:
        """(dT/dP)|F in °C/GPa."""
        f = float(np.clip(f, 0.0, 0.999))
        p_lo = max(p_gpa - dp, 0.05)
        p_hi = p_gpa + dp
        return (self.t_from_f(p_hi, f) - self.t_from_f(p_lo, f)) / (p_hi - p_lo)

    def dT_dF_at_P(self, p_gpa: float, f: float, *, df: float = 1e-4) -> float:
        """(dT/dF)|P in °C (dimensionless F)."""
        f = float(np.clip(f, 1e-5, 0.999))
        f_lo = max(f - df, 1e-5)
        f_hi = min(f + df, 0.999)
        return (self.t_from_f(p_gpa, f_hi) - self.t_from_f(p_gpa, f_lo)) / (f_hi - f_lo)

    def melt_oxides_wt(self, p_gpa: float, f: float) -> dict[str, float]:
        """Instantaneous batch melt composition (wt%)."""
        return melt_oxides(
            p_gpa,
            f,
            chemistry_backend=self.chemistry_backend,
            source=self.source_wt,
        )


def katz2003_peridotite_solidus(p_gpa: float) -> float:
    p = float(p_gpa)
    return 1085.7 + 132.899 * p - 5.104 * p * p


def katz2003_peridotite_liquidus(p_gpa: float) -> float:
    p = float(p_gpa)
    return 1788.0 + 45.095 * p - 2.0847 * p * p


def g2_pyroxenite_solidus(p_gpa: float) -> float:
    """G2 solidus ~150°C below Katz peridotite (Brown & Lesher 2016, REEBOX PRO Fig. 2)."""
    return katz2003_peridotite_solidus(p_gpa) - 150.0


def g2_pyroxenite_liquidus(p_gpa: float) -> float:
    """Narrower melting interval → higher polybaric productivity."""
    return g2_pyroxenite_solidus(p_gpa) + 0.55 * (
        katz2003_peridotite_liquidus(p_gpa) - katz2003_peridotite_solidus(p_gpa)
    )


def dry_peridotite_lithology(*, u0: float = 1.0, katz_table2: bool = True) -> Lithology:
    """Dry Katz lherzolite; ``katz_table2=True`` uses Table 2 two-segment F(P,T)."""
    from .katz2003 import KATZ_CPX_MASS

    return Lithology(
        name="peridotite",
        u0=float(u0),
        delta_s_fusion=DELTA_S_PERIDOTITE,
        source_wt=dict(HZ_DEP1_WT),
        solidus_c=katz2003_peridotite_solidus,
        liquidus_c=katz2003_peridotite_liquidus,
        katz_cpx_mass=float(KATZ_CPX_MASS) if katz_table2 else None,
    )


def g2_pyroxenite_lithology(*, u0: float) -> Lithology:
    return Lithology(
        name="g2_pyroxenite",
        u0=float(u0),
        delta_s_fusion=DELTA_S_PYROXENITE,
        source_wt=dict(G2_BULK_WT),
        solidus_c=g2_pyroxenite_solidus,
        liquidus_c=g2_pyroxenite_liquidus,
        melt_beta=1.15,
        rho_kg_m3=3300.0,
    )


def heterogeneous_source(
    *,
    pyroxenite_frac: float,
    backend: str = "native",
    peridotite_key: str = "katz_lherzolite",
    pyroxenite_key: str = "pertermann_g2",
    peridotite_h2o_wt: float = 0.0,
    pyroxenite_h2o_wt: float = 0.0,
    peridotite_chemistry: ChemistryBackend | None = None,
) -> list[Lithology]:
    """
    Two-lithology source (peridotite + pyroxenite/enriched end-member).

    ``backend="native"`` — analytic Katz + G2 (REEBOX default).
    ``backend="pymelt"`` — pyMelt lithology curves via ``pymelt_lithology_adapter``.
    """
    phi = float(np.clip(pyroxenite_frac, 0.0, 0.99))
    if backend == "pymelt":
        from .pymelt_lithology_adapter import heterogeneous_source_pymelt

        return heterogeneous_source_pymelt(
            pyroxenite_frac=phi,
            peridotite_key=peridotite_key,
            pyroxenite_key=pyroxenite_key,
            peridotite_h2o_wt=peridotite_h2o_wt,
            pyroxenite_h2o_wt=pyroxenite_h2o_wt,
            peridotite_chemistry=peridotite_chemistry,
        )
    if backend != "native":
        raise ValueError(f"unknown lithology backend: {backend}")
    if phi <= 0.0:
        return [dry_peridotite_lithology(u0=1.0)]
    if phi >= 0.99:
        return [dry_peridotite_lithology(u0=1.0)]
    return [
        dry_peridotite_lithology(u0=1.0 - phi),
        g2_pyroxenite_lithology(u0=phi),
    ]
