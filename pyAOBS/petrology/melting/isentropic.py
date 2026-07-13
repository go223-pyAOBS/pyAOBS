"""
Incrementally isentropic decompression melting (Brown & Lesher 2016, REEBOX PRO eq. 2–3).

Shared T(P) for thermally equilibrated lithologies; entropy budget couples
melting rates (heterogeneous heat transfer). Chemistry uses batch melts at
each increment; column pooling is in ``rmc.py``.

See also Brown & Lesher (2014, Nat. Geosci.) for active source buoyancy and RMC.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from petrology.active_upwelling import ADIABAT_GRAD_C_PER_GPA, adiabat_mb88_c, p0_from_tp

from .lithology import CP_J_KG_K, Lithology, RHO_KG_M3

_OXIDE_KEYS = (
    "SiO2", "TiO2", "Al2O3", "Cr2O3", "FeO", "MgO", "CaO", "Na2O", "K2O", "P2O5", "H2O",
)


@dataclass
class IsentropicStep:
    p_gpa: float
    t_c: float
    f_by_name: dict[str, float]
    df_dp_by_name: dict[str, float]
    dtdp_c_per_gpa: float


@dataclass
class IsentropicPath:
    tp_c: float
    p0_gpa: float
    pf_gpa: float
    steps: list[IsentropicStep] = field(default_factory=list)
    # Source mass fractions φ_i (pyMelt ``proportions``); used for bulk F = Σ φ_i F_i.
    u0_by_name: dict[str, float] = field(default_factory=dict)

    def f_bulk_at(self, step: IsentropicStep) -> float:
        """Bulk melt fraction Σ φ_i F_i (falls back to Σ F_i if proportions missing)."""
        if self.u0_by_name:
            return float(
                sum(
                    float(self.u0_by_name.get(name, 0.0)) * float(fi)
                    for name, fi in step.f_by_name.items()
                )
            )
        return float(sum(step.f_by_name.values()))

    @property
    def f_total(self) -> float:
        if not self.steps:
            return 0.0
        return self.f_bulk_at(self.steps[-1])

    def fbar(self) -> float:
        """Mean bulk melt fraction along the path (Σ φ_i F_i)."""
        if not self.steps:
            return 0.0
        return float(np.mean([self.f_bulk_at(st) for st in self.steps]))


def _adiabat_t(tp_c: float, p_gpa: float) -> float:
    return adiabat_mb88_c(tp_c, p_gpa)


def _bulk_rho(lithologies: list[Lithology], f: dict[str, float]) -> float:
    num = 0.0
    den = 0.0
    for lith in lithologies:
        u_solid = lith.u0 * max(1.0 - f.get(lith.name, 0.0), 0.0)
        num += u_solid * lith.rho_kg_m3
        den += u_solid
    return num / max(den, 1e-9) if den > 0 else RHO_KG_M3


def _solve_dtdp_dfdp(
    lithologies: list[Lithology],
    *,
    p_gpa: float,
    t_c: float,
    f: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """
    Entropy balance for shared dT/dP (Katz & Rudge 2011 / Phipps Morgan 2001).

    For melting lithology i on its T(P,F) curve:
        dT/dP = A_i + B_i * dF_i/dP
    Isentropic: Σ u_i [Cp dT/dP + T ΔS_i/(1-F_i) dF_i/dP] = 0
    """
    u_solid = {L.name: L.u0 * max(1.0 - f.get(L.name, 0.0), 0.0) for L in lithologies}
    melting = [
        L
        for L in lithologies
        if L.u0 > 0.0 and L.is_melting(p_gpa, t_c, f.get(L.name, 0.0))
    ]

    if not melting:
        return ADIABAT_GRAD_C_PER_GPA, {L.name: 0.0 for L in lithologies}

    a_term: dict[str, float] = {}
    b_term: dict[str, float] = {}
    for L in melting:
        fi = float(f.get(L.name, 0.0))
        a_term[L.name] = L.dT_dP_at_F(p_gpa, fi)
        b_term[L.name] = max(L.dT_dF_at_P(p_gpa, fi), 1.0)

    rho = _bulk_rho(lithologies, f)

    num = 0.0
    den = sum(u_solid[L.name] * CP_J_KG_K for L in lithologies)

    for L in melting:
        fi = float(f.get(L.name, 0.0))
        u = u_solid[L.name]
        if u <= 0.0:
            continue
        ai = a_term[L.name]
        bi = b_term[L.name]
        ds = L.delta_s_fusion
        num += u * CP_J_KG_K * ai
        num += u * t_c * ds / max(1.0 - fi, 1e-6) * (ai / bi)
        den += u * CP_J_KG_K * t_c * ds / max(1.0 - fi, 1e-6) / bi

    dtdp = num / max(den, 1e-9)
    df_dp: dict[str, float] = {L.name: 0.0 for L in lithologies}
    for L in melting:
        if u_solid[L.name] <= 0.0:
            continue
        ai = a_term[L.name]
        bi = b_term[L.name]
        df_dp[L.name] = (dtdp - ai) / bi

    return dtdp, df_dp


def integrate_isentropic_path(
    lithologies: list[Lithology],
    *,
    tp_c: float,
    pf_gpa: float,
    n_steps: int = 120,
    p0_gpa: float | None = None,
) -> IsentropicPath:
    """
    Decompress from P0 (solidus) to Pf along incrementally isentropic path.

    Returns pressure history with F_i(P) and T(P).
    """
    if p0_gpa is None:
        p0_gpa = p0_from_tp(tp_c)
    p0 = float(p0_gpa)
    pf = float(min(pf_gpa, p0 - 0.01))
    if pf >= p0:
        pf = max(p0 - 0.05, 0.05)

    f_state = {L.name: 0.0 for L in lithologies}
    t_c = _adiabat_t(tp_c, p0)
    path = IsentropicPath(
        tp_c=float(tp_c),
        p0_gpa=p0,
        pf_gpa=pf,
        u0_by_name={L.name: float(L.u0) for L in lithologies},
    )

    pressures = np.linspace(p0, pf, int(n_steps) + 1)
    for i in range(len(pressures) - 1):
        p = float(pressures[i])
        dp = float(pressures[i + 1] - pressures[i])  # negative
        dtdp, df_dp = _solve_dtdp_dfdp(lithologies, p_gpa=p, t_c=t_c, f=f_state)
        for L in lithologies:
            dfi = df_dp[L.name] * dp
            f_state[L.name] = float(np.clip(f_state[L.name] + dfi, 0.0, 0.999))
        t_c += dtdp * dp
        path.steps.append(
            IsentropicStep(
                p_gpa=p,
                t_c=t_c,
                f_by_name=dict(f_state),
                df_dp_by_name=dict(df_dp),
                dtdp_c_per_gpa=dtdp,
            )
        )

    path.steps.append(
        IsentropicStep(
            p_gpa=float(pressures[-1]),
            t_c=t_c + dtdp * (pressures[-1] - pressures[-2]) if len(pressures) > 1 else t_c,
            f_by_name=dict(f_state),
            df_dp_by_name={L.name: 0.0 for L in lithologies},
            dtdp_c_per_gpa=dtdp if path.steps else ADIABAT_GRAD_C_PER_GPA,
        )
    )
    return path
