"""KKHS02 Step-4 melting laws for H–Vp sensitivity (Fig.12–13)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class MeltingLaw(Protocol):
    """Melt fraction F and mean F̄ as functions of ΔP = P₀ − P_f (GPa)."""

    name: str

    def f_total(self, delta_p_gpa: float) -> float:
        """Total melt fraction F at the base of the melting column (ΔP)."""

    def fbar_mean(self, delta_p_gpa: float) -> float:
        """Depth-mean melt fraction F̄ integrated over 0 → ΔP."""


@dataclass(frozen=True)
class LinearMeltingLaw:
    """F = (∂F/∂P)·ΔP (eq. 6),  F̄ = ½·F  (eq. 8); §4/Fig.11–12 standard 12%/GPa."""

    dfdp_per_gpa: float = 0.12
    name: str = "linear"

    def f_total(self, delta_p_gpa: float) -> float:
        dp = max(float(delta_p_gpa), 0.0)
        return self.dfdp_per_gpa * dp

    def fbar_mean(self, delta_p_gpa: float) -> float:
        return 0.5 * self.f_total(delta_p_gpa)


# KKHS02 Fig.13b / Asimow et al. (1997) piecewise parameterization (ΔP in GPa)
ASIMOW_STAGE1_DPDP = 0.035
ASIMOW_STAGE1_DP_MAX = 1.15
ASIMOW_STAGE2_DPDP = 0.23
ASIMOW_STAGE2_F_AT_J1 = 0.04
ASIMOW_STAGE2_DP_MAX = 1.75
ASIMOW_STAGE3_DPDP = 0.113
ASIMOW_STAGE3_F_AT_J2 = 0.18


def asimow97_f_total(delta_p_gpa: float) -> float:
    """
    Three-stage F(ΔP) from KKHS02 Fig.13b caption.

    F₁ = 0.035·ΔP           (ΔP < 1.15 GPa)   — low melt productivity near solidus
    F₂ = 0.23(ΔP−1.15)+0.04 (1.15 ≤ ΔP < 1.75 GPa)
    F₃ = 0.113(ΔP−1.75)+0.18 (ΔP ≥ 1.75 GPa)  — cpx-out / higher productivity
    """
    dp = max(float(delta_p_gpa), 0.0)
    if dp < ASIMOW_STAGE1_DP_MAX:
        return ASIMOW_STAGE1_DPDP * dp
    if dp < ASIMOW_STAGE2_DP_MAX:
        return ASIMOW_STAGE2_DPDP * (dp - ASIMOW_STAGE1_DP_MAX) + ASIMOW_STAGE2_F_AT_J1
    return ASIMOW_STAGE3_DPDP * (dp - ASIMOW_STAGE2_DP_MAX) + ASIMOW_STAGE3_F_AT_J2


def asimow97_fbar_mean(delta_p_gpa: float) -> float:
    """F̄ = (1/ΔP) ∫₀^ΔP F(u) du for the piecewise linear Asimow parameterization."""
    dp = max(float(delta_p_gpa), 0.0)
    if dp <= 0.0:
        return 0.0

    j1 = ASIMOW_STAGE1_DP_MAX
    j2 = ASIMOW_STAGE2_DP_MAX
    f1 = ASIMOW_STAGE2_F_AT_J1
    f2 = ASIMOW_STAGE3_F_AT_J2

    if dp <= j1:
        return ASIMOW_STAGE1_DPDP * dp / 2.0

    int1 = ASIMOW_STAGE1_DPDP * j1 * j1 / 2.0
    if dp <= j2:
        x = dp - j1
        int2 = ASIMOW_STAGE2_DPDP * x * x / 2.0 + f1 * x
        return (int1 + int2) / dp

    x2 = j2 - j1
    int2 = ASIMOW_STAGE2_DPDP * x2 * x2 / 2.0 + f1 * x2
    x3 = dp - j2
    int3 = ASIMOW_STAGE3_DPDP * x3 * x3 / 2.0 + f2 * x3
    return (int1 + int2 + int3) / dp


@dataclass(frozen=True)
class Asimow97MeltingLaw:
    """KKHS02 Fig.13b three-stage melting function (crude Asimow et al. 1997 proxy)."""

    name: str = "asimow97"

    def f_total(self, delta_p_gpa: float) -> float:
        return asimow97_f_total(delta_p_gpa)

    def fbar_mean(self, delta_p_gpa: float) -> float:
        return asimow97_fbar_mean(delta_p_gpa)


# KKHS02 §4 / Fig.11–12 standard model: linear (∂F/∂P)_S = 12%/GPa
FIG11_MELTING = LinearMeltingLaw(dfdp_per_gpa=0.12, name="fig11_linear_12")
FIG12_MELTING = LinearMeltingLaw(dfdp_per_gpa=0.12, name="fig12_linear_12")
FIG13A_MELTING = LinearMeltingLaw(dfdp_per_gpa=0.16, name="fig13a_linear_16")
FIG13B_MELTING = Asimow97MeltingLaw()
