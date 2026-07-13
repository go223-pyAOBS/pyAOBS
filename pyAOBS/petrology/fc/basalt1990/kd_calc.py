"""
KDCALC from original BASALT.FOR (1990, 1 atm).

ARHENF(A,B,T) = EXP(2.302585*(A/T)) + B  =  10**(A/T) + B

This differs from the Langmuir (1992) extension which uses 10**(A/T + B + C*P).
"""

from __future__ import annotations

import numpy as np

from .common import DEFAULT_T_K, NCOMP, NPHAS


def arhenf_1990(a: float, b: float, t_k: float) -> float:
    """Fortran ARHENF(A,B,T) in original BASALT.FOR KDCALC."""
    tk = max(float(t_k), 1.0)
    return float(10.0 ** (float(a) / tk) + float(b))


def kd_olivine_mgo_1990(t_k: float = DEFAULT_T_K) -> float:
    return arhenf_1990(2715.0, -1.158, t_k)


def kd_olivine_feo_1990(t_k: float = DEFAULT_T_K) -> float:
    return arhenf_1990(4230.0, -2.741, t_k)


def kd_cpx_mgo_1990(t_k: float = DEFAULT_T_K) -> float:
    return arhenf_1990(3798.0, -2.28, t_k)


def kd_cpx_feo_1990(t_k: float = DEFAULT_T_K) -> float:
    return 0.24 * kd_cpx_mgo_1990(t_k)


def kd_cpx_cawo_1990(t_k: float = DEFAULT_T_K) -> float:
    return arhenf_1990(1738.0, -0.753, t_k)


def kd_cpx_tio2_1990(t_k: float = DEFAULT_T_K) -> float:
    return arhenf_1990(1034.0, -1.27, t_k)


def kd_cpx_naal_1990(t_k: float = DEFAULT_T_K) -> float:
    return arhenf_1990(2418.0, -2.30, t_k)


def kd_cpx_feo2_1990(t_k: float = DEFAULT_T_K) -> float:
    return arhenf_1990(5087.0, -4.48, t_k)


def kd_plagioclase_caa_1990(an: float, t_k: float = DEFAULT_T_K) -> float:
    b = -(1.122 + 0.2562 * float(an))
    return arhenf_1990(2446.0, b, t_k)


def kd_plagioclase_naal_1990(an: float, t_k: float = DEFAULT_T_K) -> float:
    a = 3195.0 + 3283.0 * float(an)
    b = -(2.318 + 1.885 * float(an))
    return arhenf_1990(a, b, t_k)


def an_from_clj(clj: np.ndarray) -> float:
    """AN = CL(1) / (CL(1) + 1.5*CL(2)) — Fortran KDCALC mode 3."""
    cl1 = float(clj[0])
    cl2 = float(clj[1])
    return cl1 / max(cl1 + 1.5 * cl2, 1e-30)


class KDCalc1990:
    """Mutable FKDAJ matrix (Fortran COMMON FKDAJ)."""

    def __init__(self, t_k: float = DEFAULT_T_K) -> None:
        self.t_k = float(t_k)
        self.fkda = np.zeros((NPHAS, NCOMP), dtype=float)

    def calc(self, kdmode: int, clj: np.ndarray | None = None) -> None:
        """
        KDCALC(KDMODE): 1=reset, 2=T-dependent ol+cpx, 3=plag composition.

        kdmode=0: no-op (keep current FKDAJ — for solver regression tests).
        """
        if kdmode == 0:
            return
        if kdmode == 1:
            self.fkda[:, :] = 0.0

        if kdmode >= 2:
            self.fkda[1, 2] = kd_olivine_mgo_1990(self.t_k)
            self.fkda[1, 3] = kd_olivine_feo_1990(self.t_k)
            self.fkda[2, 2] = kd_cpx_mgo_1990(self.t_k)
            self.fkda[2, 3] = kd_cpx_feo_1990(self.t_k)
            self.fkda[2, 4] = kd_cpx_cawo_1990(self.t_k)
            self.fkda[2, 5] = kd_cpx_tio2_1990(self.t_k)
            self.fkda[2, 0] = kd_cpx_naal_1990(self.t_k)
            self.fkda[2, 1] = kd_cpx_feo2_1990(self.t_k)

        if kdmode == 3:
            if clj is None:
                raise ValueError("KDCALC(3) requires liquid composition CLJ")
            an = an_from_clj(clj)
            self.fkda[0, 0] = kd_plagioclase_caa_1990(an, self.t_k)
            self.fkda[0, 1] = kd_plagioclase_naal_1990(an, self.t_k)
