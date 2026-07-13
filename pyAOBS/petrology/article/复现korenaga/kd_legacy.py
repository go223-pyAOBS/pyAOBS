"""Legacy Kd / saturation helpers for Korenaga article scripts.

Two 1-atm ARHENF forms exist in the literature and in this folder — do **not** mix them:

1. **basalt1990** — original BASALT.FOR::

       ARHENF(A,B,T) = 10**(A/T) + B

   Canonical production code: ``petrology.fc.basalt1990.kd_calc.arhenf_1990``.
   Used by ``scan_composition.py`` and ``debug_original_algorithm*.py``.

2. **modern** — ``basalt_modern.f90`` / most exploratory scripts::

       ARHENF(A,B,T) = 10**(A/T + B)   (floor at 1e-6)

   Matches Langmuir-style Arrhenius in the exponent.  Used by
   ``check_saturation.py``, ``find_demo.py``, ``check_demo.py``,
   ``debug_liquidus.py``, ``debug_step2.py``.

3. **langmuir1992** — pressure extension ``10**(A/T + B + C*P)`` (P in kbar).
   Production: ``petrology.fc.wl_kd.arhenf_kd``.  Article helpers:
   ``q_saturation_pressure`` / ``find_liquidus_pressure``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Sequence

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.fc.basalt1990.kd_calc import arhenf_1990 as _arhenf_1990_raw

CAA, NAAL, MGO, FEO, CAWO, TIO2, SIO2 = range(7)

ArhenfForm = Literal["basalt1990", "modern"]

# Stoichiometry: plag, ol, cpx (Fortran UAJ / TA; 0-based)
TA: tuple[float, float, float] = (1.0, 2.0 / 3.0, 1.0)
UAJ: tuple[tuple[float, ...], ...] = (
    (5.0 / 3.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
    (4.0 / 3.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0),
)


def arhenf_basalt1990(a: float, b: float, t_k: float) -> float:
    """Original BASALT.FOR: ``10**(A/T) + B``."""
    return float(_arhenf_1990_raw(a, b, t_k))


def arhenf_modern(a: float, b: float, t_k: float) -> float:
    """``basalt_modern.f90``: ``10**(A/T + B)``, floored at 1e-6."""
    tk = max(float(t_k), 1.0)
    kd = 10.0 ** (float(a) / tk + float(b))
    return max(kd, 1.0e-6)


def arhenf(a: float, b: float, t_k: float, *, form: ArhenfForm = "modern") -> float:
    """Dispatch ARHENF by form (default: modern, matching most article scripts)."""
    if form == "basalt1990":
        return arhenf_basalt1990(a, b, t_k)
    if form == "modern":
        return arhenf_modern(a, b, t_k)
    raise ValueError(f"Unknown ARHENF form: {form!r}")


def kdcalc(
    t_k: float,
    cs: Sequence[float],
    *,
    form: ArhenfForm = "modern",
) -> tuple[list[list[float]], float]:
    """1 atm Kd matrix (3×7) and An number.

    Default ``form="modern"`` matches ``basalt_modern.f90`` / check_saturation.
    Pass ``form="basalt1990"`` for original BASALT.FOR arithmetic.
    """
    an = cs[CAA] / (cs[CAA] + 1.5 * cs[NAAL])
    kd = [[0.0] * 7 for _ in range(3)]
    kd[0][CAA] = arhenf(2446.0, -(1.122 + 0.2562 * an), t_k, form=form)
    kd[0][NAAL] = arhenf(3195.0 + 3283.0 * an, -(2.318 + 1.885 * an), t_k, form=form)
    kd[1][MGO] = arhenf(2715.0, -1.158, t_k, form=form)
    kd[1][FEO] = arhenf(4230.0, -2.741, t_k, form=form)
    kd[2][MGO] = arhenf(3798.0, -2.28, t_k, form=form)
    kd[2][FEO] = 0.24 * kd[2][MGO]
    kd[2][CAWO] = arhenf(1738.0, -0.753, t_k, form=form)
    kd[2][TIO2] = arhenf(1034.0, -1.27, t_k, form=form)
    kd[2][CAA] = arhenf(2418.0, -2.30, t_k, form=form)
    kd[2][NAAL] = arhenf(5087.0, -4.48, t_k, form=form)
    return kd, an


def q_values(
    t_k: float,
    cs: Sequence[float],
    *,
    form: ArhenfForm = "modern",
) -> tuple[list[float], float]:
    """Saturation indices Q for plag / ol / cpx at 1 atm."""
    kd, an = kdcalc(t_k, cs, form=form)
    q: list[float] = []
    for phase in range(3):
        tq = -TA[phase]
        for j in range(7):
            tq += UAJ[phase][j] * kd[phase][j] * cs[j]
        q.append(tq)
    return q, an


def q_plag_f(
    t_k: float,
    cs: Sequence[float],
    f: float,
    *,
    form: ArhenfForm = "modern",
) -> float:
    """Plagioclase Q with fractional melt F (Rayleigh-style liquid composition)."""
    kd, _ = kdcalc(t_k, cs, form=form)
    q = -TA[0]
    for j in range(7):
        if UAJ[0][j] == 0.0:
            continue
        caj = kd[0][j] * cs[j] / (1.0 + f * (kd[0][j] - 1.0))
        q += UAJ[0][j] * caj
    return q


def cpx_stoich(
    t_k: float,
    cs: Sequence[float],
    *,
    form: ArhenfForm = "modern",
) -> float:
    """Cpx saturation stoichiometry term (debug helper)."""
    kd, _ = kdcalc(t_k, cs, form=form)
    return 2.0 * (kd[2][MGO] * cs[MGO] + kd[2][FEO] * cs[FEO]) + kd[2][CAWO] * cs[CAWO]


def q_saturation_pressure(
    cs: Sequence[float],
    t_k: float,
    p_kbar: float,
) -> tuple[float, float, float]:
    """Langmuir (1992) pressure-extended Q — plag / ol / cpx (p in kbar).

    Note: the Ca/Na terms in Q_cpx reuse plag Kd coefficients — this matches the
    original ``check_lowMgFe.py`` / ``find_liquidus.py`` exploratory scripts
    (not the full UAJ×Kd_cpx matrix in ``q_values``).
    """
    p = float(p_kbar)
    an = cs[CAA] / (cs[CAA] + 1.5 * cs[NAAL]) if (cs[CAA] + 1.5 * cs[NAAL]) > 0 else 0.4
    kd_plag_ca = 10 ** (2446 / t_k - (1.122 + 0.2562 * an) + 0.012 * p)
    kd_plag_na = 10 ** ((3195 + 3283 * an) / t_k - (2.318 + 1.885 * an) + 0.007 * p)
    kd_ol_mg = 10 ** (3740 / t_k - 1.87 + 0.0008 * p)
    kd_ol_fe = 10 ** (3911 / t_k - 2.50 + 0.0006 * p)
    kd_cpx_mg = 10 ** (3798 / t_k - 2.28 + 0.004 * p)
    kd_cpx_fe = 0.24 * kd_cpx_mg
    kd_cpx_ca = 10 ** (1738 / t_k - 0.753 + 0.009 * p)
    kd_cpx_ti = 10 ** (1034 / t_k - 1.27 + 0.005 * p)

    q_plag = -1.0 + (5.0 / 3.0) * kd_plag_ca * cs[CAA] + 2.5 * kd_plag_na * cs[NAAL]
    q_ol = -2.0 / 3.0 + kd_ol_mg * cs[MGO] + kd_ol_fe * cs[FEO]
    q_cpx = (
        -1.0
        + (4.0 / 3.0) * kd_plag_ca * cs[CAA]
        + 2.0 * kd_plag_na * cs[NAAL]
        + 2.0 * kd_ol_mg * cs[MGO]
        + 2.0 * kd_ol_fe * cs[FEO]
        + kd_cpx_ca * cs[CAWO]
        + kd_cpx_ti * cs[TIO2]
    )
    return q_plag, q_ol, q_cpx


def find_liquidus_pressure(
    cs: Sequence[float],
    p_kbar: float,
    *,
    t_min_k: float = 1200.0 + 273.16,
    t_max_k: float = 4000.0 + 273.16,
) -> float | None:
    """Highest T (K) where any phase Q ≤ 0 at pressure p (kbar)."""
    n = 1000
    dt = (t_max_k - t_min_k) / n
    tl: float | None = None
    for i in range(n + 1):
        tk = t_max_k - i * dt
        qp, qo, qc = q_saturation_pressure(cs, tk, p_kbar)
        if min(qp, qo, qc) <= 0.0:
            tl = tk
            break
    if tl is None:
        return None
    lo, hi = tl, min(tl + dt, t_max_k)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        qp, qo, qc = q_saturation_pressure(cs, mid, p_kbar)
        if min(qp, qo, qc) <= 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _an_from_clj_1based(clj_1based: Sequence[float] | None) -> float:
    if clj_1based is None:
        return 0.4
    denom = clj_1based[1] + 1.5 * clj_1based[2]
    return clj_1based[1] / denom if denom > 0 else 0.4


def kdcalc_fortran(
    t_k: float,
    kdmode: int = 3,
    clj_1based: Sequence[float] | None = None,
    *,
    form: ArhenfForm = "basalt1990",
) -> list[list[float]]:
    """4×8 Kd matrix matching BASALT.FOR layout (1-based row/col; index 0 unused).

    Default ``form="basalt1990"`` matches ``debug_original_algorithm*.py``.
    Pass ``form="modern"`` for scripts that used ``10**(A/T+B)`` inline
    (``debug_liquidus.py``, ``debug_step2.py``).
    """
    kd = [[0.0] * 8 for _ in range(4)]
    if kdmode == 1:
        return kd
    tk = float(t_k)
    if kdmode >= 2:
        kd[2][3] = arhenf(2715.0, -1.158, tk, form=form)
        kd[2][4] = arhenf(4230.0, -2.741, tk, form=form)
        kd[3][3] = arhenf(3798.0, -2.28, tk, form=form)
        kd[3][4] = 0.24 * kd[3][3]
        kd[3][5] = arhenf(1738.0, -0.753, tk, form=form)
        kd[3][6] = arhenf(1034.0, -1.27, tk, form=form)
        kd[3][1] = arhenf(2418.0, -2.30, tk, form=form)
        kd[3][2] = arhenf(5087.0, -4.48, tk, form=form)
    if kdmode == 3:
        an = _an_from_clj_1based(clj_1based)
        kd[1][1] = arhenf(2446.0, -(1.122 + 0.2562 * an), tk, form=form)
        kd[1][2] = arhenf(3195.0 + 3283.0 * an, -(2.318 + 1.885 * an), tk, form=form)
    return kd


def q_at_fa0_fortran(
    t_k: float,
    cs_1based: Sequence[float],
    *,
    form: ArhenfForm = "modern",
) -> tuple[float, float, float]:
    """Saturation Q (plag / ol / cpx) at F=0 with Fortran 1-based bulk composition.

    Default ``form="modern"`` matches original ``debug_liquidus.py`` inline Kd.
    """
    kd = kdcalc_fortran(t_k, kdmode=3, form=form)
    q_plag = -1.0 + 1.666667 * kd[1][1] * cs_1based[1] + 2.5 * kd[1][2] * cs_1based[2]
    q_ol = -0.666667 + kd[2][3] * cs_1based[3] + kd[2][4] * cs_1based[4]
    q_cpx = (
        -1.0
        + 1.333333 * kd[3][1] * cs_1based[1]
        + 2.0 * kd[3][2] * cs_1based[2]
        + 2.0 * kd[3][3] * cs_1based[3]
        + 2.0 * kd[3][4] * cs_1based[4]
        + kd[3][5] * cs_1based[5]
        + kd[3][6] * cs_1based[6]
    )
    return q_plag, q_ol, q_cpx
