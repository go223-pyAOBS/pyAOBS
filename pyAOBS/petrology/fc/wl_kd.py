"""
Weaver & Langmuir (1990) + Langmuir (1992) Kd and stoichiometry (BASALT+langmuir.FOR).

Faithful transcription of ``KDCALC`` (KDMODE=4) and ``STOICH`` from the extended
Fortran driver.  Units: T in Kelvin, P in kbar (SPARM(2)).

The full ``STATE`` Newton–Raphson solver lives in ``wl_state.py``; ``wl_partition.py`` retains oxide surrogates when ``nl=0``.
"""

from __future__ import annotations

import math
from typing import Mapping

# Default intensive variables (BLOCK DATA: T=1273.16 K, P=0 kbar).
DEFAULT_T_K = 1273.16
DEFAULT_P_KBAR = 0.0

# Olivine (phase 2) — KDCALC lines 329-332
_OL_MGO = (3740.0, -1.87, 0.0008)
_OL_FEO = (3911.0, -2.50, 0.0006)

# Clinopyroxene (phase 3) — KDCALC lines 333-344
_CPX_MGO = (3798.0, -2.28, 0.004)
_CPX_CAA = (1738.0, -0.753, 0.009)
_CPX_TIO2 = (1034.0, -1.27, 0.005)
_CPX_NAAL = (2418.0, -2.30, 0.006)
_CPX_FEO2 = (5087.0, -4.48, 0.003)

# Plagioclase (phase 1) — KDCALC lines 348-352
_PLAG_CAA_A = 2446.0
_PLAG_CAA_B0 = 1.122
_PLAG_CAA_B1 = 0.2562
_PLAG_CAA_C = 0.012
_PLAG_NAAL_A0 = 3195.0
_PLAG_NAAL_A1 = 3283.0
_PLAG_NAAL_B0 = 2.318
_PLAG_NAAL_B1 = 1.885
_PLAG_NAAL_C = 0.007

_OXIDE_MW = {"MgO": 40.305, "FeO": 71.844, "CaO": 56.078, "Na2O": 61.979}


def mpa_to_kbar(p_mpa: float) -> float:
    return float(p_mpa) / 100.0


def kbar_to_mpa(p_kbar: float) -> float:
    return float(p_kbar) * 100.0


def arhenf_kd(a: float, b: float, cp: float, t_k: float, p_kbar: float) -> float:
    """ARHENF(A,B,CP,T) = 10**(A/T + B + CP*P)  [Langmuir-extended KDCALC]."""
    tk = max(float(t_k), 300.0)
    return float(10.0 ** (float(a) / tk + float(b) + float(cp) * float(p_kbar)))


def kd_olivine_mgo(t_k: float = DEFAULT_T_K, p_kbar: float = DEFAULT_P_KBAR) -> float:
    a, b, cp = _OL_MGO
    return arhenf_kd(a, b, cp, t_k, p_kbar)


def kd_olivine_feo(t_k: float = DEFAULT_T_K, p_kbar: float = DEFAULT_P_KBAR) -> float:
    a, b, cp = _OL_FEO
    return arhenf_kd(a, b, cp, t_k, p_kbar)


def kd_cpx_mgo(t_k: float = DEFAULT_T_K, p_kbar: float = DEFAULT_P_KBAR) -> float:
    a, b, cp = _CPX_MGO
    return arhenf_kd(a, b, cp, t_k, p_kbar)


def kd_cpx_feo(t_k: float = DEFAULT_T_K, p_kbar: float = DEFAULT_P_KBAR) -> float:
    return 0.24 * kd_cpx_mgo(t_k, p_kbar)


def kd_cpx_caa(t_k: float = DEFAULT_T_K, p_kbar: float = DEFAULT_P_KBAR) -> float:
    a, b, cp = _CPX_CAA
    return arhenf_kd(a, b, cp, t_k, p_kbar)


def wl_an_component(melt: Mapping[str, float]) -> float:
    """AN = CL(1)/(CL(1)+1.5*CL(2)) on W&L feldspar join (CAA / NAAL proxy)."""
    n_caa = float(melt.get("CaO", 0.0)) / _OXIDE_MW["CaO"]
    n_naal = 2.0 * float(melt.get("Na2O", 0.0)) / _OXIDE_MW["Na2O"]
    return float(n_caa / max(n_caa + 1.5 * n_naal, 1e-12))


def kd_plagioclase_caa(
    an: float,
    t_k: float = DEFAULT_T_K,
    p_kbar: float = DEFAULT_P_KBAR,
) -> float:
    b = _PLAG_CAA_B0 + _PLAG_CAA_B1 * float(an)
    tk = max(float(t_k), 300.0)
    return float(10.0 ** (_PLAG_CAA_A / tk - b + _PLAG_CAA_C * float(p_kbar)))


def kd_plagioclase_naal(
    an: float,
    t_k: float = DEFAULT_T_K,
    p_kbar: float = DEFAULT_P_KBAR,
) -> float:
    a = _PLAG_NAAL_A0 + _PLAG_NAAL_A1 * float(an)
    b = _PLAG_NAAL_B0 + _PLAG_NAAL_B1 * float(an)
    tk = max(float(t_k), 300.0)
    return float(10.0 ** (a / tk - b + _PLAG_NAAL_C * float(p_kbar)))


def olivine_fo_pct_from_kd(
    melt: Mapping[str, float],
    *,
    t_k: float = DEFAULT_T_K,
    p_kbar: float = DEFAULT_P_KBAR,
) -> float:
    """
    Fo (%) from separate MgO/FeO Kd (olivine phase 2 in KDCALC).

    Fo_mol ≈ (Mg_liq·Kd_Mg) / (Mg_liq·Kd_Mg + Fe_liq·Kd_Fe).
    """
    mg_mol = float(melt.get("MgO", 0.0)) / _OXIDE_MW["MgO"]
    fe_mol = float(melt.get("FeO", 0.0)) / _OXIDE_MW["FeO"]
    kd_mg = kd_olivine_mgo(t_k, p_kbar)
    kd_fe = kd_olivine_feo(t_k, p_kbar)
    num = mg_mol * kd_mg
    den = num + fe_mol * kd_fe
    if den <= 1e-12:
        return 90.0
    return float(max(20.0, min(95.0, 100.0 * num / den)))


def cpx_stoich_t(caj_mgo: float, caj_feo: float, caj_caa: float) -> float:
    """STOICH JP=3: T = 2*(CAJ(3,3)+CAJ(3,4)) + CAJ(3,5)."""
    return float(2.0 * (caj_mgo + caj_feo) + caj_caa)


def cpx_stoich_stable(
    t_stoich: float,
    p_kbar: float,
) -> bool:
    """STOICH: P>4 kbar requires T>=1.1, else T>=1.0."""
    threshold = 1.1 if float(p_kbar) > 4.0 else 1.0
    return float(t_stoich) >= threshold


def cpx_stoich_index_from_melt(
    melt: Mapping[str, float],
    *,
    t_k: float = DEFAULT_T_K,
    p_kbar: float = DEFAULT_P_KBAR,
) -> float:
    """
    Proxy CAJ(3,*) from melt × Kd (for saturation index without STATE).

    Returns T / threshold so values >= 1 mean STOICH-stable.
    """
    mg_mol = float(melt.get("MgO", 0.0)) / _OXIDE_MW["MgO"]
    fe_mol = float(melt.get("FeO", 0.0)) / _OXIDE_MW["FeO"]
    ca_mol = float(melt.get("CaO", 0.0)) / _OXIDE_MW["CaO"]
    caj_mg = mg_mol * kd_cpx_mgo(t_k, p_kbar)
    caj_fe = fe_mol * kd_cpx_feo(t_k, p_kbar)
    caj_ca = ca_mol * kd_cpx_caa(t_k, p_kbar)
    # Scale melt proxy to Fortran mineral-component magnitude (~O(1) at saturation).
    t_val = cpx_stoich_t(caj_mg, caj_fe, caj_ca)
    norm = max(mg_mol + fe_mol + ca_mol, 1e-9)
    t_scaled = t_val / norm
    threshold = 1.1 if float(p_kbar) > 4.0 else 1.0
    return float(t_scaled / threshold)


def plagioclase_saturation_boost(p_kbar: float) -> float:
    """Higher P increases plag Kd (+0.012/+0.007·P in log10 Kd) → earlier saturation."""
    return float(10.0 ** (0.010 * float(p_kbar)))


def plag_an_pct_from_kd(
    melt: Mapping[str, float],
    *,
    t_k: float = DEFAULT_T_K,
    p_kbar: float = DEFAULT_P_KBAR,
) -> float:
    """Plagioclase An (%) from Kd(1,1)/Kd(1,2) weighted feldspar join."""
    n_caa = float(melt.get("CaO", 0.0)) / _OXIDE_MW["CaO"]
    n_naal = 2.0 * float(melt.get("Na2O", 0.0)) / _OXIDE_MW["Na2O"]
    an = wl_an_component(melt)
    d_caa = kd_plagioclase_caa(an, t_k, p_kbar)
    d_naal = kd_plagioclase_naal(an, t_k, p_kbar)
    w_caa = d_caa * n_caa
    w_naal = d_naal * n_naal
    total = w_caa + w_naal
    if total <= 1e-12:
        return 70.0
    return float(max(20.0, min(95.0, 100.0 * w_caa / total)))
