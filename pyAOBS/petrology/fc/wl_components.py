"""
W&L (1990) liquid components ↔ oxide wt% conversion.

Fortran explicit components (BLOCK DATA):
  1 CAA (CaAl2O4)  2 NAAL (NaAlO2)  3 MGO  4 FEO  5 CAWO (CaSiO3)  6 TIO2
  7 SIO2 implicit via CIMPL
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

_OXIDE_KEYS = ("SiO2", "TiO2", "Al2O3", "Cr2O3", "FeO", "MgO", "CaO", "Na2O", "K2O")


def normalize_melt_oxides(oxides: Mapping[str, float]) -> dict[str, float]:
    out = {k: float(max(0.0, oxides.get(k, 0.0))) for k in _OXIDE_KEYS}
    s = sum(out.values())
    if s <= 0:
        raise ValueError("Invalid melt composition")
    return {k: 100.0 * v / s for k, v in out.items()}

NCOMP = 6
NCOMPT = 7
NPHAS = 3
SIO2_IDX = 6

MW = {
    "SiO2": 60.083,
    "TiO2": 79.865,
    "Al2O3": 101.961,
    "FeO": 71.844,
    "MgO": 40.304,
    "CaO": 56.078,
    "Na2O": 61.979,
    "K2O": 94.196,
    "Cr2O3": 52.0,
}

# BLOCK DATA stoichiometry (1-based Fortran → 0-based below).
TA = np.array([1.0, 0.666667, 1.0], dtype=float)
UAJ = np.zeros((NPHAS, NCOMP), dtype=float)
UAJ[0, 0] = 1.666667
UAJ[0, 1] = 2.5
UAJ[1, 2] = 1.0
UAJ[1, 3] = 1.0
UAJ[2, 0] = 1.333333
UAJ[2, 1] = 2.0
UAJ[2, 2] = 2.0
UAJ[2, 3] = 2.0
UAJ[2, 4] = 1.0
UAJ[2, 5] = 1.0

IMPDIM = 3
IMPL = np.array([0, 1, 2], dtype=int)  # phases 1,2,3 → plag, ol, cpx
IMCL = np.full(IMPDIM, SIO2_IDX, dtype=int)
DA0 = np.zeros(IMPDIM, dtype=float)
DAJ = np.zeros((IMPDIM, NCOMP), dtype=float)
DAJ[0, :] = [0.666667, 0.0, 0.3333, 0.0, 0.0, 0.0]
DAJ[1, :] = [1.5, 0.0, 1.0, 0.0, 0.0, 0.0]
DAJ[2, :] = [0.0, 0.5, 1.0, 0.0, 0.0, 0.0]

# Oxide columns: SiO2, TiO2, Al2O3, FeO, MgO, CaO, Na2O
_COMP_TO_OX = np.array(
    [
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],  # CAA
        [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5],  # NAAL
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # MGO
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # FEO
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # CAWO
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # TIO2
    ],
    dtype=float,
)
_OX_ORDER = ("SiO2", "TiO2", "Al2O3", "FeO", "MgO", "CaO", "Na2O")


def _oxide_moles(wt: Mapping[str, float]) -> np.ndarray:
    w = normalize_melt_oxides(wt)
    return np.array([float(w[k]) / MW[k] for k in _OX_ORDER], dtype=float)


def oxides_wt_to_csj(wt: Mapping[str, float]) -> np.ndarray:
    """Map normalized oxide wt% to W&L 6-component CSJ (sum=1; SiO2 implicit via CIMPL)."""
    w = normalize_melt_oxides(wt)
    n = _oxide_moles(w)
    n_sio2, n_tio2, n_al2o3, n_feo, n_mgo, n_cao, n_na2o = n

    tio2 = max(0.0, n_tio2)
    naal = max(0.0, 2.0 * n_na2o)
    al_rem = max(0.0, n_al2o3 - 0.5 * naal)
    caa = max(0.0, min(n_cao, al_rem))
    al_rem -= caa
    mgo = max(0.0, n_mgo)
    feo = max(0.0, n_feo)
    # All Si assigned to CAWO; excess over Ca is handled implicitly by CIMPL during FC.
    cawo = max(0.0, n_sio2)
    caa = max(0.0, min(caa, max(0.0, n_cao - cawo))) if n_cao > cawo else caa

    csj = np.zeros(NCOMPT, dtype=float)
    csj[:NCOMP] = [caa, naal, mgo, feo, cawo, tio2]
    s = float(np.sum(csj[:NCOMP]))
    if s <= 1e-12:
        raise ValueError("empty composition")
    csj[:NCOMP] /= s
    return csj


def csj_to_oxides_wt(csj: np.ndarray, *, sio2: float = 0.0) -> dict[str, float]:
    """Reconstruct oxide wt% from components + implicit SiO2 (from CIMPL)."""
    c = np.asarray(csj, dtype=float).ravel()
    if c.size < NCOMP:
        c = np.pad(c, (0, NCOMP - c.size))
    ox_mol = _COMP_TO_OX.T @ c + np.array([float(sio2), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    mw_vec = np.array([MW[k] for k in _OX_ORDER], dtype=float)
    wt_core = {k: float(ox_mol[i] * MW[k]) for i, k in enumerate(_OX_ORDER)}
    wt = {k: 0.0 for k in _OXIDE_KEYS}
    wt.update(wt_core)
    return normalize_melt_oxides(wt)


def mineral_fo_an_di(caj: np.ndarray) -> tuple[float, float, float]:
    """Fo, An, Di (%) from CAJ mineral component matrix (3 x NCOMP)."""
    ol_mg = float(caj[1, 2])
    ol_fe = float(caj[1, 3])
    fo = 100.0 * ol_mg / max(ol_mg + ol_fe, 1e-12)
    pl_ca = float(caj[0, 0])
    pl_na = float(caj[0, 1])
    an = 100.0 * pl_ca / max(pl_ca + 1.5 * pl_na, 1e-12)
    cx_mg = float(caj[2, 2])
    cx_fe = float(caj[2, 3])
    di = 100.0 * cx_mg / max(cx_mg + cx_fe, 1e-12)
    return float(np.clip(fo, 20, 95)), float(np.clip(an, 20, 95)), float(np.clip(di, 20, 95))
