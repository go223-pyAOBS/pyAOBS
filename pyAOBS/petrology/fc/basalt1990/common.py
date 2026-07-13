"""
BLOCK DATA constants from Weaver & Langmuir (1990) BASALT.FOR (version 2.0).

Phases (1-based Fortran): 1=PLAG, 2=OL, 3=CPX
Components: CAA, NAAL, MGO, FEO, CAWO, TIO2 (+ implicit SIO2)
"""

from __future__ import annotations

import numpy as np

NPHAS = 3
NCOMP = 6
NCIMP = 1
NCOMPT = 7
NPDIM = 10
NCDIM = 20
SIO2_IDX = 6  # 0-based index of implicit SiO2

PNAMEA = ("PLAG", "OL", "CPX")
CNAMEJ = ("CAA", "NAAL", "MGO", "FEO", "CAWO", "TIO2", "SIO2")

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

# Fortran IMPL/1,2,3,7*0/ — 1-based phase indices; store 0-based active entries.
IMPDIM = 10
IMPL = np.array([0, 1, 2] + [0] * 7, dtype=int)
IMCL = np.full(IMPDIM, SIO2_IDX, dtype=int)
DA0 = np.zeros(IMPDIM, dtype=float)

DAJ = np.zeros((IMPDIM, NCOMP), dtype=float)
DAJ[0, :] = [0.666667, 0.0, 0.3333, 0.0, 0.0, 0.0]
DAJ[1, :] = [1.5, 0.0, 1.0, 0.0, 0.0, 0.0]
DAJ[2, :] = [0.0, 0.5, 1.0, 0.0, 0.0, 0.0]
# Rows 3–4 exist in BLOCK DATA but IMPL(4:)=0 → unused by CIMPL.
DAJ[3, :] = [0.0, 0.5, 1.0, 0.0, 0.0, 0.0]
DAJ[4, :] = [1.6, 0.0, 1.0, 0.0, 0.0, 0.0]

DEFAULT_T_K = 1273.16
MAX_ITER = 20
TOL = 1e-5
