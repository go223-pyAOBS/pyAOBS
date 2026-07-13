"""
Faithful Python port of Weaver & Langmuir (1990) BASALT.FOR (version 2.0).

Subroutines ported line-by-line from the original CP/M Fortran:

| Fortran     | Python module                         |
|-------------|---------------------------------------|
| BLOCK DATA  | ``common.py``                         |
| KDCALC      | ``kd_calc.py``  — ARHENF = 10**(A/T)+B |
| STATE       | ``solver.py``                         |
| CIMPL       | ``solver.Basalt1990System.cimpl``     |
| MATEQ       | ``solver.Basalt1990System.mateq``     |
| STOICH      | ``solver.Basalt1990System.stoich``    |
| DRIVER      | ``driver.py``                         |

Known differences from the Fortran listing (documented in code):

1. **KDCALC(2)** — Fortran STATE only calls KDCALC(1) then KDCALC(3); mode 2
   (T-dependent ol/cpx Kd) is never invoked from STATE. ``fill_temp_kd=True``
   (default) also runs mode 2 so ol/cpx Kd are non-zero.
2. **SPARM(1)=TEMP** — original DRIVER does not update SPARM before STATE;
   ``sync_sparm_temp=True`` (default) sets temperature for KDCALC.
3. **MATEQ** — pivot row is scaled once (Fortran listing double-scales the
   diagonal); fix also applied in ``wl_state.py``.

The 1990 ARHENF Kd values are much larger than the Langmuir (1992) reformulation
``10**(A/T+B)``; typical MORB may remain liquid-only at 1273 K with the original
Kd. Use ``--kd langmuir`` in ``run_basalt1990.py`` to compare.

For pressure-dependent crystallization see ``petrology.fc.wl_state`` /
``BASALT+langmuir.FOR``.

Research notes (1990 vs langmuir call flow, PARMS clashes, KDCALC semantics):
``petrology/article/复现korenaga/BASALT_FORTRAN_RESEARCH.md``.

Diagnostics: ``py -3.11 petrology/validation/research_basalt_fortran.py``.
"""

from .common import DEFAULT_T_K, NCOMP, NCOMPT, NPHAS
from .driver import DriverStep1990, driver_run
from .kd_calc import (
    KDCalc1990,
    arhenf_1990,
    kd_cpx_mgo_1990,
    kd_olivine_mgo_1990,
    kd_plagioclase_caa_1990,
)
from .solver import Basalt1990System, StateResult1990

__all__ = [
    "DEFAULT_T_K",
    "NCOMP",
    "NCOMPT",
    "NPHAS",
    "Basalt1990System",
    "DriverStep1990",
    "KDCalc1990",
    "StateResult1990",
    "arhenf_1990",
    "driver_run",
    "kd_cpx_mgo_1990",
    "kd_olivine_mgo_1990",
    "kd_plagioclase_caa_1990",
]
