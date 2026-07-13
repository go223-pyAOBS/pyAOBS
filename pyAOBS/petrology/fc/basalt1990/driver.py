"""
DRIVER from original BASALT.FOR — temperature stepping, models 1–3.

Model 1: equilibrium crystallization (FLR=1 each step)
Model 2: fractional crystallization (FLR *= FL; CSJ=CLJ)
Model 3: fractional melting
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import NCOMP, NCOMPT, NPHAS
from .solver import Basalt1990System, StateResult1990


@dataclass(frozen=True)
class DriverStep1990:
    temp_k: float
    flr: float
    model: int
    nerr: int
    nl: int
    result: StateResult1990


def _fortran_temp_done(temp_k: float, t_end_k: float, dt_k: float) -> bool:
    """IF((PARMS(2)-TEMP)/PARMS(3).LE.0.) RETURN"""
    if abs(dt_k) <= 1e-12:
        return True
    return (t_end_k - temp_k) / dt_k <= 0.0


def driver_run(
    csj: np.ndarray,
    *,
    model: int = 2,
    ti_k: float,
    tf_k: float,
    dt_k: float,
    dp_k: float = 0.0,
    temp_offset_k: float = 0.0,
    init_phases: bool = True,
    sync_sparm_temp: bool = True,
    fill_temp_kd: bool = True,
    max_steps: int = 500,
) -> list[DriverStep1990]:
    """
    Run Fortran DRIVER temperature loop for one bulk composition.

    Parameters match PARMS: Ti, Tf, dT, DP (pressure print interval — unused here),
    temp_offset_k = PARMS(5) (273.16 for °C input).

    sync_sparm_temp: set SPARM(1)=TEMP before each STATE (not in original DRIVER
        listing but required for T-dependent KDCALC).
    fill_temp_kd: run KDCALC(2) after KDCALC(1) inside STATE (see solver doc).
    """
    if model not in (1, 2, 3):
        raise ValueError("model must be 1 (eq), 2 (FC), or 3 (FM)")

    csj = np.asarray(csj, dtype=float).ravel()
    if csj.size < NCOMPT:
        csj = np.pad(csj, (0, NCOMPT - csj.size))

    dt_k = float(dt_k)
    if abs(dt_k) <= 1e-12:
        dt_k = -10.0 if tf_k < ti_k else 10.0
    dt_k = -abs(dt_k) if tf_k < ti_k else abs(dt_k)

    sys = Basalt1990System(t_k=ti_k + temp_offset_k)
    fa = np.zeros(NPHAS, dtype=float)
    if not init_phases:
        pass
    else:
        fa[:] = 0.0

    flr = 1.0
    temp = float(ti_k) + float(temp_offset_k)
    t_end = float(tf_k) + float(temp_offset_k)
    steps: list[DriverStep1990] = []
    nupage = 1  # noqa: F841 — mirrors Fortran PRNTER flag

    for _ in range(max_steps):
        if sync_sparm_temp:
            sys.sync_temperature(temp)

        res = sys.solve(csj, fa0=fa, fill_temp_kd=fill_temp_kd)
        nl, nerr = res.nl, res.nerr
        fa = res.fa.copy()

        if model == 1:
            flr = 1.0
        elif model == 2:
            flr *= res.fl
        else:
            flr = flr / max(1.0 - res.fl, 1e-12)

        steps.append(
            DriverStep1990(
                temp_k=temp,
                flr=flr,
                model=model,
                nerr=nerr,
                nl=nl,
                result=res,
            )
        )

        if nerr != 0:
            break

        temp_next = temp + dt_k
        if _fortran_temp_done(temp_next, t_end, dt_k):
            break

        temp = temp_next
        if model == 1:
            flr = 1.0
        elif model == 2:
            s = float(np.sum(res.clj[:NCOMP]))
            if s > 1e-12:
                csj[:NCOMP] = res.clj[:NCOMP] / s
        else:
            # Model 3: CSJ = (CSJ - CLJ*FL) / (1-FL) — fractional melting
            fl = res.fl
            denom = max(1.0 - fl, 1e-12)
            csj[:NCOMP] = (csj[:NCOMP] - res.clj[:NCOMP] * fl) / denom

    return steps
