"""
FC drivers using BASALT+langmuir STATE (MODEL 2 fractional crystallization).

Fortran TEMPRUN model 2: cool stepwise, ``FLR *= FL``, ``CSJ = CLJ`` (NCOMP).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .wl_components import NCOMP, NCOMPT, NPHAS, csj_to_oxides_wt, oxides_wt_to_csj
from .wl_kd import DEFAULT_T_K, mpa_to_kbar
from .wl_driver import (
    DEFAULT_DT_K as DRIVER_DEFAULT_DT_K,
    DEFAULT_T_END_K as DRIVER_DEFAULT_T_END_K,
    melt_oxides_from_result,
    polybaric_driver_model2,
    snapshots_for_f_grid,
)
from .wl_polybaric import polybaric_pressure_kbar, polybaric_pressure_levels_kbar
from .wl_state import WLStateResult, WLStateSystem, equilibrium_state
from .wl_state_stabilize import DEFAULT_FC_STABILIZE, StateStabilize

CrystallizationPathMode = Literal["fc_100", "eq_100", "polybaric_fc"]

DEFAULT_T_END_K = DRIVER_DEFAULT_T_END_K
DEFAULT_DT_K = DRIVER_DEFAULT_DT_K


@dataclass(frozen=True)
class StateFCStep:
    f_solid: float
    p_kbar: float
    t_k: float
    csj: np.ndarray
    clj: np.ndarray
    fa: np.ndarray
    fl: float
    result: WLStateResult
    melt_oxides_wt: dict[str, float]


def _p_kbar_for_path(
    path: CrystallizationPathMode,
    f_solid: float,
    *,
    p_fc_mpa: float | None,
    polybaric_p_high_mpa: float,
    polybaric_p_low_mpa: float,
    polybaric_dp_mpa: float,
) -> float:
    if path == "polybaric_fc":
        return polybaric_pressure_kbar(
            f_solid,
            p_high_kbar=polybaric_p_high_mpa / 100.0,
            p_low_kbar=polybaric_p_low_mpa / 100.0,
            dp_kbar=polybaric_dp_mpa / 100.0,
        )
    if p_fc_mpa is not None:
        return mpa_to_kbar(p_fc_mpa)
    return 1.0


def _normalize_csj(clj: np.ndarray) -> np.ndarray:
    csj = np.zeros(NCOMPT, dtype=float)
    csj[:NCOMP] = clj[:NCOMP]
    s = float(np.sum(csj[:NCOMP]))
    if s <= 1e-12:
        return csj
    csj[:NCOMP] /= s
    return csj


def temprun_fc_to_f(
    csj0: np.ndarray,
    *,
    f_target: float,
    p_kbar: float,
    t_start_k: float = DEFAULT_T_K,
    t_end_k: float = DEFAULT_T_END_K,
    dt_k: float = DEFAULT_DT_K,
    max_steps: int = 600,
    stabilize: StateStabilize | None = DEFAULT_FC_STABILIZE,
) -> tuple[np.ndarray, WLStateResult, float, float]:
    """Fortran TEMPRUN MODEL 2 until cumulative solid fraction reaches ``f_target``."""
    f_target = float(np.clip(f_target, 0.0, 0.98))
    csj = np.asarray(csj0, dtype=float).ravel()
    if csj.size < NCOMPT:
        csj = np.pad(csj, (0, NCOMPT - csj.size))

    if f_target <= 1e-9:
        res = equilibrium_state(csj, t_k=t_start_k, p_kbar=p_kbar)
        return csj.copy(), res, 0.0, t_start_k

    dt_k = -abs(dt_k) if t_end_k < t_start_k else abs(dt_k)
    sys = WLStateSystem(t_k=t_start_k, p_kbar=p_kbar)
    fa = np.zeros(NPHAS, dtype=float)
    flr = 1.0
    t_k = float(t_start_k)
    last = sys.solve(csj)

    for _ in range(max_steps):
        if (1.0 - flr) >= f_target - 1e-4:
            break
        if (dt_k < 0 and t_k <= t_end_k) or (dt_k > 0 and t_k >= t_end_k):
            break

        res = sys.solve(csj, fa0=fa, stabilize=stabilize)
        last = res
        if res.nerr != 0:
            break

        flr *= float(res.fl)
        fa = res.fa.copy()
        csj = _normalize_csj(res.clj)
        t_k += dt_k
        sys.t_k = t_k

    return csj, last, 1.0 - flr, t_k


def fractional_crystallize_isothermal(
    csj0: np.ndarray,
    *,
    f_target: float,
    t_k: float = DEFAULT_T_K,
    p_kbar: float = 1.0,
    n_substeps: int = 40,
    stabilize: StateStabilize | None = DEFAULT_FC_STABILIZE,
) -> tuple[np.ndarray, WLStateResult, float]:
    """Isothermal F-substep FC (fallback when melt is already saturated)."""
    f_target = float(np.clip(f_target, 0.0, 0.98))
    if f_target <= 1e-9:
        res = equilibrium_state(np.asarray(csj0, dtype=float), t_k=t_k, p_kbar=p_kbar)
        return np.asarray(csj0, dtype=float).ravel()[:NCOMP].copy(), res, 0.0

    csj = np.asarray(csj0, dtype=float).ravel()
    if csj.size < NCOMPT:
        csj = np.pad(csj, (0, NCOMPT - csj.size))
    fa = np.zeros(NPHAS, dtype=float)
    sys = WLStateSystem(t_k=t_k, p_kbar=p_kbar)
    f_curr = 0.0
    d_f = f_target / max(n_substeps, 1)
    last = equilibrium_state(csj, t_k=t_k, p_kbar=p_kbar)

    for _ in range(n_substeps):
        if f_curr >= f_target - 1e-9:
            break
        res = sys.solve(csj, fa0=fa, stabilize=stabilize)
        last = res
        if res.nl == 0 or res.nerr != 0:
            break
        fa = res.fa.copy()
        csj = _normalize_csj(res.clj)
        f_curr += min(d_f, f_target - f_curr)

    return csj, last, f_curr


def state_fc_path(
    primary_oxides_wt: dict,
    f_grid: np.ndarray,
    *,
    path: CrystallizationPathMode = "fc_100",
    t_fc_k: float = DEFAULT_T_K,
    t_end_fc_k: float = DEFAULT_T_END_K,
    dt_fc_k: float = DEFAULT_DT_K,
    p_fc_mpa: float | None = None,
    polybaric_p_high_mpa: float = 800.0,
    polybaric_p_low_mpa: float = 100.0,
    polybaric_dp_mpa: float = 50.0,
    n_substeps: int = 40,
) -> list[StateFCStep]:
    """Run STATE-based FC along f_grid (oxide in, oxide out)."""
    csj0 = oxides_wt_to_csj(primary_oxides_wt)
    f_grid = np.asarray(f_grid, dtype=float)
    out: list[StateFCStep] = []

    p_high_kbar = polybaric_p_high_mpa / 100.0
    p_low_kbar = polybaric_p_low_mpa / 100.0
    dp_kbar = polybaric_dp_mpa / 100.0

    if path == "polybaric_fc":
        _, _, _, driver_snaps = polybaric_driver_model2(
            csj0,
            t_start_k=t_fc_k,
            t_end_k=t_end_fc_k,
            dt_k=dt_fc_k,
            p_high_kbar=p_high_kbar,
            p_low_kbar=p_low_kbar,
            dp_kbar=dp_kbar,
        )
        rows = snapshots_for_f_grid(
            csj0,
            f_grid,
            driver_snaps,
            t_start_k=t_fc_k,
            p_high_kbar=p_high_kbar,
        )
        for f, p_kbar, t_k, res, csj in rows:
            melt = melt_oxides_from_result(res)
            out.append(
                StateFCStep(
                    f_solid=f,
                    p_kbar=p_kbar,
                    t_k=t_k,
                    csj=csj.copy(),
                    clj=res.clj.copy(),
                    fa=res.fa.copy(),
                    fl=res.fl,
                    result=res,
                    melt_oxides_wt=melt,
                )
            )
        return out

    for f in f_grid:
        f = float(f)
        p_kbar = _p_kbar_for_path(
            path,
            f,
            p_fc_mpa=p_fc_mpa,
            polybaric_p_high_mpa=polybaric_p_high_mpa,
            polybaric_p_low_mpa=polybaric_p_low_mpa,
            polybaric_dp_mpa=polybaric_dp_mpa,
        )
        if f <= 1e-9:
            res = equilibrium_state(csj0, t_k=t_fc_k, p_kbar=p_kbar)
            csj = csj0.copy()
            t_k = t_fc_k
        else:
            csj, res, _, t_k = temprun_fc_to_f(
                csj0,
                f_target=f,
                p_kbar=p_kbar,
                t_start_k=t_fc_k,
                t_end_k=t_end_fc_k,
                dt_k=dt_fc_k,
            )
        melt = csj_to_oxides_wt(res.clj[:NCOMP], sio2=float(res.clj[6]))
        out.append(
            StateFCStep(
                f_solid=f,
                p_kbar=p_kbar,
                t_k=t_k,
                csj=csj.copy(),
                clj=res.clj.copy(),
                fa=res.fa.copy(),
                fl=res.fl,
                result=res,
                melt_oxides_wt=melt,
            )
        )
    return out
