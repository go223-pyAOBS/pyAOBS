"""
Fig.2 / wl1990 FC paths using original BASALT.FOR (1990) STATE engine.

Parallel to ``wl_fc_state.state_fc_path`` but uses ``Basalt1990System`` +
``KDCalc1990`` (ARHENF = 10**(A/T)+B).  Optional ``kd_mode='langmuir'`` injects
Langmuir (1992) Kd into the same Newton solver for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .basalt1990.common import NCOMP, NCOMPT, NPHAS
from .basalt1990.driver import driver_run
from .basalt1990.solver import Basalt1990System, StateResult1990
from .wl_components import (
    csj_to_oxides_wt,
    mineral_fo_an_di,
    oxides_wt_to_csj,
)
from .wl_fc_state import CrystallizationPathMode, _p_kbar_for_path
from .wl_kd import DEFAULT_T_K, mpa_to_kbar
from .wl_polybaric import polybaric_pressure_kbar

KdMode1990 = Literal["1990", "langmuir"]
DEFAULT_T_END_K = 973.16
DEFAULT_DT_K = -5.0


@dataclass(frozen=True)
class Basalt1990FCStep:
    f_solid: float
    p_kbar: float
    t_k: float
    csj: np.ndarray
    fa: np.ndarray
    fl: float
    result: StateResult1990
    melt_oxides_wt: dict[str, float]
    fo_pct: float
    an_pct: float
    di_pct: float
    ol_frac: float
    pl_frac: float
    cpx_frac: float


def _normalize_csj(clj: np.ndarray) -> np.ndarray:
    csj = np.zeros(NCOMPT, dtype=float)
    csj[:NCOMP] = clj[:NCOMP]
    s = float(np.sum(csj[:NCOMP]))
    if s <= 1e-12:
        return csj
    csj[:NCOMP] /= s
    return csj


def _inject_kd(
    sys: Basalt1990System,
    *,
    kd_mode: KdMode1990,
    t_k: float,
    p_kbar: float,
    clj: np.ndarray | None = None,
) -> None:
    """Fill FKDAJ before STATE (1990 ARHENF or Langmuir 1992 log-linear)."""
    sys.sync_temperature(t_k)
    if kd_mode == "1990":
        sys.kd.calc(1)
        sys.kd.calc(2)
        if clj is not None:
            sys.kd.calc(3, clj[:NCOMP])
        return

    from .wl_kd import (
        arhenf_kd,
        kd_cpx_caa,
        kd_cpx_feo,
        kd_cpx_mgo,
        kd_olivine_feo,
        kd_olivine_mgo,
        kd_plagioclase_caa,
        kd_plagioclase_naal,
        wl_an_component,
    )

    melt = csj_to_oxides_wt(clj[:NCOMP] if clj is not None else sys.csj[:NCOMP])
    an = wl_an_component(melt)
    p = float(p_kbar)
    sys.kd.fkda[1, 2] = kd_olivine_mgo(t_k, p)
    sys.kd.fkda[1, 3] = kd_olivine_feo(t_k, p)
    sys.kd.fkda[2, 2] = kd_cpx_mgo(t_k, p)
    sys.kd.fkda[2, 3] = kd_cpx_feo(t_k, p)
    sys.kd.fkda[2, 4] = kd_cpx_caa(t_k, p)
    sys.kd.fkda[2, 5] = arhenf_kd(1034.0, -1.27, 0.005, t_k, p)
    sys.kd.fkda[2, 0] = arhenf_kd(2418.0, -2.30, 0.006, t_k, p)
    sys.kd.fkda[2, 1] = arhenf_kd(5087.0, -4.48, 0.003, t_k, p)
    sys.kd.fkda[0, 0] = kd_plagioclase_caa(an, t_k, p)
    sys.kd.fkda[0, 1] = kd_plagioclase_naal(an, t_k, p)


def _result_modes(res: StateResult1990) -> tuple[float, float, float]:
    fa = res.fa
    s = float(np.sum(fa))
    if s <= 1e-12:
        return 1.0, 0.0, 0.0
    return float(fa[1] / s), float(fa[0] / s), float(fa[2] / s)


def _wrap_step(
    *,
    f_solid: float,
    p_kbar: float,
    t_k: float,
    csj: np.ndarray,
    res: StateResult1990,
) -> Basalt1990FCStep:
    fo, an, di = mineral_fo_an_di(res.caj)
    ol, pl, cpx = _result_modes(res)
    melt = csj_to_oxides_wt(res.clj[:NCOMP], sio2=float(res.clj[6]))
    return Basalt1990FCStep(
        f_solid=f_solid,
        p_kbar=p_kbar,
        t_k=t_k,
        csj=csj.copy(),
        fa=res.fa.copy(),
        fl=res.fl,
        result=res,
        melt_oxides_wt=melt,
        fo_pct=fo,
        an_pct=an,
        di_pct=di,
        ol_frac=ol,
        pl_frac=pl,
        cpx_frac=cpx,
    )


def _solve(
    sys: Basalt1990System,
    csj: np.ndarray,
    *,
    kd_mode: KdMode1990,
    p_kbar: float,
    fa0: np.ndarray | None = None,
) -> StateResult1990:
    _inject_kd(sys, kd_mode=kd_mode, t_k=sys.t_k, p_kbar=p_kbar, clj=csj)
    fill = kd_mode == "1990"
    init = 1 if fill else 0
    return sys.solve(csj, fa0=fa0, kd_init_mode=init, fill_temp_kd=fill)


def temprun_fc_to_f_basalt1990(
    csj0: np.ndarray,
    *,
    f_target: float,
    p_kbar: float,
    t_start_k: float = DEFAULT_T_K,
    t_end_k: float = DEFAULT_T_END_K,
    dt_k: float = DEFAULT_DT_K,
    kd_mode: KdMode1990 = "1990",
    max_steps: int = 600,
) -> tuple[np.ndarray, StateResult1990, float, float]:
    """Cooling FC (MODEL 2) until cumulative solid fraction reaches ``f_target``."""
    f_target = float(np.clip(f_target, 0.0, 0.98))
    csj = np.asarray(csj0, dtype=float).ravel()
    if csj.size < NCOMPT:
        csj = np.pad(csj, (0, NCOMPT - csj.size))

    if f_target <= 1e-9:
        sys = Basalt1990System(t_k=t_start_k)
        res = _solve(sys, csj, kd_mode=kd_mode, p_kbar=p_kbar)
        return csj.copy(), res, 0.0, t_start_k

    dt_k = -abs(dt_k) if t_end_k < t_start_k else abs(dt_k)
    sys = Basalt1990System(t_k=t_start_k)
    fa = np.zeros(NPHAS, dtype=float)
    flr = 1.0
    t_k = float(t_start_k)
    last = _solve(sys, csj, kd_mode=kd_mode, p_kbar=p_kbar, fa0=fa)

    for _ in range(max_steps):
        if (1.0 - flr) >= f_target - 1e-4:
            break
        if (dt_k < 0 and t_k <= t_end_k) or (dt_k > 0 and t_k >= t_end_k):
            break

        res = _solve(sys, csj, kd_mode=kd_mode, p_kbar=p_kbar, fa0=fa)
        last = res
        if res.nerr != 0:
            break

        flr *= float(res.fl)
        fa = res.fa.copy()
        csj = _normalize_csj(res.clj)
        t_k += dt_k
        sys.t_k = t_k

    return csj, last, 1.0 - flr, t_k


def basalt1990_fc_path(
    primary_oxides_wt: dict,
    f_grid: np.ndarray,
    *,
    path: CrystallizationPathMode = "fc_100",
    kd_mode: KdMode1990 = "1990",
    t_fc_k: float = DEFAULT_T_K,
    t_end_fc_k: float = DEFAULT_T_END_K,
    dt_fc_k: float = DEFAULT_DT_K,
    p_fc_mpa: float | None = None,
    polybaric_p_high_mpa: float = 800.0,
    polybaric_p_low_mpa: float = 100.0,
    polybaric_dp_mpa: float = 50.0,
) -> list[Basalt1990FCStep]:
    """Run BASALT.FOR STATE FC along ``f_grid`` (oxide wt% in/out)."""
    csj0 = oxides_wt_to_csj(primary_oxides_wt)
    f_grid = np.asarray(f_grid, dtype=float)
    out: list[Basalt1990FCStep] = []

    if path == "polybaric_fc" and kd_mode == "1990":
        # Original 1990 DRIVER has no polybaric loop — step P manually on f_grid.
        csj = csj0.copy()
        fa = np.zeros(NPHAS, dtype=float)
        flr = 1.0
        sys = Basalt1990System(t_k=t_fc_k)
        f_prev = 0.0

        for f in f_grid:
            f = float(f)
            p_kbar = polybaric_pressure_kbar(
                f,
                p_high_kbar=polybaric_p_high_mpa / 100.0,
                p_low_kbar=polybaric_p_low_mpa / 100.0,
                dp_kbar=polybaric_dp_mpa / 100.0,
            )
            if f <= 1e-9:
                res = _solve(sys, csj0, kd_mode=kd_mode, p_kbar=p_kbar, fa0=fa)
                out.append(_wrap_step(f_solid=0.0, p_kbar=p_kbar, t_k=t_fc_k, csj=csj0, res=res))
                continue

            d_f = f - f_prev
            if d_f > 1e-9:
                n_sub = max(1, int(round(d_f / 0.02)))
                for i in range(n_sub):
                    f_step = f_prev + d_f * (i + 1) / n_sub
                    pk = polybaric_pressure_kbar(
                        f_step,
                        p_high_kbar=polybaric_p_high_mpa / 100.0,
                        p_low_kbar=polybaric_p_low_mpa / 100.0,
                        dp_kbar=polybaric_dp_mpa / 100.0,
                    )
                    res = _solve(sys, csj, kd_mode=kd_mode, p_kbar=pk, fa0=fa)
                    if res.nerr != 0 or float(np.sum(res.fa)) <= 1e-12:
                        break
                    fa = res.fa.copy()
                    flr *= res.fl
                    csj = _normalize_csj(res.clj)
                f_prev = f

            res = _solve(sys, csj, kd_mode=kd_mode, p_kbar=p_kbar, fa0=fa)
            out.append(_wrap_step(f_solid=f, p_kbar=p_kbar, t_k=sys.t_k, csj=csj, res=res))
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
            sys = Basalt1990System(t_k=t_fc_k)
            res = _solve(sys, csj0, kd_mode=kd_mode, p_kbar=p_kbar)
            out.append(_wrap_step(f_solid=0.0, p_kbar=p_kbar, t_k=t_fc_k, csj=csj0, res=res))
            continue

        csj, res, _, t_k = temprun_fc_to_f_basalt1990(
            csj0,
            f_target=f,
            p_kbar=p_kbar,
            t_start_k=t_fc_k,
            t_end_k=t_end_fc_k,
            dt_k=dt_fc_k,
            kd_mode=kd_mode,
        )
        out.append(_wrap_step(f_solid=f, p_kbar=p_kbar, t_k=t_k, csj=csj, res=res))

    return out
