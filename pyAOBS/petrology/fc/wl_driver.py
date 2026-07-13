"""
BASALT+langmuir.FOR DRIVER + TEMPRUN (MODEL 2 fractional crystallization).

Polybaric (MODES(6)=1): P_HIGH -> P_LOW in steps of DP; each segment runs a full
TEMPRUN (Ti -> Tf), then CSJ=CLJ.  FLR and FA carry across segments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .wl_components import NCOMP, NCOMPT, NPHAS, csj_to_oxides_wt
from .wl_kd import DEFAULT_T_K
from .wl_polybaric import polybaric_pressure_levels_kbar
from .wl_state import WLStateResult, WLStateSystem, equilibrium_state
from .wl_state_stabilize import DEFAULT_FC_STABILIZE, StateStabilize

DEFAULT_T_END_K = 973.16
DEFAULT_DT_K = -5.0


@dataclass(frozen=True)
class TemprunSnapshot:
    """One STATE equilibrium step inside TEMPRUN."""

    t_k: float
    p_kbar: float
    flr: float
    f_solid: float
    csj: np.ndarray
    result: WLStateResult


def _normalize_csj(clj: np.ndarray) -> np.ndarray:
    csj = np.zeros(NCOMPT, dtype=float)
    csj[:NCOMP] = clj[:NCOMP]
    s = float(np.sum(csj[:NCOMP]))
    if s <= 1e-12:
        return csj
    csj[:NCOMP] /= s
    return csj


def _fortran_temprun_done(t_k: float, t_end_k: float, dt_k: float) -> bool:
    """Fortran: IF((PARMS(2)-TEMP)/PARMS(3).LE.0.) RETURN"""
    if abs(dt_k) <= 1e-12:
        return True
    return (t_end_k - t_k) / dt_k <= 0.0


def temprun(
    csj: np.ndarray,
    *,
    model: int = 2,
    p_kbar: float,
    t_start_k: float,
    t_end_k: float = DEFAULT_T_END_K,
    dt_k: float = DEFAULT_DT_K,
    fa0: np.ndarray | None = None,
    flr0: float = 1.0,
    max_steps: int = 600,
    stabilize: StateStabilize | None = DEFAULT_FC_STABILIZE,
) -> tuple[np.ndarray, np.ndarray, float, float, list[TemprunSnapshot]]:
    """
    One pressure segment: TEMPRUN for MODEL 1 (eq), 2 (FC), or 3 (FM).

    Fortran BASALT+langmuir.FOR inner temperature loop.
    """
    if model not in (1, 2, 3):
        raise ValueError("model must be 1 (eq), 2 (FC), or 3 (FM)")

    csj = np.asarray(csj, dtype=float).ravel()
    if csj.size < NCOMPT:
        csj = np.pad(csj, (0, NCOMPT - csj.size))

    dt_k = -abs(dt_k) if t_end_k < t_start_k else abs(dt_k)
    sys = WLStateSystem(t_k=t_start_k, p_kbar=float(p_kbar))
    fa = np.zeros(NPHAS, dtype=float) if fa0 is None else np.asarray(fa0, dtype=float).copy()
    flr = float(flr0)
    t_k = float(t_start_k)
    snapshots: list[TemprunSnapshot] = []

    for _ in range(max_steps):
        res = sys.solve(csj, fa0=fa, stabilize=stabilize)
        if res.nerr != 0:
            break

        if model == 1:
            flr = 1.0
        elif model == 2:
            flr *= float(res.fl)
        else:
            flr = flr / max(1.0 - float(res.fl), 1e-12)

        fa = res.fa.copy()
        snapshots.append(
            TemprunSnapshot(
                t_k=t_k,
                p_kbar=float(p_kbar),
                flr=flr,
                f_solid=1.0 - flr,
                csj=csj.copy(),
                result=res,
            )
        )

        t_next = t_k + dt_k
        if _fortran_temprun_done(t_next, t_end_k, dt_k):
            break

        t_k = t_next
        sys.t_k = t_k

        if model == 1:
            flr = 1.0
        elif model == 2:
            csj = _normalize_csj(res.clj)
        else:
            fl = max(float(res.fl), 1e-12)
            csj_work = csj.copy()
            for j in range(NCOMP):
                csj_work[j] = (csj_work[j] - res.clj[j] * fl) / max(1.0 - fl, 1e-12)
            s = float(np.sum(csj_work[:NCOMP]))
            if s > 1e-12:
                csj_work[:NCOMP] /= s
            csj = csj_work

    return csj, fa, flr, t_k, snapshots


def temprun_model2(
    csj: np.ndarray,
    *,
    p_kbar: float,
    t_start_k: float,
    t_end_k: float = DEFAULT_T_END_K,
    dt_k: float = DEFAULT_DT_K,
    fa0: np.ndarray | None = None,
    flr0: float = 1.0,
    max_steps: int = 600,
    stabilize: StateStabilize | None = DEFAULT_FC_STABILIZE,
) -> tuple[np.ndarray, np.ndarray, float, float, list[TemprunSnapshot]]:
    """One pressure segment: full TEMPRUN with MODEL 2 (``FLR *= FL``, ``CSJ = CLJ``)."""
    return temprun(
        csj,
        model=2,
        p_kbar=p_kbar,
        t_start_k=t_start_k,
        t_end_k=t_end_k,
        dt_k=dt_k,
        fa0=fa0,
        flr0=flr0,
        max_steps=max_steps,
        stabilize=stabilize,
    )


def langmuir_driver_run(
    csj: np.ndarray,
    *,
    model: int = 2,
    ti_k: float,
    tf_k: float,
    dt_k: float,
    polybaric: bool = False,
    p_single_kbar: float = 0.0,
    p_high_kbar: float = 8.0,
    p_low_kbar: float = 1.0,
    dp_kbar: float = 0.5,
    init_phases: bool = True,
    stabilize: StateStabilize | None = DEFAULT_FC_STABILIZE,
) -> list[TemprunSnapshot]:
    """
    BASALT+langmuir DRIVER: single-pressure TEMPRUN or polybaric multi-P loop.

    Polybaric (MODES(6)=1) is supported for MODEL 2 only; other models use
    single pressure at ``p_single_kbar``.
    """
    if polybaric and model == 2:
        _, _, _, snaps = polybaric_driver_model2(
            csj,
            t_start_k=ti_k,
            t_end_k=tf_k,
            dt_k=dt_k,
            p_high_kbar=p_high_kbar,
            p_low_kbar=p_low_kbar,
            dp_kbar=dp_kbar,
            init_phases=init_phases,
            stabilize=stabilize,
        )
        return snaps

    _, _, _, _, snaps = temprun(
        csj,
        model=model,
        p_kbar=p_single_kbar,
        t_start_k=ti_k,
        t_end_k=tf_k,
        dt_k=dt_k,
        fa0=np.zeros(NPHAS) if init_phases else None,
        stabilize=stabilize,
    )
    return snaps


def polybaric_driver_model2(
    csj0: np.ndarray,
    *,
    t_start_k: float = DEFAULT_T_K,
    t_end_k: float = DEFAULT_T_END_K,
    dt_k: float = DEFAULT_DT_K,
    p_high_kbar: float = 8.0,
    p_low_kbar: float = 1.0,
    dp_kbar: float = 0.5,
    init_phases: bool = True,
    stabilize: StateStabilize | None = DEFAULT_FC_STABILIZE,
) -> tuple[np.ndarray, np.ndarray, float, list[TemprunSnapshot]]:
    """
    Fortran DRIVER polybaric loop (MODES(6)=1, MODEL=2).

    Returns final CSJ, FA, FLR, and all TEMPRUN snapshots (all P segments).
    """
    csj = np.asarray(csj0, dtype=float).ravel()
    if csj.size < NCOMPT:
        csj = np.pad(csj, (0, NCOMPT - csj.size))

    fa = np.zeros(NPHAS, dtype=float)
    flr = 1.0
    all_snaps: list[TemprunSnapshot] = []
    reset_fa = init_phases

    for p_kbar in polybaric_pressure_levels_kbar(
        p_high_kbar=p_high_kbar,
        p_low_kbar=p_low_kbar,
        dp_kbar=dp_kbar,
    ):
        fa0 = np.zeros(NPHAS, dtype=float) if reset_fa else fa
        csj, fa, flr, _, snaps = temprun(
            csj,
            model=2,
            p_kbar=p_kbar,
            t_start_k=t_start_k,
            t_end_k=t_end_k,
            dt_k=dt_k,
            fa0=fa0,
            flr0=flr,
            stabilize=stabilize,
        )
        all_snaps.extend(snaps)
        reset_fa = False

    return csj, fa, flr, all_snaps


def snapshot_nearest_f(
    snapshots: list[TemprunSnapshot],
    f_target: float,
) -> TemprunSnapshot | None:
    """Pick snapshot with ``f_solid`` closest to ``f_target``."""
    if not snapshots:
        return None
    f_target = float(f_target)
    return min(snapshots, key=lambda s: abs(s.f_solid - f_target))


def snapshots_for_f_grid(
    csj0: np.ndarray,
    f_grid: np.ndarray,
    snapshots: list[TemprunSnapshot],
    *,
    t_start_k: float,
    p_high_kbar: float,
) -> list[tuple[float, float, float, WLStateResult, np.ndarray]]:
    """
    Build (f_solid, p_kbar, t_k, result, csj) rows for each ``f_grid`` point.

    ``f=0`` uses initial equilibrium at ``p_high_kbar``.
    """
    f_grid = np.asarray(f_grid, dtype=float)
    out: list[tuple[float, float, float, WLStateResult, np.ndarray]] = []

    for f in f_grid:
        f = float(f)
        if f <= 1e-9:
            res = equilibrium_state(csj0, t_k=t_start_k, p_kbar=p_high_kbar)
            out.append((0.0, p_high_kbar, t_start_k, res, np.asarray(csj0, dtype=float).copy()))
            continue
        snap = snapshot_nearest_f(snapshots, f)
        if snap is None:
            res = equilibrium_state(csj0, t_k=t_start_k, p_kbar=p_high_kbar)
            out.append((f, p_high_kbar, t_start_k, res, np.asarray(csj0, dtype=float).copy()))
        else:
            out.append((f, snap.p_kbar, snap.t_k, snap.result, _normalize_csj(snap.result.clj)))

    return out


def melt_oxides_from_result(result: WLStateResult) -> dict[str, float]:
    return csj_to_oxides_wt(result.clj[:NCOMP], sio2=float(result.clj[6]))
