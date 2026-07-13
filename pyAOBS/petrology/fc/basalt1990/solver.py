"""
STATE, CIMPL, MATEQ, STOICH — original BASALT.FOR (1990).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import (
    DAJ,
    DA0,
    IMCL,
    IMPL,
    IMPDIM,
    MAX_ITER,
    NCOMP,
    NCOMPT,
    NPHAS,
    NPDIM,
    TA,
    TOL,
    UAJ,
    DEFAULT_T_K,
)
from .kd_calc import KDCalc1990


@dataclass
class StateResult1990:
    nl: int
    nerr: int
    list_phases: list[int]
    csj: np.ndarray
    clj: np.ndarray
    caj: np.ndarray
    fa: np.ndarray
    fl: float
    qa: np.ndarray
    fkda: np.ndarray


class Basalt1990System:
    """Fortran COMMON blocks for one composition / temperature step."""

    def __init__(self, t_k: float = DEFAULT_T_K) -> None:
        self.t_k = float(t_k)
        self.sparm = np.array([self.t_k, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.kd = KDCalc1990(t_k=self.t_k)
        self.csj = np.zeros(NCOMPT, dtype=float)
        self.clj = np.zeros(NCOMPT, dtype=float)
        self.caj = np.zeros((NPHAS, NCOMPT), dtype=float)
        self.fa = np.zeros(NPHAS, dtype=float)
        self.fl = 1.0
        self.rj = np.ones(NCOMPT, dtype=float)
        self.qa = np.zeros(NPHAS, dtype=float)
        self.pab = np.zeros((NPDIM, NPDIM), dtype=float)

    def sync_temperature(self, temp_k: float) -> None:
        """Set SPARM(1) and refresh KDCalc1990 (required for T-stepping; see driver)."""
        self.t_k = float(temp_k)
        self.sparm[0] = self.t_k
        self.kd.t_k = self.t_k

    @staticmethod
    def mateq(a: np.ndarray, y: np.ndarray, n: int) -> tuple[np.ndarray, float]:
        """Gaussian elimination — SUBROUTINE MATEQ (single row-scale per pivot)."""
        a = a[:n, :n].copy()
        y = y[:n].copy()
        det = 1.0
        ichg = np.arange(n, dtype=int)
        for k in range(n):
            amx = abs(a[k, k])
            imx = k
            for i in range(k, n):
                if abs(a[i, k]) > amx:
                    amx = abs(a[i, k])
                    imx = i
            if amx < 1e-8:
                return y, 0.0
            if imx != k:
                a[[k, imx]] = a[[imx, k]]
                y[k], y[imx] = y[imx], y[k]
                ichg[k] = imx
            akk = a[k, k]
            det *= akk
            a[k, :n] /= akk
            y[k] /= akk
            for i in range(n):
                if i == k:
                    continue
                aki = a[i, k]
                a[i, :n] -= aki * a[k, :n]
                y[i] -= aki * y[k]
        return y, det

    def stoich(self, jp: int) -> bool:
        """STOICH — only CPX (Fortran phase 3 → index 2) has extra check."""
        if jp != 2:
            return True
        t_val = 2.0 * (self.caj[2, 2] + self.caj[2, 3]) + self.caj[2, 4]
        return bool(t_val >= 1.0)

    def cimpl(self) -> None:
        """CIMPL — implicit SiO2 and final liquid CSJ→CLJ mass balance."""
        if NCOMP < NCOMPT:
            self.clj[NCOMP:NCOMPT] = 0.0
            self.caj[:, NCOMP:NCOMPT] = 0.0

        for i in range(IMPDIM):
            imp = IMPL[i]
            if imp == 0:
                break
            imc = IMCL[i]
            s = DA0[i]
            for j in range(NCOMP):
                s += DAJ[i, j] * self.caj[imp, j]
            self.caj[imp, imc] = s

        for j in range(NCOMPT):
            s = self.csj[j]
            for k in range(NPHAS):
                s -= self.caj[k, j] * self.fa[k]
            self.clj[j] = s

    def state(
        self,
        *,
        kd_init_mode: int = 1,
        fill_temp_kd: bool = True,
    ) -> tuple[int, int, list[int]]:
        """
        SUBROUTINE STATE.

        kd_init_mode: Fortran calls KDCALC(1) only at entry (zeros FKDAJ).
        fill_temp_kd: if True, also run KDCALC(2) so ol/cpx Kd are non-zero
            (Fortran never calls mode 2 from STATE; enable for physical runs).
        """
        self.kd.calc(kd_init_mode)
        if fill_temp_kd and kd_init_mode == 1:
            self.kd.calc(2)
            rj_boot = np.ones(NCOMP, dtype=float)
            for j in range(NCOMP):
                t = 1.0 + sum(self.fa[l] * (self.kd.fkda[l, j] - 1.0) for l in range(NPHAS))
                rj_boot[j] = 1.0 / max(t, 1e-30)
            cl_boot = self.csj[:NCOMP] * rj_boot
            self.kd.calc(3, cl_boot)

        nerr = 0
        phase_list: list[int] = []
        nl = 0

        for _ in range(MAX_ITER):
            for j in range(NCOMP):
                t = 1.0
                for l in range(NPHAS):
                    t += self.fa[l] * (self.kd.fkda[l, j] - 1.0)
                self.rj[j] = 1.0 / max(t, 1e-30)
            self.clj[:NCOMP] = self.csj[:NCOMP] * self.rj[:NCOMP]
            for l in range(NPHAS):
                for j in range(NCOMP):
                    self.caj[l, j] = self.kd.fkda[l, j] * self.clj[j]

            nl = 0
            phase_list = []
            for l in range(NPHAS):
                tq = -TA[l]
                for j in range(NCOMP):
                    tq += UAJ[l, j] * self.caj[l, j]
                self.qa[l] = tq
                if tq > 0.0:
                    continue
                if not self.stoich(l):
                    continue
                nl += 1
                phase_list.append(l)

            if nl == 0:
                self.cimpl()
                if float(np.sum(self.fa)) > TOL:
                    active = [l for l in range(NPHAS) if self.fa[l] > TOL]
                    return len(active), 0, active
                return 0, 0, []

            dfa = np.zeros(nl, dtype=float)
            pab = np.zeros((nl, nl), dtype=float)
            for j in range(nl):
                l = phase_list[j]
                dfa[j] = self.qa[l]
                for k in range(nl):
                    m = phase_list[k]
                    t = 0.0
                    for i in range(NCOMP):
                        tu = UAJ[l, i]
                        if tu == 0.0:
                            continue
                        t += (
                            tu
                            * self.caj[m, i]
                            * self.rj[i] ** 2
                            * (self.kd.fkda[m, i] - 1.0)
                        )
                    pab[j, k] = -t

            sol, det = self.mateq(pab, dfa, nl)
            if abs(det) < 1e-10:
                return nl, 1, phase_list

            tst = 0.0
            self.fl = 1.0
            for j in range(nl):
                l = phase_list[j]
                new_fa = self.fa[l] + sol[j]
                new_fa = float(np.clip(new_fa, 0.0, 0.9999))
                tst += abs(self.fa[l] - new_fa)
                self.fa[l] = new_fa
                self.fl -= new_fa

            if tst <= TOL:
                self.cimpl()
                return nl, 0, phase_list

            self.kd.calc(3, self.clj)

        return nl, 2, phase_list

    def solve(
        self,
        csj_in: np.ndarray,
        *,
        fa0: np.ndarray | None = None,
        kd_init_mode: int = 1,
        fill_temp_kd: bool = True,
    ) -> StateResult1990:
        self.csj[:] = 0.0
        c = np.asarray(csj_in, dtype=float).ravel()
        n = min(c.size, NCOMPT)
        self.csj[:n] = c[:n]
        self.fa[:] = 0.0 if fa0 is None else np.asarray(fa0, dtype=float).ravel()[:NPHAS]
        self.fl = 1.0 - float(np.sum(self.fa))
        nl, nerr, plist = self.state(kd_init_mode=kd_init_mode, fill_temp_kd=fill_temp_kd)
        return StateResult1990(
            nl=nl,
            nerr=nerr,
            list_phases=plist,
            csj=self.csj.copy(),
            clj=self.clj.copy(),
            caj=self.caj.copy(),
            fa=self.fa.copy(),
            fl=float(self.fl),
            qa=self.qa.copy(),
            fkda=self.kd.fkda.copy(),
        )
