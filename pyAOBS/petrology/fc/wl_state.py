"""
BASALT+langmuir.FOR core: STATE, CIMPL, MATEQ, STOICH, KDCALC.

Faithful port of the extended Fortran (T+P KDCALC mode 4, pressure STOICH).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .wl_components import (
    DAJ,
    DA0,
    IMCL,
    IMPL,
    IMPDIM,
    NCOMP,
    NCOMPT,
    NPHAS,
    SIO2_IDX,
    TA,
    UAJ,
    mineral_fo_an_di,
)
from .wl_kd import DEFAULT_T_K
from .wl_state_stabilize import (
    DEFAULT_FC_STABILIZE,
    NO_STABILIZE,
    StateStabilize,
    cap_nr_delta,
    cap_solid_fraction,
    filter_phases_sequential,
)

MAX_ITER = 20
TOL = 1e-5
NPDIM = 10


@dataclass
class WLStateResult:
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
    fo_pct: float
    an_pct: float
    di_pct: float
    ol_frac: float
    pl_frac: float
    cpx_frac: float


class WLStateSystem:
    """Mutable W&L state (COMMON block equivalent)."""

    def __init__(self, t_k: float = DEFAULT_T_K, p_kbar: float = 0.0) -> None:
        self.t_k = float(t_k)
        self.p_kbar = float(p_kbar)
        self.csj = np.zeros(NCOMPT, dtype=float)
        self.clj = np.zeros(NCOMPT, dtype=float)
        self.caj = np.zeros((NPHAS, NCOMPT), dtype=float)
        self.fa = np.zeros(NPHAS, dtype=float)
        self.fl = 1.0
        self.fkda = np.zeros((NPHAS, NCOMP), dtype=float)
        self.rj = np.ones(NCOMPT, dtype=float)
        self.qa = np.zeros(NPHAS, dtype=float)
        self.pab = np.zeros((NPDIM, NPDIM), dtype=float)

    def kd_calc(self, kdmode: int) -> None:
        """KDCALC — modes 1 reset, 2/4 T+P, 3 plag composition."""
        tk = self.t_k
        p = self.p_kbar
        if kdmode == 1:
            self.fkda[:, :] = 0.0

        if kdmode >= 2 or kdmode == 4:
            self.fkda[1, 2] = 10.0 ** (3740.0 / tk - 1.87 + 0.0008 * p)
            self.fkda[1, 3] = 10.0 ** (3911.0 / tk - 2.50 + 0.0006 * p)
            self.fkda[2, 2] = 10.0 ** (3798.0 / tk - 2.28 + 0.004 * p)
            self.fkda[2, 3] = 0.24 * self.fkda[2, 2]
            self.fkda[2, 4] = 10.0 ** (1738.0 / tk - 0.753 + 0.009 * p)
            self.fkda[2, 5] = 10.0 ** (1034.0 / tk - 1.27 + 0.005 * p)
            self.fkda[2, 0] = 10.0 ** (2418.0 / tk - 2.30 + 0.006 * p)
            self.fkda[2, 1] = 10.0 ** (5087.0 / tk - 4.48 + 0.003 * p)

        if kdmode in (3, 4):
            cl1 = float(self.clj[0])
            cl2 = float(self.clj[1])
            an = cl1 / max(cl1 + 1.5 * cl2, 1e-12)
            self.fkda[0, 0] = 10.0 ** (2446.0 / tk - (1.122 + 0.2562 * an) + 0.012 * p)
            self.fkda[0, 1] = 10.0 ** ((3195.0 + 3283.0 * an) / tk - (2.318 + 1.885 * an) + 0.007 * p)

    def stoich(self, jp: int) -> bool:
        """STOICH — return True if phase jp is stable."""
        if jp != 2:  # 0-based cpx = phase 3
            return True
        t_val = 2.0 * (self.caj[2, 2] + self.caj[2, 3]) + self.caj[2, 4]
        threshold = 1.1 if self.p_kbar > 4.0 else 1.0
        return bool(t_val >= threshold)

    @staticmethod
    def mateq(a: np.ndarray, y: np.ndarray, n: int) -> tuple[np.ndarray, float]:
        """Gaussian elimination (MATEQ)."""
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

    def cimpl(self) -> None:
        """CIMPL — implicit SiO2 and final CLJ."""
        if NCOMP < NCOMPT:
            self.clj[NCOMP:NCOMPT] = 0.0
            self.caj[:, NCOMP:NCOMPT] = 0.0

        for i in range(IMPDIM):
            imp = IMPL[i]
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

    def state_solve(
        self,
        *,
        stabilize: StateStabilize | None = None,
    ) -> tuple[int, int, list[int]]:
        """STATE — Newton–Raphson phase equilibrium. Returns (nl, nerr, list)."""
        stab = stabilize if stabilize is not None else NO_STABILIZE
        max_dfa = stab.max_nr_fa_delta if stab.enabled else None
        seq_phases = stab.enabled and stab.sequential_phases

        # TEMPRUN calls KDCALC(4) before STATE; inner loop ends with KDCALC(3).
        self.kd_calc(4)
        nerr = 0
        phase_list: list[int] = []
        nl = 0

        for _ in range(MAX_ITER):
            for j in range(NCOMP):
                t = 1.0
                for l in range(NPHAS):
                    t += self.fa[l] * (self.fkda[l, j] - 1.0)
                self.rj[j] = 1.0 / max(t, 1e-12)
            self.clj[:NCOMP] = self.csj[:NCOMP] * self.rj[:NCOMP]
            for l in range(NPHAS):
                for j in range(NCOMP):
                    self.caj[l, j] = self.fkda[l, j] * self.clj[j]

            nl = 0
            phase_list = []
            for l in range(NPHAS):
                tq = -TA[l]
                for j in range(NCOMP):
                    tq += UAJ[l, j] * self.caj[l, j]
                self.qa[l] = tq
                isok = self.stoich(l)
                if tq > 0.0 or not isok:
                    continue
                nl += 1
                phase_list.append(l)

            if seq_phases and nl > 1:
                phase_list = filter_phases_sequential(phase_list, stab.phase_order)
                nl = len(phase_list)

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
                        t += tu * self.caj[m, i] * self.rj[i] ** 2 * (self.fkda[m, i] - 1.0)
                    pab[j, k] = -t

            sol, det = self.mateq(pab, dfa, nl)
            if abs(det) < 1e-10:
                return nl, 1, phase_list

            tst = 0.0
            self.fl = 1.0
            for j in range(nl):
                l = phase_list[j]
                step = float(sol[j])
                if max_dfa is not None:
                    step = cap_nr_delta(step, max_dfa)
                new_fa = self.fa[l] + step
                new_fa = float(np.clip(new_fa, 0.0, 0.9999))
                tst += abs(self.fa[l] - new_fa)
                self.fa[l] = new_fa
                self.fl -= new_fa

            if tst <= TOL:
                self.cimpl()
                return nl, 0, phase_list

            self.kd_calc(3)

        return nl, 2, phase_list

    def _apply_solid_cap(self, max_solid: float) -> None:
        """After NR convergence, limit total solid for one FC step."""
        self.fa[:] = cap_solid_fraction(self.fa, max_solid)
        self.fl = 1.0 - float(np.sum(self.fa))
        self.cimpl()

    def state_solve_legacy(self) -> tuple[int, int, list[int]]:
        """Alias without stabilization (tests / literal comparison)."""
        return self.state_solve(stabilize=NO_STABILIZE)

    def solve(
        self,
        csj_in: np.ndarray,
        *,
        fa0: np.ndarray | None = None,
        stabilize: StateStabilize | None = None,
    ) -> WLStateResult:
        """Run STATE for W&L bulk CSJ (6 explicit + optional implicit SiO2)."""
        stab = stabilize if stabilize is not None else NO_STABILIZE
        self.csj[:] = 0.0
        c = np.asarray(csj_in, dtype=float).ravel()
        if c.size >= NCOMPT:
            self.csj[:] = c[:NCOMPT]
        else:
            self.csj[:NCOMP] = c[:NCOMP]
        self.fa[:] = 0.0 if fa0 is None else np.asarray(fa0, dtype=float).ravel()[:NPHAS]
        self.fl = 1.0 - float(np.sum(self.fa))
        nl, nerr, plist = self.state_solve(stabilize=stab)
        if stab.enabled and nerr == 0 and stab.max_solid_fraction < 1.0:
            self._apply_solid_cap(stab.max_solid_fraction)
            nl = int(sum(1 for l in range(NPHAS) if self.fa[l] > TOL))
            plist = [l for l in range(NPHAS) if self.fa[l] > TOL]
        fo, an, di = mineral_fo_an_di(self.caj)
        fa = self.fa.copy()
        st = float(np.sum(fa))
        if st > 1e-12 and nl > 0:
            modes = fa / st
        else:
            modes = np.array([1.0, 0.0, 0.0])
        return WLStateResult(
            nl=nl,
            nerr=nerr,
            list_phases=plist,
            csj=self.csj.copy(),
            clj=self.clj.copy(),
            caj=self.caj.copy(),
            fa=fa,
            fl=float(self.fl),
            qa=self.qa.copy(),
            fkda=self.fkda.copy(),
            fo_pct=fo,
            an_pct=an,
            di_pct=di,
            ol_frac=float(modes[1]),
            pl_frac=float(modes[0]),
            cpx_frac=float(modes[2]),
        )


def equilibrium_state(
    csj6: np.ndarray,
    *,
    t_k: float = DEFAULT_T_K,
    p_kbar: float = 0.0,
) -> WLStateResult:
    sys = WLStateSystem(t_k=t_k, p_kbar=p_kbar)
    return sys.solve(csj6)
