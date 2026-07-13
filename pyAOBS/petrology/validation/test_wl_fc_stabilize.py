"""Tests for Langmuir FC stabilization (wl_state_stabilize)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.fc.wl_components import oxides_wt_to_csj
from petrology.fc.wl_driver import temprun_model2
from petrology.fc.wl_fc_state import temprun_fc_to_f
from petrology.fc.wl_state import WLStateSystem
from petrology.fc.wl_state_stabilize import DEFAULT_FC_STABILIZE, NO_STABILIZE
from petrology.fc.wl1990 import load_kinzler1997_morb_primary, simulate_crystallization_path

USER_MORB = {
    "SiO2": 48.2,
    "TiO2": 0.94,
    "Al2O3": 16.4,
    "Cr2O3": 0.12,
    "FeO": 7.96,
    "MgO": 12.5,
    "CaO": 11.4,
    "K2O": 0.07,
    "Na2O": 2.27,
}


def test_unstabilized_morb_plag_burst():
    csj = oxides_wt_to_csj(USER_MORB)
    sys = WLStateSystem(t_k=1473.16, p_kbar=1.0)
    res = sys.solve(csj, stabilize=NO_STABILIZE)
    assert res.fa[0] > 0.9


def test_stabilized_morb_single_step_capped():
    csj = oxides_wt_to_csj(USER_MORB)
    sys = WLStateSystem(t_k=1473.16, p_kbar=1.0)
    res = sys.solve(csj, stabilize=DEFAULT_FC_STABILIZE)
    assert res.nerr == 0
    assert float(np.sum(res.fa)) <= DEFAULT_FC_STABILIZE.max_solid_fraction + 1e-6
    assert res.fl > 0.85


def test_temprun_user_morb_no_matrix_error():
    csj = oxides_wt_to_csj(USER_MORB)
    _, _, flr, _, snaps = temprun_model2(
        csj,
        p_kbar=1.0,
        t_start_k=1473.16,
        t_end_k=1173.16,
        dt_k=-10.0,
        stabilize=DEFAULT_FC_STABILIZE,
    )
    assert len(snaps) >= 5
    assert all(s.result.nerr == 0 for s in snaps)
    assert flr < 0.999
    assert flr < 1.0


def test_langmuir_fc_path_user_morb_progressive():
    st = simulate_crystallization_path(
        primary_melt_oxides_wt=USER_MORB,
        f_grid=np.array([0.0, 0.2, 0.4, 0.6, 0.8]),
        path="fc_100",
        kd_engine="langmuir",
        fig2_ab_calibrate=False,
        p_fc_mpa=100.0,
        t_fc_k=1473.16,
    )
    assert len(st) == 5
    assert st[-1].f_solid >= 0.75
    assert st[2].cum_ol + st[2].cum_pl + st[2].cum_cpx > 0.01


def test_temprun_fc_to_f_reaches_target():
    p = load_kinzler1997_morb_primary()
    csj = oxides_wt_to_csj(p["oxides_wt_percent"])
    _, res, f_act, _ = temprun_fc_to_f(
        csj,
        f_target=0.4,
        p_kbar=4.0,
        t_start_k=1473.16,
        t_end_k=973.16,
        dt_k=-5.0,
    )
    assert res.nerr == 0
    assert f_act >= 0.35


if __name__ == "__main__":
    test_unstabilized_morb_plag_burst()
    test_stabilized_morb_single_step_capped()
    test_temprun_user_morb_no_matrix_error()
    test_langmuir_fc_path_user_morb_progressive()
    test_temprun_fc_to_f_reaches_target()
    print("ok")
