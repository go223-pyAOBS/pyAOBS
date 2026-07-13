"""BASALT+langmuir.FOR call-sequence and STATE semantics tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.fc.wl_components import oxides_wt_to_csj
from petrology.fc.wl_driver import temprun_model2
from petrology.fc.wl_state import WLStateSystem


def test_state_entry_uses_kd_mode4_not_mode1():
    """wl_state matches TEMPRUN intent: non-zero Ol Kd at STATE entry."""
    wt = {
        "SiO2": 45, "MgO": 25, "CaO": 10, "FeO": 8,
        "Al2O3": 10, "Na2O": 1, "TiO2": 0.5, "K2O": 0.1, "Cr2O3": 0,
    }
    csj = oxides_wt_to_csj(wt)
    sys = WLStateSystem(t_k=1273.16, p_kbar=1.0)
    res = sys.solve(csj)
    assert res.fkda[1, 2] > 5.0
    assert res.nerr == 0


def test_fortran_literal_kdcalc1_zeros_after_mode4():
    """Document Fortran STATE: KDCALC(1) after TEMPRUN's KDCALC(4) clears Ol Kd."""
    sys = WLStateSystem(t_k=1273.16, p_kbar=1.0)
    sys.kd_calc(4)
    assert sys.fkda[1, 2] > 5.0
    sys.kd_calc(1)
    assert sys.fkda[1, 2] == 0.0


def test_temprun_model2_magnesian_crystallizes():
    wt = {
        "SiO2": 45, "MgO": 25, "CaO": 10, "FeO": 8,
        "Al2O3": 10, "Na2O": 1, "TiO2": 0.5, "K2O": 0.1, "Cr2O3": 0,
    }
    csj = oxides_wt_to_csj(wt)
    _, _, flr, _, snaps = temprun_model2(
        csj, p_kbar=1.0, t_start_k=1473.16, t_end_k=1173.16, dt_k=-25.0
    )
    assert len(snaps) >= 2
    assert flr < 0.99
    assert any(float(np.sum(s.result.fa)) > 0.01 for s in snaps)


if __name__ == "__main__":
    test_state_entry_uses_kd_mode4_not_mode1()
    test_fortran_literal_kdcalc1_zeros_after_mode4()
    test_temprun_model2_magnesian_crystallizes()
    print("ok")
