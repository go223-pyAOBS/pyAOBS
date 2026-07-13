"""Tests for BASALT+langmuir.FOR KDCALC / STOICH Python transcription."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.fc.wl_kd import (
    DEFAULT_T_K,
    arhenf_kd,
    cpx_stoich_stable,
    cpx_stoich_t,
    kd_olivine_mgo,
    kd_plagioclase_caa,
    mpa_to_kbar,
    olivine_fo_pct_from_kd,
)
from petrology.fc.wl_polybaric import (
    polybaric_pressure_kbar,
    polybaric_pressure_levels_kbar,
    polybaric_pressure_levels_mpa,
    polybaric_pressure_mpa,
)
from petrology.fc.wl1990 import load_kinzler1997_morb_primary, simulate_crystallization_path


def test_arhenf_olivine_mgo_reference():
    # At T=1273.16 K, P=0 kbar: 10**(3740/1273.16 - 1.87) ≈ 11.7
    kd = kd_olivine_mgo(DEFAULT_T_K, 0.0)
    assert 8.0 < kd < 16.0


def test_kd_increases_with_pressure():
    k0 = kd_olivine_mgo(DEFAULT_T_K, 0.0)
    k8 = kd_olivine_mgo(DEFAULT_T_K, 8.0)
    assert k8 > k0


def test_cpx_stoich_threshold():
    assert cpx_stoich_stable(1.05, 3.0)
    assert not cpx_stoich_stable(1.05, 5.0)
    assert cpx_stoich_stable(1.15, 5.0)


def test_polybaric_steps_down():
    assert polybaric_pressure_kbar(0.0) == 8.0
    assert polybaric_pressure_kbar(0.99) <= 2.0
    assert polybaric_pressure_kbar(1.0) == 1.0
    assert polybaric_pressure_mpa(0.0) == 800.0


def test_polybaric_levels_fortran():
    levels = polybaric_pressure_levels_kbar()
    assert levels[0] == 8.0
    assert levels[-1] == 1.0
    assert 7.5 in levels
    assert len(levels) == 15
    mpa = polybaric_pressure_levels_mpa()
    assert mpa[0] == 800.0
    assert mpa[-1] == 100.0


def test_simulate_polybaric_uses_discrete_p():
    p = load_kinzler1997_morb_primary()
    allowed = polybaric_pressure_levels_mpa()
    st = simulate_crystallization_path(
        primary_melt_oxides_wt=p["oxides_wt_percent"],
        f_grid=np.array([0.0, 0.4, 0.8]),
        path="polybaric_fc",
    )
    pressures = [s.p_fc_mpa for s in st]
    assert pressures[0] >= pressures[-1]
    assert max(pressures) >= 700.0
    for pr in pressures:
        assert min(abs(pr - a) for a in allowed) < 1.0


def test_fo_from_kd_reasonable():
    p = load_kinzler1997_morb_primary()
    fo = olivine_fo_pct_from_kd(p["oxides_wt_percent"], t_k=DEFAULT_T_K, p_kbar=1.0)
    assert 70.0 < fo < 92.0


if __name__ == "__main__":
    test_arhenf_olivine_mgo_reference()
    test_kd_increases_with_pressure()
    test_cpx_stoich_threshold()
    test_polybaric_levels_fortran()
    test_polybaric_steps_down()
    test_simulate_polybaric_uses_discrete_p()
    test_fo_from_kd_reasonable()
    print("ok")
