"""Validation for Weaver & Langmuir (1990) BASALT.FOR Python port."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.fc.basalt1990 import (
    DEFAULT_T_K,
    Basalt1990System,
    arhenf_1990,
    driver_run,
    kd_olivine_mgo_1990,
)
from petrology.fc.basalt1990.kd_calc import KDCalc1990
from petrology.fc.basalt1990.solver import Basalt1990System as _Sys
from petrology.fc.wl_components import oxides_wt_to_csj


def test_arhenf_1990_formula():
    """ARHENF = 10**(A/T) + B at T=1273.16 K."""
    t = DEFAULT_T_K
    val = arhenf_1990(2715.0, -1.158, t)
    expected = 10.0 ** (2715.0 / t) + (-1.158)
    assert abs(val - expected) < 1e-4
    assert val > 100.0  # distinct from Langmuir 10**(A/T+B) ~ 11.7


def test_kdcalc_mode1_zeros_mode2_fills():
    kd = KDCalc1990(t_k=DEFAULT_T_K)
    kd.calc(1)
    assert np.all(kd.fkda == 0.0)
    kd.calc(2)
    assert kd.fkda[1, 2] > 50.0
    assert kd.fkda[2, 3] == 0.24 * kd.fkda[2, 2]


def test_mateq_identity():
    a = np.eye(3, dtype=float)
    y = np.array([1.0, 2.0, 3.0])
    sol, det = Basalt1990System.mateq(a, y, 3)
    assert abs(det - 1.0) < 1e-6
    np.testing.assert_allclose(sol, y, rtol=1e-5)


def test_state_converges_tholeiite():
    """Representative MORB bulk in CSJ — STATE should converge."""
    from petrology.fc.wl1990 import load_kinzler1997_morb_primary

    csj = oxides_wt_to_csj(load_kinzler1997_morb_primary()["oxides_wt_percent"])
    sys = Basalt1990System(t_k=1273.16)
    res = sys.solve(csj, fill_temp_kd=True)
    assert res.nerr == 0
    assert 0.0 <= res.fl <= 1.0
    assert res.nl >= 0


def test_state_literal_fortran_kd_init_stays_zero_without_fill():
    """Documents Fortran STATE: KDCALC(1) only → ol/cpx Kd remain zero."""
    from petrology.fc.wl1990 import load_kinzler1997_morb_primary

    csj = oxides_wt_to_csj(load_kinzler1997_morb_primary()["oxides_wt_percent"])
    sys = Basalt1990System(t_k=1273.16)
    sys.csj[:6] = csj[:6]
    sys.kd.calc(1)
    assert sys.kd.fkda[1, 2] == 0.0


def test_state_nr_solver_with_langmuir_kd():
    """Newton–Raphson + MATEQ with Langmuir-scale Kd (solver regression)."""
    from petrology.fc.wl_kd import (
        kd_olivine_feo,
        kd_olivine_mgo,
        kd_plagioclase_caa,
        kd_plagioclase_naal,
        wl_an_component,
    )
    from petrology.fc.wl_components import csj_to_oxides_wt, oxides_wt_to_csj

    wt = {
        "SiO2": 45, "TiO2": 0.5, "Al2O3": 10, "FeO": 8,
        "MgO": 25, "CaO": 10, "Na2O": 1, "K2O": 0.1, "Cr2O3": 0,
    }
    csj = oxides_wt_to_csj(wt)
    t_k = 1273.16
    melt = csj_to_oxides_wt(csj[:6])
    an = wl_an_component(melt)

    sys = Basalt1990System(t_k=t_k)
    sys.kd.fkda[1, 2] = kd_olivine_mgo(t_k, 0.0)
    sys.kd.fkda[1, 3] = kd_olivine_feo(t_k, 0.0)
    sys.kd.fkda[0, 0] = kd_plagioclase_caa(an, t_k, 0.0)
    sys.kd.fkda[0, 1] = kd_plagioclase_naal(an, t_k, 0.0)
    res = sys.solve(csj, kd_init_mode=0, fill_temp_kd=False)
    assert res.nerr == 0
    assert res.fl < 1.0
    assert float(np.sum(res.fa)) > 0.0


def test_state_returns_active_phases_when_converged():
    """nl reflects crystallized phases after NR convergence."""
    from petrology.fc.wl_kd import (
        kd_olivine_feo,
        kd_olivine_mgo,
        kd_plagioclase_caa,
        kd_plagioclase_naal,
        wl_an_component,
    )
    from petrology.fc.wl_components import csj_to_oxides_wt, oxides_wt_to_csj

    wt = {
        "SiO2": 45, "TiO2": 0.5, "Al2O3": 10, "FeO": 8,
        "MgO": 25, "CaO": 10, "Na2O": 1, "K2O": 0.1, "Cr2O3": 0,
    }
    csj = oxides_wt_to_csj(wt)
    t_k = 1273.16
    an = wl_an_component(csj_to_oxides_wt(csj[:6]))
    sys = Basalt1990System(t_k=t_k)
    sys.kd.fkda[1, 2] = kd_olivine_mgo(t_k, 0.0)
    sys.kd.fkda[1, 3] = kd_olivine_feo(t_k, 0.0)
    sys.kd.fkda[0, 0] = kd_plagioclase_caa(an, t_k, 0.0)
    sys.kd.fkda[0, 1] = kd_plagioclase_naal(an, t_k, 0.0)
    res = sys.solve(csj, kd_init_mode=0, fill_temp_kd=False)
    assert res.nerr == 0
    assert res.nl >= 1
    assert float(np.sum(res.fa)) > 0.0


def test_driver_fc_runs_without_error():
    from petrology.fc.wl1990 import load_kinzler1997_morb_primary

    csj = oxides_wt_to_csj(load_kinzler1997_morb_primary()["oxides_wt_percent"])
    steps = driver_run(
        csj,
        model=2,
        ti_k=1000.0,
        tf_k=950.0,
        dt_k=-10.0,
        temp_offset_k=273.16,
        fill_temp_kd=True,
    )
    assert len(steps) >= 5
    assert all(s.nerr == 0 for s in steps)


def test_driver_fc_crystallizes():
    """With Langmuir-scale Kd on magnesian bulk, STATE finds crystals."""
    from petrology.fc.wl_kd import (
        kd_olivine_feo,
        kd_olivine_mgo,
        kd_plagioclase_caa,
        kd_plagioclase_naal,
        wl_an_component,
    )
    from petrology.fc.wl_components import csj_to_oxides_wt, oxides_wt_to_csj

    wt = {
        "SiO2": 45, "TiO2": 0.5, "Al2O3": 10, "FeO": 8,
        "MgO": 25, "CaO": 10, "Na2O": 1, "K2O": 0.1, "Cr2O3": 0,
    }
    csj = oxides_wt_to_csj(wt)
    melt = csj_to_oxides_wt(csj[:6])
    an = wl_an_component(melt)
    t_k = 1273.16

    sys = Basalt1990System(t_k=t_k)
    sys.kd.fkda[1, 2] = kd_olivine_mgo(t_k, 0.0)
    sys.kd.fkda[1, 3] = kd_olivine_feo(t_k, 0.0)
    sys.kd.fkda[0, 0] = kd_plagioclase_caa(an, t_k, 0.0)
    sys.kd.fkda[0, 1] = kd_plagioclase_naal(an, t_k, 0.0)
    res = sys.solve(csj, kd_init_mode=0, fill_temp_kd=False)
    assert res.nerr == 0
    assert res.fl < 1.0
    assert float(np.sum(res.fa)) > 0.0


def test_cimpl_mass_balance():
    """After STATE, sum(CAJ*FA) + CLJ ≈ CSJ for explicit components."""
    from petrology.fc.wl1990 import load_kinzler1997_morb_primary

    csj = oxides_wt_to_csj(load_kinzler1997_morb_primary()["oxides_wt_percent"])
    sys = Basalt1990System(t_k=1273.16)
    res = sys.solve(csj, fill_temp_kd=True)
    if res.nl == 0:
        return
    recon = res.clj[:6].copy()
    for k in range(3):
        recon += res.caj[k, :6] * res.fa[k]
    np.testing.assert_allclose(recon, res.csj[:6], rtol=0.05, atol=0.02)


def test_1990_vs_langmuir_kd_differ():
    """Original ARHENF differs from Langmuir 1992 log-linear form."""
    from petrology.fc.wl_kd import kd_olivine_mgo

    k1990 = kd_olivine_mgo_1990(DEFAULT_T_K)
    k1992 = kd_olivine_mgo(DEFAULT_T_K, 0.0)
    assert k1990 > 10.0 * k1992


if __name__ == "__main__":
    test_arhenf_1990_formula()
    test_kdcalc_mode1_zeros_mode2_fills()
    test_mateq_identity()
    test_state_converges_tholeiite()
    test_state_literal_fortran_kd_init_stays_zero_without_fill()
    test_state_nr_solver_with_langmuir_kd()
    test_state_returns_active_phases_when_converged()
    test_driver_fc_runs_without_error()
    test_driver_fc_crystallizes()
    test_cimpl_mass_balance()
    test_1990_vs_langmuir_kd_differ()
    print("ok")
