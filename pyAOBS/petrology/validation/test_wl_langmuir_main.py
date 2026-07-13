"""BASALT+langmuir MAIN menu (tasks 1–11) tests."""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.fc.wl_components import oxides_wt_to_csj
from petrology.fc.wl_langmuir.main import (
    BasaltLangmuirSession,
    interactive_loop,
    langmuir_driver_for_case,
)
from petrology.fc.wl_state_stabilize import NO_STABILIZE


def test_task11_toggles_polybaric():
    session = BasaltLangmuirSession()
    assert not session.polybaric
    inp = StringIO("11\n99\n")
    out = StringIO()
    interactive_loop(session, inp=inp, out=out)
    assert session.polybaric
    assert "POLYBARIC" in out.getvalue()


def test_task8_six_parms():
    session = BasaltLangmuirSession()
    session.model = 2
    session.no_model = False
    demo = {
        "SiO2": 45,
        "MgO": 25,
        "CaO": 10,
        "FeO": 8,
        "Al2O3": 10,
        "Na2O": 1,
        "TiO2": 0.5,
        "K2O": 0.1,
        "Cr2O3": 0,
    }
    session.cdat = [oxides_wt_to_csj(demo)]
    session.no_data = False
    inp = StringIO("8\n1200 900 10 0.5 8 1\n99\n")
    out = StringIO()
    interactive_loop(session, inp=inp, out=out)
    assert not session.no_temp
    assert session.parms["p_high"] == 8.0
    assert session.parms["p_low"] == 1.0
    assert session.parms["dt"] == -10.0


def test_batch_single_pressure_driver():
    session = BasaltLangmuirSession()
    session.model = 2
    session.no_model = True
    session.no_temp = False
    session.temp_offset_k = 273.16
    session.parms.update({"ti": 1200, "tf": 1150, "dt": -25})
    session.p_single_kbar = 1.0
    session.stabilize = NO_STABILIZE
    csj = oxides_wt_to_csj(
        {
            "SiO2": 45,
            "MgO": 25,
            "CaO": 10,
            "FeO": 8,
            "Al2O3": 10,
            "Na2O": 1,
            "TiO2": 0.5,
            "K2O": 0.1,
            "Cr2O3": 0,
        }
    )
    snaps = langmuir_driver_for_case(session, csj, out=StringIO())
    assert len(snaps) >= 1
    assert snaps[0].p_kbar == 1.0


def test_polybaric_driver_multiple_pressures():
    session = BasaltLangmuirSession()
    session.model = 2
    session.polybaric = True
    session.temp_offset_k = 273.16
    session.parms.update(
        {"ti": 1200, "tf": 1180, "dt": -10, "dp": 0.5, "p_high": 8.0, "p_low": 1.0}
    )
    csj = oxides_wt_to_csj(
        {
            "SiO2": 45,
            "MgO": 25,
            "CaO": 10,
            "FeO": 8,
            "Al2O3": 10,
            "Na2O": 1,
            "TiO2": 0.5,
            "K2O": 0.1,
            "Cr2O3": 0,
        }
    )
    snaps = langmuir_driver_for_case(session, csj, out=StringIO())
    pressures = {round(s.p_kbar, 1) for s in snaps}
    assert 8.0 in pressures
    assert any(p <= 2.0 for p in pressures)


if __name__ == "__main__":
    test_task11_toggles_polybaric()
    test_task8_six_parms()
    test_batch_single_pressure_driver()
    test_polybaric_driver_multiple_pressures()
    print("ok")
