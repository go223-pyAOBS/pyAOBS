"""Integration: basalt1990 STATE ↔ wl1990 Fig.2 path."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.fc.wl1990 import load_kinzler1997_morb_primary, simulate_crystallization_path


def test_kd_engine_basalt1990_langmuir_runs():
    p = load_kinzler1997_morb_primary()
    f = np.array([0.0, 0.3, 0.6])
    st = simulate_crystallization_path(
        primary_melt_oxides_wt=p["oxides_wt_percent"],
        f_grid=f,
        path="fc_100",
        fig2_ab_calibrate=False,
        kd_engine="basalt1990",
        kd_mode_1990="langmuir",
        mineral_backend="fig2",
        cipw_backend="fallback",
    )
    assert len(st) == 3
    assert st[0].f_solid == 0.0


def test_kd_engine_heuristic_default():
    p = load_kinzler1997_morb_primary()
    st = simulate_crystallization_path(
        primary_melt_oxides_wt=p["oxides_wt_percent"],
        f_grid=np.array([0.0, 0.5]),
        path="fc_100",
        fig2_ab_calibrate=False,
        kd_engine="heuristic",
    )
    assert len(st) == 2


def test_main_batch_mode():
    from petrology.fc.basalt1990.main import Basalt1990Session, run_driver_for_case, task_run
    from io import StringIO

    session = Basalt1990Session()
    session.model = 2
    session.no_model = False
    session.no_temp = False
    session.parms.update({"ti": 1273.16, "tf": 1200.0, "dt": -25.0})
    session.temp_offset_k = 0.0
    from petrology.fc.wl_components import oxides_wt_to_csj

    session.cdat = [oxides_wt_to_csj({"SiO2": 45, "MgO": 25, "CaO": 10, "FeO": 8, "Al2O3": 10, "Na2O": 1, "TiO2": 0.5, "K2O": 0.1, "Cr2O3": 0})]
    session.no_data = False
    out = StringIO()
    steps = run_driver_for_case(session, session.cdat[0], out=out)
    assert len(steps) >= 1


if __name__ == "__main__":
    test_kd_engine_basalt1990_langmuir_runs()
    test_kd_engine_heuristic_default()
    test_main_batch_mode()
    print("ok")
