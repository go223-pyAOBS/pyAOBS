"""BASALT+langmuir STATE / FC port smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.fc.wl1990 import load_kinzler1997_morb_primary, simulate_crystallization_path
from petrology.fc.wl_components import oxides_wt_to_csj
from petrology.fc.wl_state import equilibrium_state


def test_equilibrium_state_runs():
    p = load_kinzler1997_morb_primary()
    csj = oxides_wt_to_csj(p["oxides_wt_percent"])
    res = equilibrium_state(csj, t_k=1273.16, p_kbar=4.0)
    assert res.nerr == 0
    assert 0.0 <= res.fl <= 1.0


def test_state_fc_produces_valid_track():
    p = load_kinzler1997_morb_primary()
    st = simulate_crystallization_path(
        primary_melt_oxides_wt=p["oxides_wt_percent"],
        f_grid=np.array([0.0, 0.3, 0.6]),
        path="fc_100",
        p_fc_mpa=400.0,
        use_state_engine=True,
    )
    assert len(st) == 3
    assert st[0].melt_oxides_wt["SiO2"] > 40.0


def test_state_vs_heuristic_both_run():
    p = load_kinzler1997_morb_primary()
    fg = np.array([0.0, 0.5])
    kw = dict(
        primary_melt_oxides_wt=p["oxides_wt_percent"],
        f_grid=fg,
        path="fc_100",
        p_fc_mpa=400.0,
    )
    st_state = simulate_crystallization_path(**kw, use_state_engine=True, state_fc_substeps=15)
    st_heur = simulate_crystallization_path(**kw, use_state_engine=False, sub_steps_per_interval=20)
    assert st_state[-1].f_solid == st_heur[-1].f_solid == 0.5


if __name__ == "__main__":
    test_equilibrium_state_runs()
    test_state_fc_produces_valid_track()
    test_state_vs_heuristic_both_run()
    print("ok")
