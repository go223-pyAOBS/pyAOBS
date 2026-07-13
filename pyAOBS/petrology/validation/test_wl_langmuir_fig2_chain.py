"""Langmuir STATE + HS chain without Fig.2c-h anchors (Korenaga §2.1 full W&L)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.fc.cdat_library import fig2_primary_melt
from petrology.fc.wl1990 import simulate_crystallization_path


def _run(path: str, f_grid: np.ndarray):
    melt = fig2_primary_melt()["oxides_wt_percent"]
    return simulate_crystallization_path(
        primary_melt_oxides_wt=melt,
        f_grid=f_grid,
        path=path,  # type: ignore[arg-type]
        fig2_ab_calibrate=False,
        kd_engine="langmuir",
        mineral_backend="fig2",
        cipw_backend="fallback",
        p_fc_mpa=100.0,
    )


def test_langmuir_fc_no_fig2_ab_physics_vp():
    f = np.array([0.0, 0.4, 0.8])
    st = _run("fc_100", f)
    assert len(st) == 3
    assert st[0].vp_cumulate_km_s > 6.5
    assert st[-1].f_solid == 0.8
    assert st[-1].cum_ol + st[-1].cum_pl + st[-1].cum_cpx > 0.99
    assert st[1].cum_pl > 0.05
    assert st[-1].cum_cpx > 0.05


def test_langmuir_eq_100_has_eq_solid_vp():
    f = np.array([0.0, 0.4, 0.8])
    st = _run("eq_100", f)
    assert len(st) == 3
    assert st[0].vp_eq_solid_km_s is not None
    assert st[-1].vp_eq_solid_km_s is not None
    assert 6.0 < st[-1].vp_eq_solid_km_s < 8.0


def test_langmuir_polybaric_fc_runs():
    f = np.linspace(0.0, 0.8, 9)
    st = _run("polybaric_fc", f)
    assert len(st) == 9
    assert st[-1].f_solid >= 0.75
    assert all(np.isfinite(s.vp_cumulate_km_s) for s in st)
    # High-F snapshots should sit at lower crystallization P than F≈0 (800→100 MPa schedule).
    p_hi = st[1].p_fc_mpa
    p_lo = st[-1].p_fc_mpa
    assert p_hi >= p_lo


if __name__ == "__main__":
    test_langmuir_fc_no_fig2_ab_physics_vp()
    test_langmuir_eq_100_has_eq_solid_vp()
    test_langmuir_polybaric_fc_runs()
    print("ok")
