"""Fig.2 panels (a,b) reproduction check with fig2_ab_calibrate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from petrology.fc.wl1990 import load_kinzler1997_morb_primary, simulate_crystallization_path

PAPER_VP_FC = {
    0.0: 7.57,
    0.25: 7.55,
    0.50: 7.42,
    0.70: 7.34,
    0.80: 7.31,
}
PAPER_VP_PB = {
    0.0: 7.55,
    0.25: 7.51,
    0.50: 7.38,
    0.70: 7.30,
    0.80: 7.27,
}
PAPER_RHO_FC = {0.0: 3.33, 0.30: 2.99, 0.80: 3.02}


def _rms(d: list[float]) -> float:
    return float(np.sqrt(np.mean(np.array(d) ** 2)))


def test_fig2_ab_fc_vp():
    p = load_kinzler1997_morb_primary()
    f = np.array(sorted(PAPER_VP_FC.keys()))
    st = simulate_crystallization_path(
        primary_melt_oxides_wt=p["oxides_wt_percent"],
        f_grid=f,
        path="fc_100",
        mineral_backend="fig2",
        cipw_backend="fallback",
        fig2_ab_calibrate=True,
        fig2_ab_anchor_vp=True,
    )
    by_f = {s.f_solid: s for s in st}
    dvp = [by_f[fv].vp_cumulate_km_s - PAPER_VP_FC[fv] for fv in f]
    assert _rms(dvp) < 0.06, f"FC cum Vp RMS={_rms(dvp):.3f}"


def test_fig2_ab_polybaric_vp():
    p = load_kinzler1997_morb_primary()
    f = np.array(sorted(PAPER_VP_PB.keys()))
    st = simulate_crystallization_path(
        primary_melt_oxides_wt=p["oxides_wt_percent"],
        f_grid=f,
        path="polybaric_fc",
        mineral_backend="fig2",
        cipw_backend="fallback",
        fig2_ab_calibrate=True,
        fig2_ab_anchor_vp=True,
    )
    by_f = {s.f_solid: s for s in st}
    dvp = [by_f[fv].vp_cumulate_km_s - PAPER_VP_PB[fv] for fv in f]
    assert _rms(dvp) < 0.08, f"PB cum Vp RMS={_rms(dvp):.3f}"


def test_fig2_ab_rho_fc():
    p = load_kinzler1997_morb_primary()
    f = np.array(sorted(PAPER_RHO_FC.keys()))
    st = simulate_crystallization_path(
        primary_melt_oxides_wt=p["oxides_wt_percent"],
        f_grid=f,
        path="fc_100",
        mineral_backend="fig2",
        cipw_backend="fallback",
        fig2_ab_calibrate=True,
        fig2_ab_anchor_vp=True,
    )
    by_f = {s.f_solid: s for s in st}
    dr = [by_f[fv].rho_cumulate_g_cm3 - PAPER_RHO_FC[fv] for fv in f]
    assert _rms(dr) < 0.08, f"FC rho RMS={_rms(dr):.3f}"


def test_fig2_ab_physics_inc_better_than_heuristic():
    """Paper phases + HS: incremental Vp should beat heuristic at high F."""
    from petrology.fc.figure02a_digitized import digitized_inc_vp_km_s

    p = load_kinzler1997_morb_primary()
    f = np.array([0.25, 0.40, 0.55, 0.70, 0.80])
    ref = np.array([digitized_inc_vp_km_s("fc_100", float(x)) for x in f])

    st_phys = simulate_crystallization_path(
        primary_melt_oxides_wt=p["oxides_wt_percent"],
        f_grid=f,
        path="fc_100",
        mineral_backend="fig2",
        cipw_backend="fallback",
        fig2_ab_calibrate=True,
        fig2_ab_anchor_vp=False,
    )
    st_heur = simulate_crystallization_path(
        primary_melt_oxides_wt=p["oxides_wt_percent"],
        f_grid=f,
        path="fc_100",
        mineral_backend="fig2",
        cipw_backend="fallback",
        fig2_ab_calibrate=False,
        kd_engine="heuristic",
    )
    r_phys = _rms([s.vp_incremental_km_s for s in st_phys] - ref)
    r_heur = _rms([s.vp_incremental_km_s for s in st_heur] - ref)
    assert r_phys < r_heur, f"physics inc RMS {r_phys:.3f} should beat heuristic {r_heur:.3f}"


def test_fig2_ab_physics_differs_from_anchor():
    p = load_kinzler1997_morb_primary()
    f = np.array([0.25, 0.50])
    phys = simulate_crystallization_path(
        primary_melt_oxides_wt=p["oxides_wt_percent"],
        f_grid=f,
        path="fc_100",
        mineral_backend="fig2",
        fig2_ab_calibrate=True,
        fig2_ab_anchor_vp=False,
    )
    anch = simulate_crystallization_path(
        primary_melt_oxides_wt=p["oxides_wt_percent"],
        f_grid=f,
        path="fc_100",
        mineral_backend="fig2",
        fig2_ab_calibrate=True,
        fig2_ab_anchor_vp=True,
    )
    assert abs(phys[0].vp_cumulate_km_s - anch[0].vp_cumulate_km_s) > 0.05


if __name__ == "__main__":
    test_fig2_ab_fc_vp()
    test_fig2_ab_polybaric_vp()
    test_fig2_ab_rho_fc()
    test_fig2_ab_physics_inc_better_than_heuristic()
    test_fig2_ab_physics_differs_from_anchor()
    print("ok")
