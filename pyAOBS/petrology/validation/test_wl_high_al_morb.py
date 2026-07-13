"""High-Al MORB plagioclase-priority phase calibration vs KKHS02 Fig.2c."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.fc.cdat_library import fig2_primary_melt
from petrology.fc.fig2_ab import cumulative_phases_pct
from petrology.fc.wl1990 import simulate_crystallization_path
from petrology.fc.wl_high_al_morb import is_high_al_morb, mgo_plagioclase_in


def test_fig2_melt_is_high_al():
    melt = fig2_primary_melt()["oxides_wt_percent"]
    assert is_high_al_morb(melt)
    assert mgo_plagioclase_in(melt, 100.0) > 9.0


def test_fc_phases_pl_before_cpx():
    melt = fig2_primary_melt()["oxides_wt_percent"]
    f_check = np.array([0.08, 0.22, 0.40, 0.55, 0.80])
    st = simulate_crystallization_path(
        primary_melt_oxides_wt=melt,
        f_grid=f_check,
        path="fc_100",
        fig2_ab_calibrate=False,
        kd_engine="langmuir",
        mineral_backend="fig2",
        p_fc_mpa=100.0,
    )
    by_f = {float(s.f_solid): s for s in st}

    s08 = by_f[0.08]
    s22 = by_f[0.22]
    s40 = by_f[0.40]

    assert s08.cum_pl * 100 >= 10.0
    assert s22.cum_cpx * 100 < 5.0
    assert s22.cum_pl * 100 >= 50.0
    assert s40.cum_cpx * 100 >= 2.0


def test_fc_phase_rms_vs_fig2c():
    melt = fig2_primary_melt()["oxides_wt_percent"]
    f_check = np.array([0.08, 0.22, 0.40, 0.55, 0.70, 0.80])
    st = simulate_crystallization_path(
        primary_melt_oxides_wt=melt,
        f_grid=f_check,
        path="fc_100",
        fig2_ab_calibrate=False,
        kd_engine="langmuir",
        p_fc_mpa=100.0,
    )
    errs = []
    for s in st:
        ol_p, cpx_p, pl_p = cumulative_phases_pct("fc_100", float(s.f_solid))
        ol = 100 * s.cum_ol
        pl = 100 * s.cum_pl
        cpx = 100 * s.cum_cpx
        errs.append((ol - ol_p) ** 2 + (pl - pl_p) ** 2 + (cpx - cpx_p) ** 2)
    rms = float(np.sqrt(np.mean(errs)))
    assert rms < 22.0, f"phase RMS {rms:.1f} too high vs Fig.2c"


def test_fc_vp_incremental_no_spikes():
    """Incremental Vp should not oscillate Ol↔Pl on the fine Fig.2 grid."""
    melt = fig2_primary_melt()["oxides_wt_percent"]
    f = np.linspace(0.0, 0.8, 81)
    st = simulate_crystallization_path(
        primary_melt_oxides_wt=melt,
        f_grid=f,
        path="fc_100",
        fig2_ab_calibrate=False,
        kd_engine="langmuir",
        mineral_backend="fig2",
        p_fc_mpa=100.0,
    )
    vp_inc = np.array([s.vp_incremental_km_s for s in st], dtype=float)
    d_inc = np.abs(np.diff(vp_inc))
    # Ol (~7.5) vs Pl (~6.7) hard toggling used to give ΔVp > 0.8 km/s per step.
    assert float(d_inc.max()) < 0.45


if __name__ == "__main__":
    test_fig2_melt_is_high_al()
    test_fc_phases_pl_before_cpx()
    test_fc_phase_rms_vs_fig2c()
    test_fc_vp_incremental_no_spikes()
    print("ok")
