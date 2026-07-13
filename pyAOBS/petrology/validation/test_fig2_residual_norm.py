"""Fig.2 residual norm Vp calibration vs digitized frac_res_1kb."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.fc.cdat_library import fig2_primary_melt
from petrology.fc.figure02a_digitized import digitized_inc_vp_km_s, digitized_residual_vp_km_s
from petrology.fc.wl1990 import simulate_crystallization_path


def test_frac_res_1kb_rms_vs_digitized():
    melt = fig2_primary_melt()["oxides_wt_percent"]
    f = np.linspace(0.05, 0.80, 16)
    st = simulate_crystallization_path(
        primary_melt_oxides_wt=melt,
        f_grid=f,
        path="fc_100",
        kd_engine="langmuir",
        fig2_ab_calibrate=False,
        p_fc_mpa=100.0,
        mineral_backend="sb1994_fig2ol",
    )
    errs = []
    for s in st:
        ref = digitized_residual_vp_km_s("fc_100", float(s.f_solid))
        errs.append((s.vp_residual_norm_km_s - ref) ** 2)
    rms = float(np.sqrt(np.mean(errs)))
    assert rms < 0.08, f"frac_res_1kb RMS {rms:.3f} km/s too high vs digitized Fig.2(a)"


def test_inc_sol_1kb_rms_vs_digitized():
    melt = fig2_primary_melt()["oxides_wt_percent"]
    f = np.linspace(0.08, 0.80, 16)
    st = simulate_crystallization_path(
        primary_melt_oxides_wt=melt,
        f_grid=f,
        path="fc_100",
        kd_engine="langmuir",
        fig2_ab_calibrate=False,
        p_fc_mpa=100.0,
        mineral_backend="sb1994_fig2ol",
    )
    errs = []
    for s in st:
        ref = digitized_inc_vp_km_s("fc_100", float(s.f_solid))
        errs.append((s.vp_incremental_km_s - ref) ** 2)
    rms = float(np.sqrt(np.mean(errs)))
    assert rms < 0.06, f"inc_sol_1kb RMS {rms:.3f} km/s too high vs digitized Fig.2(a)"


if __name__ == "__main__":
    test_frac_res_1kb_rms_vs_digitized()
    test_inc_sol_1kb_rms_vs_digitized()
    print("ok")
