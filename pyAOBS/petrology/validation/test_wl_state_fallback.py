"""Langmuir STATE → saturation fallback for high-MgO catalog melts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.data.load_catalog import load_melt_catalog
from petrology.fc.delta_vp import FIG5_P_EVAL_MPA, FIG5_T_EVAL_C
from petrology.fc.wl1990 import simulate_crystallization_path
from petrology.norm_velocity import norm_velocity_from_record
from petrology.validation.debug_delta_vp_sign import _raw_delta_vp


def test_high_mgo_catalog_not_pure_plagioclase():
    """Walter 1998 run 70.02: STATE FA ≈ plag-only; FC must retain olivine."""
    rec = next(r for r in load_melt_catalog() if r["id"] == "w98_run70_02")
    ox = {
        k: float(rec[k])
        for k in ("SiO2", "TiO2", "Al2O3", "Cr2O3", "FeO", "MgO", "CaO", "Na2O", "K2O")
    }
    st = simulate_crystallization_path(
        primary_melt_oxides_wt=ox,
        f_grid=np.array([0.0, 0.75], dtype=float),
        path="fc_100",
        p_fc_mpa=400.0,
        kd_engine="langmuir",
        fig2_ab_calibrate=False,
    )
    s = st[-1]
    assert s.cum_ol > 0.15, f"expected olivine-rich cumulate, got {100*s.cum_ol:.0f}% Ol"
    assert s.cum_pl < 0.85, f"pure plag cumulate at F=0.75: {100*s.cum_pl:.0f}% Pl"


def test_w98_delta_vp_in_fig5_band():
    """High-MgO Walter 1998 melt: positive but not extreme ΔVp @ Fig.5 report state."""
    rec = next(r for r in load_melt_catalog() if r["id"] == "w98_run70_02")
    ox = {
        k: float(rec[k])
        for k in ("SiO2", "TiO2", "Al2O3", "Cr2O3", "FeO", "MgO", "CaO", "Na2O", "K2O")
    }
    bulk = float(
        norm_velocity_from_record(
            rec,
            p_pa=FIG5_P_EVAL_MPA * 1e6,
            t_k=FIG5_T_EVAL_C + 273.15,
            mineral_backend="sb1994_fig2ol",
        )["vp_km_s"]
    )
    r = _raw_delta_vp(
        ox,
        f_solid=0.75,
        p_fc_mpa=400.0,
        bulk_vp_km_s=bulk,
        p_eval_mpa=FIG5_P_EVAL_MPA,
        t_eval_c=FIG5_T_EVAL_C,
        mineral_backend="sb1994_fig2ol",
        cipw_backend="auto",
    )
    d = float(r["delta_raw_km_s"])
    assert 0.0 < d < 0.25, f"w98 ΔVp={d:+.3f} km/s outside Fig.5 band"


if __name__ == "__main__":
    test_high_mgo_catalog_not_pure_plagioclase()
    test_w98_delta_vp_in_fig5_band()
    print("ok")
