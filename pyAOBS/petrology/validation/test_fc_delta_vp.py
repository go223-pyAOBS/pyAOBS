"""W&L FC ΔVp wired to KKHS02 Step-3 chain."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.fc.delta_vp import delta_vp_wl_fc
from petrology.fc.wl1990 import load_kinzler1997_morb_primary
from petrology.fractionation import delta_vp_km_s, delta_vp_param_km_s


def test_wl_fc_delta_positive_at_high_f():
    """Fig.2 anchor Vp path yields paper-order ΔVp @ F=0.75 (KKHS02 Fig.5 band)."""
    p = load_kinzler1997_morb_primary()
    d = delta_vp_wl_fc(
        melt_oxides_wt=p["oxides_wt_percent"],
        f_solid=0.75,
        p_fc_mpa=400.0,
        p_eval_mpa=100.0,
        t_eval_c=100.0,
        mineral_backend="fig2",
        use_cache=False,
        fig2_ab_calibrate=True,
        fig2_ab_anchor_vp=True,
    )
    assert 0.05 < d < 0.8


def test_wl_fc_delta_raw_physical_chain():
    """Unclamped HS chain @ Fig.5 report state (R3; sb1994_fig2ol mixed HS scale)."""
    p = load_kinzler1997_morb_primary()
    d_raw = delta_vp_wl_fc(
        melt_oxides_wt=p["oxides_wt_percent"],
        f_solid=0.75,
        p_fc_mpa=400.0,
        bulk_vp_km_s=None,
        mineral_backend="sb1994_fig2ol",
        kd_engine="langmuir",
        clamp_negative=False,
        use_cache=False,
    )
    assert 0.08 < d_raw < 0.25


def test_auto_engine_uses_wl_when_melt_given():
    p = load_kinzler1997_morb_primary()
    d_auto = delta_vp_km_s(
        0.75,
        bulk_vp_km_s=7.2,
        p_fc_mpa=400.0,
        engine="auto",
        melt_oxides_wt=p["oxides_wt_percent"],
    )
    d_wl = delta_vp_wl_fc(
        melt_oxides_wt=p["oxides_wt_percent"],
        f_solid=0.75,
        p_fc_mpa=400.0,
        bulk_vp_km_s=7.2,
    )
    assert abs(d_auto - d_wl) < 1e-9


def test_param_engine_without_melt():
    d = delta_vp_km_s(0.75, bulk_vp_km_s=7.2, p_fc_mpa=400.0, engine="param")
    d_ref = delta_vp_param_km_s(0.75, bulk_vp_km_s=7.2, p_fc_mpa=400.0)
    assert abs(d - d_ref) < 1e-9


def test_pressure_weak_sensitivity():
    p = load_kinzler1997_morb_primary()
    d100 = delta_vp_wl_fc(melt_oxides_wt=p["oxides_wt_percent"], f_solid=0.75, p_fc_mpa=100.0, bulk_vp_km_s=7.17)
    d800 = delta_vp_wl_fc(melt_oxides_wt=p["oxides_wt_percent"], f_solid=0.75, p_fc_mpa=800.0, bulk_vp_km_s=7.17)
    assert abs(d100 - d800) < 0.25


if __name__ == "__main__":
    test_wl_fc_delta_positive_at_high_f()
    test_wl_fc_delta_raw_physical_chain()
    test_auto_engine_uses_wl_when_melt_given()
    test_param_engine_without_melt()
    test_pressure_weak_sensitivity()
    print("ok")
