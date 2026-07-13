"""Tests for Sobolev & Babeyko (1994) mineral backend."""

from __future__ import annotations

import numpy as np

from petrology.minerals import phase_properties
from petrology.sb1994 import P0_PA, T0_K, endmember_at_pt, load_endmembers

FIG2_P_PA = 100e6
FIG2_T_K = 373.15


def test_endmembers_load():
    em = load_endmembers()
    assert "Fo" in em and "Di" in em and "An" in em
    assert em["Fa"].k < 200.0  # typo-corrected from 1379


def test_fo90_olivine_hybrid_fig2_state():
    from petrology.fc.assemblage import assemblage_vp_rho

    raw = phase_properties(
        "olivine", fo=0.90, p_pa=FIG2_P_PA, t_k=FIG2_T_K, backend="sb1994",
    )
    vp_pure, _ = assemblage_vp_rho(
        ol_frac=1.0, pl_frac=0.0, cpx_frac=0.0,
        fo_pct=90, an_pct=50, di_pct=71,
        p_pa=FIG2_P_PA, t_k=FIG2_T_K, mineral_backend="sb1994_fig2ol",
    )
    assert abs(vp_pure - 7.58) < 0.02
    assert vp_pure < raw["vp_km_s"]

    vp_mix, _ = assemblage_vp_rho(
        ol_frac=0.4, pl_frac=0.6, cpx_frac=0.0,
        fo_pct=90, an_pct=50, di_pct=71,
        p_pa=FIG2_P_PA, t_k=FIG2_T_K, mineral_backend="sb1994_fig2ol",
    )
    vp_raw_mix, _ = assemblage_vp_rho(
        ol_frac=0.4, pl_frac=0.6, cpx_frac=0.0,
        fo_pct=90, an_pct=50, di_pct=71,
        p_pa=FIG2_P_PA, t_k=FIG2_T_K, mineral_backend="sb1994",
    )
    from petrology.sb1994 import (
        sb1994_fig2ol_mixed_hs_vp_scale,
        sb1994_fig2ol_mixed_hs_weight,
    )

    scale = sb1994_fig2ol_mixed_hs_vp_scale()
    w = sb1994_fig2ol_mixed_hs_weight(0.4)
    assert abs(vp_mix - vp_raw_mix * (1.0 + (scale - 1.0) * w)) < 0.02
    assert sb1994_fig2ol_mixed_hs_weight(0.73) == 0.0
    assert sb1994_fig2ol_mixed_hs_weight(0.16) == 1.0


def test_plagioclase_unchanged_in_hybrid():
    raw = phase_properties(
        "plagioclase", an=0.50, p_pa=FIG2_P_PA, t_k=FIG2_T_K, backend="sb1994",
    )
    hybrid = phase_properties(
        "plagioclase", an=0.50, p_pa=FIG2_P_PA, t_k=FIG2_T_K, backend="sb1994_fig2ol",
    )
    assert hybrid["vp_km_s"] == raw["vp_km_s"]


def test_fo90_olivine_raw_sb1994():
    props = phase_properties(
        "olivine", fo=0.90, p_pa=FIG2_P_PA, t_k=FIG2_T_K, backend="sb1994",
    )
    assert props["backend"] == "sb1994"
    assert 7.5 <= props["vp_km_s"] <= 8.5
    assert 3.2 <= props["rho_g_cm3"] <= 3.5


def test_plagioclase_an50():
    props = phase_properties(
        "plagioclase", an=0.50, p_pa=FIG2_P_PA, t_k=FIG2_T_K, backend="sb1994",
    )
    assert 6.5 <= props["vp_km_s"] <= 7.2


def test_quartz_at_fig2_state():
    hot = phase_properties("quartz", p_pa=FIG2_P_PA, t_k=FIG2_T_K, backend="sb1994")
    assert 5.5 <= hot["vp_km_s"] <= 6.5
    assert hot["backend"] == "sb1994"


def test_ilmenite_falls_back_to_empirical():
    props = phase_properties("ilmenite", backend="sb1994")
    assert props["backend"] == "empirical"


def test_pt_derivatives_finite():
    for key in ("Fo", "Ab", "Di"):
        p = endmember_at_pt(key, p_pa=600e6, t_k=673.15)
        for v in p.values():
            if isinstance(v, float):
                assert np.isfinite(v)


if __name__ == "__main__":
    test_endmembers_load()
    test_fo90_olivine_hybrid_fig2_state()
    test_plagioclase_unchanged_in_hybrid()
    test_fo90_olivine_raw_sb1994()
    test_plagioclase_an50()
    test_quartz_at_fig2_state()
    test_ilmenite_falls_back_to_empirical()
    test_pt_derivatives_finite()
    print("test_sb1994: OK")
