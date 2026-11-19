"""Smoke + acceptance for Fig.12a forward vs digitized RMS overlay."""

from __future__ import annotations


def test_linear_forward_matches_digitized_within_noise():
    """Printed Fig.12a is the linear §4 + eq.(1) grid; RMS should be ≲ 0.02 km/s."""
    from petrology.hvp.fig12a_background import DEFAULT_HVP_DIGITIZED, parse_hvp_digitized
    from petrology.validation.compare_fig12a_forward_vs_digitized import (
        build_linear_forward,
        compare_forward_to_digitized,
    )

    dig = parse_hvp_digitized(DEFAULT_HVP_DIGITIZED)
    main, chi1_b = build_linear_forward(tp_step=10.0)
    summary = compare_forward_to_digitized(
        dig, main, chi1_b, track_name="fig12_linear"
    )
    assert summary.n_track_curves >= 7
    assert summary.pooled_n >= 50
    assert summary.pooled_rms_dvp_km_s < 0.02
    assert abs(summary.pooled_mean_dvp_km_s) < 0.01


def test_compare_cli_imports():
    from petrology.validation import compare_fig12a_forward_vs_digitized as mod

    assert callable(mod.run)
    assert callable(mod.compare_forward_to_digitized)
