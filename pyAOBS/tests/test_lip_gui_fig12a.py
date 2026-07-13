"""Smoke test: LIP GUI imports and Fig.12a plot helpers."""

from __future__ import annotations


def test_hvp_plot_display_modes():
    from petrology.gui.hvp_plot import HVP_DISPLAY_CHOICES, HVP_DISPLAY_FIG12A

    keys = [k for k, _ in HVP_DISPLAY_CHOICES]
    assert HVP_DISPLAY_FIG12A in keys
    assert len(keys) >= 5


def test_fig12a_background_draw():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from petrology.hvp.fig12a_background import DEFAULT_HVP_DIGITIZED, draw_fig12a_background, parse_hvp_digitized

    fig, ax = plt.subplots()
    series = parse_hvp_digitized(DEFAULT_HVP_DIGITIZED)
    draw_fig12a_background(ax, series)
    plt.close(fig)
