"""Modern H–Vp curves share Fig.12 line families (χ solids + χ=1,b dashes)."""

from __future__ import annotations


def test_build_modern_hvp_curves_layout():
    from petrology.gui.hvp_plot import (
        FIG12_CHI1_B_VALUES_KM,
        FIG12_SOLID_CHI_VALUES,
        build_modern_hvp_curves,
    )
    from petrology.melting.pymelt_lithology_adapter import resolve_lithology_col_kwargs

    col_kw = resolve_lithology_col_kwargs(lithology_backend="native")
    # Tiny Tp grid for speed
    main, chi1_b = build_modern_hvp_curves(
        tp_values_c=[1300.0, 1400.0, 1500.0],
        col_kw=col_kw,
        solid_chi_values=FIG12_SOLID_CHI_VALUES,
        chi1_b_values_km=FIG12_CHI1_B_VALUES_KM,
        pyroxenite_frac=0.10,
        melting_engine="kinzler_linear",
    )
    assert {p.chi for p in main} == set(FIG12_SOLID_CHI_VALUES)
    assert all(abs(getattr(p, "b_km", 0.0) - 0.0) < 1e-9 for p in main)
    assert len(chi1_b) == len(FIG12_CHI1_B_VALUES_KM)
    for b_val, pts in chi1_b:
        assert b_val in FIG12_CHI1_B_VALUES_KM
        assert all(abs(p.chi - 1.0) < 1e-9 for p in pts)
        assert all(abs(p.b_km - b_val) < 1e-9 for p in pts)
        assert len(pts) == 3


def test_plot_modern_fig12_style_smoke():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from petrology.gui.hvp_plot import (
        FIG12_CHI1_B_VALUES_KM,
        FIG12_SOLID_CHI_VALUES,
        build_modern_hvp_curves,
        plot_hvp_fig12_style,
    )
    from petrology.melting.pymelt_lithology_adapter import resolve_lithology_col_kwargs

    col_kw = resolve_lithology_col_kwargs(lithology_backend="native")
    main, chi1_b = build_modern_hvp_curves(
        tp_values_c=[1300.0, 1400.0, 1500.0],
        col_kw=col_kw,
        solid_chi_values=FIG12_SOLID_CHI_VALUES,
        chi1_b_values_km=FIG12_CHI1_B_VALUES_KM,
        melting_engine="kinzler_linear",
    )
    fig, ax = plt.subplots()
    plot_hvp_fig12_style(
        ax,
        main,
        chi1_b_lines=chi1_b,
        tp_contour_interval_c=100.0,
        use_paper_axis_limits=True,
    )
    plt.close(fig)
