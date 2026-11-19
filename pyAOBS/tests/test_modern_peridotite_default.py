"""Modern reebox default peridotite is Matthews KLB-1; Katz and ball_* stay selectable."""

from __future__ import annotations

import argparse
import inspect

import pytest

from petrology.melting.lithology import DEFAULT_PYMELT_PERIDOTITE_LITH, heterogeneous_source
from petrology.melting.melt_chemistry import default_chemistry_for_lithology
from petrology.melting.pymelt_lithology_adapter import (
    LITHOLOGY_PRESETS,
    add_lithology_cli,
    list_pymelt_lithology_keys,
    resolve_lithology_col_kwargs,
)
from petrology.reference.kkhs02_workflow import (
    format_track_terminology_text,
    format_workflow_summary_short,
)


def test_default_key_is_matthews_klb1():
    assert DEFAULT_PYMELT_PERIDOTITE_LITH == "matthews_klb1"


def test_api_defaults_and_registry():
    from petrology.melting.column import forward_melting_column
    from petrology.melting.heterogeneous import forward_heterogeneous_column
    from petrology.melting.hvp_scan import scan_hvp_lip
    from petrology.melting.lip_scan import scan_hvp_lip_phi
    from petrology.melting.pymelt_bridge import build_mantle
    from petrology.melting.pymelt_lithology_adapter import heterogeneous_source_pymelt

    for fn in (
        forward_melting_column,
        forward_heterogeneous_column,
        scan_hvp_lip,
        scan_hvp_lip_phi,
    ):
        params = inspect.signature(fn).parameters
        assert params["peridotite_lith"].default == "matthews_klb1"
        assert params["lithology_backend"].default == "native"

    col_sig = inspect.signature(forward_melting_column).parameters
    assert col_sig["melting_engine"].default == "kinzler_linear"
    assert inspect.signature(heterogeneous_source).parameters["peridotite_key"].default == "matthews_klb1"
    assert inspect.signature(heterogeneous_source_pymelt).parameters["peridotite_key"].default == "matthews_klb1"
    assert inspect.signature(build_mantle).parameters["peridotite_key"].default == "matthews_klb1"

    assert LITHOLOGY_PRESETS["lip_default"].peridotite_key == "matthews_klb1"
    assert LITHOLOGY_PRESETS["lip_default"].pyroxenite_key == "pertermann_g2"
    assert LITHOLOGY_PRESETS["lip_default"].peridotite_h2o_wt == 0.0
    assert LITHOLOGY_PRESETS["lip_kg1"].peridotite_key == "katz_lherzolite"
    assert LITHOLOGY_PRESETS["greenland_kg1"].peridotite_key == "katz_lherzolite"
    assert LITHOLOGY_PRESETS["lip_ball"].peridotite_key == "ball_primitive_mantle"

    keys = set(list_pymelt_lithology_keys())
    for key in (
        "katz_lherzolite",
        "matthews_klb1",
        "ball_depleted_mantle",
        "ball_mixed_mantle",
        "ball_primitive_mantle",
    ):
        assert key in keys

    assert default_chemistry_for_lithology(family="peridotite", lithology_backend="pymelt") == "pmelts_klb1"


def test_resolve_and_cli_default_to_matthews():
    kw = resolve_lithology_col_kwargs(lithology_backend="pymelt")
    assert kw["lithology_backend"] == "pymelt"
    assert kw["peridotite_lith"] == "matthews_klb1"
    assert "peridotite_chemistry" not in kw

    preset = resolve_lithology_col_kwargs(lithology_preset="lip_default")
    assert preset["peridotite_lith"] == "matthews_klb1"
    assert preset["pyroxenite_lith"] == "pertermann_g2"
    assert preset["lithology_backend"] == "pymelt"

    parser = argparse.ArgumentParser()
    add_lithology_cli(parser)
    args = parser.parse_args([])
    assert args.peridotite_lith == "matthews_klb1"
    assert args.lithology_backend == "native"


def test_user_facing_modern_label_is_matthews():
    text = format_track_terminology_text()
    summary = format_workflow_summary_short()
    assert "Modern（Katz）" not in text
    assert "Matthews" in text
    assert "Katz 熔融" not in summary
    assert "matthews_klb1" in summary or "Matthews" in summary


def test_native_backend_keeps_analytic_katz():
    liths = heterogeneous_source(
        pyroxenite_frac=0.0,
        backend="native",
        peridotite_key="matthews_klb1",
    )
    assert len(liths) == 1
    assert liths[0].name == "peridotite"
    assert liths[0].chemistry_backend == "kinzler1997"
    assert liths[0]._pymelt is None
    assert liths[0].katz_cpx_mass is not None


def test_native_reebox_column_ignores_default_key():
    from petrology.melting.heterogeneous import forward_heterogeneous_column

    col = forward_heterogeneous_column(
        tp_c=1350.0,
        b_km=0.0,
        chi=2.0,
        pyroxenite_frac=0.0,
        n_isentropic_steps=8,
        lithology_backend="native",
    )
    assert col.lithologies[0].name == "peridotite"
    assert col.lithologies[0]._pymelt is None
    assert col.h_km > 0.0


def test_pymelt_reebox_default_is_matthews_and_katz_still_runs():
    from petrology.melting.pymelt_bridge import ensure_pymelt_scipy_compat

    ensure_pymelt_scipy_compat()
    pytest.importorskip("pyMelt")
    from petrology.melting.heterogeneous import forward_heterogeneous_column

    common = dict(tp_c=1400.0, b_km=0.0, chi=4.0, pyroxenite_frac=0.0, n_isentropic_steps=12)
    default = forward_heterogeneous_column(**common, lithology_backend="pymelt")
    katz = forward_heterogeneous_column(
        **common,
        lithology_backend="pymelt",
        peridotite_lith="katz_lherzolite",
    )
    assert default.lithologies[0].name == "matthews_klb1"
    assert default.lithologies[0].chemistry_backend == "pmelts_klb1"
    assert katz.lithologies[0].name == "katz_lherzolite"
    assert katz.lithologies[0].chemistry_backend == "pmelts_klb1"
    d_tsol = abs(default.lithologies[0].solidus_gpa(2.0) - katz.lithologies[0].solidus_gpa(2.0))
    assert d_tsol > 1.0
    assert default.h_km > 0.0
    assert katz.h_km > 0.0


def test_gui_startup_selects_lip_default():
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from petrology.gui.mainwindow import LipMainWindow

    app = QApplication.instance() or QApplication([])
    assert app is not None
    win = LipMainWindow()
    try:
        assert win._backend_combo.currentText() == "pymelt"
        assert win._engine_combo.currentText() == "reebox"
        assert win._preset_combo.currentText() == "lip_default"
        assert win._per_combo.currentText() == "matthews_klb1"
        items = [win._per_combo.itemText(i) for i in range(win._per_combo.count())]
        assert "katz_lherzolite" in items
        assert any(name.startswith("ball_") for name in items)
    finally:
        win.close()
