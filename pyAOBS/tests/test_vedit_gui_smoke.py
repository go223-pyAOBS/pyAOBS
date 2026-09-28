"""vedit Qt GUI 轻量冒烟：可 import，不弹窗。"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_vedit_gui_module_importable() -> None:
    pytest.importorskip("PySide6")
    from pyAOBS.modeling.vedit.gui import main
    from pyAOBS.modeling.vedit.gui.app import main as app_main
    from pyAOBS.modeling.vedit.gui.mainwindow import VeditMainWindow

    assert callable(main) and callable(app_main)
    assert VeditMainWindow is not None


def test_vedit_gui_canvas_redraw_agg() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from pyAOBS.modeling.vedit.core import Model
    from pyAOBS.modeling.vedit.gui.canvas import redraw_model_profile, redraw_vplot_profile

    vin = REPO_ROOT / "modeling" / "vedit" / "examples" / "v1.in"
    model = Model.load(str(vin))
    fig = Figure()
    ax = fig.add_subplot(111)
    lines = redraw_model_profile(ax, model, {(0, 0): [1]})
    assert len(lines[0]) == model.nlayer
    lines2, vnodes_overlay = redraw_model_profile(
        ax, model, show_velocity_nodes=True, show_velocity_fill=True, velocity_grid=40
    )
    assert len(lines2) == model.nlayer
    assert len(vnodes_overlay) >= 2
    vnodes = redraw_vplot_profile(ax, model, [0], (0, True, 1))
    assert len(vnodes) >= 2
