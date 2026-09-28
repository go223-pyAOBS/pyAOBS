"""tomo2d Qt GUI 轻量冒烟：可 import，可选 offscreen 建窗。"""

from __future__ import annotations

import os

import pytest


def test_tomo2d_gui_module_importable() -> None:
    pytest.importorskip("PySide6")
    from pyAOBS.modeling.tomo2d import launch_tomo2d_gui
    from pyAOBS.modeling.tomo2d.gui import main
    from pyAOBS.modeling.tomo2d.gui.app import main as app_main
    from pyAOBS.modeling.tomo2d.gui.main_window import Tomo2DMainWindow

    assert callable(launch_tomo2d_gui)
    assert callable(main) and callable(app_main)
    assert Tomo2DMainWindow is not None


def test_tomo2d_package_main_delegates_to_gui() -> None:
    pytest.importorskip("PySide6")
    from pyAOBS.modeling.tomo2d.__main__ import main as pkg_main
    from pyAOBS.modeling.tomo2d.gui.app import main as gui_main

    assert callable(pkg_main) and callable(gui_main)


def test_tomo2d_main_window_offscreen() -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from pyAOBS.modeling.tomo2d.gui.main_window import Tomo2DMainWindow

    app = QApplication.instance() or QApplication([])
    win = Tomo2DMainWindow()
    assert win.tabs.count() == 13
    assert win.state.get_str("work_dir")
    win.close()
    del app
