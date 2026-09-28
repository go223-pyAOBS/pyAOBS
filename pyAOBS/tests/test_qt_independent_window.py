# -*- coding: utf-8 -*-
from __future__ import annotations

import os

import pytest


def test_independent_dialog_gets_visible_frame() -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QDialog, QLabel, QVBoxLayout

    from pyAOBS.utils.qt_independent_window import (
        FRAME_BORDER_COLOR,
        configure_independent_window,
    )

    app = QApplication.instance() or QApplication([])
    dlg = QDialog()
    dlg.setWindowTitle("frame-test")
    lay = QVBoxLayout(dlg)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(QLabel("hi"))
    configure_independent_window(dlg)

    flags = dlg.windowFlags()
    assert flags & Qt.WindowType.Window
    assert flags & Qt.WindowType.WindowTitleHint
    assert flags & Qt.WindowType.WindowSystemMenuHint
    assert dlg.testAttribute(Qt.WidgetAttribute.WA_StyledBackground)
    assert FRAME_BORDER_COLOR in (dlg.styleSheet() or "")
    m = dlg.layout().contentsMargins()
    assert m.left() >= 3 and m.top() >= 3

    configure_independent_window(dlg)
    assert (dlg.styleSheet() or "").count("pyaobsIndependentWindow") == 1
    dlg.close()
    del app


def test_independent_main_window_wraps_central() -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QLabel, QMainWindow

    from pyAOBS.utils.qt_independent_window import (
        FRAME_BORDER_COLOR,
        apply_independent_window_frame,
    )

    app = QApplication.instance() or QApplication([])
    win = QMainWindow()
    inner = QLabel("plot")
    inner.setObjectName("plotInner")
    win.setCentralWidget(inner)
    apply_independent_window_frame(win)
    frame = win.centralWidget()
    assert frame is not None
    assert frame.objectName() == "PyAobsIndependentFrame"
    assert FRAME_BORDER_COLOR in (frame.styleSheet() or "")
    apply_independent_window_frame(win)
    assert win.centralWidget() is frame
    win.close()
    del app


def test_configure_independent_window_keeps_accept_drops() -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QWidget

    from pyAOBS.utils.qt_independent_window import configure_independent_window

    app = QApplication.instance() or QApplication([])
    win = QWidget()
    win.setAcceptDrops(True)
    configure_independent_window(win)
    assert win.acceptDrops()
    win.close()
    del app


def test_message_box_skips_minmax_flags() -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QMessageBox

    from pyAOBS.utils.qt_independent_window import configure_independent_window

    app = QApplication.instance() or QApplication([])
    box = QMessageBox()
    box.setText("hi")
    configure_independent_window(box)
    flags = box.windowFlags()
    assert flags & Qt.WindowType.Window
    assert flags & Qt.WindowType.WindowCloseButtonHint
    assert not (flags & Qt.WindowType.WindowMinMaxButtonsHint)
    box.close()
    del app


def test_prefer_xcb_on_wayland_respects_existing(monkeypatch) -> None:
    from pyAOBS.utils import qt_platform as qp

    monkeypatch.setenv("QT_QPA_PLATFORM", "wayland")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(qp.sys, "platform", "linux")
    assert qp.prefer_xcb_on_wayland() is False
    assert os.environ.get("QT_QPA_PLATFORM") == "wayland"


def test_prefer_xcb_on_wayland_sets_xcb(monkeypatch) -> None:
    from pyAOBS.utils import qt_platform as qp

    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(qp.sys, "platform", "linux")
    assert qp.prefer_xcb_on_wayland() is True
    assert os.environ.get("QT_QPA_PLATFORM") == "xcb"
