"""ui_prefs：最近路径与窗口布局。"""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QSplitter, QWidget

from pyAOBS.modeling.tomo2d.gui.services.ui_prefs import (
    filter_recent_for_mode,
    list_recent_paths,
    push_recent_path,
    restore_window_layout,
    save_window_layout,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_push_and_list_recent(tmp_path: Path) -> None:
    ini = tmp_path / "prefs.ini"
    s = QSettings(str(ini), QSettings.Format.IniFormat)
    a = tmp_path / "a.dat"
    b = tmp_path / "b.dat"
    a.write_text("1", encoding="utf-8")
    b.write_text("2", encoding="utf-8")
    push_recent_path(a, settings=s, max_items=5)
    push_recent_path(b, settings=s, max_items=5)
    push_recent_path(a, settings=s, max_items=5)  # 置顶
    items = list_recent_paths(settings=s, max_items=5)
    assert items[0] == str(a.resolve())
    assert str(b.resolve()) in items
    files = filter_recent_for_mode(items, mode="open_file")
    assert str(a.resolve()) in files


def test_save_restore_geometry(qapp, tmp_path: Path) -> None:
    ini = tmp_path / "win.ini"
    s = QSettings(str(ini), QSettings.Format.IniFormat)
    w = QWidget()
    w.resize(640, 400)
    sp = QSplitter()
    sp.addWidget(QWidget())
    sp.addWidget(QWidget())
    sp.setSizes([200, 400])
    save_window_layout(w, "test_win", splitters={"main": sp}, settings=s)
    w2 = QWidget()
    sp2 = QSplitter()
    sp2.addWidget(QWidget())
    sp2.addWidget(QWidget())
    ok = restore_window_layout(
        w2, "test_win", splitters={"main": sp2}, settings=s
    )
    assert ok
    w.close()
    w2.close()
