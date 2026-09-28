# -*- coding: utf-8 -*-
"""非模态对话框 helper（对齐工程规则：禁止对长生命周期窗用 exec()）。"""

from __future__ import annotations

from typing import List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog

from pyAOBS.utils.qt_independent_window import (
    apply_native_window_border,
    configure_independent_window,
)

_registry: List[QDialog] = []


def configure_independent_dialog(dlg: QDialog) -> None:
    configure_independent_window(dlg)


def show_modeless_dialog(dlg: QDialog, *, activate: bool = False) -> None:
    configure_independent_dialog(dlg)
    dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    _registry.append(dlg)

    def _on_finished(_code: int = 0) -> None:
        try:
            _registry.remove(dlg)
        except ValueError:
            pass

    try:
        dlg.finished.connect(_on_finished)
    except Exception:
        pass
    dlg.show()
    apply_native_window_border(dlg)
    if activate:
        dlg.raise_()
        dlg.activateWindow()
