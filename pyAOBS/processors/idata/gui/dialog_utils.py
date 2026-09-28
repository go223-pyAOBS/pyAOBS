# -*- coding: utf-8 -*-
"""非模态对话框 helper（对齐工程规则：禁止对工具/结果窗用 exec()）。"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

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


def show_modeless_message(
    title: str,
    text: str,
    *,
    detailed: str = "",
    icon: QMessageBox.Icon = QMessageBox.Icon.Information,
    activate: bool = False,
) -> QMessageBox:
    """结果/提示用非模态消息框，不冻结主窗口。"""
    box = QMessageBox()
    box.setWindowTitle(title)
    box.setIcon(icon)
    box.setText(text)
    if detailed:
        box.setDetailedText(detailed)
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    show_modeless_dialog(box, activate=activate)
    return box


def show_modeless_text(
    title: str,
    body: str,
    *,
    summary: str = "",
    activate: bool = False,
    width: int = 960,
    height: int = 780,
) -> QDialog:
    """长文本报告窗（非模态）。"""
    dlg = QDialog()
    dlg.setWindowTitle(title)
    dlg.resize(width, height)
    dlg.setMinimumSize(480, 520)
    lay = QVBoxLayout(dlg)
    if summary:
        lay.addWidget(QLabel(summary))
    edit = QPlainTextEdit()
    edit.setReadOnly(True)
    edit.setPlainText(body)
    edit.setMinimumHeight(420)
    lay.addWidget(edit, stretch=1)
    row = QHBoxLayout()
    row.addStretch(1)
    btn = QPushButton("关闭")
    btn.clicked.connect(dlg.close)
    row.addWidget(btn)
    lay.addLayout(row)
    show_modeless_dialog(dlg, activate=activate)
    return dlg
