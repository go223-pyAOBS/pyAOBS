# -*- coding: utf-8 -*-
"""非模态对话框 helper（对齐工程规则：禁止对工具/结果窗用 exec()）。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from pyAOBS.utils.qt_independent_window import (
    apply_native_window_border,
    configure_independent_window,
)

_registry: List[QWidget] = []


def file_dialog_options(
    *extra: QFileDialog.Option,
) -> QFileDialog.Option:
    """Linux 上不用 GTK/Nautilus 原生框（避免 Tracker/GVFS 刷屏，WSLg 也更稳）。"""
    opts = QFileDialog.Option(0)
    for e in extra:
        opts |= e
    if sys.platform.startswith("linux"):
        opts |= QFileDialog.Option.DontUseNativeDialog
    return opts


def configure_independent_dialog(dlg: QWidget) -> None:
    configure_independent_window(dlg)


def style_wrapping_caption(lbl: QLabel) -> None:
    """图区上方说明：按栏宽换行，避免单行裁切。"""
    lbl.setWordWrap(True)
    lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
    lbl.setStyleSheet("color:#111111;")


def set_wrapping_caption(lbl: QLabel, *lines: str) -> None:
    parts = [str(s).strip() for s in lines if s and str(s).strip()]
    text = "\n".join(parts)
    lbl.setText(text)
    lbl.setToolTip(text)


def add_combo_path_item(combo: QComboBox, label: str, path: str) -> int:
    """下拉显示短标签，条目与收起状态均可悬停看完整路径。"""
    combo.addItem(str(label), str(path))
    i = combo.count() - 1
    combo.setItemData(i, str(path), Qt.ItemDataRole.ToolTipRole)
    return i


def refresh_combo_path_tooltip(combo: QComboBox, *, hint: str = "") -> None:
    """收起的下拉框悬停显示当前项完整路径（再附操作说明）。"""
    data = combo.currentData()
    path = str(data).strip() if data else ""
    if path and hint:
        combo.setToolTip(f"{path}\n{hint}")
    elif path:
        combo.setToolTip(path)
    else:
        combo.setToolTip(hint)


def wire_path_combo_tooltips(combo: QComboBox, *, hint: str) -> None:
    """弹出列表与收起状态都显示路径；在填充条目后也会随 currentIndex 更新。"""
    try:
        combo.view().setMouseTracking(True)
    except Exception:
        pass
    combo.setToolTipDuration(15000)

    def _refresh(*_a) -> None:
        refresh_combo_path_tooltip(combo, hint=hint)

    combo.currentIndexChanged.connect(_refresh)
    _refresh()


def show_modeless_dialog(dlg: QWidget, *, activate: bool = False) -> None:
    """非模态工具/结果窗（QDialog 或普通 QWidget）。"""
    configure_independent_dialog(dlg)
    dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    _registry.append(dlg)

    def _drop(*_a) -> None:
        try:
            _registry.remove(dlg)
        except ValueError:
            pass

    if isinstance(dlg, QDialog):
        try:
            dlg.finished.connect(_drop)
        except Exception:
            pass
    try:
        dlg.destroyed.connect(_drop)
    except Exception:
        pass
    dlg.show()
    apply_native_window_border(dlg)
    if bool(dlg.property("_pyaobsWantDrops")) or dlg.acceptDrops():
        dlg.setAcceptDrops(True)
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
    box = QMessageBox()
    box.setWindowTitle(title)
    box.setIcon(icon)
    box.setText(text)
    if detailed:
        box.setDetailedText(detailed)
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    show_modeless_dialog(box, activate=activate)
    return box


def open_containing_directory(path: Path | str | None) -> bool:
    """在资源管理器中打开文件所在目录（若本身是目录则打开该目录）。"""
    if path is None or not str(path).strip():
        show_modeless_message("打开所在目录", "当前没有关联文件。")
        return False
    p = Path(path)
    folder = p if p.is_dir() else p.parent
    if not folder.is_dir():
        show_modeless_message("打开所在目录", f"找不到目录:\n{folder}")
        return False
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder.resolve())))
    return True


def show_modeless_text(
    title: str,
    body: str,
    *,
    summary: str = "",
    activate: bool = False,
    width: int = 960,
    height: int = 780,
    monospace: bool = False,
) -> QDialog:
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
    if monospace:
        from PySide6.QtGui import QFontDatabase

        edit.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
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
