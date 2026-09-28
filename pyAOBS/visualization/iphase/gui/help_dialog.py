# -*- coding: utf-8 -*-
"""帮助：从 docs/HELP.md 加载（含快捷键与关于），非模态显示。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import QDialog, QTextBrowser, QVBoxLayout

from .dialog_utils import show_modeless_dialog

_HELP_MD = Path(__file__).resolve().parents[1] / "docs" / "HELP.md"
_open_help: Optional[QDialog] = None


def help_markdown_path() -> Path:
    return _HELP_MD


def load_help_text() -> str:
    path = help_markdown_path()
    if not path.is_file():
        return (
            "# 未找到帮助文件\n\n"
            f"期望路径：`{path}`\n\n"
            "请确认 `visualization/iphase/docs/HELP.md` 存在。"
        )
    return path.read_text(encoding="utf-8")


def show_help_dialog(*, activate: bool = False) -> QDialog:
    """打开帮助（非模态；已打开则前置，不重复建窗）。"""
    global _open_help
    if _open_help is not None:
        try:
            if _open_help.isVisible():
                if activate:
                    _open_help.raise_()
                    _open_help.activateWindow()
                return _open_help
        except RuntimeError:
            _open_help = None

    dlg = QDialog()
    dlg.setWindowTitle("iphase — 帮助")
    dlg.resize(920, 720)
    dlg.setMinimumSize(560, 480)
    lay = QVBoxLayout(dlg)
    browser = QTextBrowser()
    browser.setOpenExternalLinks(True)
    font = QFont("Consolas")
    if not font.exactMatch():
        font = QFont("Microsoft YaHei UI")
    font.setPointSize(10)
    browser.setFont(font)
    text = load_help_text()
    try:
        browser.setMarkdown(text)
    except Exception:
        browser.setPlainText(text)
    lay.addWidget(browser)
    show_modeless_dialog(dlg, activate=activate)
    _open_help = dlg

    def _clear(_code: int = 0) -> None:
        global _open_help
        if _open_help is dlg:
            _open_help = None

    dlg.finished.connect(_clear)
    return dlg


def install_help_shortcut(window) -> None:
    """在工程主窗上绑定 F1 → 帮助。"""
    try:
        sc = QShortcut(QKeySequence("F1"), window)
        sc.activated.connect(lambda: show_help_dialog(activate=True))
        window._help_f1_shortcut = sc
    except Exception:
        pass
