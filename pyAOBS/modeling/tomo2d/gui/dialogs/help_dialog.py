# -*- coding: utf-8 -*-
"""帮助：对齐 vedit/idata — docs/HELP.md（Markdown）+ 可选 TomoHelp 章节；非模态单例；F1。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QTextBrowser,
    QVBoxLayout,
)

from ..dialog_utils import show_modeless_dialog

_HELP_MD = Path(__file__).resolve().parents[2] / "docs" / "HELP.md"
_open_help: Optional[QDialog] = None

_GUI_SECTION = "GUI 快速说明 (HELP.md)"


def help_markdown_path() -> Path:
    return _HELP_MD


def load_help_text() -> str:
    path = help_markdown_path()
    if not path.is_file():
        return (
            "# 未找到帮助文件\n\n"
            f"期望路径：`{path}`\n\n"
            "请确认 `modeling/tomo2d/docs/HELP.md` 存在。"
        )
    return path.read_text(encoding="utf-8")


def _tomo_help_sections() -> list[tuple[str, Callable[[], str]]]:
    try:
        from ...help_docs import TomoHelp
    except ImportError:
        from pyAOBS.modeling.tomo2d.help_docs import TomoHelp

    return [
        ("Python 封装总览", TomoHelp.python_wrapper_help),
        ("gen_smesh", TomoHelp.gen_smesh_help),
        ("gen_damp", TomoHelp.gen_damp_help),
        ("gen_vcorr", TomoHelp.gen_vcorr_help),
        ("gen_dcorr", TomoHelp.gen_dcorr_help),
        ("tt_forward", TomoHelp.tt_forward_help),
        ("tt_inverse", TomoHelp.tt_inverse_help),
        ("tt_inverse -L 日志列说明", TomoHelp.tt_inverse_logfile_format_help),
        ("stat_smesh", TomoHelp.stat_smesh_help),
        ("edit_smesh / edit_smesh_HHB", TomoHelp.edit_smesh_help),
    ]


def show_help_dialog(
    *,
    activate: bool = False,
    default_section: str | None = None,
) -> QDialog:
    """打开帮助（非模态；已打开则前置，不重复建窗）。"""
    global _open_help
    if _open_help is not None:
        try:
            if _open_help.isVisible():
                if default_section and hasattr(_open_help, "select_section"):
                    _open_help.select_section(default_section)
                if activate:
                    _open_help.raise_()
                    _open_help.activateWindow()
                return _open_help
        except RuntimeError:
            _open_help = None

    dlg = QDialog()
    dlg.setWindowTitle("tomo2d — 帮助")
    dlg.resize(920, 720)
    dlg.setMinimumSize(560, 480)
    lay = QVBoxLayout(dlg)

    head = QHBoxLayout()
    head.addWidget(QLabel("章节:"))
    combo = QComboBox()
    sections = [(_GUI_SECTION, None)] + list(_tomo_help_sections())
    combo.addItems([n for n, _ in sections])
    section_map = {n: fn for n, fn in sections}
    head.addWidget(combo, stretch=1)
    lay.addLayout(head)

    browser = QTextBrowser()
    browser.setOpenExternalLinks(True)
    font = QFont("Consolas")
    if not font.exactMatch():
        font = QFont("Microsoft YaHei UI")
    font.setPointSize(10)
    browser.setFont(font)
    lay.addWidget(browser, stretch=1)

    def _refresh(name: str = "") -> None:
        key = name or combo.currentText()
        fn = section_map.get(key)
        if fn is None:
            text = load_help_text()
            try:
                browser.setMarkdown(text)
            except Exception:
                browser.setPlainText(text)
            return
        try:
            body = (fn() or "").strip()
        except Exception as e:
            body = f"加载帮助失败: {e}"
        # TomoHelp 多为纯文本；用 Markdown 代码块更易读
        try:
            browser.setMarkdown(f"```\n{body}\n```")
        except Exception:
            browser.setPlainText(body)

    def select_section(name: str | None) -> None:
        if not name:
            return
        idx = combo.findText(name)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        elif name == "GUI" or name.startswith("GUI"):
            combo.setCurrentIndex(0)

    dlg.select_section = select_section  # type: ignore[attr-defined]
    combo.currentTextChanged.connect(_refresh)

    if default_section and combo.findText(default_section) >= 0:
        combo.setCurrentText(default_section)
    else:
        combo.setCurrentIndex(0)
    _refresh(combo.currentText())

    show_modeless_dialog(dlg, activate=activate)
    _open_help = dlg

    def _clear(_code: int = 0) -> None:
        global _open_help
        if _open_help is dlg:
            _open_help = None

    dlg.finished.connect(_clear)
    return dlg


def open_program_help(
    parent=None, *, default_section: str | None = None
) -> QDialog:
    """兼容旧调用名；等同 ``show_help_dialog``。"""
    return show_help_dialog(activate=True, default_section=default_section)


def install_help_shortcut(window) -> None:
    """在主窗上绑定 F1 → 帮助（对齐 vedit / idata）。"""
    try:
        sc = QShortcut(QKeySequence("F1"), window)
        sc.activated.connect(lambda: show_help_dialog(activate=True))
        window._help_f1_shortcut = sc
    except Exception:
        pass
