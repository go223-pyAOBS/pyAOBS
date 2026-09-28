# -*- coding: utf-8 -*-
"""非模态对话框 — 与 petrology / 工程规则一致。"""

from __future__ import annotations

import re
from typing import Optional

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QLineEdit,
    QWidget,
)

try:
    from pyAOBS.petrology.gui.dialog_utils import (
        configure_independent_dialog,
        show_modeless_dialog,
        connect_combo_deferred,
        defer_after_combo_popup,
    )
except ImportError:
    # 轻量回退（无 petrology 时）
    from PySide6.QtWidgets import QDialog

    from pyAOBS.utils.qt_combo import (
        connect_combo_deferred,
        defer_after_combo_popup,
    )
    from pyAOBS.utils.qt_independent_window import (
        apply_native_window_border,
        configure_independent_window,
    )

    _registry: list = []

    def configure_independent_dialog(dlg: QDialog) -> None:
        configure_independent_window(dlg)

    def show_modeless_dialog(dlg: QDialog, *, activate: bool = False, singleton: bool = False) -> None:
        configure_independent_dialog(dlg)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        _registry.append(dlg)

        def _on_finished(_code: int = 0) -> None:
            try:
                _registry.remove(dlg)
            except ValueError:
                pass

        dlg.finished.connect(_on_finished)
        dlg.show()
        apply_native_window_border(dlg)
        if activate:
            dlg.raise_()
            dlg.activateWindow()


_NUM_RE = re.compile(
    r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
)


def _normalize_paste_number(text: str) -> Optional[str]:
    """从剪贴板提取可写入 SpinBox 的数字（兼容中文逗号/全角/带单位）。"""
    if not text:
        return None
    s = (
        str(text)
        .strip()
        .replace("\u3000", " ")
        .replace("，", ".")
        .replace(",", ".")  # 欧式千分位/小数：先统一成点，再取首个数字
    )
    # 全角数字 → 半角
    out = []
    for ch in s:
        o = ord(ch)
        if 0xFF10 <= o <= 0xFF19:
            out.append(chr(o - 0xFF10 + ord("0")))
        elif ch in ("．", "。"):
            out.append(".")
        elif ch in ("－", "—"):
            out.append("-")
        else:
            out.append(ch)
    s = "".join(out)
    m = _NUM_RE.search(s)
    return m.group(0) if m else None


class _SpinPasteFilter(QObject):
    """让 QSpinBox/QDoubleSpinBox 稳定支持 Ctrl+V / 右键粘贴。"""

    def eventFilter(self, obj, event):  # noqa: N802
        if not isinstance(obj, QLineEdit):
            return super().eventFilter(obj, event)
        if event.type() != QEvent.Type.KeyPress:
            return super().eventFilter(obj, event)
        if not event.matches(QKeySequence.StandardKey.Paste):
            return super().eventFilter(obj, event)
        clip = QApplication.clipboard()
        raw = clip.text() if clip is not None else ""
        num = _normalize_paste_number(raw)
        if num is None:
            return super().eventFilter(obj, event)
        # 选中全部再插入，避免叠在旧数字上
        obj.selectAll()
        obj.insert(num)
        event.accept()
        return True


_spin_paste_filter: Optional[_SpinPasteFilter] = None


def _patch_one_spinbox(spin: QAbstractSpinBox) -> None:
    global _spin_paste_filter
    if _spin_paste_filter is None:
        _spin_paste_filter = _SpinPasteFilter(spin)
    le = spin.lineEdit()
    if le is None:
        # 部分样式下 lineEdit 延迟创建
        from PySide6.QtCore import QTimer

        QTimer.singleShot(0, lambda s=spin: _patch_one_spinbox(s))
        return
    if getattr(le, "_obs_rtm_paste", False):
        return
    le._obs_rtm_paste = True  # type: ignore[attr-defined]
    le.installEventFilter(_spin_paste_filter)

    _orig_paste = le.paste

    def _paste_norm(_le=le, _orig=_orig_paste) -> None:
        clip = QApplication.clipboard()
        raw = clip.text() if clip is not None else ""
        num = _normalize_paste_number(raw)
        if num is None:
            _orig()
            return
        _le.selectAll()
        _le.insert(num)

    le.paste = _paste_norm  # type: ignore[method-assign]


def enable_spinbox_paste(root: QWidget) -> None:
    """
    为 ``root`` 下所有 SpinBox 启用稳健粘贴。

    Windows + 样式表 max-height / 带 suffix 时，原生粘贴常失败或只贴进一半。
    """
    spins = list(root.findChildren(QAbstractSpinBox))
    # findChildren 不含自身
    if isinstance(root, QAbstractSpinBox):
        spins.insert(0, root)
    for spin in spins:
        _patch_one_spinbox(spin)


def focus_is_text_input(widget: Optional[QWidget] = None) -> bool:
    """焦点在可编辑文本/数字框时，全局快捷键应让路。"""
    fw = widget if widget is not None else QApplication.focusWidget()
    if fw is None:
        return False
    if isinstance(fw, (QLineEdit, QAbstractSpinBox)):
        return True
    # SpinBox 内部 lineEdit
    p = fw.parent()
    while p is not None:
        if isinstance(p, QAbstractSpinBox):
            return True
        p = p.parent()
    from PySide6.QtWidgets import QPlainTextEdit, QTextEdit

    return isinstance(fw, (QTextEdit, QPlainTextEdit))


__all__ = [
    "configure_independent_dialog",
    "show_modeless_dialog",
    "connect_combo_deferred",
    "defer_after_combo_popup",
    "enable_spinbox_paste",
    "focus_is_text_input",
]
