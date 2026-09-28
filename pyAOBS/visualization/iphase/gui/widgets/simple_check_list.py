# -*- coding: utf-8 -*-
"""通用可勾选 ID 列表（对齐 tomo2d SimpleCheckList）。"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class SimpleCheckList(QWidget):
    """勾选列表：每条记录含 ``id``/``key``/``label``；全选时 ``selected_ids()`` 为 ``None``。"""

    selection_changed = Signal()

    def __init__(self, heading: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._applying = False
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.lbl_head = QLabel(heading)
        self.lbl_head.setWordWrap(True)
        root.addWidget(self.lbl_head)
        bar = QHBoxLayout()
        btn_all = QPushButton("全选")
        btn_none = QPushButton("全不选")
        btn_all.clicked.connect(lambda: self.set_all_checked(True))
        btn_none.clicked.connect(lambda: self.set_all_checked(False))
        bar.addWidget(btn_all)
        bar.addWidget(btn_none)
        bar.addStretch(1)
        root.addLayout(bar)
        self.list_w = QListWidget()
        self.list_w.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.list_w.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.list_w.setMinimumHeight(0)
        self.list_w.itemChanged.connect(self._on_item_changed)
        root.addWidget(self.list_w, stretch=1)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

    def set_heading(self, text: str) -> None:
        self.lbl_head.setText(text)

    def set_records(self, records: list[dict[str, Any]]) -> None:
        self._applying = True
        self.list_w.blockSignals(True)
        self.list_w.clear()
        for rec in records:
            item = QListWidgetItem(str(rec.get("label") or rec.get("id", "")))
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, rec)
            self.list_w.addItem(item)
        self.list_w.blockSignals(False)
        self._applying = False

    def set_all_checked(self, checked: bool) -> None:
        if self.list_w.count() == 0:
            return
        self._applying = True
        self.list_w.blockSignals(True)
        st = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self.list_w.count()):
            item = self.list_w.item(i)
            if item is not None:
                item.setCheckState(st)
        self.list_w.blockSignals(False)
        self._applying = False
        self.selection_changed.emit()

    def checked_keys(self) -> list[str]:
        out: list[str] = []
        for i in range(self.list_w.count()):
            item = self.list_w.item(i)
            if item is None or item.checkState() != Qt.CheckState.Checked:
                continue
            rec = item.data(Qt.ItemDataRole.UserRole) or {}
            out.append(str(rec.get("key", rec.get("id", ""))))
        return out

    def checked_ids(self) -> list[int]:
        out: list[int] = []
        for i in range(self.list_w.count()):
            item = self.list_w.item(i)
            if item is None or item.checkState() != Qt.CheckState.Checked:
                continue
            rec = item.data(Qt.ItemDataRole.UserRole) or {}
            out.append(int(rec.get("id", rec.get("code", 0))))
        return out

    def selected_ids(self) -> set[int] | None:
        chosen = set(self.checked_ids())
        n = self.list_w.count()
        if not chosen:
            return set()
        if len(chosen) == n:
            return None
        return chosen

    def selected_keys(self) -> set[str] | None:
        chosen = set(self.checked_keys())
        n = self.list_w.count()
        if not chosen:
            return set()
        if len(chosen) == n:
            return None
        return chosen

    def _on_item_changed(self, _item: QListWidgetItem) -> None:
        if self._applying:
            return
        self.selection_changed.emit()
