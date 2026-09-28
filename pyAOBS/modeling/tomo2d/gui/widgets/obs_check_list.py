"""OBS 勾选列表：预览 tx.in 与转换页共用，勾选写入 ``tx.obs_ids``。"""

from __future__ import annotations

from typing import Any
from weakref import WeakSet

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

from ...tx2tomo2d import format_obs_id_spec, parse_obs_id_spec
from ..state.form_state import FormState


class ObsCheckList(QWidget):
    """可勾选 OBS 列表；可选持久化到 FormState，多实例之间同步勾选。"""

    selection_changed = Signal()

    _live: WeakSet[ObsCheckList] = WeakSet()

    def __init__(
        self,
        *,
        state: FormState | None = None,
        persist_key: str | None = None,
        heading: str = "OBS（可多选）",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._persist_key = persist_key
        self._records: list[dict[str, Any]] = []
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
        self.list_obs = QListWidget()
        self.list_obs.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.list_obs.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.list_obs.setMinimumHeight(0)
        self.list_obs.itemChanged.connect(self._on_item_changed)
        root.addWidget(self.list_obs, stretch=1)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color:#666; font-size:11px;")
        self.lbl_status.hide()
        root.addWidget(self.lbl_status)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        if state is not None and persist_key:
            ObsCheckList._live.add(self)

    def set_heading(self, text: str) -> None:
        self.lbl_head.setText(text)

    def set_status(self, text: str) -> None:
        self.lbl_status.setText(text)
        self.lbl_status.setVisible(bool((text or "").strip()))

    def set_records(self, records: list[dict[str, Any]], *, status: str | None = None) -> None:
        self._records = [dict(r) for r in records]
        self._applying = True
        self.list_obs.blockSignals(True)
        self.list_obs.clear()
        prev = self._persisted_ids()
        for rec in self._records:
            item = QListWidgetItem(str(rec.get("label") or ""))
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            oid = rec.get("obs_id")
            if prev is None or oid is None:
                checked = True
            else:
                checked = int(oid) in prev
            item.setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )
            item.setData(Qt.ItemDataRole.UserRole, rec)
            self.list_obs.addItem(item)
        self.list_obs.blockSignals(False)
        self._applying = False
        if status is not None:
            self.set_status(status)
        self._persist_and_broadcast(broadcast=True)

    def set_all_checked(self, checked: bool) -> None:
        if self.list_obs.count() == 0:
            return
        self._applying = True
        self.list_obs.blockSignals(True)
        st = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self.list_obs.count()):
            item = self.list_obs.item(i)
            if item is not None:
                item.setCheckState(st)
        self.list_obs.blockSignals(False)
        self._applying = False
        self._persist_and_broadcast()
        self.selection_changed.emit()

    def selected_xobs(self) -> set[float] | None:
        """勾选台站的模型距离（0.001 km）；全选时 ``None``；全不选时空集。"""
        chosen: set[float] = set()
        n = self.list_obs.count()
        for i in range(n):
            item = self.list_obs.item(i)
            if item is None or item.checkState() != Qt.CheckState.Checked:
                continue
            rec = item.data(Qt.ItemDataRole.UserRole) or {}
            chosen.add(round(float(rec.get("xobs", 0.0)), 3))
        if not chosen:
            return set()
        if len(chosen) == n:
            return None
        return chosen

    def apply_persisted_selection(self) -> None:
        """按 FormState 更新有 OBS 号的项；未匹配行保持现状。"""
        prev = self._persisted_ids()
        self._applying = True
        self.list_obs.blockSignals(True)
        for i in range(self.list_obs.count()):
            item = self.list_obs.item(i)
            if item is None:
                continue
            rec = item.data(Qt.ItemDataRole.UserRole) or {}
            oid = rec.get("obs_id")
            if oid is None:
                continue
            checked = True if prev is None else int(oid) in prev
            item.setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )
        self.list_obs.blockSignals(False)
        self._applying = False
        self.selection_changed.emit()

    def _persisted_ids(self) -> set[int] | None:
        if self._state is None or not self._persist_key:
            return None
        return parse_obs_id_spec(self._state.get_str(self._persist_key))

    def _on_item_changed(self, _item: QListWidgetItem) -> None:
        if self._applying:
            return
        self._persist_and_broadcast()
        self.selection_changed.emit()

    def _persist_and_broadcast(self, *, broadcast: bool = True) -> None:
        if self._state is None or not self._persist_key:
            return
        catalog_ids: list[int] = []
        chosen: list[int] = []
        for i in range(self.list_obs.count()):
            item = self.list_obs.item(i)
            if item is None:
                continue
            rec = item.data(Qt.ItemDataRole.UserRole) or {}
            oid = rec.get("obs_id")
            if oid is None:
                continue
            catalog_ids.append(int(oid))
            if item.checkState() == Qt.CheckState.Checked:
                chosen.append(int(oid))
        if not catalog_ids:
            return
        if len(chosen) == len(catalog_ids):
            spec = "all"
        else:
            spec = format_obs_id_spec(chosen, catalog_ids=catalog_ids)
        self._state.set(self._persist_key, spec)
        if not broadcast:
            return
        for other in list(ObsCheckList._live):
            if other is self:
                continue
            if other._state is self._state and other._persist_key == self._persist_key:
                other.apply_persisted_selection()
