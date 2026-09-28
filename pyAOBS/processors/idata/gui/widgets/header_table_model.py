# -*- coding: utf-8 -*-
"""全字段道头 QAbstractTableModel。"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QColor, QBrush

from ..services.segy_dataset import COLUMN_NAMES, GEOM_PRIORITY, SegyDataset

_GEOM_SET = set(GEOM_PRIORITY)
_GEOM_BG = QBrush(QColor(232, 244, 248))
_DIRTY_FG = QBrush(QColor(180, 80, 0))


class HeaderTableModel(QAbstractTableModel):
    """cell_edited(row, field, new_value) — 单元格手工编辑成功时发出。"""

    cell_edited = Signal(int, str, int)

    def __init__(self, dataset: Optional[SegyDataset] = None, parent=None) -> None:
        super().__init__(parent)
        self._ds = dataset or SegyDataset()
        self._columns = list(COLUMN_NAMES)

    @property
    def dataset(self) -> SegyDataset:
        return self._ds

    def set_dataset(self, ds: SegyDataset) -> None:
        self.beginResetModel()
        self._ds = ds
        self.endResetModel()

    def refresh(self) -> None:
        self.beginResetModel()
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return self._ds.ntraces

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self._columns)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self._columns):
                return self._columns[section]
            return None
        return str(section)

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:  # noqa: N802
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return (
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsEditable
        )

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if not index.isValid() or self._ds.ntraces == 0:
            return None
        row, col = index.row(), index.column()
        if row < 0 or row >= self._ds.ntraces or col < 0 or col >= len(self._columns):
            return None
        name = self._columns[col]
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return int(self._ds.headers[row].get(name, 0) or 0)
        if role == Qt.ItemDataRole.BackgroundRole and name in _GEOM_SET:
            return _GEOM_BG
        if role == Qt.ItemDataRole.ForegroundRole and row in self._ds.dirty:
            return _DIRTY_FG
        if role == Qt.ItemDataRole.ToolTipRole:
            return f"trace {row} / {name}"
        return None

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if role != Qt.ItemDataRole.EditRole or not index.isValid():
            return False
        name = self._columns[index.column()]
        try:
            ival = int(float(value))
        except (TypeError, ValueError):
            return False
        self._ds.set_header_value(index.row(), name, ival)
        self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ForegroundRole])
        self.cell_edited.emit(index.row(), name, ival)
        return True

    def field_name(self, column: int) -> str:
        return self._columns[column]

    def selected_field_names(self, columns: Sequence[int]) -> list[str]:
        return [self._columns[c] for c in columns if 0 <= c < len(self._columns)]
