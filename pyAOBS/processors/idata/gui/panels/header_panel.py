# -*- coding: utf-8 -*-
"""全字段道头编辑面板。"""

from __future__ import annotations

from typing import List

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

try:
    from pyAOBS.utils.qt_combo import connect_combo_deferred
except ImportError:
    def connect_combo_deferred(combo, slot):  # type: ignore
        combo.currentIndexChanged.connect(slot)

from ...project import DEFAULT_GEOM
from ..dialog_utils import show_modeless_dialog, show_modeless_message, show_modeless_text
from ..services import header_edit
from ..services.segy_dataset import COLUMN_NAMES, SegyDataset
from ..widgets.gather_preview import GatherPreviewWidget
from ..widgets.header_table_model import HeaderTableModel
from ..widgets.trace_preview import TracePreviewWidget

try:
    from pyAOBS.geometry_roles import physical_shot_obs_xyz, resolve_geom
except ImportError:
    physical_shot_obs_xyz = None  # type: ignore
    resolve_geom = None  # type: ignore


class HeaderPanel(QWidget):
    selection_changed = Signal(int)  # trace row
    dataset_modified = Signal()
    geom_mode_changed = Signal(str)
    log_message = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._ds = SegyDataset()
        self._model = HeaderTableModel(self._ds)
        self._clip_block: List[dict] = []
        self._clip_fields: List[str] = []
        self._build_ui()
        self._model.cell_edited.connect(self._on_cell_edited)
        self.gather.log_message.connect(self.log_message.emit)

    def _log(self, text: str) -> None:
        self.log_message.emit(text)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        tools = QHBoxLayout()
        self.geom_combo = QComboBox()
        self.geom_combo.addItem("炮=sx/sy，OBS=gx/gy（约定）", DEFAULT_GEOM)
        self.geom_combo.addItem("旧对调：炮=gx/gy，OBS=sx/sy", "obs")
        self.geom_combo.setCurrentIndex(0)  # 默认约定
        connect_combo_deferred(self.geom_combo, self._on_geom_changed)
        tools.addWidget(QLabel("解释模式"))
        tools.addWidget(self.geom_combo)

        for text, slot in (
            ("复制", self.copy_selection),
            ("粘贴", self.paste_selection),
            ("列填充…", self.fill_column_dialog),
            ("条件批量…", self.batch_where_dialog),
            ("重算 offset", self.recompute_offset_selected),
            ("offset 检查", self.offset_check_dialog),
            ("交换字段…", self.swap_fields_dialog),
            ("交换 s*/g*", self.swap_slots),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            tools.addWidget(b)
        tools.addStretch(1)
        root.addLayout(tools)

        self.phys_label = QLabel("物理坐标：—")
        self.phys_label.setStyleSheet("color:#334155;")
        root.addWidget(self.phys_label)

        split = QSplitter(Qt.Orientation.Horizontal)

        # 左侧：道头表
        self.table = QTableView()
        self.table.setModel(self._model)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectItems)
        self.table.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.selectionModel().selectionChanged.connect(self._on_sel)
        split.addWidget(self.table)

        # 右侧：道集预览 + 单道波形
        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)
        right_split = QSplitter(Qt.Orientation.Vertical)
        self.gather = GatherPreviewWidget()
        self.gather.trace_clicked.connect(self.select_row)
        right_split.addWidget(self.gather)
        self.preview = TracePreviewWidget()
        right_split.addWidget(self.preview)
        right_split.setStretchFactor(0, 3)
        right_split.setStretchFactor(1, 1)
        right_split.setSizes([420, 160])
        rlay.addWidget(right_split)
        split.addWidget(right)

        # 道头表默认约占 1/3，预览约 2/3
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 2)
        split.setSizes([360, 720])
        root.addWidget(split, stretch=1)

        QShortcut(QKeySequence.StandardKey.Copy, self, self.copy_selection)
        QShortcut(QKeySequence.StandardKey.Paste, self, self.paste_selection)

    def set_dataset(self, ds: SegyDataset) -> None:
        self._ds = ds
        self._model.set_dataset(ds)
        self.preview.clear()
        self.phys_label.setText("物理坐标：—")
        self.gather.set_dataset(ds if ds.is_open else None)
        if ds.is_open:
            self.select_row(0)

    def notify_saved(self) -> None:
        """保存后轻量刷新：表格重绘；道集仅在横轴依赖道头时重排，不重读样本。"""
        self._model.refresh()
        # 样本未变：若已有道集缓存则只更新高亮；否则保持
        rows = self._selected_rows()
        if rows:
            self.gather.set_highlight(rows[0])
            self._update_phys(rows[0])

    def dataset(self) -> SegyDataset:
        return self._ds

    def geom_mode(self) -> str:
        return str(self.geom_combo.currentData() or DEFAULT_GEOM)

    def set_geom_mode(self, mode: str) -> None:
        mode = (mode or DEFAULT_GEOM).lower()
        if mode in ("literal_segy", "约定"):
            mode = DEFAULT_GEOM
        idx = self.geom_combo.findData(mode)
        if idx < 0:
            idx = 0  # 回退约定
        if self.geom_combo.currentIndex() != idx:
            self.geom_combo.blockSignals(True)
            self.geom_combo.setCurrentIndex(idx)
            self.geom_combo.blockSignals(False)
            self._update_phys_for_current()

    def _on_geom_changed(self, _idx: int = 0) -> None:
        mode = self.geom_mode()
        self.geom_mode_changed.emit(mode)
        self._update_phys_for_current()
        label = (
            "炮=sx/sy，OBS=gx/gy"
            if mode == DEFAULT_GEOM
            else "旧对调：炮=gx/gy，OBS=sx/sy"
        )
        self._log(f"[道头] 解释模式 → {mode}（{label}）")

    def _on_cell_edited(self, row: int, field: str, value: int) -> None:
        self._log(
            f"[道头] 编辑 trace={row}  {field}={value}  "
            f"dirty={self._ds.dirty_count}"
        )
        self.dataset_modified.emit()

    def _selected_rows(self) -> List[int]:
        rows = sorted({i.row() for i in self.table.selectionModel().selectedIndexes()})
        return rows

    def current_row(self) -> int:
        rows = self._selected_rows()
        return rows[0] if rows else -1

    def _selected_columns(self) -> List[int]:
        return sorted({i.column() for i in self.table.selectionModel().selectedIndexes()})

    def _on_sel(self, *_args) -> None:
        rows = self._selected_rows()
        if not rows:
            return
        row = rows[0]
        self.selection_changed.emit(row)
        self._show_preview(row)
        self._update_phys(row)

    def _show_preview(self, row: int) -> None:
        if not self._ds.is_open:
            return
        try:
            self.gather.ensure_row_visible(row)
            self.gather.set_highlight(row)
            y = self._ds.read_samples(row)
            th = self._ds.get_header(row)
            self.preview.show_trace(
                y,
                dt_us=int(th.get("dt", 0) or 0),
                title=(
                    f"trace {row}  trid={th.get('trid')}  fldr={th.get('fldr')}  "
                    f"cdp={th.get('cdp')}  ep={th.get('ep')}  npts={y.size}"
                ),
            )
        except Exception as exc:
            self.preview.clear()
            self.phys_label.setText(f"预览失败: {exc}")

    def _update_phys_for_current(self) -> None:
        rows = self._selected_rows()
        if rows:
            self._update_phys(rows[0])

    def _update_phys(self, row: int) -> None:
        if not self._ds.is_open or physical_shot_obs_xyz is None:
            xy = self._ds.physical_xy(row) if self._ds.is_open else {}
            self.phys_label.setText(
                f"槽位(scalco后): sx=({xy.get('sx', 0):.2f},{xy.get('sy', 0):.2f}) "
                f"gx=({xy.get('gx', 0):.2f},{xy.get('gy', 0):.2f})"
            )
            return
        th = dict(self._ds.get_header(row))
        # 物理显示用已缩放坐标覆盖 sx..gy
        phy = self._ds.physical_xy(row)
        th.update(phy)
        mode = resolve_geom(self.geom_mode(), [th]) if resolve_geom else self.geom_mode()
        shot, obs = physical_shot_obs_xyz(th, geom=mode, use_utm=True)
        self.phys_label.setText(
            f"geom={mode} | 物理炮=({shot[0]:.2f},{shot[1]:.2f}) "
            f"OBS=({obs[0]:.2f},{obs[1]:.2f}) | "
            f"槽位 sx=({phy['sx']:.2f},{phy['sy']:.2f}) gx=({phy['gx']:.2f},{phy['gy']:.2f})"
        )

    def copy_selection(self) -> None:
        rows = self._selected_rows()
        cols = self._selected_columns()
        if not rows:
            return
        fields = self._model.selected_field_names(cols) if cols else None
        self._clip_block = header_edit.copy_headers(self._ds, rows, fields)
        self._clip_fields = list(fields) if fields else list(self._clip_block[0].keys())
        tsv = header_edit.clipboard_to_tsv(self._clip_block, self._clip_fields)
        QApplication.clipboard().setText(tsv)

    def paste_selection(self) -> None:
        rows = self._selected_rows()
        start = rows[0] if rows else 0
        text = QApplication.clipboard().text()
        fields, block = header_edit.tsv_to_block(text)
        if not block and self._clip_block:
            block = self._clip_block
            fields = self._clip_fields
        if not block:
            show_modeless_message("粘贴", "剪贴板无道头数据。")
            return
        n = header_edit.paste_headers(self._ds, start, block, fields or None)
        self._model.refresh()
        self.dataset_modified.emit()
        self._log(
            f"[道头] 粘贴：自 trace={start} 写入 {n} 道  "
            f"字段数={len(fields or self._clip_fields)}  dirty={self._ds.dirty_count}"
        )
        show_modeless_message("粘贴", f"已写入 {n} 道。")

    def fill_column_dialog(self) -> None:
        cols = self._selected_columns()
        if not cols:
            show_modeless_message("列填充", "请先选中目标列。")
            return
        field = self._model.field_name(cols[0])
        dlg = QDialog()
        dlg.setWindowTitle("列填充")
        form = QFormLayout(dlg)
        spin = QSpinBox()
        spin.setRange(-2_000_000_000, 2_000_000_000)
        spin.setValue(0)
        form.addRow(f"将选中行的 {field} 设为：", spin)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        form.addRow(buttons)

        def _apply() -> None:
            rows = self._selected_rows() or None
            n = header_edit.fill_column(self._ds, field, int(spin.value()), rows)
            self._model.refresh()
            self.dataset_modified.emit()
            dlg.close()
            scope = f"选中 {len(rows)} 道" if rows is not None else "全部道"
            self._log(
                f"[道头] 列填充 {field}={int(spin.value())}  "
                f"更新 {n} 道（{scope}）  dirty={self._ds.dirty_count}"
            )
            show_modeless_message("列填充", f"已更新 {n} 道。")

        buttons.accepted.connect(_apply)
        buttons.rejected.connect(dlg.close)
        show_modeless_dialog(dlg)

    def batch_where_dialog(self) -> None:
        dlg = QDialog()
        dlg.setWindowTitle("条件批量")
        form = QFormLayout(dlg)
        e_mf = QLineEdit("fldr")
        e_mv = QLineEdit("1")
        e_sf = QLineEdit("ep")
        e_sv = QLineEdit("1")
        form.addRow("match_field", e_mf)
        form.addRow("match_value", e_mv)
        form.addRow("set_field", e_sf)
        form.addRow("set_value", e_sv)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        form.addRow(buttons)

        def _apply() -> None:
            try:
                n = header_edit.batch_set_where(
                    self._ds,
                    match_field=e_mf.text().strip(),
                    match_value=int(float(e_mv.text().strip())),
                    set_field=e_sf.text().strip(),
                    set_value=int(float(e_sv.text().strip())),
                )
            except Exception as exc:
                show_modeless_message(
                    "条件批量", str(exc), icon=QMessageBox.Icon.Warning
                )
                return
            self._model.refresh()
            self.dataset_modified.emit()
            dlg.close()
            self._log(
                f"[道头] 条件批量  {e_mf.text().strip()}={e_mv.text().strip()} "
                f"→ {e_sf.text().strip()}={e_sv.text().strip()}  "
                f"更新 {n} 道  dirty={self._ds.dirty_count}"
            )
            show_modeless_message("条件批量", f"已更新 {n} 道。")

        buttons.accepted.connect(_apply)
        buttons.rejected.connect(dlg.close)
        show_modeless_dialog(dlg)

    def recompute_offset_selected(self) -> None:
        rows = self._selected_rows() or None
        n = self._ds.recompute_offset(rows)
        self._model.refresh()
        self.dataset_modified.emit()
        scope = f"选中 {len(rows)} 道" if rows is not None else "全部道"
        self._log(
            f"[道头] 重算 offset：修改 {n} 道（{scope}）  "
            f"dirty={self._ds.dirty_count}"
        )
        show_modeless_message("offset", f"重算并修改 {n} 道。")

    def offset_check_dialog(self) -> None:
        if not self._ds.is_open:
            show_modeless_message("offset 检查", "请先打开数据。")
            return
        dlg = QDialog()
        dlg.setWindowTitle("offset 检查")
        form = QFormLayout(dlg)
        tol = QDoubleSpinBox()
        tol.setRange(0.0, 1.0e9)
        tol.setDecimals(1)
        tol.setValue(5.0)
        form.addRow("容差 (m)", tol)
        form.addRow(QLabel("有表格选中行则只检查选中道，否则全部道。"))
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        form.addRow(buttons)

        def _run() -> None:
            rows = self._selected_rows() or None
            result = header_edit.check_offset_vs_xy(
                self._ds, tol_m=float(tol.value()), rows=rows
            )
            text = header_edit.format_offset_check_report(result)
            if rows is not None:
                text = f"（仅选中 {len(rows)} 道）\n" + text
            dlg.close()
            scope = f"选中 {len(rows)} 道" if rows is not None else "全部道"
            self._log(
                f"[道头] offset 检查（{scope}）tol={tol.value():g} m → "
                f"一致 {result.get('ok', 0)} / 不一致 {result.get('bad', 0)} / "
                f"共 {result.get('n', 0)}  "
                f"Δ均={float(result.get('mean_diff', 0)):.3f}m  "
                f"Δ最大={float(result.get('max_diff', 0)):.3f}m"
            )
            show_modeless_text(
                "offset 检查",
                text,
                summary=(
                    f"一致 {result.get('ok', 0)} / 不一致 {result.get('bad', 0)} / "
                    f"共 {result.get('n', 0)} 道（tol={tol.value():g} m）"
                ),
            )

        buttons.accepted.connect(_run)
        buttons.rejected.connect(dlg.close)
        show_modeless_dialog(dlg)

    def swap_fields_dialog(self) -> None:
        """任选两个道头字段互换（非模态）。"""
        if not self._ds.is_open:
            show_modeless_message("交换字段", "请先打开数据。")
            return
        names = list(COLUMN_NAMES)
        dlg = QDialog()
        dlg.setWindowTitle("交换道头字段")
        form = QFormLayout(dlg)
        combo_a = QComboBox()
        combo_b = QComboBox()
        combo_a.addItems(names)
        combo_b.addItems(names)
        cols = self._selected_columns()
        if len(cols) >= 2:
            combo_a.setCurrentText(self._model.field_name(cols[0]))
            combo_b.setCurrentText(self._model.field_name(cols[1]))
        elif len(cols) == 1:
            combo_a.setCurrentText(self._model.field_name(cols[0]))
            fa = self._model.field_name(cols[0])
            pair = {"sx": "gx", "gx": "sx", "sy": "gy", "gy": "sy"}.get(fa)
            if pair and pair in names:
                combo_b.setCurrentText(pair)
        else:
            if "sx" in names:
                combo_a.setCurrentText("sx")
            if "gx" in names:
                combo_b.setCurrentText("gx")
        form.addRow("字段 A", combo_a)
        form.addRow("字段 B", combo_b)
        form.addRow(QLabel("范围：有表格选中行则只交换选中道，否则全部道。"))
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        form.addRow(buttons)

        def _apply() -> None:
            fa = combo_a.currentText().strip()
            fb = combo_b.currentText().strip()
            if fa == fb:
                show_modeless_message("交换字段", "请选择两个不同字段。")
                return
            rows = self._selected_rows() or None
            try:
                n = header_edit.swap_fields(self._ds, fa, fb, rows)
            except Exception as exc:
                show_modeless_message(
                    "交换字段", str(exc), icon=QMessageBox.Icon.Warning
                )
                return
            self._model.refresh()
            self.dataset_modified.emit()
            dlg.close()
            scope = f"选中 {n} 道" if rows is not None else f"全部 {n} 道"
            self._log(
                f"[道头] 交换字段 {fa} ↔ {fb}  （{scope}）  "
                f"dirty={self._ds.dirty_count}"
            )
            show_modeless_message("交换字段", f"已在{scope}交换 {fa} ↔ {fb}。")

        buttons.accepted.connect(_apply)
        buttons.rejected.connect(dlg.close)
        show_modeless_dialog(dlg)

    def swap_slots(self) -> None:
        rows = self._selected_rows() or None
        n = self._ds.swap_source_group_slots(rows)
        self._model.refresh()
        self.dataset_modified.emit()
        scope = f"选中 {len(rows)} 道" if rows is not None else "全部道"
        self._log(
            f"[道头] 交换 s*/g* 槽  修改 {n} 道（{scope}）  "
            f"dirty={self._ds.dirty_count}"
        )
        show_modeless_message("交换", f"已交换 {n} 道的 s*/g* 槽。")

    def select_row(self, row: int) -> None:
        """选中行并纵向滚入视野；保留用户当前的横向滚动位置。"""
        if row < 0 or row >= self._ds.ntraces:
            return
        hbar = self.table.horizontalScrollBar()
        hpos = hbar.value()
        self.table.selectRow(row)
        # 用当前可见列附近的 index 做纵向定位，避免强制滚到第 0 列
        left_col = self.table.columnAt(1)
        if left_col < 0:
            left_col = 0
        self.table.scrollTo(
            self._model.index(row, left_col),
            QAbstractItemView.ScrollHint.PositionAtCenter,
        )
        hbar.setValue(hpos)
