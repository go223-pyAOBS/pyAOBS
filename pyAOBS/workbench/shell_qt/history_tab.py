"""Run history tab (Qt)."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..shell.logic.helpers import with_text_padding
from .styles import (
    hint_label,
    run_summary_label,
    status_panel,
    style_table_item_for_run_status,
    style_table_row_for_run_status,
)


class RunHistoryTab(QWidget):
    """Workspace run history table + detail panel."""

    open_run_folder = Signal(str)
    prefill_runner = Signal(dict)
    rerun_failed = Signal()
    batch_rerun_failed = Signal()
    delete_run = Signal(str)
    bulk_delete_non_success = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._records: list[dict] = []
        self._run_id_to_dir: dict[str, Path] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(8)

        filt_grp = QGroupBox("筛选")
        filt = QHBoxLayout(filt_grp)
        filt.addWidget(QLabel("Status"))
        self._filter_status = QComboBox()
        self._filter_status.addItem("全部")
        self._filter_status.currentIndexChanged.connect(self._apply_filter)
        filt.addWidget(self._filter_status)

        filt.addWidget(QLabel("Node"))
        self._filter_node = QComboBox()
        self._filter_node.addItem("全部")
        self._filter_node.currentIndexChanged.connect(self._apply_filter)
        filt.addWidget(self._filter_node)

        filt.addWidget(QLabel("关键词"))
        self._filter_kw = QLineEdit()
        self._filter_kw.setPlaceholderText("run_id / command / error …")
        self._filter_kw.textChanged.connect(self._apply_filter)
        filt.addWidget(self._filter_kw, stretch=1)

        self._failed_only = QCheckBox("仅失败")
        self._failed_only.toggled.connect(self._apply_filter)
        filt.addWidget(self._failed_only)

        btn_clear = QPushButton("清空筛选")
        btn_clear.clicked.connect(self._clear_filters)
        filt.addWidget(btn_clear)
        filt.addWidget(
            hint_label("success=绿  failed=红  cancelled=黄")
        )
        root.addWidget(filt_grp)

        self._summary_row = QHBoxLayout()
        self._summary_host = QWidget()
        self._summary_lay = QHBoxLayout(self._summary_host)
        self._summary_lay.setContentsMargins(0, 0, 0, 0)
        self._summary_row.addWidget(self._summary_host, stretch=1)
        root.addLayout(self._summary_row)

        root.addWidget(hint_label("节点工作区历史（runs/<run_id>/manifest.json）"))

        split = QSplitter(Qt.Orientation.Vertical)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Workspace ID", "Node", "Status", "Finished At", "Elapsed(s)"]
        )
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        split.addWidget(self._table)

        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        actions = QHBoxLayout()
        for txt, slot, obj in (
            ("刷新详情", self._refresh_detail, None),
            ("打开工作区目录", self._emit_open_folder, None),
            ("回填到运行节点", self._emit_prefill, None),
            ("失败一键复跑", lambda: self.rerun_failed.emit(), "WorkbenchWarnButton"),
            ("批量复跑失败(筛选)", lambda: self.batch_rerun_failed.emit(), "WorkbenchWarnButton"),
            ("删除选中工作区", self._emit_delete, "WorkbenchDangerButton"),
            ("一键清理非成功", lambda: self.bulk_delete_non_success.emit(), "WorkbenchDangerButton"),
        ):
            b = QPushButton(txt)
            if obj:
                b.setObjectName(obj)
            b.clicked.connect(slot)
            actions.addWidget(b)
        actions.addStretch()
        bl.addLayout(actions)

        detail_grp = QGroupBox("Manifest 详情")
        dg = QVBoxLayout(detail_grp)
        self._detail = QPlainTextEdit()
        self._detail.setReadOnly(True)
        self._detail.setPlaceholderText("选择一条运行记录查看 manifest 详情…")
        dg.addWidget(self._detail)
        bl.addWidget(detail_grp, stretch=1)
        split.addWidget(bottom)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        root.addWidget(split, stretch=1)

    def _refresh_summary(self, records: list[dict] | None = None) -> None:
        recs = records if records is not None else self._records
        while self._summary_lay.count():
            item = self._summary_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._summary_lay.addWidget(run_summary_label(recs))
        self._summary_lay.addStretch()

    def set_records(
        self,
        records: list[dict],
        *,
        statuses: set[str],
        nodes: set[str],
    ) -> None:
        self._records = list(records)
        self._run_id_to_dir = {
            str(r["run_id"]): Path(r["run_dir"]) for r in records if r.get("run_dir")
        }

        cur_st = self._filter_status.currentText()
        self._filter_status.blockSignals(True)
        self._filter_status.clear()
        self._filter_status.addItem("全部")
        for s in sorted(statuses):
            self._filter_status.addItem(s)
        idx = self._filter_status.findText(cur_st)
        self._filter_status.setCurrentIndex(idx if idx >= 0 else 0)
        self._filter_status.blockSignals(False)

        cur_nd = self._filter_node.currentText()
        self._filter_node.blockSignals(True)
        self._filter_node.clear()
        self._filter_node.addItem("全部")
        for n in sorted(nodes):
            self._filter_node.addItem(n)
        idx = self._filter_node.findText(cur_nd)
        self._filter_node.setCurrentIndex(idx if idx >= 0 else 0)
        self._filter_node.blockSignals(False)

        self._refresh_summary(records)
        self._apply_filter()

    def selected_run_id(self) -> str:
        row = self._table.currentRow()
        if row < 0:
            return ""
        item = self._table.item(row, 0)
        return item.text() if item else ""

    def select_run_id(self, run_id: str) -> None:
        if not run_id:
            return
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item and item.text() == run_id:
                self._table.selectRow(row)
                return

    def _clear_filters(self) -> None:
        self._filter_status.setCurrentIndex(0)
        self._filter_node.setCurrentIndex(0)
        self._filter_kw.clear()
        self._failed_only.setChecked(False)

    def _apply_filter(self) -> None:
        from ..shell.logic.run_history import filter_run_records

        filtered = filter_run_records(
            self._records,
            status=self._filter_status.currentText(),
            node=self._filter_node.currentText(),
            keyword=self._filter_kw.text(),
            failed_only=self._failed_only.isChecked(),
        )
        self._table.setRowCount(0)
        for rec in filtered:
            row = self._table.rowCount()
            self._table.insertRow(row)
            status = str(rec.get("status", ""))
            for col, key in enumerate(
                ("run_id", "node_id", "status", "finished_at", "elapsed_s")
            ):
                item = QTableWidgetItem(str(rec.get(key, "")))
                if col == 2:
                    style_table_item_for_run_status(item, status, status_column=True)
                self._table.setItem(row, col, item)
            style_table_row_for_run_status(self._table, row, status, status_col=2)
        self._refresh_summary(filtered)

    def _on_selection_changed(self) -> None:
        self._refresh_detail()

    def _refresh_detail(self) -> None:
        run_id = self.selected_run_id()
        if not run_id:
            self._detail.setPlainText("")
            return
        run_dir = self._run_id_to_dir.get(run_id)
        if run_dir is None:
            self._detail.setPlainText(f"(未找到 run 目录: {run_id})")
            return
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            self._detail.setPlainText("(manifest.json 不存在)")
            return
        try:
            text = json.dumps(
                json.loads(manifest_path.read_text(encoding="utf-8")),
                ensure_ascii=False,
                indent=2,
            )
        except Exception as exc:
            text = f"(读取失败: {exc})"
        self._detail.setPlainText(with_text_padding(text))

    def _emit_open_folder(self) -> None:
        rid = self.selected_run_id()
        if rid:
            self.open_run_folder.emit(rid)

    def _emit_prefill(self) -> None:
        run_id = self.selected_run_id()
        if not run_id:
            return
        run_dir = self._run_id_to_dir.get(run_id)
        if run_dir is None:
            return
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return
        cmd = manifest.get("command") or []
        payload = {
            "node_id": str(manifest.get("node_id", "")),
            "executable": str(cmd[0]) if cmd else "",
            "args": " ".join(str(x) for x in cmd[1:]) if len(cmd) > 1 else "",
            "cwd": str(manifest.get("cwd", "")),
            "env_text": "",
            "inputs_text": "",
        }
        self.prefill_runner.emit(payload)

    def _emit_delete(self) -> None:
        rid = self.selected_run_id()
        if rid:
            self.delete_run.emit(rid)

    def filter_state(self) -> dict:
        return {
            "run_filter_status": self._filter_status.currentText(),
            "run_filter_node": self._filter_node.currentText(),
            "run_filter_keyword": self._filter_kw.text(),
            "run_filter_failed_only": self._failed_only.isChecked(),
        }

    def apply_filter_state(self, state: dict) -> None:
        st = str(state.get("run_filter_status", "全部") or "全部")
        nd = str(state.get("run_filter_node", "全部") or "全部")
        self._filter_status.setCurrentText(st)
        self._filter_node.setCurrentText(nd)
        self._filter_kw.setText(str(state.get("run_filter_keyword", "") or ""))
        self._failed_only.setChecked(bool(state.get("run_filter_failed_only", False)))
