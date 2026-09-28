"""阶段 3：输出目录浏览与快捷操作。"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class OutputPanel(QWidget):
    open_outputs_requested = Signal()
    open_runs_requested = Signal()
    open_inputs_requested = Signal()
    refresh_requested = Signal()
    plot_smesh_requested = Signal()
    inv_analysis_requested = Signal()
    model_picker_requested = Signal()
    model_compare_requested = Signal()
    file_activated = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        tip = QLabel(
            "浏览工区 outputs/ 与 runs/。双击列表项用系统方式打开；"
            "跑完任务后会自动刷新。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#475569;")
        lay.addWidget(tip)

        row = QHBoxLayout()
        for text, sig in (
            ("刷新", self.refresh_requested),
            ("inputs/", self.open_inputs_requested),
            ("outputs/", self.open_outputs_requested),
            ("runs/", self.open_runs_requested),
            ("绘制 smesh…", self.plot_smesh_requested),
            ("反演分析…", self.inv_analysis_requested),
            ("模型挑选…", self.model_picker_requested),
            ("模型对比…", self.model_compare_requested),
        ):
            btn = QPushButton(text)
            btn.clicked.connect(sig.emit)
            row.addWidget(btn)
        row.addStretch(1)
        lay.addLayout(row)

        box = QGroupBox("outputs/ · runs/ 文件")
        bl = QVBoxLayout(box)
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(self._on_dbl)
        bl.addWidget(self.list)
        lay.addWidget(box, stretch=1)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color:#64748b;")
        lay.addWidget(self.lbl_status)

    def _on_dbl(self, item) -> None:
        path = item.data(256) if item else None  # Qt.UserRole
        if path:
            self.file_activated.emit(str(path))

    def set_listing(self, work: Path, entries: list[tuple[str, Path]]) -> None:
        """entries: (display_label, abs_path)。"""
        self.list.clear()
        for label, path in entries:
            from PySide6.QtWidgets import QListWidgetItem
            from PySide6.QtCore import Qt

            it = QListWidgetItem(label)
            it.setData(Qt.ItemDataRole.UserRole, str(path))
            self.list.addItem(it)
        self.lbl_status.setText(
            f"工区：{work}  ·  共 {len(entries)} 项"
            if work
            else "未设置工区 / work_dir"
        )


def list_output_tree(work: Path, *, max_files: int = 400) -> list[tuple[str, Path]]:
    """列出 outputs/ 与 runs/ 下文件（相对标签）。"""
    out: list[tuple[str, Path]] = []
    if not work.is_dir():
        return out
    for sub in ("outputs", "runs"):
        root = work / sub
        if not root.is_dir():
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in sorted(filenames):
                p = Path(dirpath) / name
                try:
                    rel = p.relative_to(work).as_posix()
                except ValueError:
                    rel = str(p)
                out.append((rel, p))
                if len(out) >= max_files:
                    return out
    return out
