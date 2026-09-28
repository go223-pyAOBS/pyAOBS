# -*- coding: utf-8 -*-
"""阶段 3：输出路径与导出。"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...project import ZplotProject


class OutputPanel(QWidget):
    project_changed = Signal()
    request_export_all = Signal()
    request_open_outputs = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        tip = QLabel(
            "输出默认写在工区 outputs/ 下（相对路径）。保存工程时会同步写出这些产物。"
            "姿态联合反演请另启：python -m pyAOBS.processors.relocation.gui。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#475569;")
        lay.addWidget(tip)

        box = QGroupBox("输出相对路径", self)
        form = QFormLayout(box)
        self.ed_waveop = QLineEdit(box)
        self.ed_picks = QLineEdit(box)
        self.ed_params = QLineEdit(box)
        form.addRow("V段 waveop", self.ed_waveop)
        form.addRow("拾取 picks", self.ed_picks)
        form.addRow("查看器参数", self.ed_params)
        lay.addWidget(box)

        row = QHBoxLayout()
        self.btn_export = QPushButton("立即导出全部产物", self)
        self.btn_export.clicked.connect(self.request_export_all.emit)
        self.btn_open = QPushButton("打开 outputs 文件夹", self)
        self.btn_open.clicked.connect(self.request_open_outputs.emit)
        row.addWidget(self.btn_export)
        row.addWidget(self.btn_open)
        row.addStretch(1)
        lay.addLayout(row)

        self.lbl_status = QLabel("", self)
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color:#0f766e;")
        lay.addWidget(self.lbl_status)
        lay.addStretch(1)

        for w in (self.ed_waveop, self.ed_picks, self.ed_params):
            w.textChanged.connect(lambda *_: self.project_changed.emit())

    def apply_to_project(self, project: ZplotProject) -> None:
        project.workflow.waveop_path = self.ed_waveop.text().strip() or project.workflow.waveop_path
        project.workflow.picks_path = self.ed_picks.text().strip() or project.workflow.picks_path
        project.workflow.viewer_params_path = (
            self.ed_params.text().strip() or project.workflow.viewer_params_path
        )

    def load_from_project(self, project: ZplotProject) -> None:
        self.ed_waveop.setText(project.workflow.waveop_path or "")
        self.ed_picks.setText(project.workflow.picks_path or "")
        self.ed_params.setText(project.workflow.viewer_params_path or "")
        if project.workdir:
            self.lbl_status.setText(f"工区：{project.workdir}")
        else:
            self.lbl_status.setText("尚未设置工区目录（请先新建/打开工程）")

    def set_status(self, text: str) -> None:
        self.lbl_status.setText(text)
