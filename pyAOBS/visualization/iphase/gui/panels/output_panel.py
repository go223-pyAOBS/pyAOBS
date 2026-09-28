# -*- coding: utf-8 -*-
"""阶段 3：输出（保存图像 / 导出 PSP / 打开 outputs）。"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pyAOBS.utils.qt_combo import connect_combo_deferred, configure_combo_list_view


class OutputPanel(QWidget):
    request_save_figure = Signal()
    request_export_psp = Signal()
    request_open_outputs = Signal()
    psp_mode_changed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        tip = QLabel(
            "输出建议写到工区 outputs/ 下。保存图像导出当前走时图；"
            "导出 PSP 按下方模式写出派生 tx 文件。"
        )
        tip.setWordWrap(True)
        tip.setObjectName("IphaseCaption")
        lay.addWidget(tip)

        box = QGroupBox("导出选项", self)
        form = QFormLayout(box)
        self.cmb_psp_mode = QComboBox(box)
        configure_combo_list_view(self.cmb_psp_mode)
        self.cmb_psp_mode.addItems(["picked", "theory2d", "theory_pss", "theory2Dequi"])
        connect_combo_deferred(
            self.cmb_psp_mode,
            lambda *_: self.psp_mode_changed.emit(self.cmb_psp_mode.currentText()),
        )
        form.addRow("PSP导出模式", self.cmb_psp_mode)
        lay.addWidget(box)

        row = QHBoxLayout()
        self.btn_save_fig = QPushButton("保存走时图…", self)
        self.btn_export_psp = QPushButton("导出PSP文件…", self)
        self.btn_open_out = QPushButton("打开 outputs 文件夹", self)
        self.btn_save_fig.clicked.connect(self.request_save_figure.emit)
        self.btn_export_psp.clicked.connect(self.request_export_psp.emit)
        self.btn_open_out.clicked.connect(self.request_open_outputs.emit)
        row.addWidget(self.btn_save_fig)
        row.addWidget(self.btn_export_psp)
        row.addWidget(self.btn_open_out)
        row.addStretch(1)
        lay.addLayout(row)

        self.lbl_status = QLabel("", self)
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setObjectName("IphaseCaption")
        lay.addWidget(self.lbl_status)
        lay.addStretch(1)

    def set_psp_mode(self, mode: str) -> None:
        idx = self.cmb_psp_mode.findText(str(mode))
        if idx >= 0:
            self.cmb_psp_mode.setCurrentIndex(idx)

    def psp_mode(self) -> str:
        return self.cmb_psp_mode.currentText()

    def set_status(self, text: str) -> None:
        self.lbl_status.setText(text)
