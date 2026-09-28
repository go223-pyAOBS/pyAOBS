# -*- coding: utf-8 -*-
"""阶段 1：数据加载（SU → 按炮 RSF）。"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..dialog_utils import connect_combo_deferred
from ..project import ObsRtmProject
from ..styles import compact_form, hint_label, primary_button, side_panel_layout


class DataPanel(QWidget):
    request_import = Signal()
    project_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        side = QWidget()
        side.setObjectName("ObsRtmSidePanel")
        side_l = side_panel_layout(side)

        box = QGroupBox("工区与 SU")
        form = compact_form(QFormLayout(box))

        row_w = QHBoxLayout()
        self.ed_workdir = QLineEdit()
        self.ed_workdir.setToolTip(
            "工区根目录：shots/、工程 JSON、速度与 RTM 工作目录均写于此下"
        )
        btn_w = QPushButton("浏览…")
        btn_w.setToolTip("选择工区根目录")
        btn_w.clicked.connect(self._pick_workdir)
        row_w.addWidget(self.ed_workdir)
        row_w.addWidget(btn_w)
        form.addRow("工区目录", row_w)

        row_s = QHBoxLayout()
        self.ed_su = QLineEdit()
        self.ed_su.setToolTip("野外 SU 路径；4C 务必再选分量，避免一次读入全部分量")
        btn_s = QPushButton("浏览…")
        btn_s.setToolTip("选择 SU 文件")
        btn_s.clicked.connect(self._pick_su)
        row_s.addWidget(self.ed_su)
        row_s.addWidget(btn_s)
        form.addRow("SU 文件", row_s)

        self.cmb_component = QComboBox()
        self.cmb_component.addItems(["hydro", "z", "radial", "trans", "z_raw", "h1", "h2", ""])
        self.cmb_component.setToolTip(
            "su_to_shots --component；推荐 hydro。空=不过滤别名（4C 易爆内存）"
        )
        connect_combo_deferred(
            self.cmb_component, lambda *_: self.project_changed.emit()
        )
        form.addRow("分量", self.cmb_component)

        self.cmb_endian = QComboBox()
        self.cmb_endian.addItems(["little", "big"])
        self.cmb_endian.setToolTip("SU 字节序；须与采集/转储一致，错了道头会乱")
        connect_combo_deferred(
            self.cmb_endian, lambda *_: self.project_changed.emit()
        )
        form.addRow("endian", self.cmb_endian)

        self.cmb_group = QComboBox()
        self.cmb_group.addItems(["fldr", "ep", "gx", "sx", "file"])
        self.cmb_group.setToolTip(
            "按道头字段拆炮（fldr/ep/…）；决定 shot_*.rsf 如何分组"
        )
        connect_combo_deferred(
            self.cmb_group, lambda *_: self.project_changed.emit()
        )
        form.addRow("分炮键", self.cmb_group)

        self.chk_native = QCheckBox("纯 Python 读 SU（--native）")
        self.chk_native.setChecked(True)
        self.chk_native.setToolTip(
            "不依赖 Madagascar 读 SU；4C / Windows 推荐勾选"
        )
        self.chk_native.toggled.connect(lambda: self.project_changed.emit())
        form.addRow(self.chk_native)

        side_l.addWidget(box)
        side_l.addWidget(
            hint_label(
                "4C 数据务必选分量（推荐 hydro），否则易内存溢出/段错误。\n"
                "勾选「纯 Python 读 SU」；几何在「工区几何」页（推荐 offset）。"
            )
        )
        btn = primary_button("导入 SU → 按炮 RSF")
        btn.setToolTip("后台运行 su_to_shots，写出 shots/ 与 shots_xz.txt")
        btn.clicked.connect(self.request_import.emit)
        side_l.addWidget(btn)
        side_l.addStretch(1)
        root.addWidget(side)

    def _pick_workdir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择工区目录")
        if d:
            self.ed_workdir.setText(d)
            self.project_changed.emit()

    def _pick_su(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 SU", "", "SU (*.su);;All (*.*)"
        )
        if path:
            self.ed_su.setText(path)
            self.project_changed.emit()

    def apply_to_project(self, project: ObsRtmProject) -> None:
        project.workdir = self.ed_workdir.text().strip()
        project.su_path = self.ed_su.text().strip()
        project.geometry.component = self.cmb_component.currentText().strip()
        project.geometry.endian = self.cmb_endian.currentText()
        project.geometry.group = self.cmb_group.currentText()
        project.geometry.native = self.chk_native.isChecked()

    def load_from_project(self, project: ObsRtmProject) -> None:
        self.ed_workdir.setText(project.workdir or "")
        self.ed_su.setText(project.su_path or "")
        idx = self.cmb_component.findText(project.geometry.component or "")
        if idx >= 0:
            self.cmb_component.setCurrentIndex(idx)
        idx = self.cmb_endian.findText(project.geometry.endian)
        if idx >= 0:
            self.cmb_endian.setCurrentIndex(idx)
        idx = self.cmb_group.findText(project.geometry.group)
        if idx >= 0:
            self.cmb_group.setCurrentIndex(idx)
        self.chk_native.setChecked(bool(project.geometry.native))
