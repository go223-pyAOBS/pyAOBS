# -*- coding: utf-8 -*-
"""阶段 1：输入数据与几何约定。"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...project import ZplotProject


class InputPanel(QWidget):
    project_changed = Signal()
    request_load_workbench = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        tip = QLabel(
            "指定工区输入：Z / HDR / REC / 水深。几何默认 geom=obs（与波形/RTM 习惯一致）；"
            "也可选 segy / auto。水深文件会在位置 Map 中自动共用。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#475569;")
        lay.addWidget(tip)

        box = QGroupBox("输入文件", self)
        form = QFormLayout(box)
        self.ed_dfile = QLineEdit(box)
        self.ed_hfile = QLineEdit(box)
        self.ed_rfile = QLineEdit(box)
        self.ed_terrain = QLineEdit(box)
        form.addRow("数据 .z", self._browse_row(self.ed_dfile, "Z files (*.z);;All (*)"))
        form.addRow("道头 .hdr", self._browse_row(self.ed_hfile, "HDR (*.hdr);;All (*)"))
        self.ed_rfile.setToolTip(
            "可选。ASCII 记录文件（.rec / .rsp）：每行 ishnum xmod ymod az [title]。\n"
            "作用：把炮号映射到模型坐标与方位；写 tx.in 时预填 xmod；"
            "理论走时生成 r.in 时可作炮点参考。\n"
            "浏览、按 ishoti 换炮、拾取不依赖此文件（道头已有 ishoti）。"
        )
        form.addRow("记录 .rec", self._browse_row(self.ed_rfile, "REC (*.rec *.rsp);;All (*)"))
        form.addRow(
            "水深地形",
            self._browse_row(self.ed_terrain, "Terrain (*.xyz *.txt *.nc *.grd);;All (*)"),
        )
        lay.addWidget(box)

        gbox = QGroupBox("几何 / 备注", self)
        gform = QFormLayout(gbox)
        self.cmb_geom = QComboBox(gbox)
        self.cmb_geom.addItems(["obs", "segy", "auto"])
        self.cmb_geom.setToolTip("obs=本工区/RTM；segy=字面 Source/Group；auto=唯一 XY 启发式")
        gform.addRow("geom", self.cmb_geom)
        self.ed_name = QLineEdit(gbox)
        gform.addRow("工程名", self.ed_name)
        self.txt_notes = QPlainTextEdit(gbox)
        self.txt_notes.setPlaceholderText("备注（可选）")
        self.txt_notes.setMaximumHeight(90)
        gform.addRow("备注", self.txt_notes)
        lay.addWidget(gbox)

        row = QHBoxLayout()
        self.btn_load = QPushButton("加载到波形工作台", self)
        self.btn_load.setDefault(True)
        self.btn_load.clicked.connect(self.request_load_workbench.emit)
        row.addWidget(self.btn_load)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addStretch(1)

        for w in (self.ed_dfile, self.ed_hfile, self.ed_rfile, self.ed_terrain, self.ed_name):
            w.textChanged.connect(lambda *_: self.project_changed.emit())
        self.cmb_geom.currentIndexChanged.connect(lambda *_: self.project_changed.emit())
        self.txt_notes.textChanged.connect(self.project_changed.emit)

    def _browse_row(self, edit: QLineEdit, filt: str) -> QWidget:
        wrap = QWidget(self)
        h = QHBoxLayout(wrap)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, stretch=1)
        btn = QPushButton("…", wrap)
        btn.setFixedWidth(36)

        def _pick() -> None:
            path, _ = QFileDialog.getOpenFileName(self, "选择文件", edit.text(), filt)
            if path:
                edit.setText(path)

        btn.clicked.connect(_pick)
        h.addWidget(btn)
        return wrap

    def apply_to_project(self, project: ZplotProject) -> None:
        project.name = self.ed_name.text().strip() or project.name or "untitled"
        project.inputs.dfile = self.ed_dfile.text().strip()
        project.inputs.hfile = self.ed_hfile.text().strip()
        project.inputs.rfile = self.ed_rfile.text().strip()
        project.inputs.terrain_path = self.ed_terrain.text().strip()
        project.inputs.geom = self.cmb_geom.currentText().strip() or "obs"
        project.inputs.notes = self.txt_notes.toPlainText().strip()
        project.notes = project.inputs.notes

    def load_from_project(self, project: ZplotProject) -> None:
        self.ed_name.setText(project.name or "")
        self.ed_dfile.setText(project.inputs.dfile or "")
        self.ed_hfile.setText(project.inputs.hfile or "")
        self.ed_rfile.setText(project.inputs.rfile or "")
        self.ed_terrain.setText(project.inputs.terrain_path or "")
        geom = (project.inputs.geom or "obs").strip()
        idx = max(0, self.cmb_geom.findText(geom))
        self.cmb_geom.setCurrentIndex(idx)
        self.txt_notes.setPlainText(project.inputs.notes or project.notes or "")
