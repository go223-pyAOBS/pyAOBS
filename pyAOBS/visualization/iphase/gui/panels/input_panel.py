# -*- coding: utf-8 -*-
"""阶段 1：输入文件（走时 / 地形 / OBS 深度 / r.in）+ 工区剖面预览。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .survey_map import SurveyMapWidget


class InputPanel(QWidget):
    """收集输入路径；由主窗加载并切换到走时图。"""

    request_apply = Signal()
    request_browse_tx = Signal()
    paths_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        tip = QLabel(
            "指定工区输入：走时 tx.in、海底地形、OBS/炮点深度表、可选 r.in。"
            "填好后点「加载到走时图」进入绘图页。"
        )
        tip.setWordWrap(True)
        tip.setObjectName("IphaseCaption")
        lay.addWidget(tip)

        box_tx = QGroupBox("走时文件 tx.in", self)
        tx_l = QVBoxLayout(box_tx)
        self.list_tx = QListWidget(box_tx)
        self.list_tx.setMinimumHeight(100)
        tx_l.addWidget(self.list_tx)
        row_tx = QHBoxLayout()
        self.btn_add_tx = QPushButton("添加走时文件…", box_tx)
        self.btn_add_tx.clicked.connect(self._browse_add_tx)
        self.btn_clear_tx = QPushButton("清空列表", box_tx)
        self.btn_clear_tx.clicked.connect(self._clear_tx)
        self.btn_remove_tx = QPushButton("移除选中", box_tx)
        self.btn_remove_tx.clicked.connect(self._remove_selected_tx)
        row_tx.addWidget(self.btn_add_tx)
        row_tx.addWidget(self.btn_remove_tx)
        row_tx.addWidget(self.btn_clear_tx)
        row_tx.addStretch(1)
        tx_l.addLayout(row_tx)
        lay.addWidget(box_tx)

        box = QGroupBox("其它输入", self)
        form = QFormLayout(box)
        self.ed_seafloor = QLineEdit(box)
        self.ed_obs_depth = QLineEdit(box)
        self.ed_rin = QLineEdit(box)
        form.addRow(
            "海底地形",
            self._browse_row(
                self.ed_seafloor,
                "Depth text (*.txt *.dat *.asc *.csv);;All (*)",
                "选择海底深度文件",
            ),
        )
        self.ed_seafloor.setToolTip("两列文本：x(km)  depth(km)。至少 2 行，按 x 排序后插值。")
        form.addRow(
            "OBS/炮点深度",
            self._browse_row(
                self.ed_obs_depth,
                "Depth text (*.txt *.dat *.asc *.csv *.lis);;All (*)",
                "选择 OBS/炮点深度表",
            ),
        )
        self.ed_obs_depth.setToolTip(
            "用于同步 r.in 的 xshot/zshot，并绘制工区剖面 OBS。\n"
            "• 两列：xshot(km)  zshot(km)（如 offset_depth_OBS.txt）\n"
            "• 三列 station.lis：站号  x(km)  z(km)（取第 2、3 列）"
        )
        form.addRow(
            "r.in",
            self._browse_row(
                self.ed_rin,
                "RAYINVR input (r.in);;All (*)",
                "选择 r.in",
            ),
        )
        lay.addWidget(box)

        row = QHBoxLayout()
        self.btn_apply = QPushButton("加载到走时图", self)
        self.btn_apply.setDefault(True)
        self.btn_apply.clicked.connect(self.request_apply.emit)
        self.btn_rin_editor = QPushButton("打开 r.in 相位组编辑器", self)
        row.addWidget(self.btn_apply)
        row.addWidget(self.btn_rin_editor)
        row.addStretch(1)
        lay.addLayout(row)

        self.lbl_status = QLabel("", self)
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setObjectName("IphaseCaption")
        lay.addWidget(self.lbl_status)

        box_map = QGroupBox("工区剖面预览", self)
        map_l = QVBoxLayout(box_map)
        self.survey_map = SurveyMapWidget(box_map)
        map_l.addWidget(self.survey_map)
        lay.addWidget(box_map, stretch=1)

        self._survey_timer = QTimer(self)
        self._survey_timer.setSingleShot(True)
        self._survey_timer.setInterval(220)
        self._survey_timer.timeout.connect(self.refresh_survey_from_paths)

        for ed in (self.ed_seafloor, self.ed_obs_depth, self.ed_rin):
            ed.textChanged.connect(self._on_paths_changed)
        self.list_tx.model().rowsInserted.connect(lambda *_: self._on_paths_changed())
        self.list_tx.model().rowsRemoved.connect(lambda *_: self._on_paths_changed())

    def _on_paths_changed(self) -> None:
        self.paths_changed.emit()
        self._survey_timer.start()

    def refresh_survey_from_paths(self) -> None:
        """根据输入框路径自行解析并刷新剖面（无需先点「加载到走时图」）。"""
        from .._business import IPhaseBusinessMixin

        sea_x = sea_z = None
        obs_x = obs_z = None
        labels = None
        sea = self.seafloor_path()
        if sea and Path(sea).is_file():
            try:
                sea_x, sea_z = IPhaseBusinessMixin._parse_xz_depth_table(Path(sea))
            except Exception:
                sea_x = sea_z = None
        obs = self.obs_depth_path()
        if obs and Path(obs).is_file():
            try:
                obs_x, obs_z, labels = IPhaseBusinessMixin._parse_xz_depth_table(
                    Path(obs), return_ids=True
                )
            except Exception:
                obs_x = obs_z = labels = None
        self.survey_map.update_map(
            seafloor_x=sea_x,
            seafloor_z=sea_z,
            obs_x=obs_x,
            obs_z=obs_z,
            obs_labels=labels,
        )

    def update_survey_map(
        self,
        *,
        seafloor_x=None,
        seafloor_z=None,
        obs_x=None,
        obs_z=None,
        obs_labels=None,
    ) -> None:
        """主窗已加载数组时直接刷新（优先于重新读盘）。"""
        self.survey_map.update_map(
            seafloor_x=seafloor_x,
            seafloor_z=seafloor_z,
            obs_x=obs_x,
            obs_z=obs_z,
            obs_labels=obs_labels,
        )

    def _browse_row(self, edit: QLineEdit, filt: str, title: str) -> QWidget:
        wrap = QWidget(self)
        h = QHBoxLayout(wrap)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, stretch=1)
        btn = QPushButton("…", wrap)
        btn.setFixedWidth(36)

        def _pick() -> None:
            path, _ = QFileDialog.getOpenFileName(self, title, edit.text(), filt)
            if path:
                edit.setText(path)
                self._on_paths_changed()

        btn.clicked.connect(_pick)
        h.addWidget(btn)
        return wrap

    def _browse_add_tx(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择走时文件 tx.in",
            "",
            "tx files (*.in);;all files (*.*)",
        )
        if not paths:
            return
        existing = {self.list_tx.item(i).text() for i in range(self.list_tx.count())}
        for p in paths:
            if p not in existing:
                self.list_tx.addItem(p)
        self._on_paths_changed()

    def _clear_tx(self) -> None:
        self.list_tx.clear()
        self._on_paths_changed()

    def _remove_selected_tx(self) -> None:
        for item in self.list_tx.selectedItems():
            self.list_tx.takeItem(self.list_tx.row(item))
        self._on_paths_changed()

    def tx_paths(self) -> list[str]:
        return [self.list_tx.item(i).text() for i in range(self.list_tx.count())]

    def set_tx_paths(self, paths: list[str]) -> None:
        self.list_tx.clear()
        for p in paths:
            if p:
                self.list_tx.addItem(str(p))

    def seafloor_path(self) -> str:
        return self.ed_seafloor.text().strip()

    def obs_depth_path(self) -> str:
        return self.ed_obs_depth.text().strip()

    def rin_path(self) -> str:
        return self.ed_rin.text().strip()

    def set_seafloor_path(self, path: str) -> None:
        self.ed_seafloor.setText(path or "")

    def set_obs_depth_path(self, path: str) -> None:
        self.ed_obs_depth.setText(path or "")

    def set_rin_path(self, path: str) -> None:
        self.ed_rin.setText(path or "")

    def set_status(self, text: str) -> None:
        self.lbl_status.setText(text)

    def sync_from_paths(
        self,
        *,
        tx_files: list[Path | str],
        seafloor: Path | str | None,
        obs_depth: Path | str | None,
        rin: Path | str | None,
    ) -> None:
        self.set_tx_paths([str(p) for p in tx_files])
        self.set_seafloor_path(str(seafloor) if seafloor else "")
        self.set_obs_depth_path(str(obs_depth) if obs_depth else "")
        self.set_rin_path(str(rin) if rin else "")
        self.refresh_survey_from_paths()
