# -*- coding: utf-8 -*-
"""阶段 2：走时图（参数条 + 绘图区 + 分析动作）。"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from ..plot_canvas import IPhasePlotCanvas
from ..toolbar_panel import CollapsibleParamStrip


class PlotPanel(QWidget):
    """走时图页：分析参数 + 2×2 主图。"""

    request_theory2d = Signal()
    request_inversion = Signal()
    request_diagnostics = Signal()
    request_rin_editor = Signal()
    request_export_tx = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        tools = QHBoxLayout()
        self.btn_theory2d = QPushButton("运行2D正演", self)
        self.btn_theory2d.setToolTip(
            "先把界面 pois（默认左支）写回 r.in，再强制 RAYINVR 正演并刷新左右两侧理论缓存。"
        )
        self.btn_inv = QPushButton("1D反演校正", self)
        self.btn_inv.setToolTip("用 1D 薄层模型做反演校正，估计厚度与 Vp/Vs 剖面。")
        self.btn_diag = QPushButton("1D诊断对比", self)
        self.btn_diag.setToolTip("弹出 1D 下观测/理论 PPS−PPP 与 PSS−PSP 诊断对比图。")
        self.btn_rin = QPushButton("r.in相位组", self)
        self.btn_rin.setToolTip("打开 r.in 相位组编辑器（独立进程）；保存后主窗可重载过滤。")
        self.btn_export_tx = QPushButton("预览/筛选 tx.in…", self)
        self.btn_export_tx.setToolTip(
            "对齐 tomo2d「预览 tx.in」：左侧勾选 OBS/震相即时预览，并可导出筛选后的 tx_*_sel.in。"
        )
        self.btn_theory2d.clicked.connect(self.request_theory2d.emit)
        self.btn_inv.clicked.connect(self.request_inversion.emit)
        self.btn_diag.clicked.connect(self.request_diagnostics.emit)
        self.btn_rin.clicked.connect(self.request_rin_editor.emit)
        self.btn_export_tx.clicked.connect(self.request_export_tx.emit)
        for b in (
            self.btn_theory2d,
            self.btn_inv,
            self.btn_diag,
            self.btn_rin,
            self.btn_export_tx,
        ):
            tools.addWidget(b)
        tools.addStretch(1)
        lay.addLayout(tools)

        self.param_strip = CollapsibleParamStrip(self)
        lay.addWidget(self.param_strip)

        self.plot = IPhasePlotCanvas(self, figsize=(13, 8), dpi=100)
        lay.addWidget(self.plot, stretch=1)
