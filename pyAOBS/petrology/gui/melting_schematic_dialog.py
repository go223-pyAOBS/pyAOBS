"""Preview active vs passive melting schematic (KKHS02 / REEBOX mode figure)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
)

from petrology.validation.plot_active_passive_schematic import ensure_schematic_png, main as regenerate_schematic

from .app import UI_FONT_PT
from .dialog_utils import show_modeless_dialog
from .forward_plot import get_last_forward_context

_CAPTION = (
    "被动 vs 主动熔融 — REEBOX / Modern LIP 模式图。"
    "含 Tp–P (c)、F–T (d)、H–Vp 等九面板。"
    "若刚运行过「正演」，点击「按正演参数生成…」使 (c)(d) 与主图一致。"
    " CLI: python petrology/validation/plot_active_passive_schematic.py"
)


class MeltingSchematicDialog(QDialog):
    """Scrollable PNG preview of ``active_passive_melting_schematic.png``."""

    def __init__(self) -> None:
        super().__init__(None)
        self.setWindowTitle("主动 / 被动熔融模式图")
        self.resize(980, 820)
        self.setMinimumSize(640, 480)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)

        hint = QLabel(_CAPTION)
        hint.setWordWrap(True)
        hint.setStyleSheet(f"font-size: {UI_FONT_PT - 1}pt; color: #444;")
        root.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll.setWidget(self._img_label)
        root.addWidget(scroll, stretch=1)

        bar = QHBoxLayout()
        btn_fwd = QPushButton("按正演参数生成…")
        btn_fwd.setToolTip("使用最近一次「正演」的 Tp、χ、Φ 重绘完整模式图（(c)(d) 与主图联动）")
        btn_fwd.clicked.connect(self._reload_from_forward)
        bar.addWidget(btn_fwd)
        btn_refresh = QPushButton("重新生成 PNG…")
        btn_refresh.setToolTip("调用 plot_active_passive_schematic 覆盖写入 figures/ 目录")
        btn_refresh.clicked.connect(self._reload_regenerate)
        bar.addWidget(btn_refresh)
        bar.addStretch()
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        bar.addWidget(btn_close)
        root.addLayout(bar)

        self._load_image(regenerate=False)

    def _load_image(self, *, regenerate: bool) -> None:
        try:
            if regenerate:
                regenerate_schematic()
            path = ensure_schematic_png()
        except Exception as exc:
            self._img_label.setText(f"无法加载模式图：\n{exc}")
            return
        pix = QPixmap(str(path))
        if pix.isNull():
            self._img_label.setText(f"无法读取图像：\n{path}")
            return
        self._img_label.setPixmap(pix)
        self._img_label.setToolTip(str(path))

    def _reload_from_forward(self) -> None:
        ctx = get_last_forward_context()
        if ctx is None:
            QMessageBox.information(
                self,
                "模式图",
                "尚无正演记录。请先在主界面运行一次「正演」。",
            )
            return
        try:
            regenerate_schematic(
                ref_tp=ctx.tp_c,
                ref_chi=ctx.chi,
                phi=ctx.phi,
                col_kw=ctx.lith_kwargs,
            )
            self._load_image(regenerate=False)
        except Exception as exc:
            QMessageBox.critical(self, "模式图", f"按正演参数生成失败：\n{exc}")

    def _reload_regenerate(self) -> None:
        try:
            self._load_image(regenerate=True)
        except Exception as exc:
            QMessageBox.critical(self, "模式图", f"重新生成失败：\n{exc}")


def show_melting_schematic_dialog() -> None:
    show_modeless_dialog(MeltingSchematicDialog(), singleton=True)
