"""调用预览面板。"""

from __future__ import annotations

from PySide6.QtWidgets import QGroupBox, QVBoxLayout

from ..widgets.terminal_view import TerminalView


class PreviewPanel(QGroupBox):
    def __init__(self, parent=None) -> None:
        super().__init__("调用预览", parent)
        lay = QVBoxLayout(self)
        self.view = TerminalView()
        lay.addWidget(self.view)

    def set_preview(self, text: str) -> None:
        self.view.set_text(text)
