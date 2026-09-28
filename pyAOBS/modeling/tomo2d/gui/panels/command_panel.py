"""底栏「命令」：最近一次解析后的 argv 命令行。"""

from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..widgets.terminal_view import TerminalView


class CommandPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.view = TerminalView()
        self.view.setPlaceholderText("运行或预览后，此处显示解析后的命令行…")
        lay.addWidget(self.view)

    def set_command(self, text: str) -> None:
        self.view.set_text(text or "")
