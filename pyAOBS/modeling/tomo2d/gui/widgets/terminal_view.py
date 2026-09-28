"""终端风格只读/可追加文本区。"""

from __future__ import annotations

from PySide6.QtWidgets import QPlainTextEdit


class TerminalView(QPlainTextEdit):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("tomoTerminal")
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

    def set_text(self, text: str) -> None:
        from PySide6.QtGui import QTextCursor

        self.setPlainText(text)
        self.moveCursor(QTextCursor.MoveOperation.End)

    def append_line(self, text: str) -> None:
        self.appendPlainText(text)
