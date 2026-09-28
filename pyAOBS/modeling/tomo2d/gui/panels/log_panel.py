"""执行日志面板（可选同步写入 tomo2d_gui.log）。"""

from __future__ import annotations

from datetime import datetime
from typing import Callable

from PySide6.QtWidgets import QGroupBox, QVBoxLayout

from ..widgets.terminal_view import TerminalView


class LogPanel(QGroupBox):
    def __init__(self, parent=None) -> None:
        super().__init__("执行日志", parent)
        lay = QVBoxLayout(self)
        self.view = TerminalView()
        lay.addWidget(self.view)
        self._file_sink: Callable[[str], None] | None = None

    def set_file_sink(self, sink: Callable[[str], None] | None) -> None:
        """设置写入文件日志的回调（摘要行）；子进程正文可传 write_file=False。"""
        self._file_sink = sink

    def log(self, text: str, *, write_file: bool = True) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        self.view.append_line(f"[{now}] {text}")
        if write_file and self._file_sink is not None:
            try:
                self._file_sink(text)
            except Exception:
                pass
