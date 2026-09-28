"""tomo2d Qt chrome / 字体。"""

from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget

UI_FONT_PT = 12


def apply_tomo2d_font(widget: QWidget, *, point_size: int = UI_FONT_PT) -> None:
    font = QFont(widget.font())
    font.setPointSize(point_size)
    widget.setFont(font)


def apply_tomo2d_chrome(window: QWidget) -> None:
    apply_tomo2d_font(window)
    window.setStyleSheet(
        """
        QMainWindow { background: #f5f6f8; }
        QToolBar#tomoTopChrome {
            background: #e8eef6;
            border-bottom: 2px solid #9bb0c9;
            spacing: 8px;
            padding: 6px 10px;
        }
        QToolBar#tomoTopChrome QPushButton#tomoExitBtn {
            background: #d9534f;
            color: #fff;
            font-weight: 600;
            padding: 4px 12px;
            border: 1px solid #b33b37;
            border-radius: 3px;
        }
        QToolBar#tomoTopChrome QPushButton#tomoExitBtn:hover {
            background: #c9302c;
        }
        QFrame#tomoParallelFrame {
            background: #f0f4fa;
            border: 1px solid #b8c6d8;
            border-radius: 4px;
        }
        QFrame#tomoParallelFrame QCheckBox,
        QFrame#tomoParallelFrame QLabel {
            font-size: 13px;
        }
        QFrame#tomoParallelFrame QLineEdit {
            font-size: 13px;
            padding: 1px 3px;
        }
        QListWidget#tomoCmdNav {
            background: #e8eef6;
            border: none;
            border-right: 2px solid #9bb0c9;
            outline: none;
            font-size: 14px;
            padding: 4px 0;
        }
        QListWidget#tomoCmdNav::item {
            padding: 8px 12px;
            color: #243447;
        }
        QListWidget#tomoCmdNav::item:selected {
            background: #2f6fed;
            color: #ffffff;
            font-weight: 600;
        }
        QListWidget#tomoCmdNav::item:hover:!selected {
            background: #d5e2f4;
        }
        QWidget#tomoFormCol {
            background: #f5f6f8;
        }
        QWidget#tomoRightDock QGroupBox {
            font-size: 12px;
        }
        QWidget#tomoRightDock QPlainTextEdit {
            font-size: 12px;
        }
        QStatusBar {
            background: #2b3340;
            color: #f0f3f7;
            font-weight: 600;
        }
        QPlainTextEdit#tomoTerminal {
            background: #1e1e1e;
            color: #d4d4d4;
            font-family: Consolas, "Courier New", monospace;
        }
        """
    )
