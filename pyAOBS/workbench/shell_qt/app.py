"""Qt Workbench entry."""

from __future__ import annotations

import sys
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    try:
        from PySide6.QtWidgets import QApplication

        from .main_window import WorkbenchMainWindow
    except ImportError as exc:
        raise RuntimeError(
            "PySide6 未就绪。请执行： pip install 'pyAOBS[gui-qt]' 或 pip install PySide6"
        ) from exc

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("pyAOBS Workbench")
    win = WorkbenchMainWindow()
    win.show()
    return app.exec()


def launch_workbench_qt() -> int:
    return main()
