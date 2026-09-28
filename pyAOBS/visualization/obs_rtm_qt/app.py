# -*- coding: utf-8 -*-
"""Qt 入口：延迟加载 PySide6。"""

from __future__ import annotations

import sys
from typing import Optional


def main(argv: Optional[list] = None) -> int:
    try:
        from .mainwindow import run_application
    except ImportError as exc:
        msg = str(exc).lower()
        if "pyside6" in msg or "qt" in msg:
            raise RuntimeError(
                "需要 PySide6。请执行: pip install 'pyAOBS[gui-qt]' 或 pip install PySide6"
            ) from exc
        raise
    return run_application(argv if argv is not None else sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
