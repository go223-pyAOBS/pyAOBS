"""Qt 入口：延迟加载 PySide6。"""

from __future__ import annotations

import sys
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    try:
        from .main_window import run_application
    except ImportError as exc:
        msg = str(exc).lower()
        if "pyside6" in msg or "qt" in msg:
            raise RuntimeError(
                "Qt 依赖未就绪。请执行： pip install PySide6"
            ) from exc
        raise RuntimeError(f"tomo2d Qt 依赖未就绪：{exc}") from exc
    return run_application(argv if argv is not None else sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
