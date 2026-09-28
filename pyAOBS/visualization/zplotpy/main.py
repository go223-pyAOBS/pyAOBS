#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""兼容启动脚本 → ``run.py`` / ``python -m …zplotpy.gui``。"""

from __future__ import annotations

try:
    from .run import main
except ImportError:
    from run import main  # type: ignore

if __name__ == "__main__":
    raise SystemExit(main())
