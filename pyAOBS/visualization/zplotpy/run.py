#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""zplotpy 启动脚本（对齐 idata/run.py）。

推荐::

    python -m pyAOBS.visualization.zplotpy.gui
    # 或
    python pyAOBS/visualization/zplotpy/run.py
"""

from __future__ import annotations

import sys


def main() -> int:
    from pyAOBS.visualization.zplotpy.gui.project_window import main as gui_main

    return int(gui_main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
