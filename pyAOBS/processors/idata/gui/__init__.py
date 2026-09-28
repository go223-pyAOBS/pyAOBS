# -*- coding: utf-8 -*-
"""idata GUI（工程化工区 + 阶段页，对齐 relocation）。

入口::

    python -m pyAOBS.processors.idata.gui
    python pyAOBS/processors/idata/run.py

工区：新建/打开/保存 → meta/idata_project.json
"""

from .main_window import IdataMainWindow, run_idata_app

__all__ = ["IdataMainWindow", "run_idata_app", "main"]


def main() -> int:
    return int(run_idata_app() or 0)
