# -*- coding: utf-8 -*-
"""idata：数据转换与 SEGY/SU 道头编辑工区（工程化）。

GUI 入口::

    python -m pyAOBS.processors.idata.gui
    # 或（Workbench / 无 pygmt 环境）
    python pyAOBS/processors/idata/run.py

工区：``meta/idata_project.json`` + inputs/outputs/。
转换后端：``processors.raw2sac``。
"""

from .project import IdataProject

__all__ = ["__version__", "IdataProject"]
__version__ = "0.1.0"
