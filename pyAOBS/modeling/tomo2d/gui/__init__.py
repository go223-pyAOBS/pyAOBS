"""TOMO2D Qt GUI 包（PySide6）。

启动::

    python -m pyAOBS.modeling.tomo2d
    python -m pyAOBS.modeling.tomo2d.gui

Workbench 插件 ``tomo2d.gui`` 经
``pyAOBS.workbench.gui_audit_launchers.tomo2d_gui`` 调用本包。

文档：``../README.md``、``../docs/HELP.md``。
工区：``meta/tomo2d_project.json`` + inputs/outputs/runs/cache。

子包:
- ``state`` / ``services``：无 UI 业务层
- ``panels`` / ``dialogs`` / ``plots``：Qt 界面与绘图
"""

from __future__ import annotations

from .app import main

__all__ = ["main"]
