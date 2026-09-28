"""
TOMO2D module for 2D traveltime tomography
二维初至波走时层析成像模块

This module provides tools for:
本模块提供以下功能：
- Traveltime tomography (初至波和反射波联合走时层析成像)

工区目录含 ``meta/tomo2d_project.json``、``inputs/``、``outputs/``、``runs/``、``cache/``。

Main Components 主要组件:
    - TomoAnd: Python wrapper for tomo2d commands
    - TomoHelp: Help documentation
    - launch_tomo2d_gui / Tomo2DMainWindow: Qt GUI
    - Tomo2dProject: 工区工程（meta/tomo2d_project.json）

Configuration 配置:
    - TomoAnd(bin_path="path/to/tomo2d/bin")
    - environment variable ``PYAOBS_TOMO2D_BIN`` / ``TOMO2D_BIN``
"""

from __future__ import annotations

from .tomand import TomoAnd
from .help_docs import TomoHelp

# Create a default instance for easy access
tomo2d = TomoAnd()

# Create a help instance for easy access
help_docs = TomoHelp()


def launch_tomo2d_gui():
    """启动 TOMO2D Qt GUI。

    等价于 ``python -m pyAOBS.modeling.tomo2d`` /
    ``python -m pyAOBS.modeling.tomo2d.gui``。
    """
    from .gui.app import main

    return main()


def __getattr__(name):
    if name == "Tomo2DMainWindow":
        from .gui.main_window import Tomo2DMainWindow

        return Tomo2DMainWindow
    if name == "Tomo2dProject":
        from .gui.project import Tomo2dProject

        return Tomo2dProject
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "TomoAnd",
    "tomo2d",
    "TomoHelp",
    "help_docs",
    "Tomo2DMainWindow",
    "Tomo2dProject",
    "launch_tomo2d_gui",
]
