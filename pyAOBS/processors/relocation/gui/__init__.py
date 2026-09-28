"""OBS 姿态校正 GUI（工程化工区 + zplotpy 工作台）。

入口::

    python -m pyAOBS.processors.relocation.gui

主窗为 ``RelocationProjectWindow``（对齐 RTM：新建/打开/保存工区、阶段页）。
波形底座仍为 ``RelocationViewer(QtFastViewer)``，嵌在「波形/拾取」阶段。
"""

from .main_window import AttitudeMainWindow, RelocationViewer
from .position_shift_dialog import show_position_shift_preview
from .project_window import RelocationProjectWindow, main

__all__ = [
    "RelocationProjectWindow",
    "RelocationViewer",
    "AttitudeMainWindow",
    "show_position_shift_preview",
    "main",
]
