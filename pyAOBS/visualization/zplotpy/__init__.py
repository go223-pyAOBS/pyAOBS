"""
zplotpy - Python implementation of ZPLOT seismic phase picking tool

A modern Python GUI for interactive seismic phase picking, based on the original
ZPLOT Fortran code by Colin A. Zelt (1994), modified by Haibo Huang (2023).
"""

__version__ = '0.1.0'
__author__ = 'Haibo Huang'

from .qt_fast_viewer import ZPlotGUI
__all__ = ['ZPlotGUI']

# DPG 版本为可选组件：缺失时不影响 Qt 主入口。
try:
    from .zplot_dpg import ZPlotDPGApp
    __all__.append('ZPlotDPGApp')
except Exception:
    ZPlotDPGApp = None
