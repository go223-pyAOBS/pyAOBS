"""
TOMO2D module for 2D traveltime tomography
二维初至波走时层析成像模块

This module provides tools for:
本模块提供以下功能：
- Traveltime tomography (初至波和反射波联合走时层析成像)

Main Components 主要组件:
- TomoAnd: Python wrapper for tomo2d commands
  - edit_smesh: Edit slowness mesh file (编辑慢度网格文件)
  - gen_smesh: Generate slowness mesh (生成慢度网格)
  - gen_damp: Generate damping file (生成阻尼文件)
  - gen_vcorr: Generate velocity correlation file (生成速度相关文件)
  - tt_forward: Forward ray tracing (正演射线追踪)
  - tt_inverse: Traveltime inversion (走时反演)
  - stat_smesh: Mesh statistics (网格统计)

- Help Documentation (帮助文档)
  - TomoHelp: Help documentation class (帮助文档类)
    - edit_smesh_help(): Edit mesh help (编辑网格帮助)
    - gen_smesh_help(): Generate mesh help (生成网格帮助)
    - gen_damp_help(): Generate damping help (生成阻尼帮助)
    - gen_vcorr_help(): Generate correlation help (生成相关文件帮助)
    - tt_forward_help(): Forward modeling help (正演模拟帮助)
    - tt_inverse_help(): Inversion help (反演帮助)
    - stat_smesh_help(): Statistics help (统计帮助)

Configuration 配置:
- You can specify executable directory by:
  - TomoAnd(bin_path="path/to/tomo2d/bin")
  - environment variable `PYAOBS_TOMO2D_BIN`
  - environment variable `TOMO2D_BIN`

Quick Start 快速示例:
    from pyAOBS.modeling.tomo2d import TomoAnd
    tomo = TomoAnd()  # auto resolve bin path
    # Show help:
    tomo.gen_smesh()
    # Build mesh:
    tomo.gen_smesh(
        vel_opt="uniform", v0=1.5, gradient=0.1,
        grid_opt="uniform", nx=101, nz=51, xmax=100.0, zmax=30.0
    )
"""

from .tomand import TomoAnd
from .help_docs import TomoHelp

# Create a default instance for easy access
tomo2d = TomoAnd()

# Create a help instance for easy access
help_docs = TomoHelp()


def launch_tomo2d_gui():
    """Lazy import GUI launcher to avoid runpy module pre-import warning."""
    from .tomo2d_gui import launch_tomo2d_gui as _launch
    return _launch()


def __getattr__(name):
    if name == "Tomo2DGui":
        from .tomo2d_gui import Tomo2DGui
        return Tomo2DGui
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # Main class
    'TomoAnd',
    'tomo2d',
    
    # Help documentation
    'TomoHelp',
    'help_docs',

    # GUI
    'Tomo2DGui',
    'launch_tomo2d_gui',
] 