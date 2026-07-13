"""
pyAOBS - Python Active-source Ocean Bottom Seismology package
Python主动源海底地震学处理包

Author: Haibo Huang
Date: 2025

This package provides tools for processing and analyzing active-source ocean bottom seismology data.
本包提供了处理和分析主动源海底地震数据的工具。

Main components 主要组件:
- Processors: Data processing tools including SU format support
  数据处理工具，包括SU格式支持
  - SUProcessor: Seismic Unix data processing
  - supython: Enhanced SU file handling

- Model Building: Tools for building and parameterizing velocity models
  速度模型构建和参数化工具
  - Point2d, ZNode2d: Basic geometry elements
  - TrapezoidCell2d: Model cell definitions
  - ZeltVelocityModel2d: Zelt format models

- Modeling: Forward modeling and inversion tools
  正演模拟和反演工具
  - TOMO2D: First-arrival and reflection traveltime tomography
  - RAYINVR: Ray tracing and velocity inversion
  - HYBRID: Hybrid modeling approaches

- Visualization: Data and model visualization tools
  数据和模型可视化工具
  - ZeltModelVisualizer: For Zelt velocity models
  - GridModelVisualizer: For grid format models
  - iphase: Phase pick (tx.in) processing and visualization

- Utils: Utility functions and tools
  实用工具和函数
  - Rock property utilities
  - Logging configuration
  - Helper functions
"""

__version__ = '3.0.0rc2'
__author__ = 'Haibo Huang'


def _ensure_legacy_import_paths() -> None:
    """Allow bare ``import petrology`` used by GUI / KKHS02 code.

    Installed layouts expose ``pyAOBS.petrology``; many modules still import
    ``petrology.*``. Prepend this package directory so both styles work.
    """
    import sys
    from pathlib import Path

    root = str(Path(__file__).resolve().parent)
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_legacy_import_paths()

__all__ = [
    'model_building',
    'visualization',
    'processors',
    'modeling',
    'utils',
    'workbench',
    'petrology',
    'iphase',
]

_LAZY_SYMBOL_IMPORTS = {
    'readsu': ('.processors.supython', 'readsu'),
    'writesu': ('.processors.supython', 'writesu'),
    'readsuamp': ('.processors.supython', 'readsuamp'),
    'readsuhdr': ('.processors.supython', 'readsuhdr'),
    'makehdr': ('.processors.supython', 'makehdr'),
    'plotsu': ('.processors.supython', 'plotsu'),
    'Point2d': ('.model_building', 'Point2d'),
    'ZNode2d': ('.model_building', 'ZNode2d'),
    'TrapezoidCell2d': ('.model_building', 'TrapezoidCell2d'),
    'ZeltVelocityModel2d': ('.model_building', 'ZeltVelocityModel2d'),
}
__all__.extend(list(_LAZY_SYMBOL_IMPORTS.keys()))


def __getattr__(name):
    if name == 'iphase':
        from . import visualization
        return visualization.iphase
    if name in _LAZY_SYMBOL_IMPORTS:
        import importlib
        module_name, symbol_name = _LAZY_SYMBOL_IMPORTS[name]
        module = importlib.import_module(module_name, __package__)
        return getattr(module, symbol_name)
    if name in __all__:
        import importlib
        return importlib.import_module(f".{name}", __package__)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# NOTE:
# Keep package import lightweight. Submodules and convenience symbols are loaded lazily
# via __getattr__ to avoid blocking startup paths (e.g. `python -m pyAOBS.workbench.app`).

# Use lazy imports to avoid circular dependencies (merged with iphase alias) 