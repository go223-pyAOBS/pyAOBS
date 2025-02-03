"""
pyAOBS - Python Active-source Ocean Bottom Seismology package

Author: Haibo Huang
Date: 2025
"""

from . import model_building
from . import visualization

__version__ = '0.1.0'
__author__ = 'Haibo Huang'
__all__ = ['visualization', 'model_building', 'processors', 'utils']

# Use lazy imports to avoid circular dependencies
def __getattr__(name):
    if name in __all__:
        import importlib
        return importlib.import_module(f".{name}", __package__)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'") 