"""
Modeling module for seismic forward modeling and inversion

This module provides various seismic modeling tools and methods:
- TOMO2D: 2D first-arrival traveltime tomography
- RAYINVR: Ray tracing and velocity inversion
- HYBRID: Hybrid modeling approaches combining different methods
"""

from . import tomo2d
from . import rayinvr
from . import hybrid

__all__ = ['tomo2d', 'rayinvr', 'hybrid']

# Use lazy imports to avoid circular dependencies
def __getattr__(name):
    if name in __all__:
        import importlib
        return importlib.import_module(f".{name}", __package__)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'") 