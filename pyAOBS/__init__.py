"""
Seismic Processing Package

This package provides tools for seismic data processing and visualization.

Main components:
- visualization: Tools for visualizing seismic data and velocity models
    - ZeltModelVisualizer: For Zelt velocity models
    - GridModelVisualizer: For grid format models
    - GridModelProcessor: For processing grid models
- model_building: Tools for building and setting up velocity models
- processors: Tools for seismic data processing
- utils: Utility functions and helpers
"""

__version__ = '0.1.0'
__author__ = 'Haibo Huang'
__all__ = ['visualization', 'model_building', 'processors', 'utils']

# Use lazy imports to avoid circular dependencies
def __getattr__(name):
    if name in __all__:
        import importlib
        return importlib.import_module(f".{name}", __package__)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'") 