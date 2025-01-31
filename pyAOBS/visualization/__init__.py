"""
Visualization module for seismic processing.

This module provides visualization tools for seismic data and models.
It includes:
- ZeltModelVisualizer: For visualizing Zelt velocity models
- GridModelVisualizer: For visualizing grid format velocity models
- GridModelProcessor: For processing and converting grid format velocity models
"""

__all__ = ['ZeltModelVisualizer', 'GridModelVisualizer', 'GridModelProcessor']

# Implement lazy loading to avoid import issues
def __getattr__(name):
    if name in __all__:
        from .show_model import ZeltModelVisualizer, GridModelVisualizer, GridModelProcessor
        return locals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")