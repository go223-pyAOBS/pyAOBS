"""
Hybrid modeling module combining different seismic modeling approaches

This module provides:
- Machine learning based modeling (ml/)
- Fast sweeping methods (solve/)
- Example implementations and case studies (examples/)
"""

from . import ml
from . import solve

__all__ = ['ml', 'solve']

# Use lazy imports to avoid circular dependencies
def __getattr__(name):
    if name in __all__:
        import importlib
        return importlib.import_module(f".{name}", __package__)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'") 