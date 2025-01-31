"""
Model Building Module

This module provides tools and classes for building and setting up seismic velocity models.
It focuses on model construction and parameterization, not forward modeling or inversion.

Key components:
- Basic geometry elements (Point2d, ZNode2d)
- Model cell definitions (TrapezoidCell2d)
- Zelt format velocity model construction (ZeltVelocityModel2d)
- Enhanced model building capabilities (EnhancedZeltModel)
"""

from .models import Point2d, ZNode2d, TrapezoidCell2d
from .zeltform import ZeltVelocityModel2d, EnhancedZeltModel

__all__ = [
    'Point2d',
    'ZNode2d',
    'TrapezoidCell2d',
    'ZeltVelocityModel2d',
    'EnhancedZeltModel',
] 