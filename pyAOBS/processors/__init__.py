"""
Processors module for handling different types of seismic data

This module provides tools for processing seismic data in various formats:
- Seismic Unix (SU) format processing through SUProcessor and supython
- Additional seismic data format support can be added here
"""

from .su_processor import SUProcessor
from .supython import (
    readsu,
    writesu,
    readsuamp,
    readsuhdr,
    makehdr,
    plotsu,
    suhdr,
    suhdrscale
)

__all__ = [
    'SUProcessor',
    'readsu',
    'writesu',
    'readsuamp',
    'readsuhdr',
    'makehdr',
    'plotsu',
    'suhdr',
    'suhdrscale'
] 