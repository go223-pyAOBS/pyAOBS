"""
Denoise processors package.

Core denoise implementations live in this directory and are designed
to be called by visualization modules such as zplotpy.
"""

from .pipeline import denoise_section, denoise_trace
from .types import DenoiseDebug, DenoiseResult

__all__ = [
    "DenoiseDebug",
    "DenoiseResult",
    "denoise_trace",
    "denoise_section",
]

