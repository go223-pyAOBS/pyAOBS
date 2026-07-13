"""
BASALT+langmuir.FOR (1992) interactive MAIN — Langmuir pressure-dependent extension.

Run::

    py -3.11 -m petrology.fc.wl_langmuir
"""

from .main import BasaltLangmuirSession, interactive_loop, langmuir_driver_for_case, main

__all__ = [
    "BasaltLangmuirSession",
    "interactive_loop",
    "langmuir_driver_for_case",
    "main",
]
