"""Interactive velocity-model analysis (imodel).

Layout (aligned with iphase / zplotpy):
  - ``engine`` / ``velocity_anomaly`` / ``gravity_obs_grid`` / ``talwani_optional``
    — shared core (no Qt)
  - ``gui`` — PySide6 interactive application
  - ``docs/HELP.md`` — GUI 帮助（F1 / 工具栏「帮助」）
  - ``_archive`` — legacy Tk GUI and old docs (not supported)

Launch (same pattern as zplotpy / iphase)::

    python -m pyAOBS.visualization.imodel.gui
"""

from __future__ import annotations

from .engine import (
    GravityCalculator,
    InteractiveModelViewer,
    PointSelector,
    ProfileExtractor,
    PropertyCalculator,
    interactive_model_viewer,
)

__all__ = [
    "GravityCalculator",
    "InteractiveModelViewer",
    "PointSelector",
    "ProfileExtractor",
    "PropertyCalculator",
    "interactive_model_viewer",
]
