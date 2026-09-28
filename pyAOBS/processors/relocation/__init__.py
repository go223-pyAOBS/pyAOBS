"""
Relocation processors.

算法 API 在本包根；GUI / 服务层见子包（勿在此 import gui，以免拖入 Qt）::

    pyAOBS.processors.relocation.services
    pyAOBS.processors.relocation.gui   # python -m pyAOBS.processors.relocation.gui

详见 GUI_MVP.md。
"""

from .orientation_correction import (
    DIRECT_WATER_PICK_WORD,
    OrientationCorrectionInput,
    OrientationCorrectionResult,
    OrientationObservation,
    is_direct_water_phase,
    resolve_phase_policy,
    run_orientation_correction,
    split_observations_by_phase,
)
from .bathymetry_sampler import BathymetrySampler, build_bathymetry_sampler

__all__ = [
    "DIRECT_WATER_PICK_WORD",
    "OrientationCorrectionInput",
    "OrientationCorrectionResult",
    "OrientationObservation",
    "is_direct_water_phase",
    "resolve_phase_policy",
    "run_orientation_correction",
    "split_observations_by_phase",
    "BathymetrySampler",
    "build_bathymetry_sampler",
]

