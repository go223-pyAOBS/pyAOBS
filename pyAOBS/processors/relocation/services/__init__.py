"""姿态校正 GUI 的无 Qt 服务层。"""

from .attitude_runner import AttitudeRunner
from .auto_pick_service import AutoPickService
from .depth_sampler import make_depth_sampler, sample_initial_depth_km
from .geometry import GeometryResolver
from .models import (
    AttitudeSolution,
    AttitudeUiParams,
    AutoPickParams,
    SessionState,
    StackResult,
    WaveformSelection,
)
from .observation_builder import OrientationObservationBuilder
from .pick_helpers import get_shared_pick, remove_shared_pick, set_shared_pick
from .preview_apply import (
    apply_orientation_to_gather,
    commit_orientation_to_loaded,
    rotate_components,
    rotate_from_result,
)
from .stack_service import WaveformStackService
from .position_shift_preview import ObsShiftSummary, build_obs_shift_summary
from .terrain_io import load_terrain_as_utm, load_terrain_meta
from .waveform_preprocess import preprocess_summary, preprocess_window, preprocess_zrt
from .waveform_selection import WaveformSelectionStore

__all__ = [
    "AttitudeRunner",
    "AttitudeSolution",
    "AttitudeUiParams",
    "AutoPickParams",
    "AutoPickService",
    "GeometryResolver",
    "OrientationObservationBuilder",
    "SessionState",
    "StackResult",
    "WaveformSelection",
    "WaveformSelectionStore",
    "WaveformStackService",
    "get_shared_pick",
    "remove_shared_pick",
    "set_shared_pick",
    "apply_orientation_to_gather",
    "commit_orientation_to_loaded",
    "rotate_components",
    "rotate_from_result",
    "sample_initial_depth_km",
    "make_depth_sampler",
    "load_terrain_as_utm",
    "load_terrain_meta",
    "ObsShiftSummary",
    "build_obs_shift_summary",
    "preprocess_window",
    "preprocess_zrt",
    "preprocess_summary",
]
