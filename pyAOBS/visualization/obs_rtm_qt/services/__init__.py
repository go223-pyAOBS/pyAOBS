# -*- coding: utf-8 -*-
"""后端服务：薄封装现有脚本，GUI 不复制算法。"""

from .paths import madagascar_rtm_dir, script_path
from .su_import import build_su_to_shots_cmd, run_su_to_shots
from .geometry import (
    check_landing,
    check_offset_sign_consistency,
    load_xz_txt,
    run_obs_geometry_check,
)
from .preprocess import apply_bandpass_mute
from .rsf_io import read_gather, write_gather, load_offsets_table
from .batch_preprocess import (
    process_filter_shots,
    process_mute_shots,
    process_rtm_shots,
    process_shots,
)
from .velocity import build_velocity_model, read_vel_rsf
from .rtm_job import build_rtm_loop_cmd, build_rtm_run_cmd, run_rtm_loop, stack_shot_images
from .sconstruct_gen import generate_sconstruct_obs_rtm

__all__ = [
    "madagascar_rtm_dir",
    "script_path",
    "build_su_to_shots_cmd",
    "run_su_to_shots",
    "check_landing",
    "check_offset_sign_consistency",
    "run_obs_geometry_check",
    "load_xz_txt",
    "apply_bandpass_mute",
    "read_gather",
    "write_gather",
    "load_offsets_table",
    "process_shots",
    "process_mute_shots",
    "process_filter_shots",
    "process_rtm_shots",
    "build_velocity_model",
    "read_vel_rsf",
    "build_rtm_loop_cmd",
    "run_rtm_loop",
    "stack_shot_images",
]
