"""
RAYINVR module for ray tracing and velocity inversion
射线追踪模块，基于 RAYINVR 程序的 Python 实现

This module provides:
主要功能：
- Velocity model definition (速度模型定义)
- Ray tracing (射线追踪)
- Travel time calculation (走时计算)
- Result visualization (结果可视化)

Key components:
主要组件：
- VelocityModel: Velocity model definition (速度模型定义)
- RayTracer: Ray tracing engine (射线追踪引擎)
- RayTracerConfig: Configuration for ray tracing (追踪配置)
"""

from .models import (
    VelocityModel,
    PhaseType
)

from .ray_tracer import (
    RayTracer,
    RayTracerConfig
)

from .rayinvr_wrapper import RayinvrWrapper
from .service import (
    RayinvrInputSpec,
    RayinvrResult,
    parse_rin_input_files,
    prepare_rayinvr_workdir,
    run_rayinvr,
    run_rayinvr_collect,
    validate_rayinvr_inputs,
)
from .tx_io import (
    TxDataset,
    TxPick,
    TxShotBlock,
    parse_tx_file_by_shot,
    read_tx_file,
    validate_tx_file,
    write_tx_file,
    write_tx_from_picks,
)
from .vin_io import (
    is_vin_file,
    load_edit_model,
    load_zelt_model,
    read_vin_dict,
    write_vin_dict,
)

__version__ = '0.1.0'
__author__ = 'Haibo Huang'

__all__ = [
    # Models
    'VelocityModel',
    'PhaseType',
    
    # Ray Tracing
    'RayTracer',
    'RayTracerConfig',
    
    # RAYINVR Interface
    'RayinvrWrapper',

    # Shared service (vedit / zplotpy / iphase)
    'RayinvrInputSpec',
    'RayinvrResult',
    'parse_rin_input_files',
    'prepare_rayinvr_workdir',
    'run_rayinvr',
    'run_rayinvr_collect',
    'validate_rayinvr_inputs',

    # tx.in / tx.out I/O
    'TxDataset',
    'TxPick',
    'TxShotBlock',
    'parse_tx_file_by_shot',
    'read_tx_file',
    'validate_tx_file',
    'write_tx_file',
    'write_tx_from_picks',

    # v.in I/O
    'is_vin_file',
    'load_edit_model',
    'load_zelt_model',
    'read_vin_dict',
    'write_vin_dict',
] 