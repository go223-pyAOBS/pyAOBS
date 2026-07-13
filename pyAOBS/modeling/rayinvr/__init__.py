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
    'RayinvrWrapper'
] 