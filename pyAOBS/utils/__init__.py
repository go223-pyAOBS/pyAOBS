"""
Utility functions for the pyAOBS package.

This module provides various utility functions for seismic data processing:
- Logging configuration
- Rock property utilities
- General helper functions
"""

import logging
from .rocks import RockProperties, load_rock_database, Rock, RockDatabase
from .isrock import identify_rock_type
from .empirical_formulas import (
    # Correction parameters
    CorrectionParameters,
    DEFAULT_CORRECTION,
    # Density formulas
    calculate_density,
    calculate_density_gardner,
    calculate_density_nafe_drake,
    calculate_density_brocher,
    calculate_density_castagna,
    calculate_density_lindseth,
    # S-wave velocity formulas
    calculate_vs,
    calculate_vs_brocher,
    calculate_vs_castagna,
    # Inverse calculation
    calculate_vp_from_vs_brocher,
    # Correction functions
    correct_velocity,
    correct_velocity_pressure,
    correct_velocity_temperature,
    # Elastic moduli
    calculate_elastic_moduli,
    # Utility functions
    calculate_vp_vs_ratio,
    calculate_poisson_ratio,
    calculate_vp_vs_ratio_from_poisson,
    # Serpentinization relationships
    calculate_serpentinization_from_water_content,
    calculate_water_content_from_serpentinization,
    calculate_vp_from_serpentinization,
    calculate_vs_from_serpentinization,
    calculate_serpentinization_from_vp,
    calculate_serpentinization_from_vs,
    # Water content and porosity relationships
    calculate_water_content_from_porosity,
    calculate_porosity_from_water_content,
    # DEM model
    calculate_dem_effective_moduli,
    calculate_velocity_from_moduli,
    calculate_dem_velocity,
    # Depth-based calculations
    calculate_pressure_from_depth,
    calculate_temperature_from_depth,
)

# 导入简化接口（推荐使用）
try:
    from .simple_rock_classifier import (
        SimpleRockClassifier,
        classify_velocity_model,
        classify_from_file
    )
    SIMPLE_CLASSIFIER_AVAILABLE = True
except ImportError:
    SIMPLE_CLASSIFIER_AVAILABLE = False
    SimpleRockClassifier = None
    classify_velocity_model = None
    classify_from_file = None

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.
    
    Args:
        name (str): Name of the logger
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only add handler if logger doesn't already have handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Set level to INFO if not set
        if not logger.level:
            logger.setLevel(logging.INFO)
    
    return logger

__all__ = [
    'get_logger',
    'RockProperties',
    'Rock',
    'RockDatabase',
    'load_rock_database',
    'identify_rock_type',
    # 简化接口（推荐）
    'SimpleRockClassifier',
    'classify_velocity_model',
    'classify_from_file',
    # 经验公式库
    'CorrectionParameters',
    'DEFAULT_CORRECTION',
    'calculate_density',
    'calculate_density_gardner',
    'calculate_density_nafe_drake',
    'calculate_density_brocher',
    'calculate_density_castagna',
    'calculate_density_lindseth',
    'calculate_vs',
    'calculate_vs_brocher',
    'calculate_vs_castagna',
    'calculate_vp_from_vs_brocher',
    'correct_velocity',
    'correct_velocity_pressure',
    'correct_velocity_temperature',
    'calculate_elastic_moduli',
    'calculate_vp_vs_ratio',
    'calculate_poisson_ratio',
    'calculate_vp_vs_ratio_from_poisson',
    # Serpentinization relationships
    'calculate_serpentinization_from_water_content',
    'calculate_water_content_from_serpentinization',
    'calculate_vp_from_serpentinization',
    'calculate_vs_from_serpentinization',
    'calculate_serpentinization_from_vp',
    'calculate_serpentinization_from_vs',
    # Water content and porosity relationships
    'calculate_water_content_from_porosity',
    'calculate_porosity_from_water_content',
    # DEM model
    'calculate_dem_effective_moduli',
    'calculate_velocity_from_moduli',
    'calculate_dem_velocity',
    # Depth-based calculations
    'calculate_pressure_from_depth',
    'calculate_temperature_from_depth',
] 