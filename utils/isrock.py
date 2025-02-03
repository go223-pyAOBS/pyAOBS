"""
isrock.py - 基于地震速度模型参数的岩石识别模块

该模块提供了一系列用于岩石识别的类和函数，可以根据地震速度模型参数
（如P波速度、S波速度、密度等）来预测岩石类型。

Author: Haibo Huang
Date: 2024
"""

import numpy as np
from typing import Optional, Union, Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from .rocks import RockDatabase, RockProperties, Rock, RockClassifier

@dataclass
class CorrectionParameters:
    """温度压力校正参数类"""
    # 温度校正参数 (基于Christensen, 1979)
    vp_temp_coefficients = {
        'granite_granodiorite': -0.39e-3,  # 花岗岩-花岗闪长岩,花岗片麻岩,英云闪长片麻岩
        'gabbro_norite': -0.57e-3,   # 辉长岩-紫苏辉长岩-斜长岩
        'basalt': -0.39e-3,   # 玄武岩,变质玄武岩
        'slate': -0.40e-3,    # 板岩,千枚岩,石英云母片岩
        'mafic_granulite': -0.52e-3,  # 镁铁质麻粒岩,镁铁质石榴石麻粒岩
        'felsic_granulite': -0.49e-3, # 长英质麻粒岩,副麻粒岩
        'amphibolite': -0.55e-3,      # 角闪岩
        'anorthosite': -0.41e-3,      # 斜长岩
        'dunite': -0.56e-3,           # 橄榄岩,辉石岩,角闪岩
        'eclogite': -0.53e-3,         # 榴辉岩
        'serpentinite': -0.68e-3,     # 蛇纹岩
        'quartzite': -0.54e-3,        # 石英岩
        'marble': -0.41e-3,           # 大理岩
    }
    
    # 默认温度校正系数(当岩石类型未知时使用)
    vp_temp_alpha: float = -0.50e-3  # 平均值
    vs_temp_alpha: float = -0.40e-3  
    density_temp_alpha: float = -0.10e-3
    
    # 压力校正参数
    vp_pressure_beta: float = 0.0002
    vs_pressure_beta: float = 0.00015
    density_pressure_beta: float = 0.0001
    
    # 校正参数来源
    source: str = "laboratory"  # 'empirical' 或 'laboratory'
    lab_name: Optional[str] = "Christensen"  # 实验室名称
    reference: Optional[str] = "Christensen [1979, also unpublished data, 1980]"  # 参考文献
    
    def get_vp_temp_alpha(self, rock_type: str) -> float:
        """获取特定岩石类型的P波温度校正系数"""
        # 处理复合岩石类型
        rock_type = rock_type.lower()
        if 'granite' in rock_type or 'granodiorite' in rock_type or 'gneiss' in rock_type:
            return self.vp_temp_coefficients['granite_granodiorite']
        elif 'gabbro' in rock_type or 'norite' in rock_type:
            return self.vp_temp_coefficients['gabbro_norite']
        elif 'granulite' in rock_type:
            if 'mafic' in rock_type:
                return self.vp_temp_coefficients['mafic_granulite']
            else:
                return self.vp_temp_coefficients['felsic_granulite']
        return self.vp_temp_coefficients.get(rock_type, self.vp_temp_alpha)

class RockIdentifier:
    """岩石识别器类"""
    
    def __init__(self, database_file: str):
        """初始化岩石识别器
        
        Args:
            database_file: 岩石物性数据库文件路径
        """
        self.database = RockDatabase(database_file)
        self.classifier = RockClassifier(self.database)
        self.trained = False
        self.correction_params = {}  # 岩石类型到校正参数的映射
        
    def add_correction_parameters(self, rock_type: str, params: CorrectionParameters):
        """添加特定岩石类型的校正参数
        
        Args:
            rock_type: 岩石类型
            params: 校正参数
        """
        self.correction_params[rock_type] = params
        
    def _get_correction_parameters(self, rock_type: str) -> CorrectionParameters:
        """获取特定岩石类型的校正参数
        
        Args:
            rock_type: 岩石类型
            
        Returns:
            CorrectionParameters: 校正参数
        """
        return self.correction_params.get(rock_type, CorrectionParameters())
        
    def _normalize_to_standard_conditions(self, properties: RockProperties,
                                        rock_type: Optional[str] = None,
                                        target_temp: float = 25.0,
                                        target_pressure: float = 200.0) -> RockProperties:
        """将岩石物性参数归一化到标准温度压力条件
        
        Args:
            properties: 原始岩石物性
            rock_type: 岩石类型（用于获取特定的校正参数）
            target_temp: 目标温度 (°C)
            target_pressure: 目标压力 (MPa)
            
        Returns:
            RockProperties: 校正后的岩石物性
        """
        # 获取校正参数
        params = self._get_correction_parameters(rock_type) if rock_type else CorrectionParameters()
        
        # 获取原始条件
        original_temp = properties.temperature if properties.temperature is not None else 25.0
        original_pressure = properties.pressure if properties.pressure is not None else 200.0
        
        # 计算温度和压力变化
        delta_T = target_temp - original_temp
        delta_P = target_pressure - original_pressure
        
        # 校正P波速度
        corrected_vp = properties.vp
        if corrected_vp is not None:
            # 使用特定岩石类型的温度校正系数
            vp_temp_alpha = params.get_vp_temp_alpha(rock_type) if rock_type else params.vp_temp_alpha
            corrected_vp *= (1 + vp_temp_alpha * delta_T)
            corrected_vp *= (1 + params.vp_pressure_beta * delta_P)
        
        # 校正S波速度
        corrected_vs = properties.vs
        if corrected_vs is not None:
            corrected_vs *= (1 + params.vs_temp_alpha * delta_T)
            corrected_vs *= (1 + params.vs_pressure_beta * delta_P)
        
        # 校正密度
        corrected_density = properties.density
        if corrected_density is not None:
            corrected_density *= (1 + params.density_temp_alpha * delta_T)
            corrected_density *= (1 + params.density_pressure_beta * delta_P)
        
        return RockProperties(
            vp=corrected_vp,
            vs=corrected_vs,
            density=corrected_density,
            porosity=properties.porosity,
            temperature=target_temp,
            pressure=target_pressure,
            fluid_saturation=properties.fluid_saturation
        )