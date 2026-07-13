"""
rocks.py - Python implementation of rock properties

This module provides classes and functions for handling rock properties,
including reading/writing velocity model files, AI training rock properties, and 
generating rock properties. It is designed to work with the isrock.py module.

Author: Haibo Huang
Date: 2025
"""

import numpy as np
from typing import Optional, Union, Dict, List, Tuple, Any
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
from enum import Enum, auto
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

# 导入经验公式库
try:
    from .empirical_formulas import (
        calculate_density as _calculate_density,
        calculate_vs as _calculate_vs,
        calculate_elastic_moduli as _calculate_elastic_moduli,
        correct_velocity as _correct_velocity,
        correct_velocity_pressure as _correct_velocity_pressure,
        correct_velocity_temperature as _correct_velocity_temperature
    )
except ImportError:
    from empirical_formulas import (
        calculate_density as _calculate_density,
        calculate_vs as _calculate_vs,
        calculate_elastic_moduli as _calculate_elastic_moduli,
        correct_velocity as _correct_velocity,
        correct_velocity_pressure as _correct_velocity_pressure,
        correct_velocity_temperature as _correct_velocity_temperature
    )

class TectonicSetting(Enum):
    """构造背景枚举类"""
    OROGENIC_BELT = auto()      # 造山带
    PASSIVE_MARGIN = auto()     # 被动陆缘
    SUBDUCTION_ZONE = auto()    # 俯冲带
    RIFT = auto()               # 裂谷
    CRATON = auto()             # 克拉通
    BASIN = auto()              # 沉积盆地
    VOLCANIC_ARC = auto()       # 火山弧
    OCEANIC_CRUST = auto()      # 大洋地壳
    CONTINENTAL_CRUST = auto()  # 大陆地壳
    UNKNOWN = auto()            # 未知

@dataclass
class RockProperties:
    """岩石物性基本参数类"""
    vp: float  # P波速度 (km/s)
    vs: Optional[float] = None  # S波速度 (km/s)
    density: Optional[float] = None  # 密度 (g/cm³)
    porosity: Optional[float] = None  # 孔隙度 (0-1)
    temperature: Optional[float] = 25    # 温度 (°C)
    pressure: Optional[float] =  200  # 压力 (MPa)
    fluid_saturation: Optional[float] = 0.5  # 流体饱和度 (0-1)
    tectonic_setting: Optional[TectonicSetting] = TectonicSetting.UNKNOWN  # 构造背景
    
@dataclass
class RockMeasurement:
    """岩石测量数据类，包含测量信息和质量控制"""
    properties: RockProperties  # 岩石物性参数
    source: str  # 数据来源（实验室/研究者）
    method: str  # 测量方法
    date: Optional[str] = None  # 测量日期
    uncertainty: Optional[Dict[str, float]] = None  # 测量不确定度
    quality_score: Optional[float] = None  # 数据质量分数
    notes: Optional[str] = None  # 备注信息
    location: Optional[str] = None  # 采样位置
    tectonic_description: Optional[str] = None  # 构造背景描述

class Rock:
    """岩石类，包含岩石物性和相关计算方法"""
    
    def __init__(self, name: str, properties: RockProperties):
        """初始化岩石实例
        
        Args:
            name: 岩石名称
            properties: 岩石物性参数
        """
        self.name = name
        self.properties = properties
        
    @property
    def vp_vs_ratio(self) -> Optional[float]:
        """计算Vp/Vs比值"""
        if self.properties.vs is not None and self.properties.vs > 0:
            return self.properties.vp / self.properties.vs
        return None
        
    @property
    def poisson_ratio(self) -> Optional[float]:
        """计算泊松比"""
        vp_vs = self.vp_vs_ratio
        if vp_vs is not None:
            return (vp_vs**2 - 2) / (2 * (vp_vs**2 - 1))
        return None
    
    def calculate_density(self, method: str = 'gardner') -> float:
        """使用经验公式计算密度
        
        Args:
            method: 使用的经验公式，可选：
                   'gardner' - Gardner公式 (适用于沉积岩)
                   'nafe_drake' - Nafe-Drake公式 (适用于海洋沉积物)
                   'brocher' - Brocher公式 (适用于地壳岩石)
                   
        Returns:
            float: 计算得到的密度 (g/cm³)
        """
        # 使用统一的经验公式库
        return _calculate_density(self.properties.vp, method=method)
    
    def calculate_vs(self, method: str = 'brocher') -> float:
        """使用经验公式计算S波速度
        
        Args:
            method: 使用的经验公式，可选：
                   'brocher' - Brocher公式
                   'castagna' - Castagna公式 (泥岩)
                   
        Returns:
            float: 计算得到的S波速度 (km/s)
        """
        # 使用统一的经验公式库
        return _calculate_vs(self.properties.vp, method=method)
    
    def calculate_elastic_moduli(self) -> Dict[str, float]:
        """计算弹性模量
        
        Returns:
            Dict[str, float]: 包含以下弹性参数：
                - bulk_modulus: 体积模量 (GPa)
                - shear_modulus: 剪切模量 (GPa)
                - young_modulus: 杨氏模量 (GPa)
                - lame_lambda: 拉梅常数λ (GPa)
        """
        if self.properties.vs is None:
            self.properties.vs = self.calculate_vs()
            
        if self.properties.density is None:
            self.properties.density = self.calculate_density()
            
        # 使用统一的经验公式库
        moduli = _calculate_elastic_moduli(
            self.properties.vp,
            self.properties.vs,
            self.properties.density
        )
        
        # 转换为float（如果是标量）
        return {k: float(v) for k, v in moduli.items()}
    
    def calculate_temperature_effect(self, 
                                  reference_vp: float,
                                  delta_T: float,
                                  alpha: Optional[float] = None) -> float:
        """计算温度对P波速度的影响
        
        使用统一的经验公式库进行校正。
        注意：delta_T应该是 (原始温度 - 目标温度)，公式使用 1 - alpha * delta_T。
        isrock.py中alpha定义为负数（-0.50e-4），公式使用减号。
        
        Args:
            reference_vp: 参考温度下的P波速度 (km/s)
            delta_T: 温度变化 (°C)，应该是 (原始温度 - 目标温度)
            alpha: 温度系数，如果为None则使用默认值-0.50e-4/°C（与CorrectionParameters一致）
            
        Returns:
            float: 温度校正后的P波速度 (km/s)
        """
        # 使用统一的经验公式库
        # 注意：correct_velocity_temperature需要原始温度和目标温度，而不是delta_T
        # 这里为了保持接口兼容性，我们仍然使用delta_T
        # 假设参考温度是目标温度，原始温度 = 目标温度 + delta_T
        target_temp = 25.0  # 默认目标温度
        original_temp = target_temp + delta_T
        return _correct_velocity_temperature(
            reference_vp,
            temperature=original_temp,
            target_temperature=target_temp,
            is_s_wave=False
        )
    
    def calculate_pressure_effect(self,
                               reference_vp: float,
                               delta_P: float,
                               beta: Optional[float] = None) -> float:
        """计算压力对P波速度的影响
        
        使用统一的经验公式库进行校正。
        注意：delta_P应该是 (目标压力 - 原始压力)，压力升高速度增加。
        
        Args:
            reference_vp: 参考压力下的P波速度 (km/s)
            delta_P: 压力变化 (MPa)，应该是 (目标压力 - 原始压力)
            beta: 压力系数，如果为None则使用默认值0.0002/MPa（与CorrectionParameters一致）
            
        Returns:
            float: 压力校正后的P波速度 (km/s)
        """
        # 使用统一的经验公式库
        # 注意：correct_velocity_pressure需要原始压力和目标压力，而不是delta_P
        # 这里为了保持接口兼容性，我们仍然使用delta_P
        # 假设参考压力是原始压力，目标压力 = 原始压力 + delta_P
        original_pressure = 200.0  # 默认原始压力
        target_pressure = original_pressure + delta_P
        return _correct_velocity_pressure(
            reference_vp,
            pressure=original_pressure,
            target_pressure=target_pressure,
            is_s_wave=False
        )
    
    def __str__(self) -> str:
        """返回岩石信息的字符串表示"""
        props = []
        for key, value in self.properties.__dict__.items():
            if value is not None:
                props.append(f"{key}: {value}")
        return f"Rock: {self.name}\n" + "\n".join(props)

def create_common_rock(rock_type: str) -> Rock:
    """创建常见岩石类型的实例
    
    Args:
        rock_type: 岩石类型，可选：
                  'sandstone' - 砂岩
                  'limestone' - 石灰岩
                  'granite' - 花岗岩
                  'basalt' - 玄武岩
                  'shale' - 页岩
                  
    Returns:
        Rock: 对应的岩石实例
    """
    rock_properties = {
        'sandstone': RockProperties(vp=3.5, vs=2.0, density=2.35),
        'limestone': RockProperties(vp=4.5, vs=2.5, density=2.55),
        'granite': RockProperties(vp=5.5, vs=3.0, density=2.65),
        'basalt': RockProperties(vp=6.0, vs=3.5, density=2.85),
        'shale': RockProperties(vp=3.0, vs=1.5, density=2.40)
    }
    
    if rock_type not in rock_properties:
        raise ValueError(f"不支持的岩石类型: {rock_type}")
        
    return Rock(rock_type, rock_properties[rock_type])

class RockQualityControl:
    """岩石物性数据质量控制类"""
    
    @staticmethod
    def check_measurement_consistency(measurements: List[RockMeasurement]) -> Dict[str, float]:
        """检查同一岩石不同测量结果的一致性
        
        Args:
            measurements: 测量结果列表
            
        Returns:
            Dict[str, float]: 各物性参数的变异系数(CV)
        """
        cv_dict = {}
        
        # 获取所有可能的物性参数
        all_props = set()
        for m in measurements:
            for key, value in m.properties.__dict__.items():
                if value is not None and not isinstance(value, TectonicSetting):
                    all_props.add(key)
        
        # 计算每个参数的变异系数
        for prop in all_props:
            values = [getattr(m.properties, prop) for m in measurements 
                     if getattr(m.properties, prop) is not None]
            if values and not isinstance(values[0], TectonicSetting):
                mean = np.mean(values)
                std = np.std(values)
                cv = std / mean if mean != 0 else float('inf')
                cv_dict[prop] = cv
                
        return cv_dict
    
    @staticmethod
    def calculate_quality_score(measurement: RockMeasurement) -> float:
        """计算测量数据的质量分数
        
        Args:
            measurement: 测量数据
            
        Returns:
            float: 质量分数 (0-1)
        """
        score = 1.0
        deductions = []
        
        # 检查必要参数是否存在
        required_props = ['vp', 'vs', 'density']
        for prop in required_props:
            if getattr(measurement.properties, prop) is None:
                deductions.append(0.2)
        
        # 检查不确定度信息
        if measurement.uncertainty is None:
            deductions.append(0.1)
        
        # 检查测量方法和日期
        if measurement.method is None:
            deductions.append(0.1)
        if measurement.date is None:
            deductions.append(0.05)
            
        # 应用扣分
        for deduction in deductions:
            score -= deduction
            
        return max(0.0, min(1.0, score))

class RockDatabase:
    """岩石物性数据库类，用于管理实验室测量的岩石物性数据"""
    
    def __init__(self, database_file: Optional[str] = None):
        """初始化岩石数据库
        
        Args:
            database_file: Excel文件路径，包含岩石物性数据
        """
        self.rocks: Dict[str, List[RockMeasurement]] = {}
        if database_file:
            self.load_from_excel(database_file)
            
    def load_from_excel(self, file_path: str) -> None:
        """从Excel文件加载岩石物性数据
        
        Args:
            file_path: Excel文件路径
            
        Excel文件格式要求：
            - 必须包含的列：rock_type, vp, source, method
            - 可选列：vs, density, porosity, temperature, pressure, 
                    fluid_saturation, date, uncertainty, notes,
                    location, tectonic_setting, tectonic_description
        """
        df = pd.read_excel(file_path)
        required_cols = ['rock_type', 'vp', 'source', 'method']
        optional_cols = ['vs', 'density', 'porosity', 'temperature', 
                        'pressure', 'fluid_saturation', 'date', 'notes',
                        'location', 'tectonic_setting', 'tectonic_description']
        
        # 检查必需列
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Excel文件必须包含以下列: {required_cols}")
            
        # 处理每一行数据
        for _, row in df.iterrows():
            # 创建RockProperties实例
            props = {'vp': row['vp']}
            for col in optional_cols:
                if col in df.columns and not pd.isna(row[col]):
                    if col == 'tectonic_setting':
                        try:
                            props[col] = TectonicSetting[row[col].upper()]
                        except (KeyError, AttributeError):
                            props[col] = TectonicSetting.UNKNOWN
                    else:
                        props[col] = row[col]
            rock_props = RockProperties(**props)
            
            # 创建RockMeasurement实例
            measurement = RockMeasurement(
                properties=rock_props,
                source=row['source'],
                method=row['method'],
                date=row['date'] if 'date' in df.columns and not pd.isna(row['date']) else None,
                notes=row['notes'] if 'notes' in df.columns and not pd.isna(row['notes']) else None,
                location=row['location'] if 'location' in df.columns and not pd.isna(row['location']) else None,
                tectonic_description=row['tectonic_description'] if 'tectonic_description' in df.columns and not pd.isna(row['tectonic_description']) else None
            )
            
            # 计算质量分数
            measurement.quality_score = RockQualityControl.calculate_quality_score(measurement)
            
            # 将数据添加到数据库
            rock_type = row['rock_type']
            if rock_type not in self.rocks:
                self.rocks[rock_type] = []
            self.rocks[rock_type].append(measurement)
            
    def get_consensus_properties(self, rock_type: str, 
                               min_quality_score: float = 0.7,
                               max_cv: float = 0.2) -> Optional[RockProperties]:
        """获取同一岩石不同测量结果的共识值
        
        Args:
            rock_type: 岩石类型
            min_quality_score: 最低质量分数
            max_cv: 最大允许变异系数
            
        Returns:
            Optional[RockProperties]: 共识物性值，如果数据质量不足则返回None
        """
        if rock_type not in self.rocks:
            return None
            
        # 筛选高质量数据
        quality_measurements = [m for m in self.rocks[rock_type] 
                              if m.quality_score >= min_quality_score]
        
        if not quality_measurements:
            return None
            
        # 检查数据一致性
        cv_dict = RockQualityControl.check_measurement_consistency(quality_measurements)
        
        # 计算共识值
        consensus_props = {}
        for prop, cv in cv_dict.items():
            if cv <= max_cv:
                values = [getattr(m.properties, prop) for m in quality_measurements 
                         if getattr(m.properties, prop) is not None]
                if values:
                    # 使用加权平均，权重为质量分数
                    weights = [m.quality_score for m in quality_measurements 
                             if getattr(m.properties, prop) is not None]
                    consensus_props[prop] = np.average(values, weights=weights)
                    
        if not consensus_props:
            return None
            
        return RockProperties(**consensus_props)
        
    def get_measurement_sources(self, rock_type: str) -> List[str]:
        """获取特定岩石类型的所有数据来源
        
        Args:
            rock_type: 岩石类型
            
        Returns:
            List[str]: 数据来源列表
        """
        if rock_type not in self.rocks:
            return []
        return list(set(m.source for m in self.rocks[rock_type]))
        
    def plot_measurement_comparison(self, rock_type: str, 
                                  property_name: str,
                                  output_file: Optional[str] = None) -> None:
        """绘制不同来源的测量结果对比图
        
        Args:
            rock_type: 岩石类型
            property_name: 物性参数名称
            output_file: 输出文件路径
        """
        if rock_type not in self.rocks:
            raise ValueError(f"数据库中不存在岩石类型: {rock_type}")
            
        import seaborn as sns
        
        # 收集数据
        sources = []
        values = []
        qualities = []
        
        for m in self.rocks[rock_type]:
            value = getattr(m.properties, property_name)
            if value is not None:
                sources.append(m.source)
                values.append(value)
                qualities.append(m.quality_score)
                
        if not values:
            raise ValueError(f"没有找到属性 {property_name} 的有效数据")
            
        # 创建图形
        plt.figure(figsize=(10, 6))
        
        # 绘制箱线图
        sns.boxplot(x=sources, y=values)
        
        # 添加散点，大小表示质量分数
        sizes = np.array(qualities) * 100
        plt.scatter(range(len(sources)), values, s=sizes, alpha=0.6)
        
        plt.title(f"{rock_type} - {property_name} Measurements by Source")
        plt.xlabel("Source")
        plt.ylabel(property_name)
        plt.xticks(rotation=45)
        
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def add_measurement(self, rock_type: str, measurement: RockMeasurement) -> None:
        """添加单个岩石测量数据
        
        Args:
            rock_type: 岩石类型
            measurement: 岩石测量数据
        """
        # 计算质量分数
        if measurement.quality_score is None:
            measurement.quality_score = RockQualityControl.calculate_quality_score(measurement)
            
        # 添加到数据库
        if rock_type not in self.rocks:
            self.rocks[rock_type] = []
        self.rocks[rock_type].append(measurement)
        
    def add_measurements_from_excel(self, file_path: str, merge_strategy: str = 'append') -> None:
        """从Excel文件添加新的测量数据
        
        Args:
            file_path: Excel文件路径
            merge_strategy: 合并策略，可选：
                          'append' - 直接添加新数据
                          'update' - 更新已存在的数据（基于source和date）
                          'replace' - 替换已存在的岩石类型的所有数据
        """
        new_db = RockDatabase()
        new_db.load_from_excel(file_path)
        
        for rock_type, measurements in new_db.rocks.items():
            if merge_strategy == 'replace' and rock_type in self.rocks:
                # 替换已存在的数据
                self.rocks[rock_type] = measurements
                
            elif merge_strategy == 'update':
                # 更新已存在的数据
                if rock_type not in self.rocks:
                    self.rocks[rock_type] = measurements
                else:
                    existing_measurements = {
                        (m.source, m.date): m for m in self.rocks[rock_type]
                        if m.date is not None
                    }
                    
                    for new_m in measurements:
                        key = (new_m.source, new_m.date)
                        if key in existing_measurements:
                            # 更新已存在的测量数据
                            idx = self.rocks[rock_type].index(existing_measurements[key])
                            self.rocks[rock_type][idx] = new_m
                        else:
                            # 添加新的测量数据
                            self.rocks[rock_type].append(new_m)
                            
            else:  # 'append'
                # 直接添加新数据
                if rock_type not in self.rocks:
                    self.rocks[rock_type] = []
                self.rocks[rock_type].extend(measurements)
                
    def save_to_excel(self, file_path: str) -> None:
        """将数据库保存到Excel文件
        
        Args:
            file_path: 输出Excel文件路径
        """
        data = []
        for rock_type, measurements in self.rocks.items():
            for m in measurements:
                row = {
                    'rock_type': rock_type,
                    'source': m.source,
                    'method': m.method,
                    'date': m.date,
                    'quality_score': m.quality_score,
                    'notes': m.notes
                }
                
                # 添加物性参数
                for key, value in m.properties.__dict__.items():
                    if value is not None:
                        row[key] = value
                        
                # 添加不确定度
                if m.uncertainty:
                    for key, value in m.uncertainty.items():
                        row[f'{key}_uncertainty'] = value
                        
                data.append(row)
                
        df = pd.DataFrame(data)
        df.to_excel(file_path, index=False)
        
    def remove_measurement(self, rock_type: str, source: str, date: Optional[str] = None) -> bool:
        """删除特定的测量数据
        
        Args:
            rock_type: 岩石类型
            source: 数据来源
            date: 测量日期（可选）
            
        Returns:
            bool: 是否成功删除
        """
        if rock_type not in self.rocks:
            return False
            
        initial_length = len(self.rocks[rock_type])
        if date:
            self.rocks[rock_type] = [
                m for m in self.rocks[rock_type]
                if not (m.source == source and m.date == date)
            ]
        else:
            self.rocks[rock_type] = [
                m for m in self.rocks[rock_type]
                if m.source != source
            ]
            
        return len(self.rocks[rock_type]) < initial_length
        
    def filter_measurements(self, 
                          rock_type: str,
                          min_quality_score: float = 0.0,
                          source: Optional[str] = None,
                          date_range: Optional[Tuple[str, str]] = None) -> List[RockMeasurement]:
        """筛选特定条件的测量数据
        
        Args:
            rock_type: 岩石类型
            min_quality_score: 最低质量分数
            source: 数据来源
            date_range: 日期范围元组 (start_date, end_date)
            
        Returns:
            List[RockMeasurement]: 符合条件的测量数据列表
        """
        if rock_type not in self.rocks:
            return []
            
        filtered = self.rocks[rock_type]
        
        # 按质量分数筛选
        filtered = [m for m in filtered if m.quality_score >= min_quality_score]
        
        # 按来源筛选
        if source:
            filtered = [m for m in filtered if m.source == source]
            
        # 按日期范围筛选
        if date_range:
            from datetime import datetime
            start_date = datetime.strptime(date_range[0], '%Y-%m-%d')
            end_date = datetime.strptime(date_range[1], '%Y-%m-%d')
            
            filtered = [
                m for m in filtered
                if m.date and start_date <= datetime.strptime(m.date, '%Y-%m-%d') <= end_date
            ]
            
        return filtered
        
    def get_database_summary(self) -> Dict[str, Dict[str, Union[int, List[str], float]]]:
        """获取数据库摘要信息
        
        Returns:
            Dict: 包含每种岩石类型的统计信息
        """
        summary = {}
        for rock_type, measurements in self.rocks.items():
            stats = {
                'total_measurements': len(measurements),
                'sources': list(set(m.source for m in measurements)),
                'avg_quality_score': np.mean([m.quality_score for m in measurements if m.quality_score]),
                'date_range': [
                    min((m.date for m in measurements if m.date), default=None),
                    max((m.date for m in measurements if m.date), default=None)
                ]
            }
            summary[rock_type] = stats
            
        return summary

    def get_rocks_by_tectonic_setting(self, setting: TectonicSetting) -> Dict[str, List[RockMeasurement]]:
        """获取特定构造背景下的岩石数据
        
        Args:
            setting: 构造背景
            
        Returns:
            Dict[str, List[RockMeasurement]]: 岩石类型到测量数据的映射
        """
        result = {}
        for rock_type, measurements in self.rocks.items():
            filtered = [m for m in measurements 
                       if m.properties.tectonic_setting == setting]
            if filtered:
                result[rock_type] = filtered
        return result
        
    def get_tectonic_distribution(self) -> Dict[str, Dict[TectonicSetting, int]]:
        """获取各岩石类型在不同构造背景下的分布
        
        Returns:
            Dict[str, Dict[TectonicSetting, int]]: 岩石类型到构造背景分布的映射
        """
        distribution = {}
        for rock_type, measurements in self.rocks.items():
            setting_counts = {}
            for m in measurements:
                setting = m.properties.tectonic_setting
                setting_counts[setting] = setting_counts.get(setting, 0) + 1
            distribution[rock_type] = setting_counts
        return distribution
        
    def plot_tectonic_distribution(self, output_file: Optional[str] = None) -> None:
        """绘制构造背景分布图
        
        Args:
            output_file: 输出文件路径
        """
        distribution = self.get_tectonic_distribution()
        
        # 准备数据
        rock_types = list(distribution.keys())
        settings = list(TectonicSetting)
        data = np.zeros((len(rock_types), len(settings)))
        
        for i, rock_type in enumerate(rock_types):
            for j, setting in enumerate(settings):
                data[i, j] = distribution[rock_type].get(setting, 0)
                
        # 创建热力图
        plt.figure(figsize=(12, 8))
        sns.heatmap(data, 
                   xticklabels=[s.name for s in settings],
                   yticklabels=rock_types,
                   annot=True, 
                   fmt='g',
                   cmap='YlOrRd')
        
        plt.title('Rock Types Distribution in Different Tectonic Settings')
        plt.xlabel('Tectonic Setting')
        plt.ylabel('Rock Type')
        plt.xticks(rotation=45)
        
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

class RockClassifier:
    """岩石分类器类"""

    @staticmethod
    def _read_table_with_fallback(file_path: Path) -> pd.DataFrame:
        """读取数据库表格，支持 Excel/CSV/TXT。"""
        suffix = file_path.suffix.lower()
        if suffix in {'.xlsx', '.xlsm'}:
            return pd.read_excel(file_path)

        if suffix in {'.txt', '.tsv'}:
            for enc in ('utf-8-sig', 'utf-8', 'gb18030', 'latin-1'):
                try:
                    return pd.read_csv(file_path, sep=None, engine='python', encoding=enc)
                except Exception:
                    continue
            raise ValueError(f"无法解析文本数据库文件: {file_path}")

        for enc in ('utf-8-sig', 'utf-8', 'gb18030', 'latin-1'):
            try:
                return pd.read_csv(file_path, encoding=enc)
            except Exception:
                continue
        raise ValueError(f"无法解析CSV数据库文件: {file_path}")
    
    def __init__(
        self,
        database,
        use_felsic_mafic: bool = False,
        use_rock_facies: bool = False,
        use_sio2: bool = False,
    ):
        """初始化分类器
        
        Args:
            database: str或pd.DataFrame，岩石数据库文件路径或DataFrame
        """
        if isinstance(database, (str, Path)):
            self.rock_database = self._read_table_with_fallback(Path(database))
        else:
            self.rock_database = database
        self.use_felsic_mafic = bool(use_felsic_mafic)
        self.use_rock_facies = bool(use_rock_facies)
        self.use_sio2 = bool(use_sio2)
            
        # 检查必需的列是否存在，支持多种列名格式和中文列名
        # 定义列名映射规则（标准列名 -> 可能的变体）
        col_name_variants = {
            'vp': ['vp', 'Vp', 'VP', 'v_p', 'v_p (km/s)', 'vp (km/s)', 'vp(km/s)', 'p-wave', 'pwave', 'p波', 'p波速度', 'v', 'V'],
            'vs': ['vs', 'Vs', 'VS', 'v_s', 'v_s (km/s)', 'vs (km/s)', 'vs(km/s)', 's-wave', 'swave', 's波', 's波速度'],
            'density': ['density', 'Density', 'DENSITY', 'ρ', 'rho', 'Rho', 'RHO', 'density (g/cm³)', 'density(g/cm³)', '密度', '密度 (g/cm³)'],
            'pressure': ['pressure', 'Pressure', 'PRESSURE', 'P', 'p', '压力', '压力 (MPa)', 'pressure (MPa)'],
            'rock_type': ['rock_type', 'Rock Type', 'RockType', 'ROCK_TYPE', 'rocktype', 'rock type', 'type', 'Type', 'TYPE', '岩石类型', '岩性', '中文', '中文名'],
            'felsic_or_mafic': ['felsic_or_mafic', 'felsic or mafic', 'composition', 'composition_class'],
            'rock_facies': ['rock_facies', 'rock facies', 'facies'],
            'sio2_wt': ['sio2_wt', 'sio2', 'sio2, wt.%', 'sio2 wt']
        }
        
        col_mapping = {}
        available_cols_lower = [str(col).lower().strip() for col in self.rock_database.columns]
        available_cols = list(self.rock_database.columns)
        
        # 创建列名映射（支持多种变体）
        for std_col, variants in col_name_variants.items():
            found = False
            # 首先尝试精确匹配（大小写不敏感）
            for variant in variants:
                variant_lower = variant.lower().strip()
                if variant in available_cols:
                    col_mapping[std_col] = variant
                    found = True
                    break
                elif variant_lower in available_cols_lower:
                    idx = available_cols_lower.index(variant_lower)
                    col_mapping[std_col] = available_cols[idx]
                    found = True
                    break
            
            # 如果没找到，对于density和pressure列，允许缺失（将使用经验公式估算）
            if not found:
                if std_col in ['density', 'pressure', 'felsic_or_mafic', 'rock_facies', 'sio2_wt']:
                    # 密度和压力列可选，如果缺失将在后续处理中估算
                    col_mapping[std_col] = None
                else:
                    # 其他列是必需的
                    raise ValueError(
                        f"Database missing required column: '{std_col}'. "
                        f"Tried variants: {variants}. "
                        f"Available columns: {list(self.rock_database.columns)}"
                    )
        
        # 使用映射后的列名
        vp_col = col_mapping['vp']
        vs_col = col_mapping['vs']
        density_col = col_mapping.get('density')  # 可能为None
        pressure_col = col_mapping.get('pressure')  # 可能为None
        rock_type_col = col_mapping['rock_type']
        felsic_or_mafic_col = col_mapping.get('felsic_or_mafic')
        rock_facies_col = col_mapping.get('rock_facies')
        sio2_col = col_mapping.get('sio2_wt')
        
        # 目标校正条件（标准条件）
        self.target_pressure = 200.0  # MPa
        self.target_temperature = 25.0  # °C
        
        # 检查是否有足够的非空数据
        # 对于缺失的vs或density，使用经验公式估算
        valid_data = self.rock_database.copy()
        
        # 如果vs缺失，使用Brocher公式估算（使用经验公式库）
        if vs_col and valid_data[vs_col].isna().any():
            mask = valid_data[vs_col].isna()
            for idx in valid_data[mask].index:
                vp_val = valid_data.loc[idx, vp_col]
                valid_data.loc[idx, vs_col] = _calculate_vs(vp_val, method='brocher')
        
        # 如果density列不存在或缺失，创建新列并使用Gardner公式估算
        if density_col is None:
            # 创建density列
            valid_data['density'] = np.nan
            density_col = 'density'
        
        # 如果density缺失，使用Gardner公式估算（使用经验公式库）
        if density_col in valid_data.columns:
            if valid_data[density_col].isna().any() or valid_data[density_col].isna().all():
                mask = valid_data[density_col].isna()
                for idx in valid_data[mask].index:
                    vp_val = valid_data.loc[idx, vp_col]
                    valid_data.loc[idx, density_col] = _calculate_density(vp_val, method='gardner')
        else:
            # 如果density列完全不存在，为所有行估算（使用经验公式库）
            valid_data[density_col] = np.nan
            for idx in valid_data.index:
                vp_val = valid_data.loc[idx, vp_col]
                valid_data.loc[idx, density_col] = _calculate_density(vp_val, method='gardner')
        
        # 处理压力列：如果不存在，假设为25°C（默认温度），压力需要根据深度估算
        # 但数据库中没有深度信息，所以假设压力为200MPa（标准条件）
        if pressure_col is None:
            valid_data['pressure'] = self.target_pressure  # 假设已经是标准条件
            pressure_col = 'pressure'
        elif pressure_col in valid_data.columns:
            # 如果压力列存在但有空值，用目标压力填充
            # 使用推荐的方式避免FutureWarning
            if valid_data[pressure_col].isna().any():
                valid_data[pressure_col] = valid_data[pressure_col].fillna(self.target_pressure)
        else:
            valid_data['pressure'] = self.target_pressure
            pressure_col = 'pressure'
        
        # 假设数据库中的温度都是25°C（默认测量温度）
        # 如果数据库中有温度列，可以读取，否则假设为25°C
        valid_data['temperature'] = self.target_temperature  # 假设测量温度为25°C
        
        # 对数据库中的速度值进行压力和温度校正到标准条件（200MPa, 25°C）
        # 校正公式（基于Christensen, 1979）：
        # 压力校正: V_corrected = V_original * (1 + beta * delta_P)
        #   其中 delta_P = target_pressure - original_pressure
        #   beta = 0.0002/MPa (P波) 或 0.00015/MPa (S波)
        # 温度校正: V_corrected = V_original * (1 - alpha * delta_T)
        #   其中 delta_T = original_temperature - target_temperature
        #   alpha = 0.50e-4/°C (P波，即0.00005/°C) 或 0.40e-4/°C (S波，即0.00004/°C)
        #   注意：温度升高速度降低，所以使用减号
        
        # P波速度校正系数（与isrock.py中的CorrectionParameters保持一致）
        beta_p = 0.0002  # P波压力系数 (1/MPa)
        alpha_p = -0.50e-4  # P波温度系数 (1/°C)，注意：isrock.py中定义为负数
        
        for idx in valid_data.index:
            vp_orig = valid_data.loc[idx, vp_col]
            p_orig = valid_data.loc[idx, pressure_col]
            t_orig = valid_data.loc[idx, 'temperature']
            
            # 压力校正
            delta_p = self.target_pressure - p_orig
            vp_pressure_corrected = vp_orig * (1 + beta_p * delta_p)
            
            # 温度校正（与isrock.py中的CorrectionParameters一致）
        # 注意：isrock.py中alpha定义为负数，公式使用减号
            delta_t = t_orig - self.target_temperature  # 原始温度 - 目标温度
            vp_corrected = vp_pressure_corrected * (1 - alpha_p * delta_t)
            
            valid_data.loc[idx, vp_col] = vp_corrected
        
        # S波速度校正（如果存在）
        if vs_col:
            beta_s = 0.00015  # S波压力系数 (1/MPa)
            alpha_s = -0.40e-4  # S波温度系数 (1/°C)，注意：isrock.py中定义为负数
            
            for idx in valid_data.index:
                vs_orig = valid_data.loc[idx, vs_col]
                p_orig = valid_data.loc[idx, pressure_col]
                t_orig = valid_data.loc[idx, 'temperature']
                
                # 压力校正
                delta_p = self.target_pressure - p_orig
                vs_pressure_corrected = vs_orig * (1 + beta_s * delta_p)
                
                # 温度校正（与isrock.py中的CorrectionParameters一致）
                # 注意：isrock.py中alpha定义为负数，公式使用减号
                delta_t = t_orig - self.target_temperature  # 原始温度 - 目标温度
                vs_corrected = vs_pressure_corrected * (1 - alpha_s * delta_t)
                
                valid_data.loc[idx, vs_col] = vs_corrected
        
        # 先转换为标准列名，便于后续扩展特征
        rename_dict = {
            vp_col: 'vp',
            rock_type_col: 'rock_type',
            pressure_col: 'pressure',
            'temperature': 'temperature',
        }
        if vs_col:
            rename_dict[vs_col] = 'vs'
        if density_col:
            rename_dict[density_col] = 'density'
        if felsic_or_mafic_col:
            rename_dict[felsic_or_mafic_col] = 'felsic_or_mafic'
        if rock_facies_col:
            rename_dict[rock_facies_col] = 'rock_facies'
        if sio2_col:
            rename_dict[sio2_col] = 'sio2_wt'

        standard_data = valid_data.rename(columns=rename_dict)

        # 基础特征：保持默认行为（vp/vs/density）
        feature_cols = ['vp']
        if 'vs' in standard_data.columns:
            feature_cols.append('vs')
        if 'density' in standard_data.columns:
            feature_cols.append('density')

        # 可选增强特征：默认关闭，仅作为后续入口
        if self.use_felsic_mafic and 'felsic_or_mafic' in standard_data.columns:
            feature_cols.append('felsic_or_mafic')
        if self.use_rock_facies and 'rock_facies' in standard_data.columns:
            feature_cols.append('rock_facies')
        if self.use_sio2 and 'sio2_wt' in standard_data.columns:
            feature_cols.append('sio2_wt')

        # 清洗增强特征
        if 'felsic_or_mafic' in standard_data.columns:
            standard_data['felsic_or_mafic'] = (
                standard_data['felsic_or_mafic']
                .astype(str)
                .str.strip()
                .replace({'': 'UNKNOWN', 'nan': 'UNKNOWN'})
            )
        if 'rock_facies' in standard_data.columns:
            standard_data['rock_facies'] = (
                standard_data['rock_facies']
                .astype(str)
                .str.strip()
                .replace({'': 'UNKNOWN', 'nan': 'UNKNOWN'})
            )
        if 'sio2_wt' in standard_data.columns:
            standard_data['sio2_wt'] = pd.to_numeric(standard_data['sio2_wt'], errors='coerce')
            if standard_data['sio2_wt'].isna().all():
                standard_data['sio2_wt'] = np.nan
                if 'sio2_wt' in feature_cols:
                    feature_cols.remove('sio2_wt')
            else:
                standard_data['sio2_wt'] = standard_data['sio2_wt'].fillna(standard_data['sio2_wt'].median())

        cols_to_use = ['rock_type'] + feature_cols
        train_data = standard_data[cols_to_use].copy()
        train_data = train_data.dropna(subset=['rock_type', 'vp'])
        if len(train_data) == 0:
            raise ValueError("Database has no valid data rows (all rows have NaN values)")

        self.classifier = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight='balanced_subsample',
            min_samples_leaf=2,
        )
        self.scaler = StandardScaler()
        self.col_mapping = col_mapping
        self.feature_cols = feature_cols
        self._categorical_feature_cols = [
            col for col in ['felsic_or_mafic', 'rock_facies'] if col in self.feature_cols
        ]
        self._category_maps: Dict[str, Dict[str, int]] = {}
        self._feature_defaults: Dict[str, Any] = {}

        # 为可选特征建立编码和默认值
        for col in self._categorical_feature_cols:
            values = train_data[col].astype(str).str.strip().replace({'': 'UNKNOWN'})
            unique_vals = sorted(values.unique())
            self._category_maps[col] = {name: i for i, name in enumerate(unique_vals)}
            train_data[col] = values.map(self._category_maps[col]).astype(float)
            self._feature_defaults[col] = values.mode().iloc[0] if not values.mode().empty else 'UNKNOWN'

        if 'sio2_wt' in self.feature_cols:
            self._feature_defaults['sio2_wt'] = float(train_data['sio2_wt'].median())

        if 'density' in self.feature_cols:
            self._feature_defaults['density'] = float(train_data['density'].median())
        if 'vs' in self.feature_cols:
            self._feature_defaults['vs'] = float(train_data['vs'].median())

        X_np = train_data[self.feature_cols].to_numpy(dtype=float)
        y = train_data['rock_type']
        # Use ndarray for scaler so inference with ndarray (classify_by_features) does not
        # trigger sklearn's "without feature names" vs "fitted with feature names" warning.
        X_scaled = self.scaler.fit_transform(X_np)
        self.classifier.fit(X_scaled, y)
        self.probability_model = self.classifier
        try:
            calibrator = CalibratedClassifierCV(
                estimator=self.classifier,
                method='sigmoid',
                cv=3,
            )
            calibrator.fit(X_scaled, y)
            self.probability_model = calibrator
        except Exception:
            self.probability_model = self.classifier
        self.rock_database_reference = standard_data.copy()
        self.rock_database_standard = train_data
    
    def correct_velocity(self, velocity, pressure=None, temperature=None, 
                        is_s_wave=False, target_pressure=200.0, target_temperature=25.0):
        """将速度值校正到目标压力和温度条件
        
        使用统一的经验公式库进行校正。
        
        Args:
            velocity: 原始速度值 (km/s)
            pressure: 原始压力值 (MPa)，如果为None则假设已经是目标压力
            temperature: 原始温度值 (°C)，如果为None则假设已经是目标温度
            is_s_wave: 是否为S波速度
            target_pressure: 目标压力 (MPa)，默认200.0
            target_temperature: 目标温度 (°C)，默认25.0
            
        Returns:
            float: 校正后的速度值 (km/s)
        """
        # 使用统一的经验公式库
        return _correct_velocity(
            velocity,
            pressure=pressure,
            temperature=temperature,
            target_pressure=target_pressure,
            target_temperature=target_temperature,
            is_s_wave=is_s_wave
        )

    def _encode_categorical_value(self, feature: str, value: Any) -> float:
        text = str(value).strip() if value is not None else ""
        if not text:
            text = str(self._feature_defaults.get(feature, "UNKNOWN"))
        mapping = self._category_maps.get(feature, {})
        if text not in mapping:
            mapping[text] = len(mapping)
            self._category_maps[feature] = mapping
        return float(mapping[text])

    def _build_feature_vector(
        self,
        vp: float,
        vs: Optional[float] = None,
        density: Optional[float] = None,
        felsic_or_mafic: Optional[str] = None,
        rock_facies: Optional[str] = None,
        sio2_wt: Optional[float] = None,
    ) -> np.ndarray:
        values: list[float] = []
        for feature in self.feature_cols:
            if feature == 'vp':
                values.append(float(vp))
            elif feature == 'vs':
                v = vs if vs is not None else self._feature_defaults.get('vs', _calculate_vs(vp, method='brocher'))
                values.append(float(v))
            elif feature == 'density':
                d = density if density is not None else self._feature_defaults.get('density', _calculate_density(vp, method='gardner'))
                values.append(float(d))
            elif feature == 'sio2_wt':
                s = sio2_wt if sio2_wt is not None else self._feature_defaults.get('sio2_wt', np.nan)
                if s is None or (isinstance(s, float) and np.isnan(s)):
                    s = 0.0
                values.append(float(s))
            elif feature == 'felsic_or_mafic':
                values.append(self._encode_categorical_value('felsic_or_mafic', felsic_or_mafic))
            elif feature == 'rock_facies':
                values.append(self._encode_categorical_value('rock_facies', rock_facies))
            else:
                values.append(0.0)
        return np.asarray(values, dtype=float).reshape(1, -1)

    def classify_by_features(
        self,
        vp: float,
        vs: Optional[float] = None,
        density: Optional[float] = None,
        felsic_or_mafic: Optional[str] = None,
        rock_facies: Optional[str] = None,
        sio2_wt: Optional[float] = None,
    ) -> str:
        X = self._build_feature_vector(
            vp=vp,
            vs=vs,
            density=density,
            felsic_or_mafic=felsic_or_mafic,
            rock_facies=rock_facies,
            sio2_wt=sio2_wt,
        )
        X_scaled = self.scaler.transform(X)
        pred = self.probability_model.predict(X_scaled)[0]
        return str(pred)

    def classify_probabilities_by_features(
        self,
        vp: float,
        vs: Optional[float] = None,
        density: Optional[float] = None,
        felsic_or_mafic: Optional[str] = None,
        rock_facies: Optional[str] = None,
        sio2_wt: Optional[float] = None,
    ) -> Dict[str, float]:
        X = self._build_feature_vector(
            vp=vp,
            vs=vs,
            density=density,
            felsic_or_mafic=felsic_or_mafic,
            rock_facies=rock_facies,
            sio2_wt=sio2_wt,
        )
        X_scaled = self.scaler.transform(X)
        probs = self.probability_model.predict_proba(X_scaled)[0]
        return {str(cls): float(p) for cls, p in zip(self.classifier.classes_, probs)}

    def get_auxiliary_attributes(
        self,
        rock_type: str,
        vp: Optional[float] = None,
        vs: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not hasattr(self, 'rock_database_reference'):
            return {}
        ref = self.rock_database_reference
        if 'rock_type' not in ref.columns or ref.empty:
            return {}

        mask = ref['rock_type'].astype(str).str.strip().str.lower() == str(rock_type).strip().lower()
        subset = ref[mask]
        if subset.empty:
            return {}

        chosen = subset.iloc[0]
        if vp is not None and 'vp' in subset.columns:
            distances = (subset['vp'].astype(float) - float(vp)).abs()
            if vs is not None and 'vs' in subset.columns:
                distances = distances + (subset['vs'].astype(float) - float(vs)).abs()
            chosen = subset.loc[distances.idxmin()]

        out: Dict[str, Any] = {}
        if 'felsic_or_mafic' in subset.columns:
            v = str(chosen.get('felsic_or_mafic', '')).strip()
            if v and v.lower() != 'nan':
                out['felsic_or_mafic'] = v
        if 'rock_facies' in subset.columns:
            v = str(chosen.get('rock_facies', '')).strip()
            if v and v.lower() != 'nan':
                out['rock_facies'] = v
        if 'sio2_wt' in subset.columns:
            try:
                s = float(chosen.get('sio2_wt'))
                if not np.isnan(s):
                    out['sio2_wt'] = s
            except Exception:
                pass
        # Extended geology descriptors from merged dictionary table
        for col, key in [
            ('岩石属性', 'rock_attributes_cn'),
            ('岩石分类', 'rock_classification'),
            ('变质程度', 'metamorphic_grade'),
            ('地质意义', 'geological_meaning'),
        ]:
            if col in subset.columns:
                v = str(chosen.get(col, '')).strip()
                if v and v.lower() != 'nan':
                    out[key] = v
        return out
    
    def classify_by_vp(self, vp):
        """基于P波速度的岩石分类
        
        Args:
            vp: float, P波速度 (km/s)
            
        Returns:
            str: 岩石类型
        """
        try:
            return self.classify_by_features(vp=vp)
        except Exception:
            distances = abs(self.rock_database_standard['vp'] - vp)
            closest_idx = distances.idxmin()
            return self.rock_database_standard.loc[closest_idx, 'rock_type']
    
    def classify_by_vp_vs(self, vp, vs):
        """基于P波和S波速度的岩石分类
        
        Args:
            vp: float, P波速度 (km/s)
            vs: float, S波速度 (km/s)
            
        Returns:
            str: 岩石类型
        """
        try:
            return self.classify_by_features(vp=vp, vs=vs)
        except Exception:
            distances = np.sqrt(
                (self.rock_database_standard['vp'] - vp)**2 +
                (self.rock_database_standard['vs'] - vs)**2
            )
            closest_idx = distances.idxmin()
            return self.rock_database_standard.loc[closest_idx, 'rock_type']
    
    def classify_with_uncertainty(self, vp, threshold=0.2):
        """带不确定性的岩石分类
        
        Args:
            vp: float, P波速度 (km/s)
            threshold: float, 速度差异阈值
            
        Returns:
            dict: 可能的岩石类型及其概率
        """
        try:
            probabilities = self.classify_probabilities_by_features(vp=vp)
            if not probabilities:
                return {self.classify_by_vp(vp): 1.0}
            max_prob = max(probabilities.values())
            if threshold is None or float(threshold) <= 0:
                return probabilities
            filtered = {
                k: v for k, v in probabilities.items()
                if v >= max(0.0, max_prob - float(threshold))
            }
            return filtered or probabilities
        except Exception:
            return {self.classify_by_vp(vp): 1.0}
    
    def classify_by_vp_and_setting(self, vp, setting):
        """考虑构造环境的岩石分类
        
        Args:
            vp: float, P波速度 (km/s)
            setting: TectonicSetting, 构造环境
            
        Returns:
            str: 岩石类型
        """
        # 检查是否有tectonic_setting列（在原始数据库中）
        if 'tectonic_setting' in self.rock_database.columns:
            # 需要找到对应的标准化数据库中的行
            setting_mask = self.rock_database['tectonic_setting'] == setting.value
            setting_indices = self.rock_database[setting_mask].index
            # 获取标准化数据库中对应的行
            setting_samples = self.rock_database_standard.loc[
                self.rock_database_standard.index.intersection(setting_indices)
            ]
        else:
            # 如果没有构造环境列，使用全部数据
            setting_samples = self.rock_database_standard
        
        if setting_samples.empty:
            # 如果在特定构造环境中没有找到，使用全部数据
            return self.classify_by_vp(vp)
        
        # 使用标准化的数据库
        distances = abs(setting_samples['vp'] - vp)
        closest_idx = distances.idxmin()
        return setting_samples.loc[closest_idx, 'rock_type']
    
    def plot_classification_results(self, results, output_file=None):
        """绘制分类结果
        
        Args:
            results: dict, 分类结果
            output_file: str, 输出文件路径
        """
        plt.figure(figsize=(10, 6))
        
        # 绘制概率分布
        rock_types = list(results.keys())
        probabilities = list(results.values())
        
        plt.bar(rock_types, probabilities)
        plt.xlabel('Rock Type')
        plt.ylabel('Probability')
        plt.title('Rock Classification Results')
        plt.xticks(rotation=45)
        
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()

def load_rock_database(database_file: Optional[str] = None) -> 'RockDatabase':
    """加载岩石数据库。

    Args:
        database_file (str, optional): 数据库文件路径。如果不提供，将创建一个空的数据库。

    Returns:
        RockDatabase: 岩石数据库实例
    """
    db = RockDatabase(database_file)
    return db