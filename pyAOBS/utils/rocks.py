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
        vp = self.properties.vp
        
        if method == 'gardner':
            # Gardner et al. (1974)
            return 1.74 * (vp * 1000)**0.25  # vp需要转换为m/s
            
        elif method == 'nafe_drake':
            # Nafe & Drake (1963)
            return 1.6612 * vp - 0.4721 * vp**2 + 0.0671 * vp**3 - 0.0043 * vp**4 + 0.000106 * vp**5
            
        elif method == 'brocher':
            # Brocher (2005)
            return 1.6612 * vp - 0.4721 * vp**2 + 0.0671 * vp**3 - 0.0043 * vp**4 + 0.000106 * vp**5
            
        raise ValueError(f"不支持的密度计算方法: {method}")
    
    def calculate_vs(self, method: str = 'brocher') -> float:
        """使用经验公式计算S波速度
        
        Args:
            method: 使用的经验公式，可选：
                   'brocher' - Brocher公式
                   'castagna' - Castagna公式 (泥岩)
                   
        Returns:
            float: 计算得到的S波速度 (km/s)
        """
        vp = self.properties.vp
        
        if method == 'brocher':
            # Brocher (2005)
            return 0.7858 - 1.2344 * vp + 0.7949 * vp**2 - 0.1238 * vp**3 + 0.0064 * vp**4
            
        elif method == 'castagna':
            # Castagna et al. (1985)
            return (vp - 1.36) / 1.16
            
        raise ValueError(f"不支持的S波速度计算方法: {method}")
    
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
            
        vp = self.properties.vp * 1000  # 转换为m/s
        vs = self.properties.vs * 1000
        rho = self.properties.density * 1000  # 转换为kg/m³
        
        # 计算弹性模量
        mu = rho * vs**2 * 1e-9  # 剪切模量 (GPa)
        lambda_ = rho * (vp**2 - 2*vs**2) * 1e-9  # 拉梅常数λ (GPa)
        K = lambda_ + 2*mu/3  # 体积模量 (GPa)
        E = mu * (3*lambda_ + 2*mu)/(lambda_ + mu)  # 杨氏模量 (GPa)
        
        return {
            'bulk_modulus': K,
            'shear_modulus': mu,
            'young_modulus': E,
            'lame_lambda': lambda_
        }
    
    def calculate_temperature_effect(self, 
                                  reference_vp: float,
                                  delta_T: float,
                                  alpha: float = -0.0005) -> float:
        """计算温度对P波速度的影响
        
        Args:
            reference_vp: 参考温度下的P波速度 (km/s)
            delta_T: 温度变化 (°C)
            alpha: 温度系数 (默认 -0.0005/°C)
            
        Returns:
            float: 温度校正后的P波速度 (km/s)
        """
        return reference_vp * (1 + alpha * delta_T)
    
    def calculate_pressure_effect(self,
                               reference_vp: float,
                               delta_P: float,
                               beta: float = 0.0002) -> float:
        """计算压力对P波速度的影响
        
        Args:
            reference_vp: 参考压力下的P波速度 (km/s)
            delta_P: 压力变化 (MPa)
            beta: 压力系数 (默认 0.0002/MPa)
            
        Returns:
            float: 压力校正后的P波速度 (km/s)
        """
        return reference_vp * (1 + beta * delta_P)
    
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
            
        import matplotlib.pyplot as plt
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
    
    def __init__(self, database: RockDatabase):
        """初始化分类器
        
        Args:
            database: 岩石数据库实例
        """
        self.database = database
        self.classifier = None
        self.scaler = None
        self.features = ['vp', 'vs', 'density', 'porosity']
        self.required_features = ['vp']  # 只有P波速度是必需的
        
    def train_classifier(self):
        """训练分类器"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # 准备训练数据
        X = []
        y = []
        
        for rock_type, measurements in self.database.rocks.items():
            for m in measurements:
                features = []
                skip = False
                for feature in self.features:
                    value = getattr(m.properties, feature)
                    if feature in self.required_features and value is None:
                        skip = True
                        break
                    features.append(value if value is not None else 0.0)  # 使用0.0填充缺失值
                
                if not skip:
                    X.append(features)
                    y.append(rock_type)
        
        if not X:
            raise ValueError("没有足够的训练数据")
            
        X = np.array(X)
        
        # 标准化特征
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练分类器
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight='balanced'
        )
        self.classifier.fit(X_scaled, y)
        
    def predict_proba(self, properties: RockProperties) -> Dict[str, float]:
        """预测所有岩石类型的概率
        
        Args:
            properties: 岩石物性参数
            
        Returns:
            Dict[str, float]: 岩石类型到预测概率的映射
        """
        if self.classifier is None:
            raise ValueError("请先训练分类器")
            
        # 提取特征
        feature_values = []
        for feature in self.features:
            value = getattr(properties, feature)
            if feature in self.required_features and value is None:
                raise ValueError(f"缺少必需特征: {feature}")
            feature_values.append(value if value is not None else 0.0)  # 使用0.0填充缺失值
            
        # 标准化特征
        X = np.array([feature_values])
        X_scaled = self.scaler.transform(X)
        
        # 获取所有类别的预测概率
        probabilities = self.classifier.predict_proba(X_scaled)[0]
        rock_types = self.classifier.classes_
        
        return dict(zip(rock_types, probabilities))

    def identify_rock(self, vp: float, vs: Optional[float] = None,
                     density: Optional[float] = None,
                     porosity: Optional[float] = None,
                     min_probability: float = 0.2,
                     max_candidates: int = 3) -> Optional[Dict[str, Any]]:
        """识别岩石类型，返回多个可能的候选岩石
        
        Args:
            vp: P波速度 (km/s)
            vs: S波速度 (km/s)，可选
            density: 密度 (g/cm³)，可选
            porosity: 孔隙度 (0-1)，可选
            min_probability: 最小可信概率
            max_candidates: 最大候选岩石数量
            
        Returns:
            Optional[Dict[str, Any]]: 识别结果，包含：
                - candidates: 候选岩石列表，每个候选包含：
                    - rock_type: 岩石类型
                    - probability: 预测概率
                    - similarity_score: 相似度分数
                - properties: 输入的岩石物性参数
        """
        if self.classifier is None:
            self.train_classifier()
        
        properties = RockProperties(vp=vp, vs=vs, density=density, porosity=porosity)
        
        try:
            # 获取所有岩石类型的预测概率
            probabilities = self.predict_proba(properties)
            
            # 筛选概率大于阈值的候选岩石
            candidates = []
            for rock_type, prob in probabilities.items():
                if prob >= min_probability:
                    # 计算相似度分数
                    consensus = self.database.get_consensus_properties(rock_type)
                    if consensus:
                        similarity = self._calculate_similarity(properties, consensus)
                        candidates.append({
                            'rock_type': rock_type,
                            'probability': prob,
                            'similarity_score': similarity
                        })
            
            # 按综合分数排序（概率和相似度的加权）
            candidates.sort(
                key=lambda x: 0.7 * x['probability'] + 0.3 * x['similarity_score'],
                reverse=True
            )
            
            # 限制候选数量
            candidates = candidates[:max_candidates]
            
            return {
                'candidates': candidates,
                'properties': properties
            } if candidates else None
            
        except Exception as e:
            print(f"岩石识别过程中出错: {str(e)}")
            return None

    def _calculate_similarity(self, props1: RockProperties,
                            props2: RockProperties) -> float:
        """计算两个岩石物性的相似度
        
        Args:
            props1: 第一个岩石物性
            props2: 第二个岩石物性
            
        Returns:
            float: 相似度分数 (0-1)
        """
        features = ['vp', 'vs', 'density', 'porosity']
        weights = {'vp': 0.4, 'vs': 0.3, 'density': 0.2, 'porosity': 0.1}
        
        total_weight = 0
        similarity = 0
        
        for feature in features:
            value1 = getattr(props1, feature)
            value2 = getattr(props2, feature)
            
            if value1 is not None and value2 is not None:
                weight = weights[feature]
                # 计算归一化差异
                max_val = max(abs(value1), abs(value2))
                diff = abs(value1 - value2) / max_val if max_val > 0 else 0
                similarity += weight * (1 - diff)
                total_weight += weight
                
        return similarity / total_weight if total_weight > 0 else 0