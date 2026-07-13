"""
isrock.py - Python implementation of rock type identification

This module provides functions for identifying rock types based on their physical properties.
It works in conjunction with the rocks.py module.

Author: Haibo Huang
Date: 2025
"""

import numpy as np
from typing import Optional, Union, Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from .rocks import RockDatabase, RockProperties, Rock, RockClassifier, TectonicSetting, create_common_rock
import pandas as pd
from enum import Enum
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

@dataclass
class CorrectionParameters:
    """温度压力校正参数类"""
    # 温度校正参数 (基于Christensen, 1979)
    vp_temp_coefficients = {
        'granite_granodiorite': -0.39e-4,  # 花岗岩-花岗闪长岩,花岗片麻岩,英云闪长片麻岩
        'gabbro_norite': -0.57e-4,   # 辉长岩-紫苏辉长岩-斜长岩
        'basalt': -0.39e-4,   # 玄武岩,变质玄武岩
        'slate': -0.40e-4,    # 板岩,千枚岩,石英云母片岩
        'mafic_granulite': -0.52e-4,  # 镁铁质麻粒岩,镁铁质石榴石麻粒岩
        'felsic_granulite': -0.49e-4, # 长英质麻粒岩,副麻粒岩
        'amphibolite': -0.55e-4,      # 角闪岩
        'anorthosite': -0.41e-4,      # 斜长岩
        'dunite': -0.56e-4,           # 橄榄岩,辉石岩,角闪岩
        'eclogite': -0.53e-4,         # 榴辉岩
        'serpentinite': -0.68e-4,     # 蛇纹岩
        'quartzite': -0.54e-4,        # 石英岩
        'marble': -0.41e-4,           # 大理岩
    }
    
    # 默认温度校正系数(当岩石类型未知时使用)
    vp_temp_alpha: float = -0.50e-4  # 平均值，降低一个数量级
    vs_temp_alpha: float = -0.40e-4  # 降低一个数量级
    density_temp_alpha: float = -0.10e-4  # 降低一个数量级
    
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

class TectonicSetting(Enum):
    """构造环境枚举类"""
    CONTINENTAL_CRUST = "CONTINENTAL_CRUST"
    OCEANIC_CRUST = "OCEANIC_CRUST"
    SUBDUCTION_ZONE = "SUBDUCTION_ZONE"

class RockIdentifier(RockClassifier):
    """岩石识别器类，继承自RockClassifier并提供更高级的识别功能"""
    
    def __init__(self, database):
        """初始化岩石识别器
        
        Args:
            database: str或pd.DataFrame，岩石数据库文件路径或DataFrame
        """
        # 调用父类的初始化方法
        super().__init__(database)
        
        # 添加额外的分类器用于更复杂的识别任务
        self.advanced_classifier = None
        self.advanced_scaler = None
        
        # 初始化校正参数
        self.correction_params = CorrectionParameters()
        
        # 校正训练数据
        self._correct_training_data()
    
    def _correct_training_data(self):
        """校正训练数据到标准条件（25°C，200MPa）"""
        # 获取原始数据
        original_vp = self.rock_database['vp'].values
        original_vs = self.rock_database['vs'].values
        original_density = self.rock_database['density'].values
        original_pressure = self.rock_database['pressure'].values
        original_temperature = self.rock_database['temperature'].values
        rock_types = self.rock_database['rock_type'].values
        
        # 进行压力校正
        corrected_vp = self.pressure_correction(
            original_vp,
            original_pressure,
            target_pressure=200.0,
            rock_type=None
        )
        corrected_vs = self.pressure_correction(
            original_vs,
            original_pressure,
            target_pressure=200.0,
            rock_type=None,
            is_s_wave=True
        )
        
        # 进行温度校正
        corrected_vp = self.temperature_correction(
            corrected_vp,
            original_temperature,
            target_temperature=25.0,
            rock_type=None
        )
        corrected_vs = self.temperature_correction(
            corrected_vs,
            original_temperature,
            target_temperature=25.0,
            rock_type=None,
            is_s_wave=True
        )
        
        # 进行密度校正
        corrected_density = self.density_correction(
            original_density,
            original_pressure,
            original_temperature,
            target_pressure=200.0,
            target_temperature=25.0
        )
        
        # 第二次校正（使用岩石类型特定的系数）
        for i, rock_type in enumerate(rock_types):
            corrected_vp[i] = self.pressure_correction(
                original_vp[i],
                original_pressure[i],
                target_pressure=200.0,
                rock_type=rock_type
            )
            corrected_vp[i] = self.temperature_correction(
                corrected_vp[i],
                original_temperature[i],
                target_temperature=25.0,
                rock_type=rock_type
            )
            
            corrected_vs[i] = self.pressure_correction(
                original_vs[i],
                original_pressure[i],
                target_pressure=200.0,
                rock_type=rock_type,
                is_s_wave=True
            )
            corrected_vs[i] = self.temperature_correction(
                corrected_vs[i],
                original_temperature[i],
                target_temperature=25.0,
                rock_type=rock_type,
                is_s_wave=True
            )
        
        # 更新数据库中的值
        self.rock_database['vp'] = corrected_vp
        self.rock_database['vs'] = corrected_vs
        self.rock_database['density'] = corrected_density
        self.rock_database['pressure'] = 200.0  # 标准压力
        self.rock_database['temperature'] = 25.0  # 标准温度
    
    def train_classifier(self):
        """训练高级分类器"""
        # 准备特征和标签
        # 检查哪些特征列存在
        available_features = []
        for feature in ['vp', 'vs', 'density', 'porosity']:
            if feature in self.rock_database.columns:
                available_features.append(feature)
            elif hasattr(self, 'rock_database') and hasattr(self.rock_database, 'columns'):
                # 尝试查找变体列名
                col_lower = [str(c).lower().strip() for c in self.rock_database.columns]
                if feature.lower() in col_lower:
                    idx = col_lower.index(feature.lower())
                    available_features.append(self.rock_database.columns[idx])
        
        # 如果porosity列不存在，使用默认值0.02
        if 'porosity' not in available_features and 'porosity' not in [str(c).lower() for c in self.rock_database.columns]:
            # 添加porosity列，使用默认值
            if 'porosity' not in self.rock_database.columns:
                self.rock_database['porosity'] = 0.02
            available_features.append('porosity')
        
        # 确保至少有一些特征
        if not available_features:
            raise ValueError("No valid features found in database for training classifier")
        
        X = self.rock_database[available_features]
        y = self.rock_database['rock_type']
        
        # 标准化特征
        self.advanced_scaler = StandardScaler()
        X_scaled = self.advanced_scaler.fit_transform(X)
        
        # 训练随机森林分类器
        self.advanced_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        self.advanced_classifier.fit(X_scaled, y)

    def identify_rock(self, vp, vs, density, porosity, 
                     tectonic_setting=None, min_probability=0.1,
                     max_candidates=5):
        """识别岩石类型，提供更详细的结果
        
        Args:
            vp: float, P波速度 (km/s)
            vs: float, S波速度 (km/s)
            density: float, 密度 (g/cm³)
            porosity: float, 孔隙度
            tectonic_setting: TectonicSetting, 构造环境
            min_probability: float, 最小概率阈值
            max_candidates: int, 最大候选数量
            
        Returns:
            dict: 识别结果，包含属性和候选岩石列表
        """
        if self.advanced_classifier is None:
            self.train_classifier()
        
        # 准备输入数据，添加特征名称
        X = pd.DataFrame([[vp, vs, density, porosity]], 
                        columns=['vp', 'vs', 'density', 'porosity'])
        X_scaled = self.advanced_scaler.transform(X)
        
        # 获取预测概率
        probabilities = self.advanced_classifier.predict_proba(X_scaled)[0]
        classes = self.advanced_classifier.classes_
        
        # 整理结果
        candidates = []
        for prob, rock_type in zip(probabilities, classes):
            if prob >= min_probability:
                candidates.append({
                    'rock_type': rock_type,
                    'probability': prob
                })
        
        # 按概率排序
        candidates.sort(key=lambda x: x['probability'], reverse=True)
        
        # 如果提供了构造环境，使用父类的方法进行验证
        if tectonic_setting:
            tectonic_result = self.classify_by_vp_and_setting(vp, tectonic_setting)
            # 调整具有相同构造环境的岩石的概率
            for candidate in candidates:
                if candidate['rock_type'] == tectonic_result:
                    candidate['probability'] *= 1.2  # 增加20%的概率
        
        # 限制候选数量
        candidates = candidates[:max_candidates]
        
        return {
            'properties': {
                'vp': vp,
                'vs': vs,
                'density': density,
                'porosity': porosity,
                'tectonic_setting': tectonic_setting.value if tectonic_setting else None
            },
            'candidates': candidates
        }
    
    def pressure_correction(self, velocity, pressure, target_pressure=200.0, rock_type=None, is_s_wave=False):
        """
        将速度值校正到目标压力条件
        
        参数:
            velocity (numpy.ndarray): 原始速度值数组
            pressure (numpy.ndarray): 原始压力值数组（MPa）
            target_pressure (float): 目标压力值（MPa），默认200.0 MPa
            rock_type (str, optional): 岩石类型，用于选择合适的校正系数
            is_s_wave (bool): 是否为S波速度，默认False
            
        返回:
            numpy.ndarray: 校正后的速度值数组
        """
        # 选择合适的压力校正系数
        beta = (self.correction_params.vs_pressure_beta if is_s_wave 
               else self.correction_params.vp_pressure_beta)
        
        pressure_diff = target_pressure - pressure
        correction_factor = 1 + beta * pressure_diff
        return velocity * correction_factor

    def temperature_correction(self, velocity, temperature, target_temperature=25.0, 
                             rock_type=None, is_s_wave=False):
        """
        将速度值校正到目标温度条件
        
        参数:
            velocity (numpy.ndarray): 原始速度值数组
            temperature (numpy.ndarray): 原始温度值数组（°C）
            target_temperature (float): 目标温度值（°C），默认25.0°C
            rock_type (str, optional): 岩石类型，用于选择合适的校正系数
            is_s_wave (bool): 是否为S波速度，默认False
            
        返回:
            numpy.ndarray: 校正后的速度值数组
        """
        # 选择合适的温度校正系数
        if is_s_wave:
            alpha = self.correction_params.vs_temp_alpha
        else:
            alpha = (self.correction_params.get_vp_temp_alpha(rock_type) 
                    if rock_type else self.correction_params.vp_temp_alpha)
        
        temperature_diff = temperature - target_temperature  # 注意：这里改变了差值的计算顺序
        correction_factor = 1 - alpha * temperature_diff  # 使用减号，因为温度升高，速度降低
        
        # 限制校正因子的范围，防止出现不合理的值
        correction_factor = np.clip(correction_factor, 0.7, 1.3)
        
        return velocity * correction_factor

    def density_correction(self, density, pressure, temperature, 
                         target_pressure=200.0, target_temperature=25.0):
        """
        将密度值校正到目标压力和温度条件
        
        参数:
            density (numpy.ndarray): 原始密度值数组
            pressure (numpy.ndarray): 原始压力值数组（MPa）
            temperature (numpy.ndarray): 原始温度值数组（°C）
            target_pressure (float): 目标压力值（MPa），默认200.0 MPa
            target_temperature (float): 目标温度值（°C），默认25.0°C
            
        返回:
            numpy.ndarray: 校正后的密度值数组
        """
        # 压力校正
        pressure_diff = target_pressure - pressure
        pressure_factor = 1 + self.correction_params.density_pressure_beta * pressure_diff
        
        # 温度校正
        temperature_diff = target_temperature - temperature
        temperature_factor = 1 + self.correction_params.density_temp_alpha * temperature_diff
        
        # 组合校正
        return density * pressure_factor * temperature_factor

    def identify_velocity_model(self, model_data, min_probability=0.1):
        """识别速度模型中的岩石类型，包括温度压力校正"""
        results = {}
        
        # 第一步：使用标准系数进行初步校正
        corrected_vp = self.pressure_correction(
            model_data['vp'], 
            model_data['pressure'],
            rock_type=None,  # 使用标准系数
            is_s_wave=False
        )
        corrected_vp = self.temperature_correction(
            corrected_vp, 
            model_data['temperature'],
            rock_type=None,  # 使用标准系数
            is_s_wave=False
        )
        
        corrected_vs = self.pressure_correction(
            model_data['vs'], 
            model_data['pressure'],
            rock_type=None,  # 使用标准系数
            is_s_wave=True
        )
        corrected_vs = self.temperature_correction(
            corrected_vs, 
            model_data['temperature'],
            rock_type=None,  # 使用标准系数
            is_s_wave=True
        )
        
        corrected_density = self.density_correction(
            model_data['density'],
            model_data['pressure'],
            model_data['temperature']
        )
        
        # 第二步：使用校正后的数据进行初步识别
        preliminary_results = []
        for i in range(len(corrected_vp)):
            result = self.identify_rock(
                vp=corrected_vp[i],
                vs=corrected_vs[i],
                density=corrected_density[i],
                porosity=model_data['porosity'][i],
                tectonic_setting=model_data['tectonic_setting'][i],  # 添加构造环境
                min_probability=min_probability
            )
            preliminary_results.append(result)
            
        # 第三步：根据初步识别结果进行第二次校正
        for i in range(len(corrected_vp)):
            # 获取最可能的岩石类型
            most_likely_rock = preliminary_results[i]['candidates'][0]['rock_type']
            
            # 使用特定岩石类型的系数进行第二次校正
            final_vp = self.pressure_correction(
                model_data['vp'][i], 
                model_data['pressure'][i],
                rock_type=most_likely_rock,
                is_s_wave=False
            )
            final_vp = self.temperature_correction(
                final_vp, 
                model_data['temperature'][i],
                rock_type=most_likely_rock,
                is_s_wave=False
            )
            
            final_vs = self.pressure_correction(
                model_data['vs'][i], 
                model_data['pressure'][i],
                rock_type=most_likely_rock,
                is_s_wave=True
            )
            final_vs = self.temperature_correction(
                final_vs, 
                model_data['temperature'][i],
                rock_type=most_likely_rock,
                is_s_wave=True
            )
            
            final_density = self.density_correction(
                model_data['density'][i],
                model_data['pressure'][i],
                model_data['temperature'][i]
            )
            
            # 使用最终校正的数据进行识别，加入构造环境信息
            final_result = self.identify_rock(
                vp=final_vp,
                vs=final_vs,
                density=final_density,
                porosity=model_data['porosity'][i],
                tectonic_setting=model_data['tectonic_setting'][i],  # 添加构造环境
                min_probability=min_probability
            )
            
            # 根据构造环境调整概率
            if model_data['tectonic_setting'][i] == TectonicSetting.OCEANIC_CRUST:
                # 在大洋地壳中，增加辉长岩和玄武岩的概率
                for candidate in final_result['candidates']:
                    if candidate['rock_type'] in ['GABBRO', 'BASALT']:
                        candidate['probability'] *= 1.3
            elif model_data['tectonic_setting'][i] == TectonicSetting.SUBDUCTION_ZONE:
                # 在俯冲带中，增加橄榄岩和榴辉岩的概率
                for candidate in final_result['candidates']:
                    if candidate['rock_type'] in ['PERIDOTITE', 'ECLOGITE']:
                        candidate['probability'] *= 1.3
            
            # 重新排序候选岩石
            final_result['candidates'].sort(key=lambda x: x['probability'], reverse=True)
            
            # 整理结果
            for candidate in final_result['candidates']:
                rock_type = candidate['rock_type']
                if rock_type not in results:
                    results[rock_type] = []
                    
                results[rock_type].append({
                    'properties': final_result['properties'],
                    'probability': candidate['probability'],
                    'original_properties': {
                        'vp': model_data['vp'][i],
                        'vs': model_data['vs'][i],
                        'density': model_data['density'][i],
                        'pressure': model_data['pressure'][i],
                        'temperature': model_data['temperature'][i],
                        'tectonic_setting': model_data['tectonic_setting'][i].value
                    },
                    'correction_info': {
                        'preliminary_rock_type': most_likely_rock,
                        'preliminary_probability': preliminary_results[i]['candidates'][0]['probability'],
                        'final_vp': final_vp,
                        'final_vs': final_vs,
                        'final_density': final_density
                    }
                })
        
        return results
    
    def plot_identification_results(self, results, output_file=None):
        """绘制更详细的识别结果可视化"""
        plt.figure(figsize=(12, 8))
        
        # 为每种岩石类型绘制概率分布
        for i, (rock_type, data) in enumerate(results.items()):
            probabilities = [d['probability'] for d in data]
            properties = [d['properties'] for d in data]
            
            # 绘制概率分布
            plt.subplot(2, 2, 1)
            plt.hist(probabilities, alpha=0.5, label=rock_type, bins=10)
            plt.xlabel('Probability')
            plt.ylabel('Count')
            plt.title('Probability Distribution')
            plt.legend()
            
            # 绘制Vp-Vs关系
            plt.subplot(2, 2, 2)
            vp = [p['vp'] for p in properties]
            vs = [p['vs'] for p in properties]
            plt.scatter(vp, vs, alpha=0.5, label=rock_type)
            plt.xlabel('Vp (km/s)')
            plt.ylabel('Vs (km/s)')
            plt.title('Vp-Vs Relationship')
            plt.legend()
            
            # 绘制密度-孔隙度关系
            plt.subplot(2, 2, 3)
            density = [p['density'] for p in properties]
            porosity = [p['porosity'] for p in properties]
            plt.scatter(density, porosity, alpha=0.5, label=rock_type)
            plt.xlabel('Density (g/cm³)')
            plt.ylabel('Porosity')
            plt.title('Density-Porosity Relationship')
            plt.legend()
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()
            
        plt.close()

def identify_rocks_from_model(model_file: str,
                            database_file: str,
                            min_probability: float = 0.6,
                            output_file: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """从速度模型文件识别岩石类型
    
    Args:
        model_file: 速度模型文件路径
        database_file: 岩石物性数据库文件路径
        min_probability: 最小可信概率
        output_file: 结果图像输出路径
        
    Returns:
        Dict[str, List[Dict[str, Any]]]: 岩石识别结果
    """
    # 创建岩石识别器
    identifier = RockIdentifier(database_file)
    
    # 训练分类器
    identifier.train_classifier()
    
    # 读取模型数据
    model_data = _load_model_data(model_file)
    
    # 识别岩石
    results = identifier.identify_velocity_model(model_data, min_probability)
    
    # 绘制结果
    if output_file:
        identifier.plot_identification_results(results, output_file)
        
    return results
    
def _load_model_data(model_file: str) -> Dict[str, np.ndarray]:
    """加载速度模型数据
    
    Args:
        model_file: 模型文件路径
        
    Returns:
        Dict[str, np.ndarray]: 模型数据
    """
    # 这里需要根据实际的模型文件格式来实现
    # 目前仅返回示例数据
    return {
        'vp': np.array([3.5, 4.5, 5.5, 6.0]),
        'vs': np.array([2.0, 2.5, 3.0, 3.5]),
        'density': np.array([2.35, 2.55, 2.65, 2.85]),
        'porosity': np.array([0.05, 0.10, 0.15, 0.20]),
        'pressure': np.array([100.0, 150.0, 200.0, 250.0]),
        'temperature': np.array([20.0, 25.0, 30.0, 35.0]),
        'tectonic_setting': np.array([TectonicSetting.CONTINENTAL_CRUST, TectonicSetting.OCEANIC_CRUST, TectonicSetting.SUBDUCTION_ZONE, TectonicSetting.CONTINENTAL_CRUST])
    }

def identify_rock_type(vp: float, 
                      vs: Optional[float] = None,
                      density: Optional[float] = None) -> Tuple[str, float]:
    """基于物性参数识别岩石类型。

    Args:
        vp (float): P波速度 (km/s)
        vs (Optional[float]): S波速度 (km/s)
        density (Optional[float]): 密度 (g/cm³)

    Returns:
        Tuple[str, float]: (岩石类型, 置信度)
    """
    # 定义常见岩石的物性范围
    rock_ranges = {
        'sandstone': {'vp': (1.5, 4.5), 'vs': (0.8, 2.8), 'density': (1.9, 2.7)},
        'limestone': {'vp': (3.5, 6.5), 'vs': (2.0, 3.5), 'density': (2.3, 2.8)},
        'granite': {'vp': (4.5, 6.5), 'vs': (2.5, 3.8), 'density': (2.5, 2.8)},
        'basalt': {'vp': (5.0, 7.0), 'vs': (2.8, 4.0), 'density': (2.7, 3.1)},
        'shale': {'vp': (1.5, 4.0), 'vs': (0.7, 2.3), 'density': (2.0, 2.7)}
    }

    # 计算每种岩石的匹配度
    scores = {}
    for rock_type, ranges in rock_ranges.items():
        score = 0.0
        count = 0

        # 检查vp
        if vp >= ranges['vp'][0] and vp <= ranges['vp'][1]:
            score += 1.0
        else:
            # 计算距离最近边界的归一化距离
            min_dist = min(abs(vp - ranges['vp'][0]), abs(vp - ranges['vp'][1]))
            range_width = ranges['vp'][1] - ranges['vp'][0]
            score += max(0, 1 - min_dist/range_width)
        count += 1

        # 检查vs（如果提供）
        if vs is not None:
            if vs >= ranges['vs'][0] and vs <= ranges['vs'][1]:
                score += 1.0
            else:
                min_dist = min(abs(vs - ranges['vs'][0]), abs(vs - ranges['vs'][1]))
                range_width = ranges['vs'][1] - ranges['vs'][0]
                score += max(0, 1 - min_dist/range_width)
            count += 1

        # 检查密度（如果提供）
        if density is not None:
            if density >= ranges['density'][0] and density <= ranges['density'][1]:
                score += 1.0
            else:
                min_dist = min(abs(density - ranges['density'][0]), 
                             abs(density - ranges['density'][1]))
                range_width = ranges['density'][1] - ranges['density'][0]
                score += max(0, 1 - min_dist/range_width)
            count += 1

        # 计算平均分数
        scores[rock_type] = score / count

    # 找出最匹配的岩石类型
    best_rock_type = max(scores.items(), key=lambda x: x[1])
    return best_rock_type[0], best_rock_type[1] 