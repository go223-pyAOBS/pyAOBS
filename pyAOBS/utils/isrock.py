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
from .rocks import RockDatabase, RockProperties, Rock, RockClassifier, TectonicSetting

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
            
        Warns:
            UserWarning: 当温度或压力超出合理范围时
        """
        import warnings
        
        # 检查极端条件
        if properties.temperature is not None:
            if properties.temperature > 800.0:
                warnings.warn(f"温度 {properties.temperature}°C 超出了实验室测量范围")
            elif properties.temperature < 0.0:
                warnings.warn(f"温度 {properties.temperature}°C 低于冰点")
                
        if properties.pressure is not None:
            if properties.pressure > 1000.0:
                warnings.warn(f"压力 {properties.pressure}MPa 超出了实验室测量范围")
            elif properties.pressure < 0.0:
                warnings.warn(f"压力不能为负值: {properties.pressure}MPa")
        
        # 获取校正参数
        params = CorrectionParameters()  # 使用新的参数实例，确保使用正确的系数
        
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
            # 温度校正
            corrected_vp = corrected_vp * (1 + vp_temp_alpha * delta_T)
            # 压力校正
            corrected_vp = corrected_vp * (1 + params.vp_pressure_beta * delta_P)
        
        # 校正S波速度
        corrected_vs = properties.vs
        if corrected_vs is not None:
            # 温度校正
            corrected_vs = corrected_vs * (1 + params.vs_temp_alpha * delta_T)
            # 压力校正
            corrected_vs = corrected_vs * (1 + params.vs_pressure_beta * delta_P)
        
        # 校正密度
        corrected_density = properties.density
        if corrected_density is not None:
            # 温度校正
            corrected_density = corrected_density * (1 + params.density_temp_alpha * delta_T)
            # 压力校正
            corrected_density = corrected_density * (1 + params.density_pressure_beta * delta_P)
        
        return RockProperties(
            vp=corrected_vp,
            vs=corrected_vs,
            density=corrected_density,
            porosity=properties.porosity,
            temperature=target_temp,
            pressure=target_pressure,
            fluid_saturation=properties.fluid_saturation
        )
        
    def plot_correction_comparison(self, rock_type: str,
                                 temp_range: Tuple[float, float] = (0, 200),
                                 pressure_range: Tuple[float, float] = (0, 500),
                                 output_file: Optional[str] = None):
        """绘制不同校正方法的对比图
        
        Args:
            rock_type: 岩石类型
            temp_range: 温度范围 (°C)
            pressure_range: 压力范围 (MPa)
            output_file: 输出文件路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 获取基准物性
        consensus = self.database.get_consensus_properties(rock_type)
        if not consensus:
            raise ValueError(f"没有找到岩石类型 {rock_type} 的共识物性值")
            
        # 温度影响
        temps = np.linspace(*temp_range, 50)
        vp_emp = []  # 经验校正
        vp_lab = []  # 实验室校正
        
        emp_params = CorrectionParameters()
        lab_params = self._get_correction_parameters(rock_type)
        
        for temp in temps:
            # 经验校正
            props_emp = self._normalize_to_standard_conditions(
                consensus, rock_type=None, target_temp=temp
            )
            vp_emp.append(props_emp.vp)
            
            # 实验室校正
            if lab_params.source == 'laboratory':
                props_lab = self._normalize_to_standard_conditions(
                    consensus, rock_type=rock_type, target_temp=temp
                )
                vp_lab.append(props_lab.vp)
        
        ax1.plot(temps, vp_emp, label='经验校正', linestyle='-')
        if lab_params.source == 'laboratory':
            ax1.plot(temps, vp_lab, label=f'实验室校正 ({lab_params.lab_name})',
                    linestyle='--')
        ax1.set_xlabel('温度 (°C)')
        ax1.set_ylabel('P波速度 (km/s)')
        ax1.set_title('温度校正对比')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 压力影响
        pressures = np.linspace(*pressure_range, 50)
        vp_emp = []
        vp_lab = []
        
        for pressure in pressures:
            # 经验校正
            props_emp = self._normalize_to_standard_conditions(
                consensus, rock_type=None, target_pressure=pressure
            )
            vp_emp.append(props_emp.vp)
            
            # 实验室校正
            if lab_params.source == 'laboratory':
                props_lab = self._normalize_to_standard_conditions(
                    consensus, rock_type=rock_type, target_pressure=pressure
                )
                vp_lab.append(props_lab.vp)
        
        ax2.plot(pressures, vp_emp, label='经验校正', linestyle='-')
        if lab_params.source == 'laboratory':
            ax2.plot(pressures, vp_lab, label=f'实验室校正 ({lab_params.lab_name})',
                    linestyle='--')
        ax2.set_xlabel('压力 (MPa)')
        ax2.set_ylabel('P波速度 (km/s)')
        ax2.set_title('压力校正对比')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
        else:
            plt.show()
        plt.close()
        
    def train_classifier(self, features: Optional[List[str]] = None,
                        test_size: float = 0.2) -> Dict[str, float]:
        """训练岩石分类器
        
        Args:
            features: 用于分类的特征列表，默认使用['vp', 'vs', 'density', 'porosity']
            test_size: 测试集比例
            
        Returns:
            Dict[str, float]: 训练结果报告
        """
        if features:
            self.classifier.features = features
            
        self.classifier.train_classifier()
        self.trained = True
        return {}
        
    def identify_rock(self, vp: float, vs: Optional[float] = None,
                     density: Optional[float] = None,
                     porosity: Optional[float] = None,
                     tectonic_setting: Optional[TectonicSetting] = None,
                     min_probability: float = 0.2,
                     max_candidates: int = 3) -> Optional[Dict[str, Any]]:
        """识别岩石类型，返回多个可能的候选岩石
        
        Args:
            vp: P波速度 (km/s)
            vs: S波速度 (km/s)，可选
            density: 密度 (g/cm³)，可选
            porosity: 孔隙度 (0-1)，可选
            tectonic_setting: 构造背景，可选
            min_probability: 最小可信概率
            max_candidates: 最大候选岩石数量
            
        Returns:
            Optional[Dict[str, Any]]: 识别结果，包含：
                - candidates: 候选岩石列表，每个候选包含：
                    - rock_type: 岩石类型
                    - probability: 预测概率
                    - similarity_score: 相似度分数
                    - tectonic_confidence: 构造背景置信度
                - properties: 输入的岩石物性参数
                - similar_rocks: 相似的岩石类型及其物性参数
        """
        if not self.trained:
            self.train_classifier()
            
        # 创建岩石物性实例
        properties = RockProperties(
            vp=vp,
            vs=vs,
            density=density,
            porosity=porosity,
            tectonic_setting=tectonic_setting
        )
        
        # 预测岩石类型
        try:
            probabilities = self.classifier.predict_proba(properties)
            
            # 获取所有概率大于阈值的岩石类型
            candidates = []
            for rock_type, prob in probabilities.items():
                if prob >= min_probability:
                    try:
                        # 计算与数据库中该类型岩石的相似度
                        consensus = self.database.get_consensus_properties(rock_type)
                        if consensus:
                            similarity = self._calculate_similarity(properties, consensus)
                            
                            # 计算构造背景置信度
                            tectonic_confidence = 1.0
                            if tectonic_setting and hasattr(self.database, 'get_tectonic_distribution'):
                                distribution = self.database.get_tectonic_distribution().get(rock_type, {})
                                total_samples = sum(distribution.values()) if distribution else 0
                                setting_samples = distribution.get(tectonic_setting, 0)
                                tectonic_confidence = setting_samples / total_samples if total_samples > 0 else 0.0
                            
                            candidates.append({
                                'rock_type': rock_type,
                                'probability': prob,
                                'similarity_score': similarity,
                                'tectonic_confidence': tectonic_confidence
                            })
                    except Exception as e:
                        print(f"处理岩石类型 {rock_type} 时出错: {str(e)}")
                        continue
            
            # 如果没有候选，返回None
            if not candidates:
                return None
                
            # 按综合分数排序（概率、相似度和构造背景置信度的加权）
            candidates.sort(
                key=lambda x: (
                    0.5 * x['probability'] + 
                    0.3 * x['similarity_score'] + 
                    0.2 * x['tectonic_confidence']
                ),
                reverse=True
            )
            
            # 限制候选数量
            candidates = candidates[:max_candidates]
            
            # 查找相似的岩石
            similar_rocks = self._find_similar_rocks(properties)
            
            return {
                'candidates': candidates,
                'properties': properties,
                'similar_rocks': similar_rocks
            }
            
        except Exception as e:
            print(f"岩石识别过程中出错: {str(e)}")
            return None
            
    def identify_velocity_model(self, model_data: Dict[str, np.ndarray],
                              min_probability: float = 0.6) -> Dict[str, List[Dict[str, Any]]]:
        """识别速度模型中的岩石类型
        
        Args:
            model_data: 速度模型数据，包含：
                - vp: P波速度数组
                - vs: S波速度数组（可选）
                - density: 密度数组（可选）
                - porosity: 孔隙度数组（可选）
            min_probability: 最小可信概率
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: 每个岩石类型的识别结果列表
        """
        if not self.trained:
            self.train_classifier()
            
        results = []
        vp = model_data['vp']
        vs = model_data.get('vs')
        density = model_data.get('density')
        porosity = model_data.get('porosity')
        
        for i in range(len(vp)):
            result = self.identify_rock(
                vp=vp[i],
                vs=vs[i] if vs is not None else None,
                density=density[i] if density is not None else None,
                porosity=porosity[i] if porosity is not None else None,
                min_probability=min_probability
            )
            if result:
                results.append(result)
                
        return self._group_results(results)
        
    def _find_similar_rocks(self, properties: RockProperties,
                           max_rocks: int = 3) -> List[Dict[str, Any]]:
        """查找相似的岩石
        
        Args:
            properties: 目标岩石物性
            max_rocks: 最大返回数量
            
        Returns:
            List[Dict[str, Any]]: 相似岩石列表
        """
        similarities = []
        
        for rock_type, measurements in self.database.rocks.items():
            consensus = self.database.get_consensus_properties(rock_type)
            if consensus:
                similarity = self._calculate_similarity(properties, consensus)
                similarities.append({
                    'rock_type': rock_type,
                    'similarity': similarity,
                    'properties': consensus
                })
                
        # 按相似度排序
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:max_rocks]
        
    def _calculate_similarity(self, props1: RockProperties,
                            props2: RockProperties) -> float:
        """计算两个岩石物性的相似度，考虑温度压力影响
        
        Args:
            props1: 第一个岩石物性
            props2: 第二个岩石物性
            
        Returns:
            float: 相似度分数 (0-1)
        """
        # 将两个物性都归一化到标准条件
        normalized_props1 = self._normalize_to_standard_conditions(props1)
        normalized_props2 = self._normalize_to_standard_conditions(props2)
        
        features = ['vp', 'vs', 'density', 'porosity']
        weights = {'vp': 0.4, 'vs': 0.3, 'density': 0.2, 'porosity': 0.1}
        
        total_weight = 0
        similarity = 0
        
        for feature in features:
            value1 = getattr(normalized_props1, feature)
            value2 = getattr(normalized_props2, feature)
            
            if value1 is not None and value2 is not None:
                weight = weights[feature]
                # 计算归一化差异
                max_val = max(abs(value1), abs(value2))
                diff = abs(value1 - value2) / max_val if max_val > 0 else 0
                similarity += weight * (1 - diff)
                total_weight += weight
                
        return similarity / total_weight if total_weight > 0 else 0
        
    def _group_results(self, results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """将识别结果按岩石类型分组
        
        Args:
            results: 识别结果列表
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: 分组后的结果
        """
        grouped = {}
        for result in results:
            if 'candidates' not in result:
                continue
            
            for candidate in result['candidates']:
                rock_type = candidate['rock_type']
                if rock_type not in grouped:
                    grouped[rock_type] = []
                # 创建包含候选信息和原始属性的完整结果
                complete_result = {
                    'rock_type': rock_type,
                    'probability': candidate['probability'],
                    'similarity_score': candidate['similarity_score'],
                    'properties': result['properties']
                }
                grouped[rock_type].append(complete_result)
            
        return grouped
        
    def plot_identification_results(self, results: Dict[str, List[Dict[str, Any]]],
                                  output_file: Optional[str] = None) -> None:
        """Plot visualization of rock identification results
        
        Args:
            results: Dictionary of identification results
            output_file: Output file path
        """
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Extract data
        vp_values = []
        temperatures = []
        pressures = []
        rock_types = []
        probabilities = []
        
        for rock_type, rock_results in results.items():
            for result in rock_results:
                vp_values.append(result['properties'].vp)
                temperatures.append(result['properties'].temperature)
                pressures.append(result['properties'].pressure)
                rock_types.append(rock_type)
                probabilities.append(result['probability'])
        
        unique_rock_types = list(set(rock_types))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_rock_types)))
        
        # Box plot
        box_data = []
        box_labels = []
        for rock_type in unique_rock_types:
            rock_vp = [vp for vp, rt in zip(vp_values, rock_types) if rt == rock_type]
            if rock_vp:  # Only add non-empty data
                box_data.append(rock_vp)
                box_labels.append(rock_type)
        
        if box_data:  # Ensure there is data to plot
            bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i])
            ax1.set_xlabel('Rock Type')
            ax1.set_ylabel('P-wave Velocity (km/s)')
            ax1.set_title('P-wave Velocity Distribution')
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Temperature distribution
        for i, rock_type in enumerate(unique_rock_types):
            temp_data = [t for t, rt in zip(temperatures, rock_types) if rt == rock_type]
            if temp_data:  # Only plot non-empty data
                ax2.hist(temp_data, alpha=0.5, label=rock_type, bins=10, color=colors[i])
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Count')
        ax2.set_title('Temperature Distribution')
        ax2.grid(True, alpha=0.3)
        if any(temp_data for rock_type in unique_rock_types
              for temp_data in [[t for t, rt in zip(temperatures, rock_types) if rt == rock_type]]):
            ax2.legend()
        
        # Pressure distribution
        for i, rock_type in enumerate(unique_rock_types):
            pressure_data = [p for p, rt in zip(pressures, rock_types) if rt == rock_type]
            if pressure_data:  # Only plot non-empty data
                ax3.hist(pressure_data, alpha=0.5, label=rock_type, bins=10, color=colors[i])
        ax3.set_xlabel('Pressure (MPa)')
        ax3.set_ylabel('Count')
        ax3.set_title('Pressure Distribution')
        ax3.grid(True, alpha=0.3)
        if any(pressure_data for rock_type in unique_rock_types
              for pressure_data in [[p for p, rt in zip(pressures, rock_types) if rt == rock_type]]):
            ax3.legend()
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
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
        'density': np.array([2.35, 2.55, 2.65, 2.85])
    } 