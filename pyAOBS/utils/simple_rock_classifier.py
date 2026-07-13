"""
simple_rock_classifier.py - 简化的岩石分类接口

提供简单易用的API，从速度模型直接进行岩石分类。
隐藏内部复杂的实现细节，用户只需提供速度数据即可。

Author: Haibo Huang
Date: 2025
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional, Tuple
from pathlib import Path
import warnings

# 延迟导入，避免循环依赖
try:
    from .isrock import RockIdentifier, TectonicSetting
    from .rocks import RockClassifier
    from .empirical_formulas import (
        calculate_density as _calculate_density,
        calculate_vs as _calculate_vs,
        calculate_pressure_from_depth,
        calculate_temperature_from_depth
    )
except ImportError:
    from isrock import RockIdentifier, TectonicSetting
    from rocks import RockClassifier
    from empirical_formulas import (
        calculate_density as _calculate_density,
        calculate_vs as _calculate_vs,
        calculate_pressure_from_depth,
        calculate_temperature_from_depth
    )


class SimpleRockClassifier:
    """简化的岩石分类器 - 一键式API
    
    使用示例:
        # 方式1: 使用默认数据库
        classifier = SimpleRockClassifier()
        
        # 方式2: 指定数据库文件
        classifier = SimpleRockClassifier(database_file='rocks.xlsx')
        
        # 分类单个点
        result = classifier.classify(vp=6.1, vs=3.5, density=2.65)
        
        # 分类速度模型
        results = classifier.classify_model(
            vp=[6.1, 6.5, 7.5],
            vs=[3.5, 3.7, 4.1],
            density=[2.65, 2.9, 3.0]
        )
    """
    
    def __init__(self, 
                 database_file: Optional[str] = None,
                 auto_correct: bool = True,
                 standard_pressure: float = 200.0,
                 standard_temperature: float = 25.0,
                 use_felsic_mafic: bool = False,
                 use_rock_facies: bool = False,
                 use_sio2: bool = False):
        """初始化简化分类器
        
        Args:
            database_file: 岩石数据库文件路径（Excel格式）
                         如果为None，尝试使用默认数据库
            auto_correct: 是否自动进行温度压力校正（默认True）
            standard_pressure: 标准压力 (MPa)，默认200.0
            standard_temperature: 标准温度 (°C)，默认25.0
            use_felsic_mafic: 是否启用 felsic/mafic 特征（默认False）
            use_rock_facies: 是否启用 rock facies 特征（默认False）
            use_sio2: 是否启用 SiO2 特征（默认False）
        """
        # 查找数据库文件
        if database_file is None:
            database_file = self._find_default_database()
        
        if database_file is None or not Path(database_file).exists():
            raise FileNotFoundError(
                f"找不到岩石数据库文件。请提供database_file参数，"
                f"或将数据库文件放在utils/rockdata目录下。"
            )
        
        self.database_file = database_file
        self.auto_correct = auto_correct
        self.standard_pressure = standard_pressure
        self.standard_temperature = standard_temperature
        self.use_felsic_mafic = bool(use_felsic_mafic)
        self.use_rock_facies = bool(use_rock_facies)
        self.use_sio2 = bool(use_sio2)
        
        # 初始化内部分类器（延迟加载）
        self._identifier = None
        self._classifier = None
    
    def _find_default_database(self) -> Optional[str]:
        """查找默认数据库文件"""
        # 优先使用统一维护目录 utils/rockdata
        rockdata_dir = Path(__file__).parent / 'rockdata'
        candidate = self._find_best_database_in_dir(rockdata_dir)
        if candidate is not None:
            return candidate

        # 回退到历史位置，保持兼容
        legacy_paths = [
            Path(__file__).parent / 'rocks.xlsx',
            Path(__file__).parent / 'rocks.csv',
            Path.cwd() / 'rocks.xlsx',
            Path.cwd() / 'rocks.csv',
        ]
        for path in legacy_paths:
            if path.exists():
                return str(path)
        return None

    def _find_best_database_in_dir(self, directory: Path) -> Optional[str]:
        """在目录中查找最合适的岩石数据库文件。"""
        if not directory.exists() or not directory.is_dir():
            return None

        # 先按常用命名做优先级，然后再尝试全量扫描
        preferred_names = [
            'rocks_merged.csv',
            'rocks.csv',
            'rocks.xlsx',
            'external_rock_data_template.csv',
        ]
        ordered_candidates: list[Path] = []
        for name in preferred_names:
            p = directory / name
            if p.exists() and p.is_file():
                ordered_candidates.append(p)

        for pattern in ('*.xlsx', '*.xlsm', '*.csv', '*.txt', '*.tsv'):
            for p in sorted(directory.rglob(pattern)):
                if p not in ordered_candidates and p.is_file():
                    ordered_candidates.append(p)

        best_path: Optional[Path] = None
        best_score: tuple[int, int] = (-1, -1)
        for path in ordered_candidates:
            score = self._score_database_file(path)
            if score > best_score:
                best_score = score
                best_path = path

        if best_path is None or best_score[0] < 3:
            return None
        return str(best_path)

    def _score_database_file(self, path: Path) -> tuple[int, int]:
        """根据字段匹配度与行数给数据库文件打分。"""
        try:
            df = self._read_table(path)
        except Exception:
            return (-1, -1)

        if df.empty:
            return (-1, -1)

        normalized_cols = {
            str(c).strip().lower().replace(' ', '_') for c in df.columns
        }
        groups = [
            {'rock_type', 'rocktype', 'rock_type_cn', '岩石类型', '岩性', 'type'},
            {'vp', 'p_wave_velocity', 'p_velocity', 'v_p', 'v'},
            {'vs', 's_wave_velocity', 's_velocity', 'v_s'},
            {'density', 'rho', '密度'},
            {'pressure', '压力'},
            {'temperature', '温度'},
        ]
        hit_count = 0
        for variants in groups:
            if any(v in normalized_cols for v in variants):
                hit_count += 1
        return (hit_count, int(len(df)))

    def _read_table(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix in {'.xlsx', '.xlsm'}:
            return pd.read_excel(path)
        if suffix in {'.txt', '.tsv'}:
            for enc in ('utf-8-sig', 'utf-8', 'gb18030', 'latin-1'):
                try:
                    return pd.read_csv(path, sep=None, engine='python', encoding=enc)
                except Exception:
                    continue
            raise ValueError(f'Unable to parse table file: {path}')

        for enc in ('utf-8-sig', 'utf-8', 'gb18030', 'latin-1'):
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                continue
        raise ValueError(f'Unable to parse csv file: {path}')
    
    @property
    def identifier(self):
        """延迟加载高级识别器"""
        if self._identifier is None:
            self._identifier = RockIdentifier(self.database_file)
            if self.auto_correct:
                # 自动训练分类器
                self._identifier.train_classifier()
        return self._identifier
    
    @property
    def classifier(self):
        """延迟加载基础分类器（用于简单分类）"""
        if self._classifier is None:
            try:
                self._classifier = RockClassifier(
                    self.database_file,
                    use_felsic_mafic=self.use_felsic_mafic,
                    use_rock_facies=self.use_rock_facies,
                    use_sio2=self.use_sio2,
                )
            except Exception as e:
                # 如果分类器初始化失败，记录错误但不抛出异常
                import warnings
                warnings.warn(f"Failed to initialize RockClassifier: {e}. Rock classification will be limited.")
                # 创建一个占位符分类器，返回UNKNOWN
                self._classifier = None
        return self._classifier
    
    def classify(self,
                 vp: float,
                 vs: Optional[float] = None,
                 density: Optional[float] = None,
                 felsic_or_mafic: Optional[str] = None,
                 rock_facies: Optional[str] = None,
                 sio2_wt: Optional[float] = None,
                 porosity: Optional[float] = None,
                 pressure: Optional[float] = None,
                 temperature: Optional[float] = None,
                 tectonic_setting: Optional[Union[str, TectonicSetting]] = None,
                 return_probabilities: bool = False) -> Union[str, Dict]:
        """分类单个样本
        
        Args:
            vp: P波速度 (km/s) - 必需
            vs: S波速度 (km/s) - 可选
            density: 密度 (g/cm³) - 可选
            felsic_or_mafic: 长英质/镁铁质标签 - 可选（需启用对应特征开关）
            rock_facies: 岩相标签 - 可选（需启用对应特征开关）
            sio2_wt: SiO2质量分数（wt.%）- 可选（需启用对应特征开关）
            porosity: 孔隙度 (0-1) - 可选
            pressure: 压力 (MPa) - 可选，用于校正
            temperature: 温度 (°C) - 可选，用于校正
            tectonic_setting: 构造环境 - 可选
            return_probabilities: 是否返回概率分布
            
        Returns:
            如果return_probabilities=False: 返回岩石类型字符串
            如果return_probabilities=True: 返回字典，包含岩石类型和概率
        """
        # 如果提供了温压信息且启用自动校正，使用高级识别器
        if self.auto_correct and (pressure is not None or temperature is not None):
            # 转换构造环境
            if isinstance(tectonic_setting, str):
                try:
                    tectonic_setting = TectonicSetting[tectonic_setting.upper()]
                except KeyError:
                    tectonic_setting = None
            
            # 使用高级识别器
            result = self.identifier.identify_rock(
                vp=vp,
                vs=vs or self._estimate_vs(vp),
                density=density or self._estimate_density(vp),
                porosity=porosity or 0.02,
                tectonic_setting=tectonic_setting,
                min_probability=0.1,
                max_candidates=5
            )
            
            aux = {}
            if self.classifier is not None:
                aux = self.classifier.get_auxiliary_attributes(
                    result['candidates'][0]['rock_type'],
                    vp=vp,
                    vs=vs,
                )

            if return_probabilities:
                return {
                    'rock_type': result['candidates'][0]['rock_type'],
                    'probability': result['candidates'][0]['probability'],
                    'all_candidates': result['candidates'],
                    'felsic_or_mafic': aux.get('felsic_or_mafic'),
                    'rock_facies': aux.get('rock_facies'),
                    'sio2_wt': aux.get('sio2_wt'),
                    'rock_classification': aux.get('rock_classification'),
                    'metamorphic_grade': aux.get('metamorphic_grade'),
                    'geological_meaning': aux.get('geological_meaning'),
                }
            else:
                return result['candidates'][0]['rock_type']
        
        # 简单分类（需要校正到标准条件）
        if self.classifier is None:
            # 如果分类器未初始化，返回UNKNOWN
            if return_probabilities:
                return {
                    'rock_type': 'UNKNOWN',
                    'probability': 0.0,
                    'all_candidates': []
                }
            else:
                return 'UNKNOWN'
        
        # 对输入速度进行压力和温度校正到标准条件（200MPa, 25°C）
        # 如果未提供压力和温度，假设已经是标准条件
        target_pressure = 200.0  # MPa
        target_temperature = 25.0  # °C
        
        vp_corrected = self.classifier.correct_velocity(
            vp, 
            pressure=pressure, 
            temperature=temperature,
            is_s_wave=False,
            target_pressure=target_pressure,
            target_temperature=target_temperature
        )
        
        vs_corrected = None
        if vs is not None:
            vs_corrected = self.classifier.correct_velocity(
                vs,
                pressure=pressure,
                temperature=temperature,
                is_s_wave=True,
                target_pressure=target_pressure,
                target_temperature=target_temperature
            )
        
        try:
            use_extended = self.use_felsic_mafic or self.use_rock_facies or self.use_sio2
            if use_extended:
                rock_type = self.classifier.classify_by_features(
                    vp=vp_corrected,
                    vs=vs_corrected,
                    density=density,
                    felsic_or_mafic=felsic_or_mafic,
                    rock_facies=rock_facies,
                    sio2_wt=sio2_wt,
                )
            elif vs_corrected is not None:
                rock_type = self.classifier.classify_by_vp_vs(vp_corrected, vs_corrected)
            else:
                rock_type = self.classifier.classify_by_vp(vp_corrected)
            
            if return_probabilities:
                # 获取概率分布（使用校正后的速度值）
                if use_extended:
                    probs = self.classifier.classify_probabilities_by_features(
                        vp=vp_corrected,
                        vs=vs_corrected,
                        density=density,
                        felsic_or_mafic=felsic_or_mafic,
                        rock_facies=rock_facies,
                        sio2_wt=sio2_wt,
                    )
                else:
                    probs = self.classifier.classify_with_uncertainty(vp_corrected, threshold=0.5)
                # 按概率排序
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                aux = self.classifier.get_auxiliary_attributes(
                    rock_type,
                    vp=vp_corrected,
                    vs=vs_corrected,
                )
                return {
                    'rock_type': rock_type,
                    'probability': probs.get(rock_type, 0.5),
                    'all_candidates': [{'rock_type': k, 'probability': v} 
                                     for k, v in sorted_probs[:5]],  # 只返回前5个
                    'felsic_or_mafic': aux.get('felsic_or_mafic'),
                    'rock_facies': aux.get('rock_facies'),
                    'sio2_wt': aux.get('sio2_wt'),
                    'rock_classification': aux.get('rock_classification'),
                    'metamorphic_grade': aux.get('metamorphic_grade'),
                    'geological_meaning': aux.get('geological_meaning'),
                }
            else:
                return rock_type
        except Exception as e:
            # 如果分类失败，返回UNKNOWN
            import warnings
            warnings.warn(f"Rock classification failed: {e}")
            if return_probabilities:
                return {
                    'rock_type': 'UNKNOWN',
                    'probability': 0.0,
                    'all_candidates': []
                }
            else:
                return 'UNKNOWN'
    
    def classify_model(self,
                      vp: Union[np.ndarray, List[float]],
                      vs: Optional[Union[np.ndarray, List[float]]] = None,
                      density: Optional[Union[np.ndarray, List[float]]] = None,
                      porosity: Optional[Union[np.ndarray, List[float]]] = None,
                      pressure: Optional[Union[np.ndarray, List[float]]] = None,
                      temperature: Optional[Union[np.ndarray, List[float]]] = None,
                      depth: Optional[Union[np.ndarray, List[float]]] = None,
                      tectonic_setting: Optional[Union[np.ndarray, List, str]] = None,
                      return_dataframe: bool = True) -> Union[pd.DataFrame, Dict]:
        """分类整个速度模型
        
        Args:
            vp: P波速度数组 (km/s) - 必需
            vs: S波速度数组 (km/s) - 可选
            density: 密度数组 (g/cm³) - 可选
            porosity: 孔隙度数组 (0-1) - 可选
            pressure: 压力数组 (MPa) - 可选
            temperature: 温度数组 (°C) - 可选
            depth: 深度数组 (km) - 可选，用于估算压力
            tectonic_setting: 构造环境 - 可选
            return_dataframe: 是否返回DataFrame格式（默认True）
            
        Returns:
            如果return_dataframe=True: 返回DataFrame，包含深度、速度、岩石类型等信息
            如果return_dataframe=False: 返回字典
        """
        # 转换为numpy数组
        vp = np.asarray(vp)
        n_points = len(vp)
        
        # 估算缺失参数
        if vs is None:
            vs = self._estimate_vs(vp)
        else:
            vs = np.asarray(vs)
        
        if density is None:
            density = self._estimate_density(vp)
        else:
            density = np.asarray(density)
        
        if porosity is None:
            porosity = np.full(n_points, 0.02)
        else:
            porosity = np.asarray(porosity)
        
        # 估算压力和温度（如果提供深度，使用统一的经验公式库）
        if pressure is None and depth is not None:
            depth = np.asarray(depth)
            pressure = calculate_pressure_from_depth(depth, pressure_gradient=30.0)
        elif pressure is None:
            pressure = np.full(n_points, self.standard_pressure)
        else:
            pressure = np.asarray(pressure)
        
        if temperature is None and depth is not None:
            depth = np.asarray(depth)
            # 如果有vp值，使用vp自动检测海水层
            if vp is not None:
                vp_array = np.asarray(vp)
                temperature = calculate_temperature_from_depth(
                    depth, temperature_gradient=30.0,
                    detect_seawater_from_vp=vp_array,
                    seawater_vp_threshold=1.6
                )
            else:
                temperature = calculate_temperature_from_depth(depth, temperature_gradient=30.0)
        elif temperature is None:
            temperature = np.full(n_points, self.standard_temperature)
        else:
            temperature = np.asarray(temperature)
        
        # 处理构造环境
        if isinstance(tectonic_setting, str):
            # 单个值，扩展到所有点
            try:
                setting = TectonicSetting[tectonic_setting.upper()]
                tectonic_setting = [setting] * n_points
            except KeyError:
                tectonic_setting = [TectonicSetting.UNKNOWN] * n_points
        elif tectonic_setting is None:
            tectonic_setting = [TectonicSetting.UNKNOWN] * n_points
        else:
            # 转换为TectonicSetting枚举
            settings = []
            for s in tectonic_setting:
                if isinstance(s, str):
                    try:
                        settings.append(TectonicSetting[s.upper()])
                    except KeyError:
                        settings.append(TectonicSetting.UNKNOWN)
                elif isinstance(s, TectonicSetting):
                    settings.append(s)
                else:
                    settings.append(TectonicSetting.UNKNOWN)
            tectonic_setting = settings
        
        # 如果启用自动校正，使用高级识别器
        if self.auto_correct:
            model_data = {
                'vp': vp,
                'vs': vs,
                'density': density,
                'porosity': porosity,
                'pressure': pressure,
                'temperature': temperature,
                'tectonic_setting': np.array(tectonic_setting)
            }
            
            results = self.identifier.identify_velocity_model(
                model_data,
                min_probability=0.1
            )
            
            # 转换为简单格式
            return self._format_model_results(
                results, vp, vs, density, depth, return_dataframe
            )
        
        # 简单分类（无校正）
        rock_types = []
        probabilities = []
        
        for i in range(n_points):
            if vs is not None and density is not None:
                rock_type = self.classifier.classify_by_vp_vs(vp[i], vs[i])
            elif vs is not None:
                rock_type = self.classifier.classify_by_vp_vs(vp[i], vs[i])
            else:
                rock_type = self.classifier.classify_by_vp(vp[i])
            
            # 获取概率
            probs = self.classifier.classify_with_uncertainty(vp[i], threshold=0.5)
            prob = probs.get(rock_type, 0.5)
            
            rock_types.append(rock_type)
            probabilities.append(prob)
        
        # 构建结果
        result_dict = {
            'vp': vp,
            'vs': vs,
            'density': density,
            'rock_type': rock_types,
            'probability': probabilities
        }
        
        if depth is not None:
            result_dict['depth'] = depth
        if pressure is not None:
            result_dict['pressure'] = pressure
        if temperature is not None:
            result_dict['temperature'] = temperature
        
        if return_dataframe:
            return pd.DataFrame(result_dict)
        else:
            return result_dict
    
    def _format_model_results(self, 
                              results: Dict,
                              vp: np.ndarray,
                              vs: np.ndarray,
                              density: np.ndarray,
                              depth: Optional[np.ndarray],
                              return_dataframe: bool) -> Union[pd.DataFrame, Dict]:
        """格式化模型识别结果
        
        identify_velocity_model返回的结果格式：
        {
            'ROCK_TYPE1': [
                {'properties': {...}, 'probability': 0.8, 'original_properties': {...}, ...},
                ...
            ],
            'ROCK_TYPE2': [...],
            ...
        }
        
        需要转换为每个点一个结果的格式
        """
        n_points = len(vp)
        
        # 为每个点找到最可能的岩石类型
        # 创建一个列表，每个元素是该点的所有候选结果
        point_results = [[] for _ in range(n_points)]
        
        # 遍历所有岩石类型的结果
        for rock_type, rock_data_list in results.items():
            for data in rock_data_list:
                # 从original_properties中找到对应的点索引
                orig_vp = data['original_properties']['vp']
                orig_vs = data['original_properties']['vs']
                
                # 找到匹配的点（允许小的数值误差）
                for i in range(n_points):
                    if (abs(orig_vp - vp[i]) < 0.1 and 
                        abs(orig_vs - vs[i]) < 0.1):
                        point_results[i].append({
                            'rock_type': rock_type,
                            'probability': data['probability']
                        })
                        break
        
        # 为每个点选择最可能的岩石类型
        rock_types = []
        probabilities = []
        
        for i, candidates in enumerate(point_results):
            if candidates:
                # 按概率排序，选择最高的
                best = max(candidates, key=lambda x: x['probability'])
                rock_types.append(best['rock_type'])
                probabilities.append(best['probability'])
            else:
                # 如果没有找到匹配，使用简单分类
                if vs is not None:
                    rock_type = self.classifier.classify_by_vp_vs(vp[i], vs[i])
                else:
                    rock_type = self.classifier.classify_by_vp(vp[i])
                rock_types.append(rock_type)
                probabilities.append(0.5)  # 默认概率
        
        # 构建结果
        result_dict = {
            'vp': vp,
            'vs': vs,
            'density': density,
            'rock_type': rock_types,
            'probability': probabilities
        }
        
        if depth is not None:
            result_dict['depth'] = depth
        
        if return_dataframe:
            return pd.DataFrame(result_dict)
        else:
            return result_dict
    
    def _estimate_vs(self, vp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """使用Brocher公式估算S波速度（使用经验公式库）"""
        return _calculate_vs(vp, method='brocher')
    
    def _estimate_density(self, vp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """使用Gardner公式估算密度（使用经验公式库）
        
        标准Gardner公式: ρ = 0.31 × (Vp in m/s)^0.25
        其中 Vp 单位是 km/s，需要转换为 m/s
        结果单位是 g/cm³
        """
        return _calculate_density(vp, method='gardner')
    
    def classify_from_file(self,
                          model_file: str,
                          vp_column: str = 'vp',
                          vs_column: Optional[str] = None,
                          density_column: Optional[str] = None,
                          depth_column: Optional[str] = None,
                          output_file: Optional[str] = None) -> pd.DataFrame:
        """从文件加载速度模型并分类
        
        Args:
            model_file: 模型文件路径（支持CSV、Excel、文本文件）
            vp_column: P波速度列名，默认'vp'
            vs_column: S波速度列名，可选
            density_column: 密度列名，可选
            depth_column: 深度列名，可选
            output_file: 输出文件路径（可选）
            
        Returns:
            DataFrame: 包含原始数据和分类结果
        """
        # 读取文件
        file_path = Path(model_file)
        if file_path.suffix.lower() == '.xlsx' or file_path.suffix.lower() == '.xls':
            df = pd.read_excel(model_file)
        elif file_path.suffix.lower() == '.csv':
            df = pd.read_csv(model_file)
        else:
            # 尝试按空格分隔的文本文件
            df = pd.read_csv(model_file, sep=r'\s+')
        
        # 提取数据
        vp = df[vp_column].values
        vs = df[vs_column].values if vs_column and vs_column in df.columns else None
        density = df[density_column].values if density_column and density_column in df.columns else None
        depth = df[depth_column].values if depth_column and depth_column in df.columns else None
        
        # 分类
        results = self.classify_model(
            vp=vp,
            vs=vs,
            density=density,
            depth=depth,
            return_dataframe=True
        )
        
        # 合并原始数据
        result_df = df.copy()
        result_df['rock_type'] = results['rock_type'].values
        result_df['rock_probability'] = results['probability'].values
        
        # 保存结果
        if output_file:
            output_path = Path(output_file)
            if output_path.suffix.lower() == '.xlsx':
                result_df.to_excel(output_file, index=False)
            else:
                result_df.to_csv(output_file, index=False)
        
        return result_df


# 便捷函数
def classify_velocity_model(vp: Union[float, np.ndarray, List[float]],
                           vs: Optional[Union[float, np.ndarray, List[float]]] = None,
                           density: Optional[Union[float, np.ndarray, List[float]]] = None,
                           database_file: Optional[str] = None,
                           **kwargs) -> Union[str, pd.DataFrame, Dict]:
    """一键式速度模型分类函数
    
    这是最简单的使用方式，只需提供速度数据即可。
    
    Args:
        vp: P波速度 (km/s) - 必需，可以是单个值或数组
        vs: S波速度 (km/s) - 可选
        density: 密度 (g/cm³) - 可选
        database_file: 数据库文件路径 - 可选
        **kwargs: 其他参数（depth, pressure, temperature等）
        
    Returns:
        如果输入是单个值: 返回岩石类型字符串
        如果输入是数组: 返回DataFrame
        
    示例:
        # 分类单个点
        rock_type = classify_velocity_model(vp=6.1, vs=3.5)
        
        # 分类模型
        results = classify_velocity_model(
            vp=[6.1, 6.5, 7.5],
            vs=[3.5, 3.7, 4.1],
            depth=[0, 10, 20]
        )
    """
    classifier = SimpleRockClassifier(database_file=database_file)
    
    # 判断是单个值还是数组
    vp_array = np.asarray(vp)
    is_single = vp_array.ndim == 0 or len(vp_array) == 1
    
    if is_single:
        # 单个值
        return classifier.classify(
            vp=float(vp_array) if vp_array.ndim == 0 else float(vp_array[0]),
            vs=float(vs) if vs is not None else None,
            density=float(density) if density is not None else None,
            **kwargs
        )
    else:
        # 数组
        return classifier.classify_model(
            vp=vp,
            vs=vs,
            density=density,
            **kwargs
        )


def classify_from_file(model_file: str,
                      vp_column: str = 'vp',
                      vs_column: Optional[str] = None,
                      density_column: Optional[str] = None,
                      database_file: Optional[str] = None,
                      output_file: Optional[str] = None) -> pd.DataFrame:
    """从文件加载并分类速度模型
    
    示例:
        results = classify_from_file(
            'velocity_model.csv',
            vp_column='Vp',
            depth_column='depth',
            output_file='classified_model.csv'
        )
    """
    classifier = SimpleRockClassifier(database_file=database_file)
    return classifier.classify_from_file(
        model_file=model_file,
        vp_column=vp_column,
        vs_column=vs_column,
        density_column=density_column,
        output_file=output_file
    )
