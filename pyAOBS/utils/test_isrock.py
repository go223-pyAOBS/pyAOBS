"""
test_isrock.py - 岩石识别模块的演示测试代码

Author: Haibo Huang
Date: 2024
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from .isrock import RockIdentifier, TectonicSetting
from .rocks import RockClassifier, RockProperties
import matplotlib.pyplot as plt

class TestRockClassificationBase(unittest.TestCase):
    """测试基础岩石分类功能（RockClassifier）"""
    
    def setUp(self):
        """设置测试环境"""
        self.output_dir = "./rock_classification_demo"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # 创建测试数据库
        self.db_file = os.path.join(self.output_dir, "demo_database.xlsx")
        
        # 创建测试数据
        test_data = {
            'rock_type': [
                'GRANITE', 'GRANITE', 'GRANITE',  # 多个花岗岩样本
                'BASALT', 'BASALT', 'BASALT',     # 多个玄武岩样本
                'GABBRO', 'GABBRO', 'GABBRO',     # 多个辉长岩样本
                'PERIDOTITE', 'PERIDOTITE'        # 多个橄榄岩样本
            ],
            'vp': [
                6.1, 5.8, 6.3,    # 花岗岩的不同Vp值
                6.5, 6.8, 6.2,    # 玄武岩的不同Vp值
                7.0, 7.2, 6.8,    # 辉长岩的不同Vp值
                8.1, 7.8          # 橄榄岩的不同Vp值
            ],
            'vs': [
                3.5, 3.3, 3.6,    # 花岗岩的不同Vs值
                3.7, 3.9, 3.5,    # 玄武岩的不同Vs值
                3.9, 4.1, 3.8,    # 辉长岩的不同Vs值
                4.5, 4.3          # 橄榄岩的不同Vs值
            ],
            'density': [
                2.65, 2.62, 2.67,    # 花岗岩的不同密度值
                2.9, 2.95, 2.85,     # 玄武岩的不同密度值
                3.0, 3.1, 2.95,      # 辉长岩的不同密度值
                3.3, 3.25            # 橄榄岩的不同密度值
            ],
            'porosity': [0.02] * 11,  # 添加孔隙度数据
            'source': ['demo'] * 11,   # 添加数据来源
            'method': ['lab'] * 11,    # 添加测量方法
            'temperature': [
                100.0, 150.0, 200.0,  # 花岗岩（高温）
                150.0, 200.0, 250.0,  # 玄武岩（更高温）
                200.0, 250.0, 300.0,  # 辉长岩（非常高温）
                300.0, 350.0          # 橄榄岩（极高温）
            ],
            'pressure': [
                300.0, 400.0, 500.0,  # 花岗岩（高压）
                400.0, 500.0, 600.0,  # 玄武岩（更高压）
                500.0, 600.0, 700.0,  # 辉长岩（非常高压）
                700.0, 800.0          # 橄榄岩（极高压）
            ],
            'tectonic_setting': [
                TectonicSetting.CONTINENTAL_CRUST.value,
                TectonicSetting.CONTINENTAL_CRUST.value,
                TectonicSetting.CONTINENTAL_CRUST.value,
                TectonicSetting.OCEANIC_CRUST.value,
                TectonicSetting.OCEANIC_CRUST.value,
                TectonicSetting.OCEANIC_CRUST.value,
                TectonicSetting.OCEANIC_CRUST.value,
                TectonicSetting.OCEANIC_CRUST.value,
                TectonicSetting.OCEANIC_CRUST.value,
                TectonicSetting.SUBDUCTION_ZONE.value,
                TectonicSetting.SUBDUCTION_ZONE.value
            ]
        }
        
        # 保存测试数据
        self.test_df = pd.DataFrame(test_data)
        self.test_df.to_excel(self.db_file, index=False)
        
        # 初始化分类器
        self.classifier = RockClassifier(database=self.db_file)
        
    def test_classify_by_vp(self):
        """测试基于P波速度的岩石分类"""
        # 使用数据集中的实际值进行测试
        vp = 6.1  # 花岗岩的实际P波速度
        result = self.classifier.classify_by_vp(vp)
        self.assertEqual(result, 'GRANITE')
        
        vp = 6.5  # 玄武岩的实际P波速度
        result = self.classifier.classify_by_vp(vp)
        self.assertEqual(result, 'BASALT')
        
        vp = 7.0  # 辉长岩的实际P波速度
        result = self.classifier.classify_by_vp(vp)
        self.assertEqual(result, 'GABBRO')
        
        vp = 8.1  # 橄榄岩的实际P波速度
        result = self.classifier.classify_by_vp(vp)
        self.assertEqual(result, 'PERIDOTITE')
    
    def test_classify_by_vp_vs(self):
        """测试基于P波和S波速度的岩石分类"""
        # 使用数据集中的实际值进行测试
        vp, vs = 6.1, 3.5  # 花岗岩的实际值
        result = self.classifier.classify_by_vp_vs(vp, vs)
        self.assertEqual(result, 'GRANITE')
        
        vp, vs = 6.5, 3.7  # 玄武岩的实际值
        result = self.classifier.classify_by_vp_vs(vp, vs)
        self.assertEqual(result, 'BASALT')
        
        vp, vs = 7.0, 3.9  # 辉长岩的实际值
        result = self.classifier.classify_by_vp_vs(vp, vs)
        self.assertEqual(result, 'GABBRO')
        
        vp, vs = 8.1, 4.5  # 橄榄岩的实际值
        result = self.classifier.classify_by_vp_vs(vp, vs)
        self.assertEqual(result, 'PERIDOTITE')
    
    def test_classify_with_uncertainty(self):
        """测试带不确定性的岩石分类"""
        # 测试边界值情况
        vp = 6.15  # 在花岗岩的多个样本范围内
        result = self.classifier.classify_with_uncertainty(vp, threshold=0.3)
        self.assertIsInstance(result, dict)
        self.assertTrue('GRANITE' in result)
        self.assertGreater(result['GRANITE'], 0.3)  # 确保花岗岩有足够高的概率
        
        # 测试明确的情况
        vp = 8.1  # 明确的橄榄岩P波速度
        result = self.classifier.classify_with_uncertainty(vp, threshold=0.2)
        self.assertIsInstance(result, dict)
        self.assertTrue('PERIDOTITE' in result)
        self.assertGreaterEqual(result['PERIDOTITE'], 0.5)  # 使用大于等于而不是严格大于
    
    def test_classify_by_tectonic_setting(self):
        """测试考虑构造环境的岩石分类"""
        # 使用数据集中的实际值进行测试
        vp = 6.1  # 花岗岩的实际P波速度
        setting = TectonicSetting.CONTINENTAL_CRUST
        result = self.classifier.classify_by_vp_and_setting(vp, setting)
        self.assertEqual(result, 'GRANITE')
        
        vp = 6.5  # 玄武岩的实际P波速度
        setting = TectonicSetting.OCEANIC_CRUST
        result = self.classifier.classify_by_vp_and_setting(vp, setting)
        self.assertEqual(result, 'BASALT')
        
        vp = 8.1  # 橄榄岩的实际P波速度
        setting = TectonicSetting.SUBDUCTION_ZONE
        result = self.classifier.classify_by_vp_and_setting(vp, setting)
        self.assertEqual(result, 'PERIDOTITE')

class TestRockIdentification(unittest.TestCase):
    """测试高级岩石识别功能（RockIdentifier）"""
    
    def setUp(self):
        """设置测试环境"""
        self.output_dir = "./rock_classification_demo"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # 创建测试数据库
        self.db_file = os.path.join(self.output_dir, "demo_database.xlsx")
        
        # 创建第一组测试数据（多个样本的基础数据）
        test_data_1 = {
            'rock_type': [
                'GRANITE', 'GRANITE', 'GRANITE',  # 多个花岗岩样本
                'BASALT', 'BASALT', 'BASALT',     # 多个玄武岩样本
                'GABBRO', 'GABBRO', 'GABBRO',     # 多个辉长岩样本
                'PERIDOTITE', 'PERIDOTITE'        # 多个橄榄岩样本
            ],
            'vp': [
                6.1, 5.8, 6.3,    # 花岗岩的不同Vp值
                6.5, 6.8, 6.2,    # 玄武岩的不同Vp值
                7.0, 7.2, 6.8,    # 辉长岩的不同Vp值
                8.1, 7.8          # 橄榄岩的不同Vp值
            ],
            'vs': [
                3.5, 3.3, 3.6,    # 花岗岩的不同Vs值
                3.7, 3.9, 3.5,    # 玄武岩的不同Vs值
                3.9, 4.1, 3.8,    # 辉长岩的不同Vs值
                4.5, 4.3          # 橄榄岩的不同Vs值
            ],
            'density': [
                2.65, 2.62, 2.67,    # 花岗岩的不同密度值
                2.9, 2.95, 2.85,     # 玄武岩的不同密度值
                3.0, 3.1, 2.95,      # 辉长岩的不同密度值
                3.3, 3.25            # 橄榄岩的不同密度值
            ],
            'porosity': [0.02] * 11,  # 添加孔隙度数据
            'source': ['demo'] * 11,   # 添加数据来源
            'method': ['lab'] * 11,    # 添加测量方法
            'temperature': [
                100.0, 150.0, 200.0,  # 花岗岩（高温）
                150.0, 200.0, 250.0,  # 玄武岩（更高温）
                200.0, 250.0, 300.0,  # 辉长岩（非常高温）
                300.0, 350.0          # 橄榄岩（极高温）
            ],
            'pressure': [
                300.0, 400.0, 500.0,  # 花岗岩（高压）
                400.0, 500.0, 600.0,  # 玄武岩（更高压）
                500.0, 600.0, 700.0,  # 辉长岩（非常高压）
                700.0, 800.0          # 橄榄岩（极高压）
            ],
            'tectonic_setting': [
                TectonicSetting.CONTINENTAL_CRUST.value,
                TectonicSetting.CONTINENTAL_CRUST.value,
                TectonicSetting.CONTINENTAL_CRUST.value,
                TectonicSetting.OCEANIC_CRUST.value,
                TectonicSetting.OCEANIC_CRUST.value,
                TectonicSetting.OCEANIC_CRUST.value,
                TectonicSetting.OCEANIC_CRUST.value,
                TectonicSetting.OCEANIC_CRUST.value,
                TectonicSetting.OCEANIC_CRUST.value,
                TectonicSetting.SUBDUCTION_ZONE.value,
                TectonicSetting.SUBDUCTION_ZONE.value
            ]
        }
        
        # 创建第二组测试数据（更多岩石类型和更复杂的物理属性）
        test_data_2 = {
            'rock_type': [
                'ECLOGITE', 'ECLOGITE', 'ECLOGITE',           # 榴辉岩
                'AMPHIBOLITE', 'AMPHIBOLITE', 'AMPHIBOLITE',  # 角闪岩
                'MARBLE', 'MARBLE', 'MARBLE',                  # 大理岩
                'QUARTZITE', 'QUARTZITE', 'QUARTZITE',        # 石英岩
                'GNEISS', 'GNEISS', 'GNEISS',                 # 片麻岩
                'SCHIST', 'SCHIST', 'SCHIST',                 # 片岩
                'SLATE', 'SLATE', 'SLATE',                    # 板岩
                'LIMESTONE', 'LIMESTONE', 'LIMESTONE',         # 石灰岩
                'SANDSTONE', 'SANDSTONE', 'SANDSTONE',        # 砂岩
                'SHALE', 'SHALE', 'SHALE'                     # 页岩
            ],
            'vp': [
                8.0, 7.8, 8.2,     # 榴辉岩
                6.8, 6.6, 7.0,     # 角闪岩
                6.2, 6.0, 6.4,     # 大理岩
                6.0, 5.8, 6.2,     # 石英岩
                6.2, 6.0, 6.4,     # 片麻岩
                5.8, 5.6, 6.0,     # 片岩
                5.5, 5.3, 5.7,     # 板岩
                5.0, 4.8, 5.2,     # 石灰岩
                4.5, 4.3, 4.7,     # 砂岩
                4.0, 3.8, 4.2      # 页岩
            ],
            'vs': [
                4.4, 4.2, 4.6,     # 榴辉岩
                3.8, 3.6, 4.0,     # 角闪岩
                3.4, 3.2, 3.6,     # 大理岩
                3.7, 3.5, 3.9,     # 石英岩
                3.5, 3.3, 3.7,     # 片麻岩
                3.3, 3.1, 3.5,     # 片岩
                3.0, 2.8, 3.2,     # 板岩
                2.8, 2.6, 3.0,     # 石灰岩
                2.5, 2.3, 2.7,     # 砂岩
                2.2, 2.0, 2.4      # 页岩
            ],
            'density': [
                3.4, 3.3, 3.5,     # 榴辉岩
                3.0, 2.9, 3.1,     # 角闪岩
                2.7, 2.6, 2.8,     # 大理岩
                2.65, 2.60, 2.70,  # 石英岩
                2.8, 2.7, 2.9,     # 片麻岩
                2.75, 2.65, 2.85,  # 片岩
                2.7, 2.6, 2.8,     # 板岩
                2.6, 2.5, 2.7,     # 石灰岩
                2.4, 2.3, 2.5,     # 砂岩
                2.3, 2.2, 2.4      # 页岩
            ],
            'porosity': [
                0.01, 0.015, 0.005,  # 榴辉岩
                0.02, 0.025, 0.015,  # 角闪岩
                0.03, 0.035, 0.025,  # 大理岩
                0.02, 0.025, 0.015,  # 石英岩
                0.01, 0.015, 0.005,  # 片麻岩
                0.03, 0.035, 0.025,  # 片岩
                0.05, 0.055, 0.045,  # 板岩
                0.10, 0.12, 0.08,    # 石灰岩
                0.15, 0.18, 0.12,    # 砂岩
                0.20, 0.25, 0.15     # 页岩
            ],
            'source': ['demo'] * 30,
            'method': ['lab'] * 30,
            'temperature': [25.0] * 30,
            'pressure': [
                500.0, 500.0, 500.0,  # 榴辉岩（高压）
                300.0, 300.0, 300.0,  # 角闪岩
                200.0, 200.0, 200.0,  # 大理岩
                200.0, 200.0, 200.0,  # 石英岩
                300.0, 300.0, 300.0,  # 片麻岩
                200.0, 200.0, 200.0,  # 片岩
                100.0, 100.0, 100.0,  # 板岩
                50.0, 50.0, 50.0,     # 石灰岩
                50.0, 50.0, 50.0,     # 砂岩
                50.0, 50.0, 50.0      # 页岩
            ],
            'tectonic_setting': [
                TectonicSetting.SUBDUCTION_ZONE.value,    # 榴辉岩
                TectonicSetting.SUBDUCTION_ZONE.value,
                TectonicSetting.SUBDUCTION_ZONE.value,
                TectonicSetting.CONTINENTAL_CRUST.value,  # 角闪岩
                TectonicSetting.CONTINENTAL_CRUST.value,
                TectonicSetting.CONTINENTAL_CRUST.value,
                TectonicSetting.CONTINENTAL_CRUST.value,  # 大理岩
                TectonicSetting.CONTINENTAL_CRUST.value,
                TectonicSetting.CONTINENTAL_CRUST.value,
                TectonicSetting.CONTINENTAL_CRUST.value,  # 石英岩
                TectonicSetting.CONTINENTAL_CRUST.value,
                TectonicSetting.CONTINENTAL_CRUST.value,
                TectonicSetting.CONTINENTAL_CRUST.value,  # 片麻岩
                TectonicSetting.CONTINENTAL_CRUST.value,
                TectonicSetting.CONTINENTAL_CRUST.value,
                TectonicSetting.CONTINENTAL_CRUST.value,  # 片岩
                TectonicSetting.CONTINENTAL_CRUST.value,
                TectonicSetting.CONTINENTAL_CRUST.value,
                TectonicSetting.CONTINENTAL_CRUST.value,  # 板岩
                TectonicSetting.CONTINENTAL_CRUST.value,
                TectonicSetting.CONTINENTAL_CRUST.value,
                TectonicSetting.CONTINENTAL_CRUST.value,  # 石灰岩
                TectonicSetting.CONTINENTAL_CRUST.value,
                TectonicSetting.CONTINENTAL_CRUST.value,
                TectonicSetting.CONTINENTAL_CRUST.value,  # 砂岩
                TectonicSetting.CONTINENTAL_CRUST.value,
                TectonicSetting.CONTINENTAL_CRUST.value,
                TectonicSetting.CONTINENTAL_CRUST.value,  # 页岩
                TectonicSetting.CONTINENTAL_CRUST.value,
                TectonicSetting.CONTINENTAL_CRUST.value
            ]
        }
        
        # 合并两组数据
        df1 = pd.DataFrame(test_data_1)
        df2 = pd.DataFrame(test_data_2)
        self.test_df = pd.concat([df1, df2], ignore_index=True)
        
        # 保存原始数据集
        original_data_file = os.path.join(self.output_dir, "original_database.xlsx")
        self.test_df.to_excel(original_data_file, index=False)
        print(f"\n原始数据已保存到: {original_data_file}")
        
        # 初始化识别器（会自动进行数据校正）
        self.identifier = RockIdentifier(self.test_df)
        
        # 保存校正后的数据集
        corrected_data_file = os.path.join(self.output_dir, "corrected_database.xlsx")
        self.identifier.rock_database.to_excel(corrected_data_file, index=False)
        print(f"校正后的数据已保存到: {corrected_data_file}")
        
        # 输出校正前后的对比
        print("\n校正前后的数据对比:")
        print("\n平均值对比:")
        original_means = self.test_df[['vp', 'vs', 'density']].mean()
        corrected_means = self.identifier.rock_database[['vp', 'vs', 'density']].mean()
        print("原始数据平均值:")
        print(original_means)
        print("\n校正后数据平均值:")
        print(corrected_means)
        
        print("\n标准差对比:")
        original_stds = self.test_df[['vp', 'vs', 'density']].std()
        corrected_stds = self.identifier.rock_database[['vp', 'vs', 'density']].std()
        print("原始数据标准差:")
        print(original_stds)
        print("\n校正后数据标准差:")
        print(corrected_stds)
        
        # 绘制校正前后的对比图
        self._plot_correction_comparison()
    
    def _plot_correction_comparison(self):
        """绘制校正前后的对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 对比Vp
        axes[0].scatter(self.test_df['vp'], self.identifier.rock_database['vp'], alpha=0.5)
        axes[0].plot([4, 9], [4, 9], 'r--')  # 对角线
        axes[0].set_xlabel('Original Vp (km/s)')
        axes[0].set_ylabel('Corrected Vp (km/s)')
        axes[0].set_title('P-wave Velocity Correction')
        
        # 对比Vs
        axes[1].scatter(self.test_df['vs'], self.identifier.rock_database['vs'], alpha=0.5)
        axes[1].plot([2, 5], [2, 5], 'r--')  # 对角线
        axes[1].set_xlabel('Original Vs (km/s)')
        axes[1].set_ylabel('Corrected Vs (km/s)')
        axes[1].set_title('S-wave Velocity Correction')
        
        # 对比密度
        axes[2].scatter(self.test_df['density'], self.identifier.rock_database['density'], alpha=0.5)
        axes[2].plot([2, 3.5], [2, 3.5], 'r--')  # 对角线
        axes[2].set_xlabel('Original Density (g/cm³)')
        axes[2].set_ylabel('Corrected Density (g/cm³)')
        axes[2].set_title('Density Correction')
        
        plt.tight_layout()
        comparison_plot_file = os.path.join(self.output_dir, "correction_comparison.png")
        plt.savefig(comparison_plot_file)
        plt.close()
        print(f"\n校正对比图已保存到: {comparison_plot_file}")

    def test_basic_identification(self):
        """测试基本的岩石识别功能"""
        # 1. 典型大陆地壳岩石 (花岗岩)
        granite_result = self.identifier.identify_rock(
            vp=6.1,
            vs=3.5,
            density=2.65,
            porosity=0.02,
            tectonic_setting=TectonicSetting.CONTINENTAL_CRUST,
            min_probability=0.1,
            max_candidates=5
        )
        
        # 2. 典型大洋地壳岩石 (玄武岩/辉长岩)
        oceanic_result = self.identifier.identify_rock(
            vp=6.5,
            vs=3.7,
            density=2.9,
            porosity=0.02,
            tectonic_setting=TectonicSetting.OCEANIC_CRUST,
            min_probability=0.1,
            max_candidates=5
        )
        
        # 3. 上地幔岩石 (橄榄岩)
        mantle_result = self.identifier.identify_rock(
            vp=8.1,
            vs=4.5,
            density=3.3,
            porosity=0.02,
            tectonic_setting=TectonicSetting.SUBDUCTION_ZONE,
            min_probability=0.1,
            max_candidates=5
        )
        
        # 保存分类结果
        results = {
            'granite': granite_result,
            'oceanic': oceanic_result,
            'mantle': mantle_result
        }
        
        # 为每个结果生成可视化
        for name, result in results.items():
            if result and 'candidates' in result:
                plot_file = os.path.join(self.output_dir, f"{name}_classification.png")
                self.identifier.plot_identification_results({
                    rock_type: [{'properties': result['properties'], 
                               'probability': candidate['probability']}]
                    for candidate in result['candidates']
                    for rock_type in [candidate['rock_type']]
                }, plot_file)
        
        # 验证结果
        self.assertIsNotNone(granite_result)
        self.assertIsNotNone(oceanic_result)
        self.assertIsNotNone(mantle_result)
        
        # 检查花岗岩结果
        granite_candidates = granite_result['candidates']
        self.assertGreater(len(granite_candidates), 0)
        self.assertEqual(granite_candidates[0]['rock_type'], 'GRANITE')
        
        # 检查大洋岩石结果
        oceanic_candidates = oceanic_result['candidates']
        self.assertGreater(len(oceanic_candidates), 0)
        self.assertIn(oceanic_candidates[0]['rock_type'], ['BASALT', 'GABBRO'])
        
        # 检查地幔岩石结果
        mantle_candidates = mantle_result['candidates']
        self.assertGreater(len(mantle_candidates), 0)
        self.assertEqual(mantle_candidates[0]['rock_type'], 'PERIDOTITE')
    
    def test_velocity_model_identification(self):
        """测试速度模型的岩石识别，包括压力和温度校正"""
        # 创建不同深度、压力和温度条件下的速度模型数据
        depths = np.array([0, 10, 20, 30])  # 深度（km）
        pressures = depths * 30  # MPa (假设压力梯度为30 MPa/km)
        temperatures = 25 + depths * 30  # °C (假设地温梯度为30°C/km)
        
        model_data = {
            'depth': depths,  # 深度（km）
            'vp': np.array([6.1, 6.5, 7.5, 8.1]),  # 原始P波速度
            'vs': np.array([3.5, 3.7, 4.1, 4.5]),  # 原始S波速度
            'density': np.array([2.65, 2.9, 3.0, 3.3]),  # 原始密度
            'porosity': np.array([0.02, 0.02, 0.02, 0.02]),  # 孔隙度
            'pressure': pressures,  # 压力值（MPa）
            'temperature': temperatures,  # 温度值（°C）
            'tectonic_setting': [
                TectonicSetting.CONTINENTAL_CRUST,  # 0-10km，大陆地壳
                TectonicSetting.CONTINENTAL_CRUST,  # 10-20km，大陆地壳
                TectonicSetting.OCEANIC_CRUST,      # 20-30km，大洋地壳
                TectonicSetting.SUBDUCTION_ZONE     # >30km，俯冲带
            ]
        }
        
        # 1. 测试压力校正函数
        corrected_vp = self.identifier.pressure_correction(
            model_data['vp'],
            model_data['pressure'],
            target_pressure=200.0  # 校正到标准压力（200 MPa）
        )
        self.assertIsNotNone(corrected_vp)
        self.assertEqual(len(corrected_vp), len(model_data['vp']))
        
        # 打印压力校正结果
        print("\n压力校正结果：")
        for i, (orig, corr) in enumerate(zip(model_data['vp'], corrected_vp)):
            print(f"深度{depths[i]}km: 原始Vp={orig:.2f}, 校正后Vp={corr:.2f}")
        
        corrected_vs = self.identifier.pressure_correction(
            model_data['vs'],
            model_data['pressure'],
            target_pressure=200.0
        )
        self.assertIsNotNone(corrected_vs)
        self.assertEqual(len(corrected_vs), len(model_data['vs']))
        
        # 2. 测试温度校正函数
        corrected_vp = self.identifier.temperature_correction(
            corrected_vp,
            model_data['temperature'],
            target_temperature=25.0  # 校正到标准温度（25°C）
        )
        self.assertIsNotNone(corrected_vp)
        
        # 打印温度校正结果
        print("\n温度校正结果：")
        for i, vp in enumerate(corrected_vp):
            print(f"深度{depths[i]}km: 最终校正Vp={vp:.2f}")
        
        corrected_vs = self.identifier.temperature_correction(
            corrected_vs,
            model_data['temperature'],
            target_temperature=25.0
        )
        self.assertIsNotNone(corrected_vs)
        
        # 3. 验证校正后的速度是否在合理范围内
        self.assertTrue(np.all(corrected_vp > 0))  # 速度应该为正
        self.assertTrue(np.all(corrected_vs > 0))
        self.assertTrue(np.all(corrected_vp > corrected_vs))  # P波速度应大于S波速度
        
        # 4. 使用校正后的数据进行岩石识别
        results = self.identifier.identify_velocity_model(
            model_data,
            min_probability=0.1
        )
        
        # 打印识别结果
        print("\n岩石识别结果：")
        for depth_idx, depth in enumerate(depths):
            print(f"\n深度{depth}km的识别结果：")
            found_rocks = []
            for rock_type, rock_data in results.items():
                if depth_idx < len(rock_data):
                    prob = rock_data[depth_idx]['probability']
                    print(f"{rock_type}: {prob:.3f}")
                    found_rocks.append((rock_type, prob))
            
            # 按概率排序
            found_rocks.sort(key=lambda x: x[1], reverse=True)
            if found_rocks:
                print(f"最可能的岩石类型: {found_rocks[0][0]} (概率: {found_rocks[0][1]:.3f})")
        
        # 保存结果图像
        plot_file = os.path.join(self.output_dir, "velocity_model_classification.png")
        self.identifier.plot_identification_results(results, plot_file)
        
        # 验证结果
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        # 检查是否识别出了主要的岩石类型
        rock_types = set(results.keys())
        expected_rocks = {'GRANITE', 'BASALT', 'GABBRO', 'PERIDOTITE'}
        
        # 至少应该识别出一个预期的岩石类型
        self.assertTrue(len(rock_types.intersection(expected_rocks)) > 0)
        
        # 5. 测试不同深度的识别结果是否合理
        for i, depth in enumerate(depths):
            depth_results = []
            for rock_type, rock_data in results.items():
                if i < len(rock_data):
                    prob = rock_data[i]['probability']
                    if prob > 0.1:  # 只考虑概率大于0.1的结果
                        depth_results.append((rock_type, prob))
            
            # 按概率排序
            depth_results.sort(key=lambda x: x[1], reverse=True)
            rock_types = [r[0] for r in depth_results]
            
            if depth < 10:  # 浅部
                self.assertTrue(
                    any(rock_type in ['GRANITE', 'BASALT'] 
                        for rock_type in rock_types),
                    f"深度{depth}km处应该有花岗岩或玄武岩，实际岩石类型：{rock_types}"
                )
            elif depth < 20:  # 中部
                self.assertTrue(
                    any(rock_type in ['BASALT', 'GABBRO'] 
                        for rock_type in rock_types),
                    f"深度{depth}km处应该有玄武岩或辉长岩，实际岩石类型：{rock_types}"
                )
            else:  # 深部
                self.assertTrue(
                    any(rock_type in ['GABBRO', 'PERIDOTITE'] 
                        for rock_type in rock_types),
                    f"深度{depth}km处应该有辉长岩或橄榄岩，实际岩石类型：{rock_types}"
                )

if __name__ == '__main__':
    unittest.main() 