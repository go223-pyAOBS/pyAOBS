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
from .isrock import RockIdentifier
from .rocks import TectonicSetting, RockProperties

class TestRockClassificationDemo(unittest.TestCase):
    """岩石分类演示测试"""
    
    def setUp(self):
        """初始化测试环境"""
        self.output_dir = "./rock_classification_demo"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # 创建测试数据库
        self.db_file = os.path.join(self.output_dir, "demo_database.xlsx")
        
        # 创建更完整的测试数据集
        data = {
            'rock_type': [
                'granite', 'basalt', 'gabbro', 'peridotite', 'eclogite',
                'amphibolite', 'marble', 'quartzite', 'gneiss', 'schist'
            ],
            'vp': [6.1, 6.5, 7.0, 8.1, 8.0, 6.8, 6.2, 6.0, 6.2, 5.8],
            'vs': [3.6, 3.8, 3.9, 4.5, 4.4, 3.8, 3.4, 3.7, 3.5, 3.3],
            'density': [2.7, 2.9, 3.0, 3.3, 3.4, 3.0, 2.7, 2.65, 2.8, 2.75],
            'porosity': [0.02, 0.05, 0.01, 0.005, 0.01, 0.02, 0.03, 0.02, 0.01, 0.03],
            'source': ['demo'] * 10,
            'method': ['lab'] * 10,
            'temperature': [25.0] * 10,
            'pressure': [200.0] * 10,
            'tectonic_setting': [
                'CONTINENTAL_CRUST', 'OCEANIC_CRUST', 'OCEANIC_CRUST', 
                'SUBDUCTION_ZONE', 'SUBDUCTION_ZONE', 'CONTINENTAL_CRUST',
                'CONTINENTAL_CRUST', 'CONTINENTAL_CRUST', 'CONTINENTAL_CRUST',
                'CONTINENTAL_CRUST'
            ]
        }
        df = pd.DataFrame(data)
        df.to_excel(self.db_file, index=False)
        
        # 初始化岩石识别器
        self.identifier = RockIdentifier(self.db_file)
        self.identifier.train_classifier()
        
    def tearDown(self):
        """保留所有测试结果"""
        pass
        
    def test_basic_classification(self):
        """测试基本的岩石分类功能"""
        # 1. 典型大陆地壳岩石 (花岗岩)
        granite_result = self.identifier.identify_rock(
            vp=6.1,
            vs=3.6,
            density=2.7,
            porosity=0.02,
            tectonic_setting=TectonicSetting.CONTINENTAL_CRUST,
            min_probability=0.1,
            max_candidates=5
        )
        
        # 2. 典型大洋地壳岩石 (玄武岩/辉长岩)
        oceanic_result = self.identifier.identify_rock(
            vp=6.8,
            vs=3.9,
            density=3.0,
            porosity=0.01,
            tectonic_setting=TectonicSetting.OCEANIC_CRUST,
            min_probability=0.1,
            max_candidates=5
        )
        
        # 3. 上地幔岩石 (橄榄岩)
        mantle_result = self.identifier.identify_rock(
            vp=8.1,
            vs=4.5,
            density=3.3,
            porosity=0.005,
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
        self.assertEqual(granite_candidates[0]['rock_type'], 'granite')
        
        # 检查大洋岩石结果
        oceanic_candidates = oceanic_result['candidates']
        self.assertGreater(len(oceanic_candidates), 0)
        self.assertIn(oceanic_candidates[0]['rock_type'], ['basalt', 'gabbro'])
        
        # 检查地幔岩石结果
        mantle_candidates = mantle_result['candidates']
        self.assertGreater(len(mantle_candidates), 0)
        self.assertEqual(mantle_candidates[0]['rock_type'], 'peridotite')
        
    def test_velocity_model_classification(self):
        """测试速度模型的岩石分类"""
        # 创建模拟的地壳-地幔速度模型
        model_data = {
            'vp': np.array([6.1, 6.5, 7.0, 8.1]),  # 从地壳到地幔
            'vs': np.array([3.6, 3.8, 3.9, 4.5]),
            'density': np.array([2.7, 2.9, 3.0, 3.3]),
            'porosity': np.array([0.02, 0.01, 0.01, 0.005])
        }
        
        # 识别岩石类型
        results = self.identifier.identify_velocity_model(
            model_data,
            min_probability=0.1
        )
        
        # 保存结果图像
        plot_file = os.path.join(self.output_dir, "velocity_model_classification.png")
        self.identifier.plot_identification_results(results, plot_file)
        
        # 验证结果
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        # 检查是否识别出了主要的岩石类型
        rock_types = set()
        for rock_type in results.keys():
            rock_types.add(rock_type)
        
        expected_rocks = {'granite', 'basalt', 'gabbro', 'peridotite'}
        self.assertTrue(any(rock in rock_types for rock in expected_rocks))

    def test_create_logo(self):
        """测试生成程序包logo"""
        from .logo import create_logo
        
        # 在演示目录中生成logo
        logo_file = os.path.join(self.output_dir, "pyAOBS_logo.png")
        create_logo(logo_file, dpi=300)
        
        # 验证logo文件是否创建成功
        self.assertTrue(os.path.exists(logo_file))

if __name__ == '__main__':
    unittest.main() 