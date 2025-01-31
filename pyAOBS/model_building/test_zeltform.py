"""
测试 Zelt 格式速度模型的功能

这个模块包含了对 ZeltVelocityModel2d 和 EnhancedZeltModel 类的测试。
"""

import unittest
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import xarray as xr

from pyAOBS.model_building.zeltform import ZeltVelocityModel2d, EnhancedZeltModel
from pyAOBS.model_building.models import Point2d

class TestZeltVelocityModel2d(unittest.TestCase):
    """测试 ZeltVelocityModel2d 类的基本功能"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境，加载示例模型文件"""
        # 假设在 tests/data 目录下有测试用的 v.in 文件
        test_data_dir = Path(__file__).parent / "tests" / "data"
        cls.model_file = str(test_data_dir / "v.in")
        
        # 创建测试数据目录（如果不存在）
        test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 如果没有测试文件，创建一个简单的测试模型
        if not Path(cls.model_file).exists():
            cls._create_test_model(cls.model_file)
    
    @staticmethod
    def _create_test_model(filename):
        """创建一个简单的测试模型文件"""
        with open(filename, 'w') as f:
            # 第一层界面
            f.write("1  0.0  10.0  20.0\n")  # x坐标
            f.write("0  0.0   0.0   0.0\n")  # z坐标
            f.write("0  0     0     0\n")    # 标志位
            
            # 第一层上部速度
            f.write("1  0.0  10.0  20.0\n")
            f.write("0  1.5   1.6   1.7\n")
            f.write("0  0     0     0\n")
            
            # 第一层下部速度
            f.write("1  0.0  10.0  20.0\n")
            f.write("0  1.8   1.9   2.0\n")
            f.write("0  0     0     0\n")
            
            # 第二层界面（底部）
            f.write("2  0.0  10.0  20.0\n")
            f.write("0  5.0   5.0   5.0\n")
            f.write("0  0     0     0\n")
    
    def setUp(self):
        """每个测试用例开始前执行"""
        self.model = ZeltVelocityModel2d(self.model_file)
    
    def test_model_loading(self):
        """测试模型加载功能"""
        self.assertIsNotNone(self.model.depth_nodes)
        self.assertIsNotNone(self.model.vupper_nodes)
        self.assertIsNotNone(self.model.vlower_nodes)
        self.assertIsNotNone(self.model.cells)
        
        # 检查层数（包括底部界面）
        self.assertEqual(len(self.model.depth_nodes), 10)  # 9个层界面加1个底部界面
        self.assertEqual(len(self.model.vupper_nodes), 9)  # 9层的上部速度
        self.assertEqual(len(self.model.vlower_nodes), 9)  # 9层的下部速度
        
        # 输出模型信息
        print("\n=== 模型信息 ===")
        print(f"模型边界: xmin={self.model.xmin:.1f}, xmax={self.model.xmax:.1f}, zmin={self.model.zmin:.1f}, zmax={self.model.zmax:.1f}")
        print(f"\n总层数: {len(self.model.depth_nodes)-1}")
        
        # 输出每层的信息
        for i in range(len(self.model.depth_nodes)-1):
            print(f"\n--- 第 {i+1} 层 ---")
            # 深度节点信息
            x_coords = self.model.depth_nodes[i].get_x_coords()
            z_coords = self.model.depth_nodes[i].get_values()
            print(f"深度节点数: {len(x_coords)}")
            print("深度节点 x 坐标:", [f"{x:.1f}" for x in x_coords])
            print("深度节点 z 坐标:", [f"{z:.1f}" for z in z_coords])
            
            # 速度节点信息
            upper_x = self.model.vupper_nodes[i].get_x_coords()
            upper_v = self.model.vupper_nodes[i].get_values()
            lower_x = self.model.vlower_nodes[i].get_x_coords()
            lower_v = self.model.vlower_nodes[i].get_values()
            print(f"\n上部速度节点数: {len(upper_x)}")
            print("上部速度节点 x 坐标:", [f"{x:.1f}" for x in upper_x])
            print("上部速度值:", [f"{v:.3f}" for v in upper_v])
            print(f"\n下部速度节点数: {len(lower_x)}")
            print("下部速度节点 x 坐标:", [f"{x:.1f}" for x in lower_x])
            print("下部速度值:", [f"{v:.3f}" for v in lower_v])
        
        # 输出底部边界信息
        print("\n--- 底部边界 ---")
        bottom_x = self.model.depth_nodes[-1].get_x_coords()
        bottom_z = self.model.depth_nodes[-1].get_values()
        print(f"底部边界节点数: {len(bottom_x)}")
        print("底部边界 x 坐标:", [f"{x:.1f}" for x in bottom_x])
        print("底部边界 z 坐标:", [f"{z:.1f}" for z in bottom_z])
    
    def test_velocity_interpolation(self):
        """测试速度插值功能"""
        # 测试模型内部的点
        v = self.model.at(200.0, 30.0)  # 在模型中间位置测试
        self.assertGreaterEqual(v, 0.0)  # 速度应该在合理范围内
        
        # 测试模型边界外的点
        with self.assertRaises(ValueError):
            self.model.at(-1.0, 0.0)    # 左边界外
        with self.assertRaises(ValueError):
            self.model.at(411.0, 30.0)  # 右边界外
        with self.assertRaises(ValueError):
            self.model.at(200.0, -1.0)  # 上边界外
        with self.assertRaises(ValueError):
            self.model.at(200.0, 61.0)  # 下边界外
    
    def test_layer_geometry(self):
        """测试层几何信息获取功能"""
        x_coords, z_coords = self.model.get_layer_geometry(0)
        self.assertEqual(len(x_coords), len(z_coords))
        self.assertGreater(len(x_coords), 0)
        
        # 检查第一层的几何信息
        self.assertEqual(x_coords[0], 0.0)   # 左边界
        self.assertEqual(x_coords[-1], 410.0) # 右边界
        self.assertEqual(z_coords[0], 0.0)   # 顶部深度
        
        # 测试无效层索引
        with self.assertRaises(ValueError):
            self.model.get_layer_geometry(-1)
        with self.assertRaises(ValueError):
            self.model.get_layer_geometry(len(self.model.depth_nodes))

class TestEnhancedZeltModel(unittest.TestCase):
    """测试 EnhancedZeltModel 类的扩展功能"""
    
    def setUp(self):
        """每个测试用例开始前执行"""
        self.test_data_dir = Path(__file__).parent / "tests" / "data"
        self.model_file = str(self.test_data_dir / "v.in")
        self.base_model = ZeltVelocityModel2d(self.model_file)
        self.model = EnhancedZeltModel(self.base_model)
    
    def test_model_comparison(self):
        """测试模型对比功能"""
        # 获取初始状态的对比结果
        initial_comparison = self.model.compare_with_base()
        
        # 检查初始状态下模型应该完全相同
        for layer_idx, ((base_top, base_bottom), (current_top, current_bottom)) in enumerate(initial_comparison['velocity_changes']):
            print(f"\n第 {layer_idx + 1} 层初始速度对比:")
            print(f"基础模型: 顶部={base_top:.3f}, 底部={base_bottom:.3f}")
            print(f"当前模型: 顶部={current_top:.3f}, 底部={current_bottom:.3f}")
            self.assertAlmostEqual(base_top, current_top, places=6)
            self.assertAlmostEqual(base_bottom, current_bottom, places=6)
        
        # 处理速度模型（计算平均速度）
        avg_velocities = self.model.process_velocity_model('average_velocity')
        
        # 获取处理后的对比结果
        processed_comparison = self.model.compare_with_base()
        
        # 检查处理后的变化
        print("\n=== 处理后的模型对比 ===")
        for layer_idx, ((base_top, base_bottom), (current_top, current_bottom)) in enumerate(processed_comparison['velocity_changes']):
            print(f"\n第 {layer_idx + 1} 层处理后速度对比:")
            print(f"基础模型: 顶部={base_top:.3f}, 底部={base_bottom:.3f}")
            print(f"当前模型: 顶部={current_top:.3f}, 底部={current_bottom:.3f}")
            print(f"变化量: 顶部={current_top-base_top:.3f}, 底部={current_bottom-base_bottom:.3f}")
            
            # 验证处理后的速度值与返回的平均速度一致
            self.assertAlmostEqual(current_top, avg_velocities[layer_idx][0], places=6)
            self.assertAlmostEqual(current_bottom, avg_velocities[layer_idx][1], places=6)
    
    def test_velocity_processing(self):
        """测试速度模型处理功能"""
        # 测试不支持的处理类型
        with self.assertRaises(ValueError):
            self.model.process_velocity_model('unsupported_type')
        
        # 测试层间速度计算
        result = self.model.process_velocity_model('interval_velocity')
        self.assertIn('interval_velocities', result)
        self.assertIn('boundaries', result)
        
        # 检查结果
        print("\n=== 层间速度信息 ===")
        for i, (velocity, boundary) in enumerate(zip(result['interval_velocities'], result['boundaries'])):
            print(f"\n层界面 {i + 1}:")
            print(f"层间速度: {velocity:.3f} km/s")
            print(f"界面深度: {boundary:.3f} km")
            
            # 验证速度值在合理范围内
            self.assertGreaterEqual(velocity, 0.0)
            
            # 验证模型节点已更新
            if i < len(self.model.vlower_nodes) - 1:
                # 检查当前层的下部速度
                current_lower = self.model.vlower_nodes[i].get_values()
                for val in current_lower:
                    self.assertAlmostEqual(val, velocity, places=6)
                
                # 检查下一层的上部速度
                next_upper = self.model.vupper_nodes[i + 1].get_values()
                for val in next_upper:
                    self.assertAlmostEqual(val, velocity, places=6)
        
        # 测试双程走时计算
        result = self.model.process_velocity_model('two_way_time')
        self.assertIn('twt', result)
        self.assertIn('velocities', result)
        
        # 检查结果
        print("\n=== 双程走时信息 ===")
        for i, (twt, velocity) in enumerate(zip(result['twt'], result['velocities'])):
            print(f"\n第 {i + 1} 层:")
            print(f"双程走时: {twt:.3f} s")
            print(f"平均速度: {velocity:.3f} km/s")
            
            # 验证走时和速度值在合理范围内
            self.assertGreaterEqual(twt, 0.0)
            self.assertGreaterEqual(velocity, 0.0)
            
            # 验证深度节点已更新为时间值
            depth_vals = self.model.depth_nodes[i].get_values()
            for val in depth_vals:
                self.assertAlmostEqual(val, twt, places=6)
    
    def test_average_velocities(self):
        """测试平均速度计算功能"""
        avg_velocities = self.model.process_velocity_model('average_velocity')
        self.assertEqual(len(avg_velocities), 9)  # 9层的平均速度
        
        print("\n=== 层速度信息 ===")
        for i, (top_avg, bottom_avg) in enumerate(avg_velocities):
            print(f"\n第 {i+1} 层:")
            print(f"顶部平均速度: {top_avg:.3f} km/s")
            print(f"底部平均速度: {bottom_avg:.3f} km/s")
            
            # 检查速度值是否在合理范围内
            self.assertGreaterEqual(top_avg, 0.0)
            self.assertGreaterEqual(bottom_avg, 0.0)
            # 检查底部速度是否大于等于顶部速度
            self.assertGreaterEqual(bottom_avg, top_avg)
            
            # 验证模型节点的速度值是否已更新
            model_top_vals = self.model.vupper_nodes[i].get_values()
            model_bottom_vals = self.model.vlower_nodes[i].get_values()
            
            # 检查所有节点是否都更新为平均值
            for val in model_top_vals:
                self.assertAlmostEqual(val, top_avg, places=6)
            for val in model_bottom_vals:
                self.assertAlmostEqual(val, bottom_avg, places=6)
    
    def test_xarray_conversion(self):
        """测试转换为 xarray 数据集的功能"""
        ds = self.model.to_xarray()
        self.assertIn('velocity', ds.data_vars)
        self.assertIn('x', ds.coords)
        self.assertIn('z', ds.coords)
        
        # 检查数据集的形状
        self.assertEqual(len(ds.x), 100)  # 默认网格大小
        self.assertEqual(len(ds.z), 100)
        
        # 检查坐标范围
        self.assertEqual(ds.x.values[0], 0.0)    # 左边界
        self.assertEqual(ds.x.values[-1], 410.0) # 右边界
        self.assertEqual(ds.z.values[0], 0.0)    # 顶部深度
        self.assertEqual(ds.z.values[-1], 60.0)  # 底部深度
        
        # 检查速度值的范围
        v = ds.velocity.values
        valid_velocities = v[~np.isnan(v)]  # 去除 NaN 值
        self.assertTrue(np.all(valid_velocities >= 0.0))  # 所有有效值应大于等于 0.0

    def test_plot_velocity_difference(self):
        """测试绘制模型速度差值"""
        # 创建增强型模型
        enhanced_model = EnhancedZeltModel(self.model)
        
        # 获取基础模型的速度分布
        base_ds = self.model.to_xarray()
        
        # 测试不同处理方法的速度差值
        process_types = ['average_velocity', 'interval_velocity', 'two_way_time']
        for process_type in process_types:
            # 处理模型
            enhanced_model.process_velocity_model(process_type)
            processed_ds = enhanced_model.to_xarray()
            
            # 计算速度差值（基础模型 - 处理后模型）
            velocity_diff = base_ds.velocity - processed_ds.velocity
            
            # 创建新的数据集来存储差值
            diff_ds = xr.Dataset(
                data_vars={
                    'velocity_difference': (['z', 'x'], velocity_diff.data)
                },
                coords={
                    'x': processed_ds.x,
                    'z': processed_ds.z
                }
            )
            
            # 绘制差值图
            output_file = str(self.test_data_dir / f"{process_type}_difference.png")
            self._plot_velocity_difference(diff_ds, output_file, process_type, enhanced_model)
            self.assertTrue(Path(output_file).exists())
            print(f"\n{process_type} 速度差值图已保存到: {output_file}")
            
            # 输出统计信息
            valid_diff = velocity_diff.data[~np.isnan(velocity_diff.data)]
            print(f"\n{process_type} 速度差值统计:")
            print(f"最小差值: {np.min(valid_diff):.3f} km/s")
            print(f"最大差值: {np.max(valid_diff):.3f} km/s")
            print(f"平均差值: {np.mean(valid_diff):.3f} km/s")
            print(f"标准差: {np.std(valid_diff):.3f} km/s")
    
    def _plot_velocity_difference(self, ds: xr.Dataset, output_file: str, process_type: str, model: ZeltVelocityModel2d):
        """绘制速度差值分布图
        
        Args:
            ds: 包含速度差值的数据集
            output_file: 输出图片文件路径
            process_type: 处理类型名称
            model: 速度模型对象，用于获取层界面信息
        """
        plt.figure(figsize=(10, 6))
        
        # 使用发散色标以便于区分正负差值
        vmax = np.nanmax(np.abs(ds.velocity_difference))
        vmin = -vmax
        plt.pcolormesh(ds.x, ds.z, ds.velocity_difference, 
                      shading='auto', cmap='seismic',
                      vmin=vmin, vmax=vmax)
        
        # 绘制层界面
        for i in range(len(model.depth_nodes)):
            x_coords, z_coords = model.get_layer_geometry(i)
            plt.plot(x_coords, z_coords, 'k-', linewidth=0.5, alpha=0.5)
        
        plt.colorbar(label='Velocity Difference (km/s)')
        plt.xlabel('Distance (km)')
        plt.ylabel('Depth (km)')
        plt.title(f'Velocity Difference ({process_type})')
        plt.gca().invert_yaxis()  # 反转 Y 轴使深度向下增加
        plt.savefig(output_file)
        plt.close()

class TestModelVisualization(unittest.TestCase):
    """测试模型可视化功能"""
    
    def setUp(self):
        """每个测试用例开始前执行"""
        self.test_data_dir = Path(__file__).parent / "tests" / "data"
        self.model_file = str(self.test_data_dir / "v.in")
        self.model = ZeltVelocityModel2d(self.model_file)
        
    def test_plot_model(self):
        """测试模型绘图功能"""
        # 测试基础模型绘图
        output_file = str(self.test_data_dir / "base_model.png")
        plot_model(self.model, output_file)
        self.assertTrue(Path(output_file).exists())
        print(f"\n基础速度模型图已保存到: {output_file}")
        
        # 测试处理后模型的绘图
        enhanced_model = EnhancedZeltModel(self.model)
        
        # 绘制平均速度处理后的模型
        enhanced_model.process_velocity_model('average_velocity')
        output_file = str(self.test_data_dir / "average_velocity_model.png")
        plot_model(enhanced_model, output_file)
        self.assertTrue(Path(output_file).exists())
        print(f"\n平均速度模型图已保存到: {output_file}")
        
        # 绘制层间速度处理后的模型
        enhanced_model.process_velocity_model('interval_velocity')
        output_file = str(self.test_data_dir / "interval_velocity_model.png")
        plot_model(enhanced_model, output_file)
        self.assertTrue(Path(output_file).exists())
        print(f"\n层间速度模型图已保存到: {output_file}")
        
        # 绘制双程走时处理后的模型
        enhanced_model.process_velocity_model('two_way_time')
        output_file = str(self.test_data_dir / "two_way_time_model.png")
        plot_model(enhanced_model, output_file)
        self.assertTrue(Path(output_file).exists())
        print(f"\n双程走时模型图已保存到: {output_file}")

def plot_model(model: ZeltVelocityModel2d, output_file: str):
    """绘制模型的速度分布图
    
    Args:
        model: ZeltVelocityModel2d 对象
        output_file: 输出图片文件路径
    """
    ds = model.to_xarray()
    
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(ds.x, ds.z, ds.velocity, shading='auto', cmap='jet')
    
    # 绘制层界面
    for i in range(len(model.depth_nodes)):
        x_coords, z_coords = model.get_layer_geometry(i)
        plt.plot(x_coords, z_coords, 'k-', linewidth=0.5, alpha=0.5)
    
    plt.colorbar(label='Velocity (km/s)')
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (km)')
    plt.title('Velocity Model')
    plt.gca().invert_yaxis()  # 反转 Y 轴使深度向下增加
    plt.savefig(output_file)
    plt.close()

if __name__ == '__main__':
    # 运行单元测试
    unittest.main(argv=[''], verbosity=2, exit=False)
    
    # 创建示例图
    test_data_dir = Path(__file__).parent / "tests" / "data"
    model = ZeltVelocityModel2d(str(test_data_dir / "v.in"))
    plot_model(model, str(test_data_dir / "velocity_model.png"))
    print(f"\n速度模型图已保存到: {test_data_dir / 'velocity_model.png'}") 