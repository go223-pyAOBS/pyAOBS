"""
测试 show_model.py 模块

这个模块包含了对 ZeltModelVisualizer 类的测试。
"""

import unittest
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import xarray as xr

from pyAOBS.model_building.zeltform import ZeltVelocityModel2d
from pyAOBS.model_building.zeltform import EnhancedZeltModel
from pyAOBS.visualization.show_model import ZeltModelVisualizer
from pyAOBS.visualization.show_model import GridModelVisualizer
from pyAOBS.visualization.show_model import GridModelProcessor

class TestZeltModelVisualizer(unittest.TestCase):
    """测试 ZeltModelVisualizer 类的网格化功能"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 使用已有的模型文件
        cls.model_file = Path('./v.in')
        if not cls.model_file.exists():
            raise FileNotFoundError(f"找不到模型文件: {cls.model_file}")
        
        print(f"\n使用的模型文件: {cls.model_file}")
        
        # 创建输出目录
        cls.output_dir = Path('./output/test_grids')
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        """每个测试用例开始前执行"""
        # # 创建可视化器，加载下层网格（速度模型）
        self.grid_visualizer = GridModelVisualizer(
            grid_file="./output/test_grids/twt_model.grd",
            output_dir=str(self.output_dir)
        )
        # 加载cpt文件
        self.cmap_lower = self.grid_visualizer.load_cpt('./scale1.cpt')
        self.cmap_upper = self.grid_visualizer.create_zero_cmap(base_cmap='gray', zero_threshold=1e-6)

    def test_plot_with_matplotlib(self):
        """测试使用 Matplotlib 绘制网格数据。"""
        # # 加载速度模型
        base_model = ZeltVelocityModel2d(self.model_file)
        enhanced_model = EnhancedZeltModel(base_model)
        enhanced_model.process_velocity_model('two_way_time')

        
        # # 使用 Matplotlib 绘制叠加图
        self.grid_visualizer.plot_with_matplotlib(
            output_fig=str(self.output_dir / "matplotlib_overlay_south.pdf"),
            upper_grid_file="./su_outputs/grid_offset_south.grd",
            lower_cmap=self.cmap_lower,  # 使用加载的cpt颜色映射
            upper_cmap=self.cmap_upper,
            upper_transparency=0.8,
            title="Seismic Section on Velocity Model (Matplotlib)",
            colorbar_label="Velocity (km/s)",
            lower_clim=[1.5, 8.0],
            upper_clim=[25, 80],
            plot_region=[0, 180, 0, 14],
            figsize=(10, 4),
            plot_interfaces=True,
            model=enhanced_model,
            interface_color='black',
            interface_linewidth=0.8,
            interface_linestyle='--',
            plot_contours_lower=True,
            contour_interval_lower=0.5,
            contour_colors_lower='black',
            contour_linewidths_lower=0.5,
            contour_linestyles_lower='-',
            contour_label_fmt_lower='%.2f',
            contour_label_fontsize_lower=8,
            contour_inline_lower=True,
            contour_inline_spacing_lower=5,
            xlabel="Distance (km)",
            ylabel="Time (s)"
        )
        
        print(f"Matplotlib 图像已保存到: {self.output_dir / 'matplotlib_overlay.pdf'}")
        
        # # 验证图像是否成功生成
        self.assertTrue(Path(self.output_dir / "matplotlib_overlay.pdf").exists())

    def test_plot_xarray(self):
        """测试使用plot_xarray绘制单个数据集。"""
        # 创建输出目录
        output_file = str(self.output_dir / "xarray_plot.pdf")
        
        # 加载速度模型用于绘制界面
        base_model = ZeltVelocityModel2d(self.model_file)
        enhanced_model = EnhancedZeltModel(base_model)
        enhanced_model.process_velocity_model('two_way_time')
        
        # 使用已有的网格文件进行测试
        self.grid_visualizer.plot_xarray(
            output_fig=output_file,
            data="./output/test_grids/twt_model.grd",
            title="Xarray Plot Test",
            colorbar_label="Value",
            clim=[5, 95],  # 使用5-95百分位数范围
            figsize=(10, 4),
            plot_interfaces=True,
            model=enhanced_model,
            interface_color='black',
            interface_linewidth=0.8,
            interface_linestyle='--',
            plot_contours=True,
            contour_interval=0.5,
            contour_colors='black',
            contour_linewidths=0.5,
            contour_linestyles='-',
            contour_label_fmt='%.2f',
            contour_label_fontsize=8,
            contour_inline=True,
            contour_inline_spacing=5
        )
        
        # 验证图像是否成功生成
        self.assertTrue(Path(output_file).exists())
        print(f"\nXarray图像已保存到: {output_file}")
        
        # 测试使用xarray数据集作为输入
        with xr.open_dataset("./output/test_grids/twt_model.grd") as ds:
            output_file_2 = str(self.output_dir / "xarray_plot_dataset.pdf")
            self.grid_visualizer.plot_xarray(
                output_fig=output_file_2,
                data=ds,
                title="Xarray Dataset Plot Test",
                colorbar_label="Value",
                clim=[1, 99],  # 使用1-99百分位数范围
                figsize=(10, 4),
                plot_interfaces=True,
                model=enhanced_model,
                interface_color='white',
                interface_linewidth=0.5,
                interface_linestyle=':',
                plot_contours=True,
                contour_interval=0.5,
                contour_colors='black',
                contour_linewidths=0.5,
                contour_linestyles='-',
                contour_label_fmt='%.2f',
                contour_label_fontsize=8,
                contour_inline=True,
                contour_inline_spacing=5
            )
            
            # 验证图像是否成功生成
            self.assertTrue(Path(output_file_2).exists())
            print(f"Xarray数据集图像已保存到: {output_file_2}")

    # def test_plot_velocity_base(self):
    #     """测试速度基础绘图功能"""
    #     # 加载基础速度模型
    #     base_model = ZeltVelocityModel2d(self.model_file)
    #     visualizer = ZeltModelVisualizer(base_model)
    #     visualizer.plot_zeltmodel(dx=2.0, 
    #                             dz=0.5,
    #                             figsize=(10, 4), 
    #                             region=[0, 410, 0, 50],
    #                             velocity_cmap=self.cmap,  # 使用加载的cpt颜色映射
    #                             output_file=str(self.output_dir / "velocity_base.png"))  
    #     print(f"速度基础绘图功能已保存到: {self.output_dir / 'velocity_base.png'}")
    #     enhanced_model = EnhancedZeltModel(base_model)
    #     enhanced_model.process_velocity_model('average_velocity')
    #     visualizer = ZeltModelVisualizer(enhanced_model)
    #     visualizer.plot_zeltmodel(dx=2.0, 
    #                             dz=0.5,
    #                             figsize=(10, 4), 
    #                             region=[0, 410, 0, 50],
    #                             velocity_cmap=self.cmap,  # 使用加载的cpt颜色映射
    #                             output_file=str(self.output_dir / "velocity_base_avg.png"))  
    #     print(f"速度基础绘图功能已保存到: {self.output_dir / 'velocity_base_avg.png'}")



    # def test_plot_velocity_difference(self):
    #     """测试速度差值绘图功能"""
    #     # 加载基础速度模型
    #     base_model = ZeltVelocityModel2d(self.model_file)
        
    #     # 创建增强型模型
    #     enhanced_model = EnhancedZeltModel(base_model)
    #     visualizer = ZeltModelVisualizer(enhanced_model)
        
    #     # 测试不同处理方法的速度差值
    #     process_types = ['average_velocity', 'interval_velocity', 'two_way_time']
    #     for process_type in process_types:
    #         output_file = str(self.output_dir / f"{process_type}_difference.png")
            
    #         # 对于双程走时模型，使用不同的y轴标签
    #         if process_type == 'two_way_time':
    #             visualizer.plot_velocity_difference(
    #                 output_file=output_file,
    #                 base_model=base_model,
    #                 process_type=process_type,
    #                 ylabel="Time (s)",
    #                 figsize=(10, 4),
    #                 dx=2.0,
    #                 dz=0.5,
    #                 velocity_cmap=self.cmap,  # 使用加载的cpt颜色映射
    #                 region=[0, 410, 0, 10]  # 时间域的深度范围不同,注意反转
    #             )
    #         else:
    #             visualizer.plot_velocity_difference(
    #                 output_file=output_file,
    #                 base_model=base_model,
    #                 process_type=process_type,
    #                 figsize=(10, 4),
    #                 dx=2.0,
    #                 dz=0.5,
    #                 region=[0, 410, 0, 50],
    #                 velocity_cmap='coolwarm' if process_type == 'interval_velocity' else 'seismic_r'
    #             )
            

    #         self.assertTrue(Path(output_file).exists())
    #         print(f"\n{process_type} 速度差值图已保存到: {output_file}")
        
    #     # 测试自定义参数
    #     output_file = str(self.output_dir / "custom_difference.png")
    #     visualizer.plot_velocity_difference(
    #         output_file=output_file,
    #         base_model=base_model,
    #         title="Custom Velocity Difference",
    #         velocity_cmap=self.cmap,  # 使用加载的cpt颜色映射
    #         figsize=(10, 4),
    #         dx=2.0,
    #         dz=0.5,
    #         plot_interfaces=True,
    #         region=[0, 410, 0, 50],
    #         interface_color='white',
    #         interface_linewidth=0.8,
    #         interface_alpha=0.3,
    #         colorbar_label="ΔV (km/s)",
    #         dpi=300
    #     )
    #     self.assertTrue(Path(output_file).exists())
    #     print(f"自定义差值图已保存到: {output_file}")

if __name__ == '__main__':
    unittest.main() 