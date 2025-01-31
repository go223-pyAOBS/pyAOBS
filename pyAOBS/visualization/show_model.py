"""
Model Visualization Module

This module provides classes for visualizing and processing seismic velocity models using PyGMT.
It supports plotting velocity fields and layer interfaces, as well as model conversions.
"""

import numpy as np
import pygmt
import xarray as xr
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Union
import os
from matplotlib.colors import LinearSegmentedColormap

from pyAOBS.model_building.zeltform import ZeltVelocityModel2d,EnhancedZeltModel

class ZeltModelVisualizer:
    """用于可视化 Zelt 速度模型的类。
    
    该类提供了将 Zelt 速度模型网格化和绘制速度场的方法。
    网格化后的模型可以使用 GridModelVisualizer 进行进一步的可视化。
    
    属性:
        model (Optional[Union[ZeltVelocityModel2d, EnhancedZeltModel]]): 要可视化的 Zelt 速度模型
        output_dir (Path): 输出文件目录
    """
    
    def __init__(self, model: Optional[Union[ZeltVelocityModel2d, EnhancedZeltModel]] = None, 
                 output_dir: Optional[str] = None):
        """初始化 Zelt 模型可视化器。
        
        参数:
            model: 要可视化的 Zelt 速度模型。如果为 None,则只能绘制外部网格文件。
            output_dir: 输出文件目录。默认为 './output'
                
        异常:
            TypeError: 如果提供的模型不是 ZeltVelocityModel2d 或 EnhancedZeltModel 的实例
        """
        if model is not None and not isinstance(model, (ZeltVelocityModel2d, EnhancedZeltModel)):
            raise TypeError("模型必须是 ZeltVelocityModel2d 或 EnhancedZeltModel 的实例")
            
        self.model = model
        self.output_dir = Path(output_dir) if output_dir else Path('./output')
        self.output_dir.mkdir(exist_ok=True)
        self.saved_files: Dict[str, Path] = {}
        
        # 初始化网格相关属性
        self.velocity_grid = None
        self.grid_file = None
        self.plot_region = None
        self.grid_spacing = None

    def grid_model(self, plot_region: Optional[Tuple[float, float, float, float]] = None,
                  grid_spacing: Optional[Tuple[float, float]] = None,
                  dx: Optional[float] = 2.0,
                  dz: Optional[float] = 0.5,
                  velocity_grid_file: Optional[str] = None) -> None:
        """将 Zelt 速度模型网格化并保存到文件。
        
        参数:
            plot_region: 绘图区域 (xmin, xmax, zmin, zmax)
            grid_spacing: x 和 z 方向的网格间距 (dx, dz)
            velocity_grid_file: 速度网格文件的名称(不含扩展名)
            
        异常:
            ValueError: 如果在初始化时未提供模型
        """
        if self.model is None:
            raise ValueError("无法网格化模型: 初始化时未提供 Zelt 模型")
            
        # 将模型转换为 xarray 数据集
        ds = self.model.to_xarray(dx, dz)
        
        # 确定绘图区域
        if plot_region is None and self.plot_region is None:
            self.plot_region = [
                float(ds.x.min()),
                float(ds.x.max()),
                float(ds.z.min()),
                float(ds.z.max())
            ]
        elif plot_region is not None:
            self.plot_region = list(plot_region)
            if not (ds.x.min() <= self.plot_region[0] <= ds.x.max() and 
                   ds.x.min() <= self.plot_region[1] <= ds.x.max() and
                   ds.z.min() <= self.plot_region[2] <= ds.z.max() and
                   ds.z.min() <= self.plot_region[3] <= ds.z.max()):
                print("警告: 绘图区域超出模型边界")
        
        # 设置网格间距
        if grid_spacing is not None:
            self.grid_spacing = grid_spacing
        elif self.grid_spacing is None:
            self.grid_spacing = (1.0, 0.5)  # 默认网格间距
            
        # 创建规则网格
        x = np.arange(self.plot_region[0], self.plot_region[1] + self.grid_spacing[0], self.grid_spacing[0])
        z = np.arange(self.plot_region[2], self.plot_region[3] + self.grid_spacing[1], self.grid_spacing[1])
        X, Z = np.meshgrid(x, z, indexing='xy')
        
        # 计算网格点上的速度
        V = np.zeros_like(X)
        for i in range(Z.shape[0]):
            for j in range(X.shape[1]):
                try:
                    V[i,j] = self.model.at(X[i,j], Z[i,j])
                    #print(f"计算速度: {X[i,j]}, {Z[i,j]}, {V[i,j]}")
                except ValueError:
                    V[i,j] = np.nan
        
        # 创建 xarray 数据集
        self.velocity_grid = xr.Dataset(
            data_vars={
                'velocity': (['z', 'x'], V)
            },
            coords={
                'x': x,
                'z': z
            }
        )
        
        # 保存速度网格到文件
        if velocity_grid_file:
            self.grid_file = self.output_dir / f"{velocity_grid_file}.grd"
        else:
            self.grid_file = self.output_dir / "velocity.grd"
            
        # 保存为 netCDF4 格式，指定数据类型和填充值
        self.velocity_grid.velocity.to_netcdf(
            self.grid_file,
            format='NETCDF4',
            encoding={'velocity': {'dtype': 'float32', '_FillValue': -9999.0}}
        )
        self.saved_files['velocity_grid'] = self.grid_file

            
    def load_cpt(self, cpt_file: str) -> str:
        """加载 CPT 文件并返回其内容。
        
        参数:
            cpt_file (str): CPT 文件的路径
            Returns:
        LinearSegmentedColormap: matplotlib 颜色映射对象
        """
        cpt_data = []
        with open(cpt_file, 'r') as file:
            lines = file.readlines()
        for line in lines:
            #skip the comment lines
            if line.startswith('#'):
                continue
            #parse the cpt data
            data = line.strip().split()
            if len(data) == 8: # standard cpt file format
                z1, r1, g1, b1, z2, r2, g2, b2 = map(float, data)
                cpt_data.append([z1, r1/255, g1/255, b1/255])
                cpt_data.append([z2, r2/255, g2/255, b2/255])
        # covert into numpy
        cpt_data = np.array(cpt_data)
        # 归一化
        z_normalized = (cpt_data[:, 0] - cpt_data[:, 0].min()) / (cpt_data[:, 0].max() - cpt_data[:, 0].min())
        colors = cpt_data[:, 1:]
        cmap = LinearSegmentedColormap.from_list('custom_cpt', list(zip(z_normalized, colors)))
        return cmap
    
    def plot_zeltmodel(self, 
                      output_file: str, 
                      title: str = 'Zelt Velocity Model',
                      xlabel: str = 'Distance (km)',
                      ylabel: str = 'Depth (km)',
                      colorbar_label: str = 'Velocity (km/s)',
                      figsize: Tuple[float, float] = (10, 8),
                      velocity_cmap: str = 'jet',
                      clim: Optional[List[float]] = None,  # [vmin, vmax]
                      region: Optional[List[float]] = None,  # [xmin, xmax, zmin, zmax]
                      plot_interfaces: bool = True,
                      interface_color: str = 'black',
                      interface_linewidth: float = 0.5,
                      interface_alpha: float = 0.5,
                      colorbar_orientation: str = 'vertical',
                      colorbar_fraction: float = 0.046,
                      colorbar_pad: float = 0.04,
                      dx: Optional[float] = 2.0,
                      dz: Optional[float] = 0.5,
                      dpi: int = 300) -> None:
        """绘制 Zelt 速度模型。
        
        该方法使用 matplotlib 绘制速度场(及其延申场，如密度、刚度等)
        和层界面。适用于原始速度模型和经过处理的模型（如平均速度、层间速度、双程走时等）。
        
        参数:
            output_file (str): 输出文件路径
            title (str): 图像标题
            xlabel (str): x轴标签
            ylabel (str): y轴标签，对于双程走时模型可设为 'Time (s)'
            colorbar_label (str): 色标标签，可根据模型类型设置不同的标签
            figsize (Tuple[float, float]): 图形大小 (宽度, 高度), 单位为英寸
            velocity_cmap (str): 颜色映射，可选值包括：
                     - 连续色板：'viridis', 'plasma', 'inferno', 'magma', 'jet', 'seismic'
                     - 单色系列：'Greys', 'Blues', 'Reds', 'Greens'
                     - 离散色板：'Paired', 'Set1', 'tab10'
                     - 添加 '_r' 后缀可以反转色板
                     - 支持 CPT 文件，文件格式如下：
                       # 标准 CPT 文件格式
                       # 深度(km) 红 绿 蓝 深度(km) 红 绿 蓝
                       0    0   0  0  10 155 155 155
                       10 155 155 155 20 255 255 255 
                       20 255 255 255 30 255 255 255
                       B	255	255	255
                       F	0	0	0
                       N	255	255	255                  
            clim (Optional[List[float]]): 色标范围 [vmin, vmax]
            region (Optional[List[float]]): 绘图区域 [xmin, xmax, zmin, zmax]
            plot_interfaces (bool): 是否绘制层界面
            interface_color (str): 界面线条颜色
            interface_linewidth (float): 界面线条宽度
            interface_alpha (float): 界面线条透明度
            colorbar_orientation (str): 色标方向，'vertical' 或 'horizontal'
            colorbar_fraction (float): 色标相对于主图的大小比例
            colorbar_pad (float): 色标与主图的间距
            dx (float): 网格间距
            dz (float): 网格间距
            dpi (int): 图像分辨率
            
        异常:
            ValueError: 如果未提供模型
            
        示例:
            >>> # 创建基础速度模型
            >>> base_model = ZeltVelocityModel2d("v.in")
            >>> visualizer = ZeltModelVisualizer(base_model)
            
            # 绘制基础速度模型
            >>> visualizer.plot_zeltmodel(
            ...     output_file="velocity_model.png",
            ...     title="Original Velocity Model",
            ...     colorbar_label="Velocity (km/s)",
            ...     region=[0, 100, 0, 30]  # 指定绘图区域
            ... )
            
            # 创建增强型模型并处理为平均速度模型
            >>> enhanced_model = EnhancedZeltModel(base_model)
            >>> enhanced_model.process_velocity_model('average_velocity')
            >>> visualizer = ZeltModelVisualizer(enhanced_model)
            >>> visualizer.plot_zeltmodel(
            ...     output_file="avg_velocity_model.png",
            ...     title="Average Velocity Model",
            ...     colorbar_label="Average Velocity (km/s)",
            ...     cmap='viridis',
            ...     region=[0, 100, 0, 30]  # 指定绘图区域
            ... )
            
            # 处理为层间速度模型
            >>> enhanced_model.process_velocity_model('interval_velocity')
            >>> visualizer = ZeltModelVisualizer(enhanced_model)
            >>> visualizer.plot_zeltmodel(
            ...     output_file="interval_velocity_model.png",
            ...     title="Interval Velocity Model",
            ...     colorbar_label="Interval Velocity (km/s)",
            ...     cmap='plasma',
            ...     region=[0, 100, 0, 30]  # 指定绘图区域
            ... )
            
            # 处理为双程走时模型
            >>> enhanced_model.process_velocity_model('two_way_time')
            >>> visualizer = ZeltModelVisualizer(enhanced_model)
            >>> visualizer.plot_zeltmodel(
            ...     output_file="twt_model.png",
            ...     title="Two-Way Time Model",
            ...     ylabel="Time (s)",
            ...     colorbar_label="Velocity (km/s)",
            ...     cmap='seismic',
            ...     region=[0, 100, 0, 10]  # 指定时间域的绘图区域
            ... )
        """
        if self.model is None:
            raise ValueError("无法绘制模型: 未提供 Zelt 模型")
            
        import matplotlib.pyplot as plt
        
        # 创建图形
        plt.figure(figsize=figsize)
        
        # 获取模型数据
        ds = self.model.to_xarray(dx, dz)
        
        # 确定绘图区域
        if region is None:
                region = [
                float(ds.x.min()),
                float(ds.x.max()),
                float(ds.z.min()),  # 注意: z 轴反转
                float(ds.z.max())
            ]

        # 解析cmap
        if isinstance(velocity_cmap, str) and velocity_cmap.lower().endswith('.cpt') and os.path.exists(velocity_cmap):
            cmap = self.load_cpt(velocity_cmap)
        else:
            cmap = velocity_cmap
        # 绘制速度场
        im = plt.pcolormesh(ds.x.values, ds.z.values, ds.velocity.values,
                       shading='auto',
                       cmap=cmap,
                       vmin=clim[0] if clim else None,
                       vmax=clim[1] if clim else None)
        
        # 添加色标
        plt.colorbar(im, 
                    label=colorbar_label,
                    orientation=colorbar_orientation,
                    fraction=colorbar_fraction,
                    pad=colorbar_pad)
        
        # 绘制层界面
        if plot_interfaces:
            for i in range(len(self.model.depth_nodes)):
                x_coords, z_coords = self.model.get_layer_geometry(i)
                plt.plot(x_coords, z_coords,
                        color=interface_color,
                        linewidth=interface_linewidth,
                        alpha=interface_alpha,
                        linestyle='-')
        
        # 设置标题和轴标签
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        # 设置绘图区域
        if region:
            plt.xlim(region[0], region[1])
            plt.ylim(region[3], region[2])  # 注意：反转顺序以保持深度向下增加
        
        # 保存图形
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()

    def get_saved_files(self) -> Dict[str, Path]:
        """获取所有保存的文件路径。
        
        返回:
            Dict[str, Path]: 包含保存的文件路径的字典
        """
        return self.saved_files

    def plot_velocity_difference(self,
                              output_file: str,
                              base_model: Union[ZeltVelocityModel2d, EnhancedZeltModel],
                              region: Optional[List[float]] = None, # [xmin, xmax, zmin, zmax]
                              process_type: Optional[str] = None,
                              title: Optional[str] = None,
                              xlabel: str = 'Distance (km)',
                              ylabel: str = 'Depth (km)',
                              colorbar_label: str = 'Velocity Difference (km/s)',
                              figsize: Tuple[float, float] = (10, 6),
                              velocity_cmap: str = 'seismic',
                              plot_interfaces: bool = True,
                              interface_color: str = 'black',
                              interface_linewidth: float = 0.5,
                              interface_alpha: float = 0.5,
                              dx: Optional[float] = 2.0,
                              dz: Optional[float] = 0.5,
                              dpi: int = 300) -> None:
        """绘制速度差值分布图。
        
        该方法用于比较基础模型和处理后模型的速度差异。对于每个处理类型，
        计算并绘制速度差值（基础模型 - 处理后模型）的分布。
        
        参数:
            output_file (str): 输出图片文件路径
            base_model (Union[ZeltVelocityModel2d, EnhancedZeltModel]): 基础模型
            region (Optional[List[float]]): 绘图区域 [xmin, xmax, zmin, zmax]
            process_type (Optional[str]): 处理类型，可选值：
                                      - 'average_velocity': 平均速度
                                      - 'interval_velocity': 层间速度
                                      - 'two_way_time': 双程走时
                                      如果为 None，则直接比较当前模型与基础模型
            title (Optional[str]): 图像标题，如果为 None 则自动生成
            xlabel (str): x轴标签
            ylabel (str): y轴标签
            colorbar_label (str): 色标标签
            figsize (Tuple[float, float]): 图像大小
            velocity_cmap (str): 颜色映射，建议使用发散色标如 'seismic', 'coolwarm', 'RdBu' 等
                        添加 '_r' 后缀可以反转色标
                        支持 CPT 文件，文件格式如下：
                        # 标准 CPT 文件格式
                        # 深度(km) 红 绿 蓝 深度(km) 红 绿 蓝
                        0    0   0  0  10 155 155 155
                        10 155 155 155 20 255 255 255 
                        20 255 255 255 30 255 255 255
                        B	255	255	255
            plot_interfaces (bool): 是否绘制层界面
            interface_color (str): 界面线条颜色
            interface_linewidth (float): 界面线条宽度
            interface_alpha (float): 界面线条透明度
            dx (float): 网格间距
            dz (float): 网格间距
            dpi (int): 图像分辨率
            
        异常:
            ValueError: 如果未提供模型或处理类型无效
            
        示例:
            >>> # 创建基础模型和可视化器
            >>> base_model = ZeltVelocityModel2d("v.in")
            >>> enhanced_model = EnhancedZeltModel(base_model)
            >>> visualizer = ZeltModelVisualizer(enhanced_model)
            
            # 比较平均速度处理的差异
            >>> enhanced_model.process_velocity_model('average_velocity')
            >>> visualizer.plot_velocity_difference(
            ...     output_file="avg_velocity_diff.png",
            ...     base_model=base_model,
            ...     process_type='average_velocity'
            ... )
            
            # 比较层间速度处理的差异
            >>> enhanced_model.process_velocity_model('interval_velocity')
            >>> visualizer.plot_velocity_difference(
            ...     output_file="interval_velocity_diff.png",
            ...     base_model=base_model,
            ...     process_type='interval_velocity',
            ...     cmap='coolwarm'
            ... )
            
            # 比较双程走时处理的差异
            >>> enhanced_model.process_velocity_model('two_way_time')
            >>> visualizer.plot_velocity_difference(
            ...     output_file="twt_velocity_diff.png",
            ...     base_model=base_model,
            ...     process_type='two_way_time',
            ...     ylabel="Time (s)"  # 注意：时间域的y轴标签不同
            ... )
        """
        if self.model is None:
            raise ValueError("无法绘制差值: 未提供当前模型")
            
        import matplotlib.pyplot as plt
        
        # 获取基础模型的速度分布
        base_ds = base_model.to_xarray(dx, dz)
         # 确定绘图区域
        if region is None:
                region = [
                float(base_ds.x.min()),
                float(base_ds.x.max()),
                float(base_ds.z.min()), 
                float(base_ds.z.max())
            ]
    
        # 如果指定了处理类型，先进行处理
        if process_type is not None:
            if not isinstance(self.model, EnhancedZeltModel):
                raise ValueError("要进行模型处理，当前模型必须是 EnhancedZeltModel 类型")
            self.model.process_velocity_model(process_type)
            
        # 获取当前模型的速度分布
        current_ds = self.model.to_xarray(dx, dz)
        
        # 计算速度差值（基础模型 - 当前模型）
        velocity_diff = base_ds.velocity - current_ds.velocity
        
        # 创建图形
        plt.figure(figsize=figsize)
        
        # 使用发散色标以便于区分正负差值
        vmax = float(np.nanmax(np.abs(velocity_diff)))
        vmin = -vmax
        if isinstance(velocity_cmap, str) and velocity_cmap.lower().endswith('.cpt') and os.path.exists(velocity_cmap):
            cmap = self.load_cpt(velocity_cmap)
        else:
            cmap = velocity_cmap
        # 绘制速度差值
        im = plt.pcolormesh(current_ds.x, current_ds.z, velocity_diff, 
                           shading='auto', 
                           cmap=cmap,
                           vmin=vmin, 
                           vmax=vmax)
        
        # 绘制层界面
        if plot_interfaces:
            for i in range(len(self.model.depth_nodes)):
                x_coords, z_coords = self.model.get_layer_geometry(i)
                plt.plot(x_coords, z_coords,
                        color=interface_color,
                        linewidth=interface_linewidth,
                        alpha=interface_alpha,
                        linestyle='-')
        
        # 添加色标
        plt.colorbar(im, label=colorbar_label)
        
        # 设置标题和轴标签
        if title is None:
            if process_type:
                title = f'Velocity Difference ({process_type})'
            else:
                title = 'Velocity Difference'
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        # 设置绘图区域
        if region:
            plt.xlim(region[0], region[1])
            plt.ylim(region[3], region[2])  # 注意：反转顺序以保持深度向下增加
        
        # 输出统计信息
        valid_diff = velocity_diff.values[~np.isnan(velocity_diff.values)]
        print(f"\n速度差值统计:")
        print(f"最小差值: {np.min(valid_diff):.3f} km/s")
        print(f"最大差值: {np.max(valid_diff):.3f} km/s")
        print(f"平均差值: {np.mean(valid_diff):.3f} km/s")
        print(f"标准差: {np.std(valid_diff):.3f} km/s")
        
        # 保存图形
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()

class GridModelVisualizer:
    """A class for visualizing grid format velocity models.
    
    This class provides methods to plot and process velocity models stored in
    grid formats (e.g., .grd, .nc). It supports various visualization options
    including contours, shading, and custom colormaps.
    
    Attributes:
        grid_file (Optional[Path]): Path to the grid file
        fig (pygmt.Figure): PyGMT figure object
        output_dir (Path): Directory for output files
        plot_region (Optional[List[float]]): Plot region boundaries [xmin, xmax, zmin, zmax]
    """
    
    def __init__(self, 
                 grid_file: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 plot_region: Optional[Tuple[float, float, float, float]] = None):
        """Initialize the grid model visualizer.
        
        Args:
            grid_file (Optional[str]): Path to the grid file (.grd or .nc format)
            output_dir (Optional[str]): Directory for output files. Defaults to './output'
            plot_region (Optional[Tuple[float, float, float, float]]): Plot region as (xmin, xmax, zmin, zmax)
        """
        self.grid_file = Path(grid_file) if grid_file else None
        self.fig = None
        self.output_dir = Path(output_dir) if output_dir else Path('./output')
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize plot region
        if plot_region:
            self.plot_region = list(plot_region)
        elif grid_file:
            # Try to get region from grid file
            with xr.open_dataset(grid_file) as ds:
                # 获取坐标变量名
                coords = list(ds.coords)
                x_coord = next((c for c in coords if c in ['x', 'lon', 'longitude']), None)
                z_coord = next((c for c in coords if c in ['z', 'y', 'depth', 't', 'time']), None)
                
                if x_coord is None or z_coord is None:
                    raise ValueError(f"无法识别坐标变量，可用的坐标: {coords}")
                    
                self.plot_region = [
                    float(ds[x_coord].min()),
                    float(ds[x_coord].max()),
                    float(ds[z_coord].min()),
                    float(ds[z_coord].max())
                ]
        else:
            self.plot_region = None
    def create_zero_cmap(self, base_cmap: str = 'gray', zero_threshold: float = 1e-6) -> LinearSegmentedColormap:
        """创建零振幅值为无色的色板。
        
        参数:
            base_cmap (str): 基础色板名称或色板对象
            zero_threshold (float): 零值阈值，[-threshold, threshold]范围内的振幅将被设为无色
            
        返回:
            LinearSegmentedColormap: 修改后的色板，零值为无色
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        # 如果输入是字符串，则获取对应的色板对象
        if isinstance(base_cmap, str):
            base_cmap = plt.get_cmap(base_cmap)
            
        # 获取色板的颜色列表
        colors = base_cmap(np.linspace(0, 1, 256))
        
        # 计算中心点位置（对应振幅为0的位置）
        center = 128
        
        # 计算阈值对应的颜色索引范围
        threshold_range = int(256 * zero_threshold)
        
        # 修改颜色列表，使零值附近的颜色为透明
        colors[center-threshold_range:center+threshold_range, 3] = 0  # 设置alpha通道为0（完全透明）
        
        # 创建新的色板
        return LinearSegmentedColormap.from_list('zero_transparent', colors)
            
    def load_cpt(self, cpt_file: str) -> str:
        """加载 CPT 文件并返回其内容。
        
        参数:
            cpt_file (str): CPT 文件的路径
            Returns:
        LinearSegmentedColormap: matplotlib 颜色映射对象
        """
        cpt_data = []
        with open(cpt_file, 'r') as file:
            lines = file.readlines()
        for line in lines:
            #skip the comment lines
            if line.startswith('#'):
                continue
            #parse the cpt data
            data = line.strip().split()
            if len(data) == 8: # standard cpt file format
                z1, r1, g1, b1, z2, r2, g2, b2 = map(float, data)
                cpt_data.append([z1, r1/255, g1/255, b1/255])
                cpt_data.append([z2, r2/255, g2/255, b2/255])
        # covert into numpy
        cpt_data = np.array(cpt_data)
        # 归一化
        z_normalized = (cpt_data[:, 0] - cpt_data[:, 0].min()) / (cpt_data[:, 0].max() - cpt_data[:, 0].min())
        colors = cpt_data[:, 1:]
        cmap = LinearSegmentedColormap.from_list('custom_cpt', list(zip(z_normalized, colors)))
        return cmap
            
    def load_grid(self, grid_file: str) -> None:
        """Load a grid file.
        
        Args:
            grid_file (str): Path to the grid file (.grd or .nc format)
        """
        self.grid_file = Path(grid_file)
        if not self.grid_file.exists():
            raise FileNotFoundError(f"Grid file not found: {self.grid_file}")
            
        # Update plot region from new grid
        with xr.open_dataset(grid_file) as ds:
            self.plot_region = [
                float(ds.x.min()),
                float(ds.x.max()),
                float(ds.z.min()),
                float(ds.z.max())
            ]
            
    def plot_grid(self,
                 output_fig: str,
                 figsize: Tuple[float, float] = (10, 8),
                 velocity_cmap: str = 'viridis',
                 cmap_reverse: bool = False,
                 cpt_series: Optional[str] = None,
                 cpt_continuous: bool = True,
                 title: Optional[str] = None,
                 colorbar_label: Optional[str] = None,
                 region: Optional[Tuple[float, float, float, float]] = None,
                 plot_contours: bool = False,
                 contour_interval: Optional[float] = None,
                 contour_annotation_interval: Optional[float] = None,
                 contour_pen: str = '0.5p,black',
                 contour_annotation_pen: str = '1p,black',
                 use_shading: bool = False,
                 azimuth: float = 45,
                 norm_method: Union[bool, str] = "e",
                 norm_amp: Optional[float] = None,
                 norm_sigma: Optional[float] = None,
                 norm_offset: Optional[float] = None,
                 ambient: Optional[float] = None,
                 frame: Optional[List[str]] = None) -> pygmt.Figure:
        """Plot the grid.
        
        Args:
            output_fig (str): Output figure path
            figsize (Tuple[float, float]): Figure size in inches (width, height)
            velocity_cmap (str): Colormap for velocity field. Can be:
                              1. Path to a .cpt file
                              2. Built-in GMT colormap name (e.g., 'polar', 'seis')
            cmap_reverse (bool): Whether to reverse the colormap
            cpt_series (Optional[str]): Series parameter for CPT creation in format "min/max/inc[+b|l|n]"
            cpt_continuous (bool): Whether to use continuous colormap
            title (Optional[str]): Title for the plot
            colorbar_label (Optional[str]): Label for the colorbar. If None, will be determined from grid content
            region (Optional[Tuple[float, float, float, float]]): Plot region for this specific plot
            plot_contours (bool): Whether to plot velocity contours
            contour_interval (Optional[float]): Interval between contour lines
            contour_annotation_interval (Optional[float]): Interval between annotated contour lines
            contour_pen (str): Pen settings for contour lines (width,color)
            contour_annotation_pen (str): Pen settings for annotated contour lines (width,color)
            use_shading (bool): Whether to apply shading effect
            azimuth (float): Azimuth angle for shading in degrees
            norm_method (str or bool): Normalization method for shading
            norm_amp (Optional[float]): Maximum output magnitude after normalization
            norm_sigma (Optional[float]): Sigma parameter for normalization
            norm_offset (Optional[float]): Offset parameter for normalization
            ambient (Optional[float]): Ambient light to add after normalization
            frame (Optional[List[str]]): Frame settings for the plot
            
        Returns:
            pygmt.Figure: The resulting figure
            
        Raises:
            ValueError: If no grid file is loaded
        """
        if not self.grid_file:
            raise ValueError("No grid file loaded. Call load_grid() first or provide grid_file during initialization.")
            
        if colorbar_label is None:
            with xr.open_dataset(self.grid_file) as ds:
                if 'twt' in ds.data_vars:
                    colorbar_label = "Two-Way Time (s)"
                elif 'velocity' in ds.data_vars:
                    colorbar_label = "Velocity (km/s)"
                elif 'density' in ds.data_vars:
                    colorbar_label = "Density (g/cm³)"
                else:
                    colorbar_label = "Value"
            
        self.fig = pygmt.Figure()
        projection = f'X{figsize[0]}i/{figsize[1]}i'
        current_region = list(region) if region else self.plot_region
        
        if isinstance(velocity_cmap, str) and velocity_cmap.lower().endswith('.cpt') and os.path.exists(velocity_cmap):
            cpt_file = velocity_cmap
        else:
            cpt_file = self.output_dir / "temp.cpt"
            pygmt.grd2cpt(
                grid=str(self.grid_file),
                cmap=velocity_cmap,
                continuous=cpt_continuous,
                reverse=cmap_reverse,
                series=cpt_series,
                output=str(cpt_file)
            )
            
        if use_shading:
            norm_str = ""
            if isinstance(norm_method, str):
                norm_str = norm_method
            if norm_amp is not None:
                norm_str += str(norm_amp)
            if norm_sigma is not None:
                norm_str += f"+s{norm_sigma}"
            if norm_offset is not None:
                norm_str += f"+o{norm_offset}"
            if ambient is not None:
                norm_str += f"+a{ambient}"
            
            shade = pygmt.grdgradient(
                grid=str(self.grid_file),
                azimuth=azimuth,
                normalize=norm_str if norm_str else norm_method
            )
            self.fig.grdimage(
                str(self.grid_file),
                region=current_region,
                projection=projection,
                frame=frame if frame is not None else ["af", f'+t"{title}"', 'x+l"Distance (km)"', 'y+l"Depth (km)"'] if title else ["af", 'x+l"Distance (km)"', 'y+l"Depth (km)"'],
                cmap=cpt_file,
                shading=shade,
                nan_transparent=True
            )
        else:
            self.fig.grdimage(
                str(self.grid_file),
                region=current_region,
                projection=projection,
                frame=frame if frame is not None else ['af', f'+t"{title}"', 'x+lDistance (km)', 'y+lDepth (km)'] if title else ['af', 'x+lDistance (km)', 'y+lDepth (km)'],
                cmap=cpt_file,
                nan_transparent=True
            )
            
        self.fig.colorbar(
            frame=[f'af+l"{colorbar_label}"'],
            position='JMR+o0.5c/0c+w8c',
            cmap=cpt_file
        )

        if plot_contours:
            if contour_interval is None:
                with xr.open_dataset(self.grid_file) as ds:
                    data_var = next(iter(ds.data_vars))
                    v_min = float(np.nanmin(ds[data_var].values))
                    v_max = float(np.nanmax(ds[data_var].values))
                    v_range = v_max - v_min
                    contour_interval = v_range / 10

            if contour_annotation_interval is None:
                contour_annotation_interval = contour_interval

            self.fig.grdcontour(
                str(self.grid_file),
                levels=contour_interval,
                annotation=contour_annotation_interval,
                pen=[f"c{contour_pen}", f"a{contour_annotation_pen}"]
            )
            
           
        self.fig.savefig(output_fig)
        return self.fig
        
    def get_velocity_range(self) -> Tuple[float, float]:
        """Get the velocity range from the grid.
        
        Returns:
            Tuple[float, float]: Minimum and maximum velocity values
            
        Raises:
            ValueError: If no grid file is loaded
        """
        if not self.grid_file:
            raise ValueError("No grid file loaded")
            
        with xr.open_dataset(self.grid_file) as ds:
            v_min = float(np.nanmin(ds.velocity.values))
            v_max = float(np.nanmax(ds.velocity.values))
        return v_min, v_max
        
    def get_plot_region(self) -> List[float]:
        """Get the plot region from the grid.
        
        Returns:
            List[float]: Plot region as [xmin, xmax, zmin, zmax]
            
        Raises:
            ValueError: If no grid file is loaded and no plot region is set
        """
        if self.plot_region is None:
            raise ValueError("No plot region available")
        return self.plot_region

    def plot_two_grids(self,
                    output_fig: str,
                    upper_grid_file: str,
                    figsize: Tuple[float, float] = (10, 8),
                    lower_cmap: str = 'seis',
                    upper_cmap: str = 'gray',
                    upper_transparency: float = 50,  # 0-100 透明度百分值
                    cmap_reverse: bool = False,
                    title: Optional[str] = None,
                    colorbar_label: Optional[str] = None,
                    region: Optional[Tuple[float, float, float, float]] = None,
                    plot_contours: bool = False,
                    contour_interval: Optional[float] = None,
                    contour_annotation_interval: Optional[float] = None,
                    contour_pen: str = '0.5p,black',
                    contour_annotation_pen: str = '1p,black',
                    frame: Optional[List[str]] = None) -> pygmt.Figure:
        """叠加绘制两个网格文件。
        
        Args:
            output_fig (str): 输出图像路径
            upper_grid_file (str): 上层网格文件路径
            figsize (Tuple[float, float]): 图像大小
            lower_cmap (str): 1. Path to a .cpt file
                              2. Built-in GMT colormap name (e.g., 'seis')
            upper_cmap (str): 1. Path to a .cpt file
                              2. Built-in GMT colormap name (e.g., 'gray')
            upper_transparency (float): 上层网格的透明度 (0-100)
            cmap_reverse (bool): 是否反转颜色映射
            title (Optional[str]): 图像标题
            colorbar_label (Optional[str]): 色标标签
            region (Optional[Tuple]): 绘图区域
            plot_contours (bool): 是否绘制等值线
            contour_interval (Optional[float]): 等值线间隔
            contour_annotation_interval (Optional[float]): 标注等值线间隔
            contour_pen (str): 等值线画笔设置
            contour_annotation_pen (str): 标注等值线画笔设置
            frame (Optional[List[str]]): 坐标轴设置
        """
        if not self.grid_file:
            raise ValueError("未加载下层网格文件")
            
        if not Path(upper_grid_file).exists():
            raise FileNotFoundError(f"找不到上层网格文件: {upper_grid_file}")
            
        # 创建新的 PyGMT 图像
        self.fig = pygmt.Figure()
        
        # 设置投影
        projection = f'X{figsize[0]}i/{figsize[1]}i'
        
        # 确定绘图区域
        current_region = list(region) if region else self.plot_region
        
        # 为下层网格创建颜色映射
        if isinstance(lower_cmap, str) and lower_cmap.lower().endswith('.cpt') and os.path.exists(lower_cmap):
            lower_cpt = lower_cmap
        else:
            lower_cpt = self.output_dir / "lower_temp.cpt"
            pygmt.grd2cpt(
            grid=str(self.grid_file),
            cmap=lower_cmap,
            continuous=True,
            reverse=cmap_reverse,
            output=str(lower_cpt)
            )
        
        # 为上层网格创建颜色映射
        if isinstance(upper_cmap, str) and upper_cmap.lower().endswith('.cpt') and os.path.exists(upper_cmap):
            upper_cpt = upper_cmap
        else:
            upper_cpt = self.output_dir / "upper_temp.cpt"
            pygmt.grd2cpt(
            grid=str(upper_grid_file),
            cmap=upper_cmap,
            continuous=True,
            reverse=cmap_reverse,
            output=str(upper_cpt)
            )
        
        # 绘制下层网格
        self.fig.grdimage(
            str(self.grid_file),
            region=current_region,
            projection=projection,
            frame=frame if frame is not None else ["af", f'+t"{title}"', 'x+l"Distance (km)"', 'y+l"Depth (km)"'] if title else ["af", 'x+l"Distance (km)"', 'y+l"Depth (km)"'],
            cmap=lower_cpt,
            nan_transparent=True
        )
        
        # 绘制上层网格（带透明度）
        # 将0-1范围的透明度转换为0-100的百分比
        self.fig.grdimage(
            str(upper_grid_file),
            region=current_region,
            projection=projection,
            cmap=upper_cpt,
            transparency=upper_transparency,
            nan_transparent=True
        )
        
        # 如果需要，添加等值线
        if plot_contours:
            if contour_interval is None:
                # 自动计算等值线间隔
                with xr.open_dataset(upper_grid_file) as ds:
                    data_var = next(iter(ds.data_vars))
                    v_min = float(np.nanmin(ds[data_var].values))
                    v_max = float(np.nanmax(ds[data_var].values))
                    v_range = v_max - v_min
                    contour_interval = v_range / 10

            if contour_annotation_interval is None:
                contour_annotation_interval = contour_interval

            # 绘制等值线
            self.fig.grdcontour(
                str(upper_grid_file),
                levels=contour_interval,
                annotation=contour_annotation_interval,
                pen=[f"c{contour_pen}", f"a{contour_annotation_pen}"]
            )
        
        # 添加色标
        if colorbar_label:
            self.fig.colorbar(
                frame=[f'af+l"{colorbar_label}"'],
                position='JMR+o0.5c/0c+w8c',
                cmap=lower_cpt
            )
        
        
        # 保存图像
        self.fig.savefig(output_fig)
        return self.fig

    def plot_with_matplotlib(self,
                         output_fig: str,
                         figsize: Tuple[float, float] = (10, 8),
                         lower_cmap: str = 'seismic',
                         upper_grid_file: Optional[str] = None,
                         upper_cmap: str = 'gray',
                         upper_transparency: float = 0.5,
                         title: Optional[str] = None,
                         colorbar_label: Optional[str] = None,
                         plot_region: Optional[Tuple[float, float, float, float]] = None,
                         lower_clim: Optional[List[float]] = None,  # [vmin, vmax] in data value
                         upper_clim: Optional[List[float]] = None,  # [clip1, clip2] in percent
                         colorbar_orientation: str = 'vertical',
                         colorbar_fraction: float = 0.046,
                         colorbar_pad: float = 0.04,
                         plot_interfaces: bool = False,
                         model: Optional[Union[ZeltVelocityModel2d, EnhancedZeltModel]] = None,
                         interface_color: str = 'black',
                         interface_linewidth: float = 0.5,
                         interface_linestyle: str = ':',
                         plot_contours_lower: bool = False,
                         contour_interval_lower: Optional[float] = None,
                         contour_levers_lower: Optional[List[float]] = None,
                         contour_colors_lower: Union[str, List[str]] = 'black',
                         contour_linewidths_lower: Union[float, List[float]] = 0.5,
                         contour_linestyles_lower: Union[str, List[str]] = '-',
                         contour_label_fmt_lower: str = '%.2f',
                         contour_label_fontsize_lower: float = 8,
                         contour_inline_lower: bool = True,
                         contour_inline_spacing_lower: float = 5,
                         xlabel: str = "Distance (km)",
                         ylabel: str = "Depth (km)") -> None:
        """使用 Matplotlib 绘制网格数据。
        
        Args:
            output_fig (str): 输出图像路径
            figsize (Tuple[float, float]): 图像大小
            lower_cmap (str): 下层数据的颜色映射，可以是：
                           1. matplotlib内置颜色映射名称
                           2. .cpt文件路径
            upper_grid_file (Optional[str]): 上层网格文件路径
            upper_cmap (str): 上层数据的颜色映射，可以是：
                           1. matplotlib内置颜色映射名称
                           2. .cpt文件路径
            upper_transparency (float): 上层数据的透明度 (0-1)
            title (Optional[str]): 图像标题
            colorbar_label (Optional[str]): 色标标签
            plot_region (Optional[Tuple]): 绘图区域 (xmin, xmax, zmin, zmax)
            lower_clim (Optional[List[float]]): 下层数据的值范围 [vmin, vmax] in data value
            upper_clim (Optional[List[float]]): 上层数据的值范围 [clip1, clip2] in percent
            colorbar_orientation (str): 色标方向，'vertical' 或 'horizontal'
            colorbar_fraction (float): 色标相对于主图的大小比例
            colorbar_pad (float): 色标与主图的间距
            plot_interfaces (bool): 是否绘制界面信息
            model (Optional[Union[ZeltVelocityModel2d, EnhancedZeltModel]]): Zelt 速度模型对象
            interface_color (str): 界面线条颜色
            interface_linewidth (float): 界面线条宽度
            interface_linestyle (str): 线条样式
            xlabel (str): x轴标签
            ylabel (str): y轴标签
        """
        import matplotlib.pyplot as plt
        
        if not self.grid_file:
            raise ValueError("未加载下层网格文件")
            
        if plot_interfaces and model is None:
            raise ValueError("要绘制界面信息，必须提供 model 参数")
            
        # 创建图形
        plt.figure(figsize=(abs(figsize[0]), abs(figsize[1])))
        
        # 读取下层数据和坐标范围
        with xr.open_dataset(self.grid_file) as ds_lower    :
            # 获取数据变量
            data_var_lower = next(iter(ds_lower.data_vars))
            lower_data = ds_lower[data_var_lower].values
            
            # 振幅剪切处理
            if lower_clim is None:
                # 默认使用1-99百分位数作为剪切范围
                vmin = np.percentile(lower_data[~np.isnan(lower_data)], 1)
                vmax = np.percentile(lower_data[~np.isnan(lower_data)], 99)
                lower_clim = [vmin, vmax]
            
            # 获取坐标变量
            coords_lower = list(ds_lower.coords)
            x_coord_lower = next((c for c in coords_lower if c in ['x', 'lon', 'longitude']), None)
            z_coord_lower = next((c for c in coords_lower if c in ['z', 'y', 'depth', 't', 'time']), None)
            
            if x_coord_lower is None or z_coord_lower is None:
                raise ValueError(f"无法识别坐标变量，可用的坐标: {coords_lower}")
            
            # 检查坐标是否存在于数据集中
            if z_coord_lower not in ds_lower:
                raise ValueError(f"坐标变量 {z_coord_lower} 不存在于数据集中。可用的变量: {list(ds.variables)}")
            
            lower_extent = [
                float(ds_lower[x_coord_lower].min()),
                float(ds_lower[x_coord_lower].max()),
                float(ds_lower[z_coord_lower].max()),  # 注意：反转z轴
                float(ds_lower[z_coord_lower].min())
            ]
            
        # 如果有上层数据，获取其坐标范围
        upper_extent = None
        if upper_grid_file:
            with xr.open_dataset(upper_grid_file) as ds_upper:
                coords_upper = list(ds_upper.coords)
                x_coord_upper = next((c for c in coords_upper if c in ['x', 'lon', 'longitude']), None)
                z_coord_upper = next((c for c in coords_upper if c in ['z', 'y', 'depth', 't', 'time']), None)
                
                if x_coord_upper is None or z_coord_upper is None:
                    raise ValueError(f"无法识别上层数据的坐标变量，可用的坐标: {coords_upper}")
                    
                # 检查坐标是否存在于数据集中
                if z_coord_upper not in ds_upper:
                    raise ValueError(f"坐标变量 {z_coord_upper} 不存在于数据集中。可用的变量: {list(ds.variables)}")
                    
                upper_extent = [
                    float(ds_upper[x_coord_upper].min()),
                    float(ds_upper[x_coord_upper].max()),
                    float(ds_upper[z_coord_upper].max()),  # 注意：反转z轴
                    float(ds_upper[z_coord_upper].min())
                ]
        
        # 确定最终的绘图区域
        if plot_region:
            extent = [
                plot_region[0],
                plot_region[1],
                plot_region[3],
                plot_region[2]
            ]
        elif upper_extent:
            # 使用两个数据集的并集作为绘图区域
            extent = [
                min(lower_extent[0], upper_extent[0]),
                max(lower_extent[1], upper_extent[1]),
                max(lower_extent[2], upper_extent[2]),  # z轴反转
                min(lower_extent[3], upper_extent[3])
            ]
        else:
            extent = lower_extent
            
        # 解析lower_cmap
        if isinstance(lower_cmap, str) and lower_cmap.lower().endswith('.cpt') and os.path.exists(lower_cmap):
            lower_cmap = self.load_cpt(lower_cmap)
            
        # 绘制下层数据
        im1 = plt.imshow(lower_data, 
                        extent=lower_extent,
                        cmap=lower_cmap,
                        aspect='auto',
                        vmin=lower_clim[0] if lower_clim else None,
                        vmax=lower_clim[1] if lower_clim else None)
        plt.colorbar(im1, 
                    label=colorbar_label if colorbar_label else "Value",
                    orientation=colorbar_orientation,
                    fraction=colorbar_fraction,
                    pad=colorbar_pad)
        if plot_contours_lower:
            # 绘制等值线
            if contour_interval_lower is not None:
                levels = contour_levers_lower
            elif contour_interval_lower is not None:
                levels = np.arange(vmin, vmax + contour_interval_lower, contour_interval_lower)
            else:
                # 自动计算10条等值线
                levels = np.linspace(vmin, vmax, 10)
            # 获取网格坐标
            x = ds_lower[x_coord_lower].values
            z = ds_lower[z_coord_lower].values
            X, Z = np.meshgrid(x, z)
            # 绘制等值线    
            CS = plt.contour(X, Z, lower_data, 
                             levels=levels, 
                             colors=contour_colors_lower, 
                             linewidths=contour_linewidths_lower, 
                             linestyles=contour_linestyles_lower
                             )
            # 添加等值线标签
            if contour_inline_lower:
                plt.clabel(CS, CS.levels,
                          fmt=contour_label_fmt_lower,
                          fontsize=contour_label_fontsize_lower,
                          inline=True,
                          inline_spacing=contour_inline_spacing_lower)
            
        # 如果有上层数据，叠加绘制
        if upper_grid_file:
            with xr.open_dataset(upper_grid_file) as ds:
                data_var = next(iter(ds.data_vars))
                upper_data = ds[data_var].values
                
                # 振幅剪切处理
                if upper_clim is None:
                    # 默认使用1-99百分位数作为剪切范围
                    vmin = np.percentile(upper_data[~np.isnan(upper_data)], 5)
                    vmax = np.percentile(upper_data[~np.isnan(upper_data)], 99)
                else:
                    vmin = np.percentile(upper_data[~np.isnan(upper_data)], upper_clim[0])
                    vmax = np.percentile(upper_data[~np.isnan(upper_data)], upper_clim[1])
                    upper_clim = [vmin, vmax]
                
                # 解析upper_cmap
                if isinstance(upper_cmap, str) and upper_cmap.lower().endswith('.cpt') and os.path.exists(upper_cmap):
                    upper_cmap = self.load_cpt(upper_cmap)
                    
                im2 = plt.imshow(upper_data,
                               extent=upper_extent,
                               cmap=upper_cmap,
                               alpha=upper_transparency,
                               aspect='auto',
                               vmin=upper_clim[0] if upper_clim else None,
                               vmax=upper_clim[1] if upper_clim else None)
        
        # 如果需要绘制界面信息
        if plot_interfaces and model:
            for i in range(len(model.depth_nodes)):
                x_coords, z_coords = model.get_layer_geometry(i)
                plt.plot(x_coords, z_coords, 
                        color=interface_color, 
                        linewidth=interface_linewidth,
                        linestyle=interface_linestyle)
        
        # 设置标题和轴标签
        if title:
            plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        # 设置绘图区域
        plt.axis(extent)
        
        # 如果需要反转y轴
        if figsize[1] < 0:
            plt.gca().invert_yaxis()
        
        # 保存图像
        plt.savefig(output_fig, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_xarray(self,
                    output_fig: str,
                    data: Union[xr.Dataset, str],
                    figsize: Tuple[float, float] = (10, 8),
                    cmap: str = 'seismic',
                    title: Optional[str] = None,
                    colorbar_label: Optional[str] = None,
                    plot_region: Optional[Tuple[float, float, float, float]] = None,
                    clim: Optional[List[float]] = None,  # [clip1, clip2] in percent
                    colorbar_orientation: str = 'vertical',
                    colorbar_fraction: float = 0.046,
                    colorbar_pad: float = 0.04,
                    xlabel: str = "Distance (km)",
                    ylabel: str = "Depth (km)",
                    plot_interfaces: bool = False,
                    model: Optional[Union[ZeltVelocityModel2d, EnhancedZeltModel]] = None,
                    interface_color: str = 'black',
                    interface_linewidth: float = 0.5,
                    interface_linestyle: str = ':',
                    plot_contours: bool = False,
                    contour_interval: Optional[float] = None,
                    contour_levels: Optional[List[float]] = None,
                    contour_colors: Union[str, List[str]] = 'black',
                    contour_linewidths: Union[float, List[float]] = 0.5,
                    contour_linestyles: Union[str, List[str]] = '-',
                    contour_label_fmt: str = '%.2f',
                    contour_label_fontsize: float = 8,
                    contour_inline: bool = True,
                    contour_inline_spacing: float = 5) -> None:
        """使用 Matplotlib 绘制 xarray 数据。
        
        Args:
            output_fig (str): 输出图像路径
            data (Union[xr.Dataset, str]): 要绘制的xarray数据集或.grd文件路径
            figsize (Tuple[float, float]): 图像大小
            cmap (str): 颜色映射，可以是：
                     1. matplotlib内置颜色映射名称
                     2. .cpt文件路径
            title (Optional[str]): 图像标题
            colorbar_label (Optional[str]): 色标标签
            plot_region (Optional[Tuple]): 绘图区域 (xmin, xmax, zmin, zmax)
            clim (Optional[List[float]]): 数据值范围 [clip1, clip2] in percent
            colorbar_orientation (str): 色标方向，'vertical' 或 'horizontal'
            colorbar_fraction (float): 色标相对于主图的大小比例
            colorbar_pad (float): 色标与主图的间距
            xlabel (str): x轴标签
            ylabel (str): y轴标签
            plot_interfaces (bool): 是否绘制界面
            model (Optional[Union[ZeltVelocityModel2d, EnhancedZeltModel]]): Zelt模型对象
            interface_color (str): 界面线条颜色
            interface_linewidth (float): 界面线条宽度
            interface_linestyle (str): 界面线条样式
            plot_contours (bool): 是否绘制等值线
            contour_interval (Optional[float]): 等值线间隔，如果未指定则自动计算
            contour_levels (Optional[List[float]]): 自定义等值线水平，优先级高于contour_interval
            contour_colors (Union[str, List[str]]): 等值线颜色，可以是单个颜色或颜色列表
            contour_linewidths (Union[float, List[float]]): 等值线宽度，可以是单个值或列表
            contour_linestyles (Union[str, List[str]]): 等值线样式，可以是单个样式或样式列表
            contour_label_fmt (str): 等值线标签格式化字符串
            contour_label_fontsize (float): 等值线标签字体大小
            contour_inline (bool): 是否在等值线上绘制标签
            contour_inline_spacing (float): 等值线标签间距
        """
        import matplotlib.pyplot as plt
        
        # 创建图形
        plt.figure(figsize=(abs(figsize[0]), abs(figsize[1])))
        
        # 如果输入是文件路径，则加载数据
        if isinstance(data, str):
            data = xr.open_dataset(data)
        
        # 获取数据变量
        data_var = next(iter(data.data_vars))
        plot_data = data[data_var].values
        
        # 振幅剪切处理
        if clim is None:
            # 默认使用1-99百分位数作为剪切范围
            vmin = np.percentile(plot_data[~np.isnan(plot_data)], 1)
            vmax = np.percentile(plot_data[~np.isnan(plot_data)], 99)
        else:
            # 使用指定的百分比范围
            vmin = np.percentile(plot_data[~np.isnan(plot_data)], clim[0])
            vmax = np.percentile(plot_data[~np.isnan(plot_data)], clim[1])
        
        # 获取坐标变量
        coords = list(data.coords)
        x_coord = next((c for c in coords if c in ['x', 'lon', 'longitude']), None)
        z_coord = next((c for c in coords if c in ['z', 'y', 'depth', 't']), None)
        
        if x_coord is None or z_coord is None:
            raise ValueError(f"无法识别坐标变量，可用的坐标: {coords}")
        
        # 设置数据范围
        extent = [
            float(data[x_coord].min()),
            float(data[x_coord].max()),
            float(data[z_coord].max()),  # 注意：反转z轴
            float(data[z_coord].min())
        ]
        
        # 确定最终的绘图区域
        if plot_region:
            extent = [
                plot_region[0],
                plot_region[1],
                plot_region[3],  # 注意：反转顺序以保持深度向下增加
                plot_region[2]
            ]
            
        # 解析cmap
        if isinstance(cmap, str) and cmap.lower().endswith('.cpt') and os.path.exists(cmap):
            cmap = self.load_cpt(cmap)
            
        # 绘制数据
        im = plt.imshow(plot_data, 
                       extent=extent,  # 使用数据的实际范围
                       cmap=cmap,
                       aspect='auto',
                       vmin=vmin,
                       vmax=vmax)
        plt.colorbar(im, 
                    label=colorbar_label if colorbar_label else "Value",
                    orientation=colorbar_orientation,
                    fraction=colorbar_fraction,
                    pad=colorbar_pad)
        
        # 绘制等值线
        if plot_contours:
            # 准备等值线水平
            if contour_levels is not None:
                levels = contour_levels
            elif contour_interval is not None:
                levels = np.arange(vmin, vmax + contour_interval, contour_interval)
            else:
                # 自动计算10条等值线
                levels = np.linspace(vmin, vmax, 10)
            
            # 获取网格坐标
            x = data[x_coord].values
            z = data[z_coord].values
            X, Z = np.meshgrid(x, z)
            
            # 绘制等值线
            CS = plt.contour(X, Z, plot_data,
                           levels=levels,
                           colors=contour_colors,
                           linewidths=contour_linewidths,
                           linestyles=contour_linestyles)
            
            # 添加等值线标签
            if contour_inline:
                plt.clabel(CS, CS.levels,
                          fmt=contour_label_fmt,
                          fontsize=contour_label_fontsize,
                          inline=True,
                          inline_spacing=contour_inline_spacing)
        
        # 如果需要绘制界面
        if plot_interfaces and model:
            for i in range(len(model.depth_nodes)):
                x_coords, z_coords = model.get_layer_geometry(i)
                plt.plot(x_coords, z_coords,
                        color=interface_color,
                        linewidth=interface_linewidth,
                        linestyle=interface_linestyle)
        
        # 设置标题和轴标签
        if title:
            plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        # 设置绘图区域
        plt.axis(extent)
        
        # 如果需要反转y轴
        if figsize[1] < 0:
            plt.gca().invert_yaxis()
        
        # 保存图像
        plt.savefig(output_fig, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 如果输入是文件，关闭数据集
        if isinstance(data, str):
            data.close()


    
class GridModelProcessor:
    """A class for processing and converting grid format velocity models.
    
    This class provides methods to process and convert velocity models to other
    physical properties (e.g., density, elastic moduli). It supports various
    empirical relationships and conversion methods.
    
    Attributes:
        grid_file (Optional[Path]): Path to the grid file
        output_dir (Path): Directory for output files
        velocity_grid (Optional[xr.Dataset]): The loaded velocity grid data
    """
    
    def __init__(self, 
                 grid_file: Optional[Union[str, xr.Dataset]] = None,
                 output_dir: Optional[str] = None):
        """初始化网格模型处理器。
        
        Args:
            grid_file (Optional[Union[str, xr.Dataset]]): 网格文件路径(.grd或.nc格式)或xarray数据集
            output_dir (Optional[str]): 输出文件目录，默认为'./output'
        """
        self.grid_file = Path(grid_file) if isinstance(grid_file, str) else None
        self.output_dir = Path(output_dir) if output_dir else Path('./output')
        self.output_dir.mkdir(exist_ok=True)
        self.velocity_grid = None
        
        if grid_file is not None:
            self.load_grid(grid_file)
            
    def load_grid(self, grid_file: Union[str, xr.Dataset]) -> None:
        """加载网格数据。
        
        Args:
            grid_file (Union[str, xr.Dataset]): 网格文件路径(.grd或.nc格式)或xarray数据集
            
        Raises:
            FileNotFoundError: 如果提供的是文件路径且文件不存在
        """
        if isinstance(grid_file, str):
            self.grid_file = Path(grid_file)
            if not self.grid_file.exists():
                raise FileNotFoundError(f"找不到网格文件: {self.grid_file}")
            self.velocity_grid = xr.open_dataset(grid_file)
        else:
            # 如果输入是xarray数据集，直接使用
            self.velocity_grid = grid_file
            self.grid_file = None  # 没有对应的文件路径
    
    def velocity_to_density(self, 
                          method: str = 'gardner',
                          output_file: Optional[str] = None,
                          model: Optional[Union[ZeltVelocityModel2d, EnhancedZeltModel]] = None,
                          seafloor_idx: int = 0,  # 海底面索引
                          basement_idx: int = 1,  # 沉积基底索引
                          moho_idx: int = -1,     # 莫霍面索引
                          **kwargs) -> xr.Dataset:
        """将速度转换为密度，使用不同的经验公式。
        
        根据不同的地质层位使用不同的速度-密度关系：
        - 海水层：固定密度 1.03 g/cm³
        - 沉积层：Gardner方程 ρ = 0.31 * v^0.25
        - 地壳层：Brocher方程（复杂多项式关系）
        - 地幔层：固定密度 3.33 g/cm³ 或 Nafe-Drake关系
        
        Args:
            method (str): 转换方法。可选项：
                       - 'gardner': Gardner方程 (ρ = a * v^b)
                       - 'castagna': Castagna方程 (ρ = 1.66 * v^0.261)
                       - 'brocher': Brocher方程 (复杂多项式关系)
                       - 'lindseth': Lindseth方程 (ρ = 0.31 * v + 1.7)
                       - 'nafe_drake': Nafe-Drake方程 (五次多项式关系)
                       - 'layered': 根据地质层位使用不同关系（需要提供model参数）
            output_file (Optional[str]): 如果提供，将结果保存到此文件
            model (Optional[Union[ZeltVelocityModel2d, EnhancedZeltModel]]): Zelt模型对象，
                  用于提供界面信息。如果method='layered'，则必须提供此参数
            seafloor_idx (int): 海底面界面的索引，默认为0
            basement_idx (int): 沉积基底界面的索引，默认为1
            moho_idx (int): 莫霍面界面的索引，默认为-1（最后一个界面）
            **kwargs: 特定方法的额外参数：
                   - Gardner方法: 'a' (默认: 0.31), 'b' (默认: 0.25)
                   
        Returns:
            xr.Dataset: 包含密度网格的数据集
            
        Raises:
            ValueError: 如果未加载速度网格或方法无效
        """
        if self.velocity_grid is None:
            raise ValueError("未加载速度网格")
            
        if method.lower() == 'layered' and model is None:
            raise ValueError("使用layered方法时必须提供model参数")
            
        velocity = self.velocity_grid.velocity.values  # 单位：km/s
        density = np.zeros_like(velocity)
        
        if method.lower() == 'layered':
            # 获取网格坐标
            x_coords = self.velocity_grid.x.values
            z_coords = self.velocity_grid.z.values
            X, Z = np.meshgrid(x_coords, z_coords)
            
            # 检查界面索引的有效性
            if len(model.depth_nodes) <= max(abs(seafloor_idx), abs(basement_idx), abs(moho_idx)):
                raise ValueError("指定的界面索引超出了模型界面的数量范围")
            
            # 获取界面几何
            seafloor_x, seafloor_z = model.get_layer_geometry(seafloor_idx)
            basement_x, basement_z = model.get_layer_geometry(basement_idx)
            moho_x, moho_z = model.get_layer_geometry(moho_idx)
            
            # 对每个网格点判断所属层位并应用相应的速度-密度关系
            for i in range(len(z_coords)):
                for j in range(len(x_coords)):
                    z = Z[i, j]
                    x = X[i, j]
                    v = velocity[i, j]
                    
                    # 插值得到当前x位置的界面深度
                    sf_z = np.interp(x, seafloor_x, seafloor_z)
                    bs_z = np.interp(x, basement_x, basement_z)
                    mh_z = np.interp(x, moho_x, moho_z)
                    
                    # 根据深度确定层位并计算密度
                    if z < sf_z:  # 海水层
                        density[i, j] = 1.03
                    elif z < bs_z:  # 沉积层
                        density[i, j] = 0.31 * (v * 1000) ** 0.25
                    elif z < mh_z:  # 地壳层
                        density[i, j] = (1.6612 * v - 0.4721 * v**2 + 0.0671 * v**3 - 
                                       0.0043 * v**4 + 0.000106 * v**5)
                    else:  # 地幔层
                        density[i, j] = 3.33  # 固定密度值
                        
        else:
            # 使用单一关系的原有逻辑
            if method.lower() == 'gardner':
                # Gardner et al. (1974)
                a = kwargs.get('a', 0.31)
                b = kwargs.get('b', 0.25)
                density = a * (velocity * 1000) ** b  # 转换为m/s
                
            elif method.lower() == 'castagna':
                # Castagna et al. (1993)
                density = 1.66 * (velocity * 1000) ** 0.261  # 转换为m/s
                
            elif method.lower() == 'brocher':
                # Brocher (2005)
                v = velocity  # 使用km/s
                density = (1.6612 * v - 0.4721 * v**2 + 0.0671 * v**3 - 
                          0.0043 * v**4 + 0.000106 * v**5)
                
            elif method.lower() == 'lindseth':
                # Lindseth (1979)
                density = 0.31 * velocity + 1.7
                
            elif method.lower() == 'nafe_drake':
                # Nafe-Drake (1963)
                v = velocity  # 使用km/s
                density = (1.6612 * v - 0.4721 * v**2 + 0.0671 * v**3 - 
                          0.0043 * v**4 + 0.000106 * v**5)
                
            else:
                raise ValueError(f"未知的转换方法: {method}")
            
        # 创建密度数据集
        density_grid = xr.Dataset(
            data_vars={
                'density': (('z', 'x'), density)
            },
            coords=self.velocity_grid.coords
        )
        
        # 如果指定了输出文件，保存结果
        if output_file:
            output_path = self.output_dir / output_file
            density_grid.to_netcdf(output_path)
            
        return density_grid
    
    def velocity_to_elastic_moduli(self,
                                 output_file: Optional[str] = None,
                                 vp_vs_ratio: float = 1.732,  # √3 for Poisson solid
                                 density_method: str = 'gardner',
                                 **kwargs) -> Tuple[xr.Dataset, xr.Dataset]:
        """Convert velocity to elastic moduli (λ, μ, K).
        
        Args:
            output_file (Optional[str]): If provided, save results to this file (without extension)
            vp_vs_ratio (float): Vp/Vs ratio to use for conversion
            density_method (str): Method to use for density calculation
            **kwargs: Additional parameters passed to density calculation
        
        Returns:
            Tuple[xr.Dataset, xr.Dataset]: Lamé parameters (λ, μ) and bulk modulus (K)
            
        Raises:
            ValueError: If no velocity grid is loaded
        """
        if self.velocity_grid is None:
            raise ValueError("No velocity grid loaded")
            
        # Get density
        density_grid = self.velocity_to_density(method=density_method, **kwargs)
        density = density_grid.density.values
        
        # Get velocities
        vp = self.velocity_grid.velocity.values * 1000  # Convert to m/s
        vs = vp / vp_vs_ratio
        
        # Calculate elastic moduli
        mu = density * vs * vs  # Shear modulus
        lambda_ = density * (vp * vp - 2 * vs * vs)  # Lamé's first parameter
        k = lambda_ + 2 * mu / 3  # Bulk modulus
        
        # Create datasets
        lame_grid = xr.Dataset(
            data_vars={
                'lambda': (('z', 'x'), lambda_),
                'mu': (('z', 'x'), mu)
            },
            coords=self.velocity_grid.coords
        )
        
        bulk_grid = xr.Dataset(
            data_vars={
                'k': (('z', 'x'), k)
            },
            coords=self.velocity_grid.coords
        )
        
        # Save if output file specified
        if output_file:
            lame_path = self.output_dir / f"{output_file}_lame.grd"
            bulk_path = self.output_dir / f"{output_file}_bulk.grd"
            lame_grid.to_netcdf(lame_path)
            bulk_grid.to_netcdf(bulk_path)
            
        return lame_grid, bulk_grid
    
    def velocity_to_twt(self, output_file: Optional[str] = None) -> xr.Dataset:
        """将速度模型转换为双层走时（Two-Way Time）。
        
        对于每个 (x,z) 位置，计算从地表到该点的双层走时：
        1. 计算从地表到该点的每个网格单元的单程走时 (dz/v)
        2. 将这些走时相加并乘以2，得到双层走时
        
        参数:
            output_file (Optional[str]): 如果提供，将结果保存到此文件
            
        返回:
            xr.Dataset: 包含双层走时网格的数据集
            
        异常:
            ValueError: 如果未加载速度网格
        """
        if self.velocity_grid is None:
            raise ValueError("未加载速度网格")
            
        # 获取速度数据和坐标
        velocity = self.velocity_grid.velocity.values  # [z, x]
        x_coords = self.velocity_grid.x.values
        z_coords = self.velocity_grid.z.values
        
        # 计算网格间距
        dz = z_coords[1] - z_coords[0]  # 假设均匀网格
        
        # 初始化双层走时数组
        twt = np.zeros_like(velocity)
        
        # 对每个x位置计算双层走时
        for ix in range(velocity.shape[1]):  # 遍历x
            # 对每个深度计算累积走时
            for iz in range(velocity.shape[0]):  # 遍历z
                # 计算从地表到当前深度的所有网格单元的走时总和
                total_time = 0
                for k in range(iz + 1):
                    # 计算当前网格单元的单程走时 (dz/v)
                    if velocity[k, ix] > 0:  # 避免除以零
                        total_time += dz / velocity[k, ix]
                    
                # 存储双层走时（乘以2表示往返）
                twt[iz, ix] = 2.0 * total_time
        
        # 创建双层走时数据集
        twt_grid = xr.Dataset(
            data_vars={
                'twt': (('z', 'x'), twt)
            },
            coords=self.velocity_grid.coords
        )
        
        # 如果指定了输出文件，保存结果
        if output_file:
            output_path = self.output_dir / output_file
            twt_grid.to_netcdf(output_path)
            
        return twt_grid
    
    def depth_to_twt(self, output_file: Optional[str] = None) -> xr.Dataset:
        """将深度域的速度模型转换到双层走时域。
        
        对于每个 x 位置：
        1. 计算每个深度点的双层走时（使用 velocity_to_twt）
        2. 将原始速度值映射到对应的双层走时位置
        
        参数:
            output_file (Optional[str]): 如果提供，将结果保存到此文件
            
        返回:
            xr.Dataset: 包含双层走时域的速度模型
        """
        if self.velocity_grid is None:
            raise ValueError("未加载速度网格")
            
        # 获取速度数据和坐标
        velocity = self.velocity_grid.velocity.values  # [z, x]
        x_coords = self.velocity_grid.x.values
        z_coords = self.velocity_grid.z.values
        
        # 计算网格间距
        dz = z_coords[1] - z_coords[0]  # 假设均匀网格
        
        # 首先计算每个点的双层走时
        twt = np.zeros_like(velocity)
        for ix in range(velocity.shape[1]):  # 遍历x
            for iz in range(velocity.shape[0]):  # 遍历z
                total_time = 0
                for k in range(iz + 1):
                    if velocity[k, ix] > 0:  # 避免除以零
                        total_time += dz / velocity[k, ix]
                twt[iz, ix] = 2.0 * total_time
                
        # 创建新的时间坐标
        t_min = 0
        t_max = np.nanmax(twt)
        dt = (t_max - t_min) / (len(z_coords) - 1)  # 使用相同数量的采样点
        t_coords = np.arange(t_min, t_max + dt, dt)
        
        # 初始化双层走时域的速度数组
        v_twt = np.zeros((len(t_coords), len(x_coords)))
        
        # 对每个x位置进行插值
        for ix in range(len(x_coords)):
            # 去除无效值
            valid = ~np.isnan(twt[:, ix])
            if np.any(valid):
                # 使用线性插值将速度值映射到新的时间坐标
                v_twt[:, ix] = np.interp(
                    t_coords,
                    twt[valid, ix],
                    velocity[valid, ix],
                    left=np.nan,
                    right=np.nan
                )
                
        # 创建双层走时域的数据集
        twt_grid = xr.Dataset(
            data_vars={
                'velocity': (('t', 'x'), v_twt)
            },
            coords={
                'x': x_coords,
                't': t_coords
            }
        )
        
        # 如果指定了输出文件，保存结果
        if output_file:
            output_path = self.output_dir / output_file
            twt_grid.to_netcdf(output_path)
            
        return twt_grid
    
    def close(self) -> None:
        """Close the loaded grid dataset."""
        if self.velocity_grid is not None:
            self.velocity_grid.close()
            self.velocity_grid = None

# Export classes in __init__.py
__all__ = ['ZeltModelVisualizer', 'GridModelVisualizer', 'GridModelProcessor']

# Usage examples
if __name__ == "__main__":
    # Create model instance
    
    # GridModelVisualizer example
    grid_visualizer = GridModelVisualizer(
        grid_file="./model_output/velocity.grd",
        output_dir="grid_output"
    )
    
    # Plot grid with various options
    fig = grid_visualizer.plot_grid(
        output_fig="grid_output/velocity_plot.png",
        figsize=(10, -4),  # Negative height for reversed y-axis
        velocity_cmap="scale1.cpt",  # Use custom CPT file
        cmap_reverse=True,
        title="Velocity Model from Grid",
        plot_contours=True,
        contour_interval=0.1,
        contour_annotation_interval=0.5,
        use_shading=True,
        azimuth=135
    )
    
    # Print velocity range
    v_min, v_max = grid_visualizer.get_velocity_range()
    print(f"\nVelocity range: {v_min:.2f} to {v_max:.2f} km/s")

    # GridModelProcessor example
    #processor = GridModelProcessor(
    #    grid_file="./model_output/velocity.grd",
    #    output_dir="processed_output"
    #)
    
    # Convert velocity to density using Gardner's equation
    #density_grid = processor.velocity_to_density(
    #    method='gardner',
    #    output_file='density.nc',
    #    a=0.31,  # Gardner's coefficient
    #    b=0.25   # Gardner's exponent
    #)
    
    
    # Clean up
    #processor.close() 