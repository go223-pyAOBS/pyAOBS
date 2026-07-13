from enum import Enum
import numpy as np
from typing import List, Optional, Tuple, Union
from ...model_building.zeltform import ZeltVelocityModel2d
from ...model_building.models import TrapezoidCell2d

class PhaseType(Enum):
    """射线震相类型"""
    DIRECT = 'direct'         # 直达波
    REFLECTION = 'reflection' # 反射波
    REFRACTION = 'refraction' # 折射波
    HEAD = 'head'            # 头波

class VelocityModel:
    """速度模型"""
    def __init__(self, model: ZeltVelocityModel2d):
        """初始化速度模型
        
        Args:
            model: ZeltVelocityModel2d对象
        """
        self._zelt_model = model  # 使用下划线表示这是内部实现细节
        # 使用ZeltVelocityModel2d的get_model_bounds函数获取边界
        x_min, x_max, z_min, z_max = self._zelt_model.get_model_bounds()
        self.x_range = (x_min, x_max)
        self.z_range = (z_min, z_max)
        
    @staticmethod
    def from_zelt_file(file_path: str) -> 'VelocityModel':
        """从Zelt格式文件创建速度模型
        
        Args:
            file_path: Zelt格式的速度模型文件路径
            
        Returns:
            VelocityModel: 速度模型对象
        """
        zelt_model = ZeltVelocityModel2d(file_path)
        return VelocityModel(zelt_model)
        
    @staticmethod
    def from_tomo_file(file_path: str) -> 'VelocityModel':
        """从Tomo格式文件创建速度模型
        
        Args:
            file_path: Tomo格式的速度模型文件路径
            
        Returns:
            VelocityModel: 速度模型对象
            
        Raises:
            NotImplementedError: 该方法尚未实现
        """
        raise NotImplementedError("从Tomo格式文件创建模型的功能尚未实现")
        
    def get_velocity(self, x: float, z: float) -> float:
        """获取指定位置的速度"""
        # 处理边界情况
        eps = 1e-6
        if (x < self.x_range[0] - eps or x > self.x_range[1] + eps or
            z < self.z_range[0] - eps or z > self.z_range[1] + eps):
            raise ValueError(f"Point ({x}, {z}) is outside model bounds")
            
        return self._zelt_model.at(x, z)
        
    def get_layer_index(self, z: float) -> int:
        """获取指定深度所在层的索引"""
        eps = 1e-6
        if z < self.z_range[0] - eps or z > self.z_range[1] + eps:
            raise ValueError(f"Depth {z} is outside model bounds")
            
        # 找到深度所在的层索引
        for i in range(len(self._zelt_model.depth_nodes) - 1):
            if self._zelt_model.depth_nodes[i].get_values()[0] <= z <= self._zelt_model.depth_nodes[i+1].get_values()[0]:
                return i
                
        return len(self._zelt_model.depth_nodes) - 2  # 最后一层
        
    def get_interface_depth(self, interface: int) -> float:
        """获取指定界面的深度"""
        if interface < 0 or interface >= len(self._zelt_model.depth_nodes):
            raise ValueError(f"Invalid interface number {interface}")
        return self._zelt_model.depth_nodes[interface].get_values()[0]
        
    def get_model_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """获取模型的范围
        
        Returns:
            ((x_min, x_max), (z_min, z_max))
        """
        return self.x_range, self.z_range
        
    def get_cells(self) -> List[TrapezoidCell2d]:
        """获取所有速度块
        
        Returns:
            List[TrapezoidCell2d]: 速度块列表
        """
        return self._zelt_model.cells
        
    def get_depth_nodes_count(self) -> int:
        """获取深度节点数量
        
        Returns:
            int: 深度节点数量
        """
        return len(self._zelt_model.depth_nodes)
        
    def plot_model(self, ax=None, nx: int = 50, nz: int = 50):
        """绘制速度模型"""
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        # 创建网格
        x = np.linspace(self.x_range[0], self.x_range[1], nx)
        z = np.linspace(self.z_range[0], self.z_range[1], nz)
        X, Z = np.meshgrid(x, z)
        
        # 计算速度
        V = np.array([[self.get_velocity(xi, zi) for xi in x] for zi in z])
        
        # 绘制速度等值线
        #cs = ax.contour(X, Z, V, levels=10, cmap='jet')
        #plt.clabel(cs)
        
        # 绘制界面
        for depth_node in self._zelt_model.depth_nodes:
            x_coords = depth_node.get_x_coords()
            z_coords = depth_node.get_values()
            ax.plot(x_coords, z_coords, 'k-', linewidth=0.5)
            
        # 设置坐标轴
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Depth (km)')
        ax.invert_yaxis()
        
        return ax
        
    def __str__(self) -> str:
        return f"VelocityModel with {len(self._zelt_model.depth_nodes)-1} layers" 