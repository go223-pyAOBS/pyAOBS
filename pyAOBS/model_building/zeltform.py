"""
Zelt Format Velocity Model Module

This module provides classes for handling Zelt format velocity models.
It includes implementations for basic and enhanced velocity models,
supporting both reading from and writing to Zelt format files.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
import xarray as xr
from pathlib import Path

from pyAOBS.model_building.models import Point2d, ZNode2d, TrapezoidCell2d
from pyAOBS.model_building.read import read_vin_model

class ZeltVelocityModel2d:
    """Base class for 2D velocity models in Zelt format.
    
    This class provides basic functionality for handling 2D velocity models
    in Zelt format, including velocity calculations and model manipulation.
    
    Attributes:
        depth_nodes (List[ZNode2d]): Depth nodes for each layer
        vupper_nodes (List[ZNode2d]): Upper velocity nodes for each layer
        vlower_nodes (List[ZNode2d]): Lower velocity nodes for each layer
        cells (List[TrapezoidCell2d]): Trapezoid cells for velocity interpolation
        bottom_boundary (ZNode2d): Bottom boundary nodes of the model
    """
    
    def __init__(self, model_file: Optional[str] = None):
        """Initialize the velocity model.
        
        Args:
            model_file (Optional[str]): Path to the Zelt format model file
        """
        self.depth_nodes: List[ZNode2d] = []
        self.vupper_nodes: List[ZNode2d] = []
        self.vlower_nodes: List[ZNode2d] = []
        self.cells: List[TrapezoidCell2d] = []
        self.bottom_boundary: Optional[ZNode2d] = None
        
        if model_file:
            self.read_model(model_file)
            
    def read_model(self, model_file: str) -> None:
        """从文件中读取速度模型。

        Args:
            model_file (str): 速度模型文件路径。
        """
        # 读取速度模型
        model_data = read_vin_model(model_file)
        
        # 转换数据到内部格式
        n_layers = len(model_data['layer_boundary_x'])
        
        # 初始化每一层的节点
        for i in range(n_layers):
            # 创建深度节点
            depth_node = ZNode2d()
            x_coords = model_data['layer_boundary_x'][i]
            z_coords = model_data['layer_boundary_z'][i]
            flags = model_data['layer_boundary_flags'][i]
            for x, z, flag in zip(x_coords, z_coords, flags):
                if not np.isnan(x) and not np.isnan(z):
                    depth_node.add_point(x, z, flag)
            self.depth_nodes.append(depth_node)
            
            # 创建上部速度节点
            vupper_node = ZNode2d()
            x_coords = model_data['upper_x_velocities'][i]
            v_coords = model_data['upper_velocities'][i]
            flags = model_data['upper_velocity_flags'][i]
            for x, v, flag in zip(x_coords, v_coords, flags):
                if not np.isnan(x) and not np.isnan(v):
                    vupper_node.add_point(x, v, flag)
            self.vupper_nodes.append(vupper_node)
            
            # 创建下部速度节点
            vlower_node = ZNode2d()
            x_coords = model_data['lower_x_velocities'][i]
            v_coords = model_data['lower_velocities'][i]
            flags = model_data['lower_velocity_flags'][i]
            for x, v, flag in zip(x_coords, v_coords, flags):
                if not np.isnan(x) and not np.isnan(v):
                    vlower_node.add_point(x, v, flag)
            self.vlower_nodes.append(vlower_node)
        
        # 添加底部边界（如果存在）
        if model_data.get('bottom_boundary_x'):
            self.bottom_boundary = ZNode2d()
            x_coords = model_data['bottom_boundary_x']
            z_coords = model_data['bottom_boundary_z']
            flags = model_data['bottom_boundary_flags']
            for x, z, flag in zip(x_coords, z_coords, flags):
                if not np.isnan(x) and not np.isnan(z):
                    self.bottom_boundary.add_point(x, z, flag)
            self.depth_nodes.append(self.bottom_boundary)
        
        # 更新模型边界
        x_coords = []
        z_coords = []
        for node in self.depth_nodes:
            x_coords.extend(node.get_x_coords())
            z_coords.extend(node.get_values())
        
        self.xmin = min(x_coords)
        self.xmax = max(x_coords)
        self.zmin = min(z_coords)
        self.zmax = max(z_coords)
        
        # 创建速度插值单元
        self._create_cells()
    
    def _create_cells(self) -> None:
        """Create trapezoid cells for velocity interpolation."""
        for ilayer in range(len(self.depth_nodes) - 1):
            # Get unique x-coordinates for this layer
            x_coords = set()
            
            # Add x-coordinates from depth nodes
            x_coords.update(self.depth_nodes[ilayer].get_x_coords())
            x_coords.update(self.depth_nodes[ilayer + 1].get_x_coords())
            
            # Add x-coordinates from velocity nodes
            x_coords.update(self.vupper_nodes[ilayer].get_x_coords())
            x_coords.update(self.vlower_nodes[ilayer].get_x_coords())
            
            # Sort x-coordinates
            x_coords = sorted(list(x_coords))
            
            # Create cells between adjacent x-coordinates
            for i in range(len(x_coords) - 1):
                x1, x2 = x_coords[i], x_coords[i + 1]
                
                # Get z-coordinates
                z1 = self.depth_nodes[ilayer].get_value_at(x1)
                z2 = self.depth_nodes[ilayer].get_value_at(x2)
                z3 = self.depth_nodes[ilayer + 1].get_value_at(x1)
                z4 = self.depth_nodes[ilayer + 1].get_value_at(x2)
                
                # Get velocities
                v1 = self.vupper_nodes[ilayer].get_value_at(x1)
                v2 = self.vupper_nodes[ilayer].get_value_at(x2)
                v3 = self.vlower_nodes[ilayer].get_value_at(x1)
                v4 = self.vlower_nodes[ilayer].get_value_at(x2)
                
                # Create cell
                cell = TrapezoidCell2d(x1, x2, z1, z2, z3, z4, v1, v2, v3, v4)
                self.cells.append(cell)
    
    def at(self, x: float, z: float) -> float:
        """获取指定位置的速度值。

        Args:
            x (float): x 坐标。
            z (float): z 坐标。

        Returns:
            float: 速度值。

        Raises:
            ValueError: 如果点超出模型边界。
        """
        # 检查模型边界
        if x < self.xmin or x > self.xmax or z < self.zmin or z > self.zmax:
            raise ValueError(f"Point ({x}, {z}) is outside model boundaries")
        
        point = Point2d(x, z)
        for cell in self.cells:
            if cell.is_in(point):
                return cell.at(point)
        raise ValueError(f"Point ({x}, {z}) is outside model boundaries")
    
    def get_layer_geometry(self, layer_idx: int) -> Tuple[List[float], List[float]]:
        """Get the geometry of a layer.
        
        Args:
            layer_idx (int): Layer index
            
        Returns:
            Tuple[List[float], List[float]]: X-coordinates and depths of layer nodes
        """
        if layer_idx < 0 or layer_idx >= len(self.depth_nodes):
            raise ValueError(f"Invalid layer index: {layer_idx}")
        
        return (self.depth_nodes[layer_idx].get_x_coords(), 
                self.depth_nodes[layer_idx].get_values())
    
    def get_model_bounds(self) -> Tuple[float, float, float, float]:
        """Get model boundaries.
        
        Returns:
            Tuple[float, float, float, float]: (x_min, x_max, z_min, z_max)
        """
        x_min = min(node.get_x_coords()[0] for node in self.depth_nodes)
        x_max = max(node.get_x_coords()[-1] for node in self.depth_nodes)
        z_min = min(node.get_values()[0] for node in self.depth_nodes)
        z_max = max(node.get_values()[-1] for node in self.depth_nodes)
        return x_min, x_max, z_min, z_max
    
    def to_xarray(self, dx: float = 10.0, dz: float = 3.0) -> xr.Dataset:
        """将模型转换为 xarray 数据集。

        Args:
            dx (float): x方向的采样间隔（km），默认1.0 km
            dz (float): z方向的采样间隔（km），默认0.25 km

        Returns:
            xr.Dataset: 包含速度场的数据集。
        """
        # 获取模型边界
        x_min, x_max, z_min, z_max = self.get_model_bounds()
        
        # 计算网格点数（向上取整以确保覆盖整个区域）
        nx = int(np.ceil((x_max - x_min) / dx)) + 1
        nz = int(np.ceil((z_max - z_min) / dz)) + 1
        
        # 创建网格
        x = np.linspace(x_min, x_max, nx)
        z = np.linspace(z_min, z_max, nz)
        X, Z = np.meshgrid(x, z)
        
        # 计算每个网格点的速度
        V = np.zeros_like(X)
        for i in range(Z.shape[0]):
            for j in range(X.shape[1]):
                try:
                    V[i,j] = self.at(X[i,j], Z[i,j])
                except ValueError:
                    V[i,j] = np.nan
        
        # 创建数据集
        ds = xr.Dataset(
            data_vars={
                'velocity': (['z', 'x'], V)
            },
            coords={
                'x': x,
                'z': z
            }
        )
        
        return ds

class EnhancedZeltModel(ZeltVelocityModel2d):
    """增强型 Zelt 速度模型类，提供额外的处理功能。"""

    def __init__(self, base_model: ZeltVelocityModel2d):
        """
        初始化增强型 Zelt 速度模型。

        参数:
            base_model: ZeltVelocityModel2d 对象，基础速度模型
        """
        # 保存原始模型的副本
        self.base_model = ZeltVelocityModel2d()
        self.base_model.depth_nodes = [node for node in base_model.depth_nodes]
        self.base_model.vupper_nodes = [node for node in base_model.vupper_nodes]
        self.base_model.vlower_nodes = [node for node in base_model.vlower_nodes]
        self.base_model.bottom_boundary = base_model.bottom_boundary
        self.base_model.xmin = base_model.xmin
        self.base_model.xmax = base_model.xmax
        self.base_model.zmin = base_model.zmin
        self.base_model.zmax = base_model.zmax
        self.base_model.cells = [cell for cell in base_model.cells]
        
        # 初始化当前模型
        self.depth_nodes = [node for node in base_model.depth_nodes]
        self.vupper_nodes = [node for node in base_model.vupper_nodes]
        self.vlower_nodes = [node for node in base_model.vlower_nodes]
        self.bottom_boundary = base_model.bottom_boundary
        self.xmin = base_model.xmin
        self.xmax = base_model.xmax
        self.zmin = base_model.zmin
        self.zmax = base_model.zmax
        self.cells = [cell for cell in base_model.cells]

    def compare_with_base(self) -> dict:
        """
        比较当前模型与基础模型的差异。

        Returns:
            dict: 包含比较结果的字典，包括：
                - 'velocity_changes': 每层速度变化的列表，每个元素是一个元组 
                  ((base_top, base_bottom), (current_top, current_bottom))
                - 'depth_changes': 每层深度变化的列表，每个元素是一个元组
                  (base_depth, current_depth)
                - 'boundary_changes': 模型边界的变化
                  {'xmin': (base, current), 'xmax': (base, current),
                   'zmin': (base, current), 'zmax': (base, current)}
        """
        velocity_changes = []
        depth_changes = []
        
        # 比较每层的速度变化
        for i in range(len(self.vupper_nodes)):
            # 基础模型的速度
            base_upper = self.base_model.vupper_nodes[i].get_values()
            base_lower = self.base_model.vlower_nodes[i].get_values()
            base_top_avg = sum(base_upper) / len(base_upper)
            base_bottom_avg = sum(base_lower) / len(base_lower)
            
            # 当前模型的速度
            current_upper = self.vupper_nodes[i].get_values()
            current_lower = self.vlower_nodes[i].get_values()
            current_top_avg = sum(current_upper) / len(current_upper)
            current_bottom_avg = sum(current_lower) / len(current_lower)
            
            velocity_changes.append(
                ((base_top_avg, base_bottom_avg),
                 (current_top_avg, current_bottom_avg))
            )
        
        # 比较每层的深度变化
        for i in range(len(self.depth_nodes)):
            base_depths = self.base_model.depth_nodes[i].get_values()
            current_depths = self.depth_nodes[i].get_values()
            
            base_avg_depth = sum(base_depths) / len(base_depths)
            current_avg_depth = sum(current_depths) / len(current_depths)
            
            depth_changes.append((base_avg_depth, current_avg_depth))
        
        # 比较边界变化
        boundary_changes = {
            'xmin': (self.base_model.xmin, self.xmin),
            'xmax': (self.base_model.xmax, self.xmax),
            'zmin': (self.base_model.zmin, self.zmin),
            'zmax': (self.base_model.zmax, self.zmax)
        }
        
        return {
            'velocity_changes': velocity_changes,
            'depth_changes': depth_changes,
            'boundary_changes': boundary_changes
        }

    def _update_model_attributes(self):
        """
        更新模型的属性，包括边界值和速度插值单元。
        在对模型节点进行修改后调用此方法。
        """
        # 更新模型边界
        x_coords = []
        z_coords = []
        for node in self.depth_nodes:
            x_coords.extend(node.get_x_coords())
            z_coords.extend(node.get_values())
        
        self.xmin = min(x_coords)
        self.xmax = max(x_coords)
        self.zmin = min(z_coords)
        self.zmax = max(z_coords)
        
        # 重新创建速度插值单元
        self.cells = []
        self._create_cells()

    def process_velocity_model(self, process_type: str):
        """
        处理速度模型并更新模型属性。

        参数:
            process_type: 处理类型，可选值包括：
                - 'average_velocity': 计算并更新为平均速度
                - 'interval_velocity': 计算并更新为层间速度
                - 'two_way_time': 计算并更新为双程走时

        返回:
            根据处理类型返回相应的结果，同时更新模型属性
        """
        # 首先检查是否是支持的处理类型
        supported_types = {'average_velocity', 'interval_velocity', 'two_way_time'}
        if process_type not in supported_types:
            raise ValueError(f'不支持的处理类型: {process_type}')
        
        # 然后检查是否是已实现的功能
        if process_type == 'average_velocity':
            result = self._compute_average_velocities()
            self._update_model_attributes()
            return result
        elif process_type == 'interval_velocity':
            result = self._compute_interval_velocities()
            self._update_model_attributes()
            return result
        elif process_type == 'two_way_time':
            result = self._compute_two_way_time()
            self._update_model_attributes()
            return result

    def _compute_average_velocities(self) -> List[Tuple[float, float]]:
        """
        计算每一层的平均速度并更新模型。

        返回:
            每一层的平均速度列表，每个元素是一个元组 (top_avg, bottom_avg)，
            分别表示该层顶部和底部的平均速度。
        """
        avg_velocities = []
        for i in range(len(self.vupper_nodes)):
            # 获取并计算上部速度节点的平均值
            upper_vals = self.vupper_nodes[i].get_values()
            top_avg = sum(upper_vals) / len(upper_vals)
            
            # 获取并计算下部速度节点的平均值
            lower_vals = self.vlower_nodes[i].get_values()
            bottom_avg = sum(lower_vals) / len(lower_vals)
            
            # 创建新的节点替换原有节点
            new_vupper = ZNode2d()
            x_coords = self.vupper_nodes[i].get_x_coords()
            for x in x_coords:
                new_vupper.add_point(x, top_avg, 0)
            self.vupper_nodes[i] = new_vupper
            
            new_vlower = ZNode2d()
            x_coords = self.vlower_nodes[i].get_x_coords()
            for x in x_coords:
                new_vlower.add_point(x, bottom_avg, 0)
            self.vlower_nodes[i] = new_vlower
            
            avg_velocities.append((top_avg, bottom_avg))
        
        return avg_velocities
    
    def _compute_interval_velocities(self) -> dict:
        """
        计算层间速度并更新模型。
        
        计算相邻两层之间的速度差，并更新模型的速度节点。
        上层的下部速度和下层的上部速度将被更新为它们的平均值。

        Returns:
            dict: 包含层间速度信息的字典，包括：
                - 'interval_velocities': 每层之间的速度值
                - 'boundaries': 速度界面的深度
        """
        interval_velocities = []
        boundaries = []
        
        # 遍历相邻的层
        for i in range(len(self.vupper_nodes) - 1):
            # 获取当前层的下部速度和下一层的上部速度
            current_lower = self.vlower_nodes[i].get_values()
            next_upper = self.vupper_nodes[i + 1].get_values()
            
            # 计算层间速度（平均值）
            interval_velocity = (sum(current_lower) + sum(next_upper)) / (len(current_lower) + len(next_upper))
            
            # 创建新的节点替换原有节点
            new_vlower = ZNode2d()
            x_coords = self.vlower_nodes[i].get_x_coords()
            for x in x_coords:
                new_vlower.add_point(x, interval_velocity, 0)
            self.vlower_nodes[i] = new_vlower
            
            new_vupper = ZNode2d()
            x_coords = self.vupper_nodes[i + 1].get_x_coords()
            for x in x_coords:
                new_vupper.add_point(x, interval_velocity, 0)
            self.vupper_nodes[i + 1] = new_vupper
            
            # 获取层界面深度
            boundary = self.depth_nodes[i + 1].get_values()
            
            interval_velocities.append(interval_velocity)
            boundaries.append(sum(boundary) / len(boundary))
        
        return {
            'interval_velocities': interval_velocities,
            'boundaries': boundaries
        }
    
    def _compute_two_way_time(self) -> dict:
        """
        计算双程走时并更新模型。
        
        将深度域的模型转换为时间域，同时保留节点的速度值：
        1. 对于每个节点，计算从地表到该点的双程走时
        2. 更新深度节点的值为双程走时
        3. 保持速度节点的值不变

        Returns:
            dict: 包含双程走时信息的字典，包括：
                - 'twt': 每层的双程走时
                - 'velocities': 对应的速度值
        """
        twt_values = []
        velocities = []
        
        # 遍历所有层
        for i in range(len(self.depth_nodes) - 1):
            # 获取当前层的深度节点
            x_coords = self.depth_nodes[i].get_x_coords()
            depths = self.depth_nodes[i].get_values()
            
            # 计算每个x位置的双程走时
            layer_twt = []
            layer_velocities_upper = []
            layer_velocities_lower = []
            
            for j, x in enumerate(x_coords):
                # 计算从地表到当前点的累积走时
                total_time = 0
                current_depth = depths[j]
                
                # 遍历上层计算走时
                for k in range(i):
                    layer_depth = self.depth_nodes[k].get_value_at(x)
                    next_depth = self.depth_nodes[k + 1].get_value_at(x)
                    layer_thickness = next_depth - layer_depth
                    
                    # 计算该层中点的深度
                    mid_depth = (layer_depth + next_depth) / 2
                    try:
                        # 尝试使用 at() 方法获取速度
                        layer_v = self.at(x, mid_depth)
                    except ValueError:
                        # 如果点在边界外，使用该层的速度节点插值
                        v_upper = self.vupper_nodes[k].get_value_at(x)
                        v_lower = self.vlower_nodes[k].get_value_at(x)
                        layer_v = (v_upper + v_lower) / 2
                    
                    total_time += layer_thickness / layer_v
                
                # 计算当前层的走时贡献
                current_layer_depth = current_depth - (self.depth_nodes[i].get_value_at(x) if i > 0 else 0)
                try:
                    # 尝试使用 at() 方法获取当前层的速度
                    mid_depth = (current_depth + (self.depth_nodes[i].get_value_at(x) if i > 0 else 0)) / 2
                    current_layer_v = self.at(x, mid_depth)
                except ValueError:
                    # 如果点在边界外，使用当前层的速度节点插值
                    v_upper = self.vupper_nodes[i].get_value_at(x)
                    v_lower = self.vlower_nodes[i].get_value_at(x)
                    current_layer_v = (v_upper + v_lower) / 2
                
                total_time += current_layer_depth / current_layer_v
                
                # 双程走时
                layer_twt.append(2.0 * total_time)
                
                # 记录速度值
                layer_velocities_upper.append(self.vupper_nodes[i].get_value_at(x))
                layer_velocities_lower.append(self.vlower_nodes[i].get_value_at(x))
            
            # 创建新的深度节点（使用双程走时替换深度值）
            new_depth = ZNode2d()
            for j, x in enumerate(x_coords):
                new_depth.add_point(x, layer_twt[j], 0)
            self.depth_nodes[i] = new_depth
            
            # 记录该层的走时和速度信息
            twt_values.append(layer_twt)
            velocities.append((layer_velocities_upper, layer_velocities_lower))
        
        return {
            'twt': twt_values,
            'velocities': velocities
        } 