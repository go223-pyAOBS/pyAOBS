import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Optional, List, Union
from .models import VelocityModel, PhaseType
import time
from ...model_building.models import Point2d, ZNode2d, TrapezoidCell2d
import matplotlib.pyplot as plt

class RayTracerConfig:
    """射线追踪配置参数"""
    def __init__(self):
        # 数值计算参数
        self.step = 1.00        # 默认步长(km)
        self.min_step = 0.01    # 最小步长(km)
        self.max_step = 5.0     # 最大步长(km)
        self.tolerance = 0.2 # 两点射线追踪的容差，默认0.2
        self.hdenom = 64.0      # 步长分母
        self.max_steps = 10000  # 最大步数，增加以适应更长的射线路径
        
        # 角度参数
        self.pi2 = np.pi/2      # π/2
        self.pi4 = np.pi/4      # π/4
        self.pi34 = 3*np.pi/4   # 3π/4
        
        # 界面参数
        self.interface_eps = 0.001  # 界面判定精度
        
        # 射线追踪参数
        self.angle_step = 1.0     # 角度搜索步长(度)
        self.time_interp_points = 1000  # 走时计算插值点数
        
        # 用于路径优化的参数
        self.smooth_window = 5    # 平滑窗口大小
        self.interp_points = 200  # 插值点数
        self.optimize_iters = 3   # 优化迭代次数
        
        # 界面参数
        self.interface_tolerance = 0.001  # 界面判定容差
        
        # 自适应步长参数
        self.gradient_factor = 0.1  # 速度梯度对步长的影响因子
        self.interface_factor = 0.2 # 接近界面时的步长缩减因子
        
    def get_step(self):
        return self.step

class RayState:
    """射线状态"""
    def __init__(self):
        self.layer = 1          # 当前层号
        self.block = 1          # 当前块号
        self.direction = 1      # 射线传播方向(1:向右,-1:向左)
        self.wave_type = 1      # 波类型(1:P波,-1:S波)
        self.x = 0.0            # 当前x坐标
        self.z = 0.0            # 当前z坐标
        self.angle = 0.0        # 当前角度
        self.velocity = 0.0     # 当前速度
        self.is_initial_point = False  # 初始点标记
        
    def update(self, x: float, z: float, angle: float, velocity: float):
        """更新射线状态"""
        self.x = x
        self.z = z
        self.angle = angle
        self.velocity = velocity
        
class RayPath:
    """射线路径"""
    def __init__(self):
        self.x_coords: List[float] = []     # x坐标序列
        self.z_coords: List[float] = []     # z坐标序列
        self.angles: List[float] = []       # 角度序列
        self.velocities: List[float] = []   # 速度序列
        self.layers: List[int] = []         # 层号序列
        self.blocks: List[int] = []         # 块号序列
        
    def add_point(self, x: float, z: float, angle: float, velocity: float, 
                 layer: int, block: int):
        """添加一个路径点"""
        self.x_coords.append(x)
        self.z_coords.append(z)
        self.angles.append(angle)
        self.velocities.append(velocity)
        self.layers.append(layer)
        self.blocks.append(block)
        
    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取坐标数组"""
        return np.array(self.x_coords), np.array(self.z_coords)
        
    def calculate_travel_time(self) -> float:
        """计算总走时"""
        time = 0.0
        for i in range(len(self.x_coords)-1):
            dx = self.x_coords[i+1] - self.x_coords[i]
            dz = self.z_coords[i+1] - self.z_coords[i]
            ds = np.sqrt(dx*dx + dz*dz)
            v_avg = (self.velocities[i] + self.velocities[i+1])/2
            time += ds/v_avg
        return time

class RayTracer:
    """射线追踪器"""
    def __init__(self, velocity_model: VelocityModel, config: Optional[RayTracerConfig] = None):
        self.model = velocity_model
        self.config = config or RayTracerConfig()
        self.state = RayState()
        self.path = RayPath()
               
    def _initialize_ray(self, x: float, z: float, angle: float) -> bool:
        """初始化射线状态"""
        # 1. 清空路径
        self.path = RayPath()
        
        # 2. 确定起始点所在层
        try:
            # 添加调试信息
            print(f"\n初始化射线:")
            print(f"位置: ({x:.3f}, {z:.3f}), 角度: {angle*180/np.pi:.2f}°")
            
            layer = self.model.get_layer_index(z)
            print(f"所在层: {layer}")
            
            # 获取并打印速度信息
            v = self.model.get_velocity(x, z)
            vx = self._get_velocity_gradient_x(x, z)
            vz = self._get_velocity_gradient_z(x, z)
            print(f"速度: {v:.3f} km/s")
            print(f"速度梯度: ({vx:.3f}, {vz:.3f})")
            
            # 3. 确定所在块
            block = None
            for i, cell in enumerate(self.model.get_cells()):
                if cell.layer == layer:
                    point = Point2d(x, z)
                    if cell.is_in_with_tolerance(point, self.config.interface_eps):
                        block = i + 1
                        break
                        
            if block is None:
                return False
                
            # 4. 初始化状态
            self.state.layer = layer
            self.state.block = block
            self.state.direction = 1 if np.cos(angle) > 0 else -1
            
            velocity = self.model.get_velocity(x, z)
            self.state.update(x, z, angle, velocity)
            
            # 5. 添加起始点
            self.path.add_point(x, z, angle, velocity, layer, block)
            
            # 6. 记录初始点是否在界面上
            self.state.is_initial_point = True
            
            return True
            
        except ValueError as e:
            print(f"初始化射线失败: {str(e)}")
            return False
        
    def _get_layer_block(self, x: float, z: float) -> Tuple[Optional[int], Optional[int]]:
        """获取点(x,z)所在的层号和块号
        
        Args:
            x: x坐标
            z: z坐标
            
        Returns:
            (layer, block): 层号和块号的元组,如果未找到则返回(None, None)
        """
        eps = 1e-6  # 边界容差
        
        # 按层号分组所有块
        layer_blocks = {}
        min_z = float('inf')
        for cell in self.model.get_cells():
            min_z = min(min_z, cell.z1, cell.z2)
            if cell.layer not in layer_blocks:
                layer_blocks[cell.layer] = []
            layer_blocks[cell.layer].append(cell)
            
        # 遍历所有层
        for layer in sorted(layer_blocks.keys()):
            blocks = layer_blocks[layer]
            
            # 遍历该层的所有块
            for i, cell in enumerate(blocks):
                # 获取块的边界
                x_min = min(cell.x1, cell.x2)
                x_max = max(cell.x1, cell.x2)
                z_top = min(cell.z1, cell.z2)
                z_bottom = max(cell.z3, cell.z4)
                # 使用带容差的方法检查点是否在块内
                point = Point2d(x, z)
                if cell.is_in_with_tolerance(point, eps):
                    return layer, i + 1
                        
        # 如果没找到,输出详细信息
        print(f"警告: 未找到点 ({x:.3f}, {z:.3f}) 所在的层和块")
        print("可能原因:")
        print("1. 点在模型边界外")
        print("2. 点在层的边界上")
        print("3. 需要调整边界容差")
            
        return None, None
            
    def _runge_kutta(self, x: float, z: float, angle: float, step: float, 
                    is_x_independent: bool) -> Tuple[float, float, float]:
        """四阶Runge-Kutta方法求解射线方程"""
        h = step / self.config.hdenom
        
        if is_x_independent:
            y = np.array([z, angle])
            x_start = x
            
            def deriv(x: float, y: np.ndarray) -> np.ndarray:
                z, theta = y
                v = self.model.get_velocity(x, z)
                vx = self._get_velocity_gradient_x(x, z)
                vz = self._get_velocity_gradient_z(x, z)
                
                # 射线方程 (以x为自变量)
                p = 1/v  # 射线参数
                dz = p * v * np.sin(theta)  # dz/dx
                dtheta = p * (vz * np.sin(theta) - vx * np.cos(theta))  # dθ/dx
                
                # 限制变化率
                max_dz = 5.0
                max_dtheta = np.pi/4
                dz = np.clip(dz, -max_dz, max_dz)
                dtheta = np.clip(dtheta, -max_dtheta, max_dtheta)
                
                return np.array([dz, dtheta])
                
            # 执行RK4积分
            for i in range(int(self.config.hdenom)):
                k1 = deriv(x_start, y)
                k2 = deriv(x_start + h/2, y + h*k1/2)
                k3 = deriv(x_start + h/2, y + h*k2/2)
                k4 = deriv(x_start + h, y + h*k3)
                
                y = y + h*(k1 + 2*k2 + 2*k3 + k4)/6
                x_start = x_start + h
                
            return x_start, y[0], y[1]
        
        else:
            y = np.array([x, angle])
            z_start = z
            
            def deriv(z: float, y: np.ndarray) -> np.ndarray:
                x, theta = y
                v = self.model.get_velocity(x, z)
                vx = self._get_velocity_gradient_x(x, z)
                vz = self._get_velocity_gradient_z(x, z)
                
                # 射线方程 (以z为自变量)
                p = 1/v * np.sin(theta)  # 射线参数
                dx = 1/(p * v**2)  # dx/dz
                dtheta = (-vx + vz*p*v**2)/(p*v**3)  # dθ/dz
                
                return np.array([dx, dtheta])
                
            # 执行RK4积分
            for i in range(int(self.config.hdenom)):
                k1 = deriv(z_start, y)
                k2 = deriv(z_start + h/2, y + h*k1/2)
                k3 = deriv(z_start + h/2, y + h*k2/2)
                k4 = deriv(z_start + h, y + h*k3)
                
                y = y + h*(k1 + 2*k2 + 2*k3 + k4)/6
                z_start = z_start + h
                
            return y[0], z_start, y[1]
            
    def _trace_step_x(self) -> bool:
        """以x为自变量的一步追踪"""
        # 计算步长
        step = self._calculate_step()
        print(f"step: {step}")
        try:
            # 使用Runge-Kutta方法求解
            x_new, z_new, angle_new = self._runge_kutta(
                self.state.x, self.state.z, self.state.angle, 
                step * self.state.direction, True
            )
            print(f"x_old: {self.state.x}, z_old: {self.state.z}, angle_old: {self.state.angle}")
            print(f"x_new: {x_new}, z_new: {z_new}, angle_new: {angle_new}")
            # 检查新点是否在模型范围内
            velocity = self.model.get_velocity(x_new, z_new)
            
            # 更新状态
            self.state.update(x_new, z_new, angle_new, velocity)
            
            # 添加到路径
            self.path.add_point(x_new, z_new, angle_new, velocity,
                              self.state.layer, self.state.block)
            
            return True
            
        except ValueError:
            return False
            
    def _trace_step_z(self) -> bool:
        """在z方向上追踪一步
        
        Returns:
            bool: 是否成功
        """
        # 1. 计算基本步长
        base_step = self.config.get_step()
        
        # 2. 根据速度梯度调整步长
        vx = self._get_velocity_gradient_x(self.state.x, self.state.z)
        vz = self._get_velocity_gradient_z(self.state.x, self.state.z)
        gradient_mag = np.sqrt(vx*vx + vz*vz)
        if gradient_mag > 0:
            step = min(base_step, base_step/gradient_mag)
        else:
            step = base_step
            
        # 3. 如果接近界面,减小步长
        layer, block = self._get_layer_block(self.state.x, self.state.z)
        if layer is not None and block is not None:
            cells = [cell for cell in self.model.get_cells() if cell.layer == layer]
            if cells and block <= len(cells):
                current_cell = cells[block-1]
                # 计算到上下界面的距离
                top_z = min(current_cell.z1, current_cell.z2)
                bottom_z = max(current_cell.z3, current_cell.z4)
                dist_to_top = abs(self.state.z - top_z)
                dist_to_bottom = abs(self.state.z - bottom_z)
                min_dist = min(dist_to_top, dist_to_bottom)
                
                # 如果接近界面,减小步长
                if min_dist < 0.1:  # 0.1 km = 100m
                    step = min(step, min_dist/2)
        
        # 4. 确保步长在允许范围内
        step = max(self.config.min_step, min(step, self.config.max_step))
        
        # 5. 根据射线方向确定实际步长
        if abs(self.state.angle) <= np.pi/2:  # 向下传播
            actual_step = step
        else:  # 向上传播
            actual_step = -step
            
        # 6. 使用Runge-Kutta方法计算新位置
        try:
            x_new, z_new, angle_new = self._runge_kutta(
                self.state.x, 
                self.state.z,
                self.state.angle,
                actual_step,
                is_x_independent=False
            )
        except ValueError:
            print(f"Runge-Kutta计算失败,当前位置:({self.state.x:.3f}, {self.state.z:.3f})")
            return False
            
        # 7. 检查新位置是否有效
        new_layer, new_block = self._get_layer_block(x_new, z_new)
        if new_layer is None or new_block is None:
            return False
            
        # 8. 更新状态
        try:
            velocity = self.model.get_velocity(x_new, z_new)
            self.state.update(x_new, z_new, angle_new, velocity)
            
            # 9. 添加到路径
            self.path.add_point(x_new, z_new, angle_new, velocity,
                              new_layer, new_block)
            
            return True
        except ValueError:
            return False
        
    def _calculate_step(self) -> float:
        """计算自适应步长"""
        try:
            # 基本步长
            step = self.config.step
            
            # 根据速度和速度梯度调整步长
            v = self.state.velocity
            vx = self._get_velocity_gradient_x(self.state.x, self.state.z)
            vz = self._get_velocity_gradient_z(self.state.x, self.state.z)
            gradient_mag = np.sqrt(vx*vx + vz*vz)
            
            # 使用相对梯度调整步长
            if gradient_mag > 0:
                relative_gradient = gradient_mag/v
                # 限制最小步长比例
                step *= max(0.1, min(1.0, np.exp(-self.config.gradient_factor * relative_gradient)))
            
            # 确保步长在允许范围内
            step = max(self.config.min_step, min(step, self.config.max_step))
            
            print(f"计算步长: {step:.3f} km")
            return step
            
        except Exception as e:
            print(f"计算步长时出错: {str(e)}")
            return self.config.min_step
        
    def _get_velocity_gradient_x(self, x: float, z: float) -> float:
        """计算水平方向速度梯度"""
        dx = 0.001  # 减小步长
        try:
            # 找到当前点所在的单元
            cell = self._get_current_cell(x, z)
            if cell is None:
                return 0.0
            
            # 获取单元的左右边界
            x_min = cell.x1
            x_max = cell.x2
            
            # 调整采样点以确保在单元内
            x1 = max(x_min, x - dx/2)
            x2 = min(x_max, x + dx/2)
            
            # 如果采样点间距太小，返回0
            if abs(x2 - x1) < 1e-6:
                return 0.0
            
            # 计算梯度
            point1 = Point2d(x1, z)
            point2 = Point2d(x2, z)
            v1 = cell.at(point1)
            v2 = cell.at(point2)
            
            return (v2 - v1) / (x2 - x1)
            
        except Exception as e:
            print(f"计算水平速度梯度时出错: {str(e)}")
            return 0.0

    def _get_velocity_gradient_z(self, x: float, z: float) -> float:
        """计算垂直方向速度梯度"""
        dz = 0.001  # 减小步长
        try:
            # 找到当前点所在的单元
            cell = self._get_current_cell(x, z)
            if cell is None:
                return 0.0
            
            # 获取单元的上下边界
            z_min = min(cell.z1, cell.z2, cell.z3, cell.z4)
            z_max = max(cell.z1, cell.z2, cell.z3, cell.z4)
            
            # 调整采样点以确保在单元内
            z1 = max(z_min, z - dz/2)
            z2 = min(z_max, z + dz/2)
            
            # 如果采样点间距太小，返回0
            if abs(z2 - z1) < 1e-6:
                return 0.0
            
            # 计算梯度
            point1 = Point2d(x, z1)
            point2 = Point2d(x, z2)
            v1 = cell.at(point1)
            v2 = cell.at(point2)
            
            return (v2 - v1) / (z2 - z1)
            
        except Exception as e:
            print(f"计算垂直速度梯度时出错: {str(e)}")
            return 0.0

    def _get_current_cell(self, x: float, z: float) -> Optional[TrapezoidCell2d]:
        """获取点(x,z)所在的单元"""
        point = Point2d(x, z)
        for cell in self.model.get_cells():
            if cell.is_in_with_tolerance(point, self.config.interface_eps):
                return cell
        return None

    def _check_boundary(self) -> str:
        """检查是否到达边界或界面"""
        try:
            # 检查是否超出模型范围
            x_range, z_range = self.model.get_model_bounds()
            if (self.state.x < x_range[0] or self.state.x > x_range[1] or
                self.state.z < z_range[0] or self.state.z > z_range[1]):
                print(f"超出模型边界: ({self.state.x:.3f}, {self.state.z:.3f})")
                return 'model'
            
            # 如果是初始点，不进行界面检查
            if self.state.is_initial_point:
                self.state.is_initial_point = False
                return 'none'
            
            # 获取当前点所在层
            current_layer = self.model.get_layer_index(self.state.z)
            if current_layer != self.state.layer:
                print(f"到达界面: 从层 {self.state.layer} 到层 {current_layer}")
                return 'interface'
            
            return 'none'
            
        except Exception as e:
            print(f"检查边界时出错: {str(e)}")
            return 'model'
        
    def _handle_interface(self) -> bool:
        """处理界面反射/折射"""
        try:
            # 获取界面法向量
            current_layer = self.model.get_layer_index(self.state.z)
            interface_z = self.model.get_interface_depth(current_layer)
            
            # 计算入射角（相对于法线）
            incident_angle = abs(self.state.angle - np.pi/2)
            
            # 获取界面两侧的速度
            eps = self.config.interface_eps
            v1 = self.model.get_velocity(self.state.x, self.state.z - eps)
            v2 = self.model.get_velocity(self.state.x, self.state.z + eps)
            
            # 根据波类型处理界面
            if self.state.wave_type == 1:  # 反射波
                # 反射角等于入射角
                self.state.angle = np.pi - self.state.angle
                return True
                
            elif self.state.wave_type == -1:  # 折射波
                # 使用斯奈尔定律
                sin_theta2 = v2/v1 * np.sin(incident_angle)
                if abs(sin_theta2) > 1:  # 全反射
                    return False
                    
                # 计算折射角
                theta2 = np.arcsin(sin_theta2)
                self.state.angle = np.pi/2 + theta2
                return True
                
            return False
            
        except Exception as e:
            print(f"处理界面时出错: {str(e)}")
            return False
        
    def _check_termination(self) -> bool:
        """检查是否终止追踪"""
        try:
            # 检查是否达到目标点
            if hasattr(self, 'target'):
                dx = self.state.x - self.target[0]
                dz = self.state.z - self.target[1]
                dist = np.sqrt(dx*dx + dz*dz)
                if dist < self.config.tolerance:
                    return True
                
            return False
            
        except Exception as e:
            print(f"检查终止条件时出错: {str(e)}")
            return True

    def trace_ray(self, x: float, z: float, angle: float, phase_type: PhaseType, **kwargs) -> Optional[RayPath]:
        """追踪单条射线
        
        Args:
            x, z: 起始点坐标
            angle: 初始射线角度(弧度)
            phase_type: 震相类型
            
        Returns:
            Optional[RayPath]: 射线路径,如果追踪失败则返回None
        """
        # 角度转换：从度转弧度
        angle_rad = angle * np.pi / 180
        
        if not self._initialize_ray(x, z, angle_rad):
            return None
            
        step_count = 0
        while step_count < self.config.max_steps:
            step_count += 1
            print(f"\n步数: {step_count}")
            
            # 根据射线角度选择追踪方法
            theta = abs(self.state.angle % (2*np.pi))
            if 0.25*np.pi <= theta <= 0.75*np.pi:
                success = self._trace_step_z()
            else:
                success = self._trace_step_x()
                
            if not success:
                print("trace_step_x or trace_step_z failed")
                break
                
            # 2.2 检查是否到达界面
            boundary = self._check_boundary()
            print("boundary",boundary)
            if boundary == 'interface':
                print("到达界面")
                # 处理界面反射/折射
                if not self._handle_interface():
                    print("handle_interface failed")
                    break
            elif boundary == 'model':
                # 到达模型边界
                print("到达模型边界")
                break
                
            # 2.3 检查终止条件
            if self._check_termination():
                print("到达终止条件")
                break
                
        return self.path if len(self.path.x_coords) > 1 else None

    def _calculate_travel_time(self, path: RayPath) -> float:
        """计算射线走时"""
        # 对路径进行重采样，使用更多的点以提高精度
        x = np.array(path.x_coords)
        z = np.array(path.z_coords)
        
        # 计算路径长度
        path_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(z)**2))
        
        # 根据配置的插值点数重采样
        t = np.linspace(0, 1, self.config.time_interp_points)
        x_interp = np.interp(t, np.linspace(0, 1, len(x)), x)
        z_interp = np.interp(t, np.linspace(0, 1, len(z)), z)
        
        # 计算每个插值点的速度
        velocities = np.array([self.model.get_velocity(xi, zi) 
                             for xi, zi in zip(x_interp[:-1], z_interp[:-1])])
        
        # 计算每段的长度
        ds = np.sqrt(np.diff(x_interp)**2 + np.diff(z_interp)**2)
        
        # 计算总走时
        return np.sum(ds / velocities)
        
    def _calculate_amplitude(self, path: RayPath) -> float:
        """计算射线振幅
        
        考虑:
        1. 几何扩散
        2. 透射/反射系数
        3. 界面处的能量分配
        
        Args:
            path: 射线路径
            
        Returns:
            float: 振幅
        """
        # 1. 计算射线管扩展因子
        dx = np.diff(path.x_coords)
        dz = np.diff(path.z_coords)
        ds = np.sqrt(dx*dx + dz*dz)
        angles = np.arctan2(dz, dx)
        
        # 2. 计算几何扩散
        spreading = 1.0 / np.sqrt(np.sum(ds))
        
        # 3. 计算界面处的能量损失
        transmission_loss = 1.0
        for i in range(len(path.x_coords)-1):
            if path.layers[i] != path.layers[i+1]:
                # 获取界面两侧的速度
                v1 = path.velocities[i]
                v2 = path.velocities[i+1]
                
                # 计算入射角和折射角
                theta1 = angles[i]
                theta2 = np.arcsin(v2 * np.sin(theta1) / v1)
                
                if path.wave_type == 1:  # 反射
                    # 反射系数
                    R = ((v2/v1 * np.cos(theta1) - np.cos(theta2)) /
                         (v2/v1 * np.cos(theta1) + np.cos(theta2)))
                    transmission_loss *= abs(R)
                else:  # 折射
                    # 透射系数
                    T = (4 * v1 * v2 * np.cos(theta1) * np.cos(theta2) /
                         ((v1 * np.cos(theta1) + v2 * np.cos(theta2))**2))
                    transmission_loss *= np.sqrt(T)
                    
        # 4. 计算总振幅
        amplitude = spreading * transmission_loss
        
        return amplitude
        
    def _calculate_phase(self, path: RayPath) -> float:
        """计算射线相位
        
        考虑:
        1. 传播相位
        2. 界面反射/透射相位
        3. 驻点相位
        
        Args:
            path: 射线路径
            
        Returns:
            float: 相位(弧度)
        """
        # 1. 传播相位
        dx = np.diff(path.x_coords)
        dz = np.diff(path.z_coords)
        ds = np.sqrt(dx*dx + dz*dz)
        k = 2 * np.pi / (path.velocities[:-1] * 0.1)  # 假设频率为10Hz
        phase = np.sum(k * ds)
        
        # 2. 界面相位变化
        for i in range(len(path.x_coords)-1):
            if path.layers[i] != path.layers[i+1]:
                if path.wave_type == 1:  # 反射波
                    phase += np.pi  # 反射相位变化π
                    
        return phase
        
    def calculate_ray_properties(self, path: RayPath) -> dict:
        """计算射线的所有属性
        
        Args:
            path: 射线路径
            
        Returns:
            dict: 包含以下信息：
            - time: 走时
            - amplitude: 振幅
            - phase: 相位
            - path_length: 路径长度
            - average_velocity: 平均速度
            - interfaces: 穿过的界面
        """
        # 1. 基本属性
        dx = np.diff(path.x_coords)
        dz = np.diff(path.z_coords)
        ds = np.sqrt(dx*dx + dz*dz)
        path_length = np.sum(ds)
        
        # 2. 走时
        time = self._calculate_travel_time(path)
        
        # 3. 振幅
        amplitude = self._calculate_amplitude(path)
        
        # 4. 相位
        phase = self._calculate_phase(path)
        
        # 5. 平均速度
        average_velocity = path_length / time
        
        # 6. 界面信息
        interfaces = []
        for i in range(len(path.x_coords)-1):
            if path.layers[i] != path.layers[i+1]:
                interfaces.append({
                    'x': path.x_coords[i],
                    'z': path.z_coords[i],
                    'from_layer': path.layers[i],
                    'to_layer': path.layers[i+1],
                    'velocity1': path.velocities[i],
                    'velocity2': path.velocities[i+1]
                })
                
        return {
            'time': time,
            'amplitude': amplitude,
            'phase': phase,
            'path_length': path_length,
            'average_velocity': average_velocity,
            'interfaces': interfaces
        }
        
    def find_minimum_time_path(self, x1: float, z1: float, x2: float, z2: float,
                             phase_types: List[PhaseType] = None) -> Tuple[Optional[RayPath], dict]:
        """寻找最小走时路径
        
        Args:
            x1, z1: 起始点坐标
            x2, z2: 终止点坐标
            phase_types: 考虑的震相类型列表，如果为None则考虑所有类型
            
        Returns:
            Tuple[RayPath, dict]: 最小走时路径和其属性
        """
        if phase_types is None:
            phase_types = [PhaseType.DIRECT, PhaseType.REFLECTION, 
                         PhaseType.REFRACTION, PhaseType.HEAD]
                         
        min_time = float('inf')
        best_path = None
        best_properties = None
        
        for phase_type in phase_types:
            if phase_type in [PhaseType.REFLECTION, PhaseType.REFRACTION, PhaseType.HEAD]:
                # 对每个界面尝试
                for i in range(1, self.model.get_depth_nodes_count()):
                    path = self.find_ray_path(x1, z1, x2, z2, phase_type, interface=i)
                    if path is None:
                        continue
                        
                    properties = self.calculate_ray_properties(path)
                    if properties['time'] < min_time:
                        min_time = properties['time']
                        best_path = path
                        best_properties = properties
                        
            else:
                path = self.find_ray_path(x1, z1, x2, z2, phase_type)
                if path is None:
                    continue
                    
                properties = self.calculate_ray_properties(path)
                if properties['time'] < min_time:
                    min_time = properties['time']
                    best_path = path
                    best_properties = properties
                    
        return best_path, best_properties

    def _optimize_ray_path(self, path: RayPath) -> RayPath:
        """优化射线路径，使用平滑和重采样"""
        # 获取原始坐标
        x = np.array(path.x_coords)
        z = np.array(path.z_coords)
        
        # 使用移动平均进行平滑
        window = self.config.smooth_window
        x_smooth = np.convolve(x, np.ones(window)/window, mode='valid')
        z_smooth = np.convolve(z, np.ones(window)/window, mode='valid')
        
        # 重采样到指定点数
        t = np.linspace(0, 1, self.config.interp_points)
        x_new = np.interp(t, np.linspace(0, 1, len(x_smooth)), x_smooth)
        z_new = np.interp(t, np.linspace(0, 1, len(z_smooth)), z_smooth)
        
        # 创建新的路径对象
        new_path = RayPath()
        for i in range(len(x_new)):
            try:
                velocity = self.model.get_velocity(x_new[i], z_new[i])
                layer, block = self._get_layer_block(x_new[i], z_new[i])
                if layer is not None and block is not None:
                    angle = np.arctan2(z_new[i+1]-z_new[i], x_new[i+1]-x_new[i]) if i < len(x_new)-1 else path.angles[-1]
                    new_path.add_point(x_new[i], z_new[i], angle, velocity, layer, block)
            except (ValueError, IndexError):
                continue
                
        return new_path 

    def find_ray_path(self, x1: float, z1: float, x2: float, z2: float,
                      phase_type: PhaseType, **kwargs) -> Optional[RayPath]:
        """寻找从(x1,z1)到(x2,z2)的射线路径"""
        print(f"\n开始寻找射线路径:")
        print(f"起点: ({x1:.3f}, {z1:.3f})")
        print(f"终点: ({x2:.3f}, {z2:.3f})")
        print(f"震相类型: {phase_type.value}")
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        
        # 绘制速度模型
        self.model.plot_model(ax)
        
        # 标记起点和终点
        plt.plot(x1, z1, 'r*', markersize=10, label='起点')
        plt.plot(x2, z2, 'g*', markersize=10, label='终点')
        
        # 根据震相类型确定搜索策略
        if phase_type == PhaseType.DIRECT:
            print("\n处理直达波:")
            angle_range = kwargs.get('angle_range', (-45, 45))
            # 计算采样点数并转换为整数
            num_angles = int(abs(angle_range[1] - angle_range[0]) / self.config.angle_step + 1)
            angles = np.linspace(angle_range[0], angle_range[1], num_angles)
            print(f"角度搜索范围: {angle_range[0]:.1f}° 到 {angle_range[1]:.1f}°")
            print(f"角度采样数: {num_angles}")
            
        elif phase_type == PhaseType.REFLECTION:
            print("\n处理反射波:")
            interface = kwargs.get('interface')
            if interface is None:
                raise ValueError("反射波需要指定界面参数")
            
            print(f"反射界面编号: {interface}")
            z_interface = self.model.get_interface_depth(interface)
            print(f"界面深度: {z_interface:.3f} km")
            
            # 计算镜像点
            z_image = 2 * z_interface - z2
            print(f"镜像点深度: {z_image:.3f} km")
            
            # 计算理论反射角
            dx = x2 - x1
            dz = z_image - z1
            reflection_angle = np.arctan2(dx, abs(dz)) * 180 / np.pi
            print(f"理论反射角: {reflection_angle:.2f}°")
            
            # 在理论反射角附近搜索
            angles = np.linspace(reflection_angle * 0.8, reflection_angle * 1.2, 30)
            print(f"搜索角度范围: {angles[0]:.2f}° 到 {angles[-1]:.2f}°")
            
        elif phase_type == PhaseType.REFRACTION:
            print("\n处理折射波:")
            interface = kwargs.get('interface')
            if interface is None:
                raise ValueError("折射波需要指定界面参数")
                
            print(f"折射界面编号: {interface}")
            z_interface = self.model.get_interface_depth(interface)
            print(f"界面深度: {z_interface:.3f} km")
            
            try:
                # 获取界面上下的速度
                eps = min(self.config.interface_eps, 0.0001)
                v1 = self.model.get_velocity(x1, z_interface - eps)
                v2 = self.model.get_velocity(x1, z_interface + eps)
                
                print(f"界面上方速度: {v1:.3f} km/s")
                print(f"界面下方速度: {v2:.3f} km/s")
                
                # 计算临界角
                critical_angle = np.arcsin(v1/v2) * 180 / np.pi
                print(f"临界角: {critical_angle:.2f}°")
                
                # 在临界角附近搜索
                max_angle = min(critical_angle * 0.99, 85.0)
                angles = np.linspace(critical_angle * 0.8, max_angle, 30)
                print(f"搜索角度范围: {angles[0]:.2f}° 到 {angles[-1]:.2f}°")
                
            except ValueError as e:
                print(f"计算临界角时出错: {str(e)}")
                return None
        else:
            raise ValueError(f"不支持的震相类型: {phase_type}")
            
        # 搜索最佳路径
        print("\n开始搜索最佳路径:")
        best_path = None
        min_error = float('inf')
        
        for i, angle in enumerate(angles):
            path = self.trace_ray(x1, z1, angle, phase_type, **kwargs)
            if path is not None:
                # 绘制射线路径（使用浅色）
                plt.plot(path.x_coords, path.z_coords, 'c-', alpha=0.3)
                
                # 计算终点误差
                error = np.sqrt((path.x_coords[-1] - x2)**2 + (path.z_coords[-1] - z2)**2)
                
                # 更新最佳路径
                if error < min_error:
                    min_error = error
                    best_path = path
                    print(f"找到更好的路径 - 角度: {angle:.2f}°, 误差: {error:.3f} km")
                    
                    # 绘制当前最佳路径（使用深色）
                    plt.plot(path.x_coords, path.z_coords, 'b-', linewidth=2)
                    
        # 完善图形
        plt.title(f'Raytracing Result - {phase_type.value}')
        plt.xlabel('Distance (km)')
        plt.ylabel('Depth (km)')
        plt.grid(True)
        plt.legend()
        
        # 保存图形
        plt.savefig('ray_tracing_result.png')
        plt.close()
        
        if best_path is None:
            print("未找到有效路径")
            return None
            
        if min_error > self.config.tolerance:
            print(f"最小误差 ({min_error:.3f} km) 超过容差 ({self.config.tolerance:.3f} km)")
            return None
            
        print(f"\n成功找到路径:")
        print(f"终点误差: {min_error:.3f} km")
        print(f"路径点数: {len(best_path.x_coords)}")
        print(f"走时: {best_path.calculate_travel_time():.3f} s")
        
        return best_path 

    def _get_interface_depth(self, x: float, interface_index: int) -> float:
        """获取指定位置的界面深度"""
        try:
            # 获取界面的所有节点
            interface_nodes = self.model.get_interface_nodes(interface_index)
            if not interface_nodes:
                raise ValueError(f"未找到界面 {interface_index}")
            
            # 对节点进行排序
            sorted_nodes = sorted(interface_nodes, key=lambda n: n.x)
            
            # 找到x坐标两侧的节点
            for i in range(len(sorted_nodes) - 1):
                if sorted_nodes[i].x <= x <= sorted_nodes[i + 1].x:
                    # 线性插值
                    x1, z1 = sorted_nodes[i].x, sorted_nodes[i].z
                    x2, z2 = sorted_nodes[i + 1].x, sorted_nodes[i + 1].z
                    return z1 + (z2 - z1) * (x - x1) / (x2 - x1)
                
            # 如果x在界面范围外
            if x < sorted_nodes[0].x:
                return sorted_nodes[0].z
            return sorted_nodes[-1].z
            
        except Exception as e:
            print(f"获取界面深度时出错: {str(e)}")
            return None 

    def _get_velocity(self, x: float, z: float) -> float:
        """获取指定位置的速度"""
        try:
            # 找到点所在的单元
            cell = self._get_current_cell(x, z)
            if cell is None:
                raise ValueError(f"点 ({x}, {z}) 不在任何单元内")
            
            # 使用单元的速度计算方法
            point = Point2d(x, z)
            return cell.at(point)  # 使用 get_velocity 方法
        
        except Exception as e:
            print(f"获取速度时出错: {str(e)}")
            return None 