"""
Talwani 2D重力异常计算模块

基于Talwani et al. (1959)的方法计算2D多边形体的重力异常
"""

import numpy as np
from typing import Union, List, Tuple, Optional
import warnings

# 物理常量
NEWTON_G = 6.67430e-11  # 万有引力常数 (m³/(kg·s²))
SI_TO_MGAL = 1.0e5      # m/s² 转换为 mGal
SI_TO_EOTVOS = 1.0e9    # (m/s²)/m 转换为 Eotvos
TOL = 1.0e-7            # 计算容差


def ensure_closed_polygon(x: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    确保多边形闭合（最后一个点等于第一个点）
    
    Args:
        x: x坐标数组
        z: z坐标数组
    
    Returns:
        (x, z): 闭合后的坐标数组
    """
    x = np.asarray(x)
    z = np.asarray(z)
    
    if len(x) < 3:
        raise ValueError("多边形至少需要3个顶点")
    
    # 检查是否已闭合
    if abs(x[0] - x[-1]) < TOL and abs(z[0] - z[-1]) < TOL:
        return x, z
    
    # 添加第一个点作为最后一个点
    x = np.append(x, x[0])
    z = np.append(z, z[0])
    
    return x, z


def talwani2d_gravity(
    x: Union[np.ndarray, List[float]],
    z: Union[np.ndarray, List[float]],
    x0: float,
    z0: float,
    rho: float,
    check_vertex: bool = True
) -> float:
    """
    计算2D多边形体的重力异常（Talwani方法）
    
    基于Talwani et al. (1959)的经典算法，计算无限延伸的2D多边形体
    在观测点产生的重力异常。
    
    Args:
        x: 多边形x坐标数组 (m)
        z: 多边形z坐标数组 (m)
        x0: 观测点x坐标 (m)
        z0: 观测点z坐标 (m)
        rho: 密度对比度 (kg/m³)，相对于背景密度
        check_vertex: 是否检查观测点是否与顶点重合
    
    Returns:
        重力异常 (mGal)
    
    Raises:
        ValueError: 如果观测点与多边形顶点重合
    
    Reference:
        Talwani, M., Worzel, J. L., & Landisman, M. (1959). 
        Rapid gravity computations for two-dimensional bodies with 
        application to the Mendocino submarine fracture zone. 
        J. Geophys. Res., 64, 49-59.
    """
    x, z = ensure_closed_polygon(x, z)
    n = len(x)
    
    sum_val = 0.0
    xi = x[0] - x0
    zi = z[0] - z0
    phi_i = np.arctan2(zi, xi)
    ri = np.hypot(xi, zi)
    
    if check_vertex and ri < TOL:
        raise ValueError(f"观测点 ({x0:.2f}, {z0:.2f}) 与多边形顶点 ({x[0]:.2f}, {z[0]:.2f}) 重合！")
    
    for i in range(n - 1):
        i1 = i + 1
        xi1 = x[i1] - x0
        zi1 = z[i1] - z0
        phi_i1 = np.arctan2(zi1, xi1)
        ri1 = np.hypot(xi1, zi1)
        
        if check_vertex and ri1 < TOL:
            raise ValueError(f"观测点 ({x0:.2f}, {z0:.2f}) 与多边形顶点 ({x[i1]:.2f}, {z[i1]:.2f}) 重合！")
        
        # Talwani公式的核心计算
        # 分子：(xi*zi1 - zi*xi1) * ((xi1-xi)*(φi-φi1) + (zi1-zi)*log(ri1/ri))
        # 分母：(xi1-xi)² + (zi1-zi)²
        cross_product = xi * zi1 - zi * xi1
        angle_term = (xi1 - xi) * (phi_i - phi_i1)
        log_term = (zi1 - zi) * np.log(ri1 / ri) if ri > TOL and ri1 > TOL else 0.0
        
        numerator = cross_product * (angle_term + log_term)
        denominator = (xi1 - xi)**2 + (zi1 - zi)**2
        
        if abs(denominator) > TOL:
            sum_val += numerator / denominator
        
        # 更新为下一个顶点
        xi = xi1
        zi = zi1
        ri = ri1
        phi_i = phi_i1
    
    # 转换为mGal
    # 公式：Δg = 2 * G * ρ * sum * SI_TO_MGAL
    gravity = 2.0 * NEWTON_G * rho * SI_TO_MGAL * sum_val
    
    return gravity


def talwani2d_gravity_profile(
    x: Union[np.ndarray, List[float]],
    z: Union[np.ndarray, List[float]],
    x_obs: Union[np.ndarray, List[float]],
    z_obs: float,
    rho: float,
    check_vertex: bool = True
) -> np.ndarray:
    """
    计算沿观测线的重力异常剖面
    
    Args:
        x: 多边形x坐标数组 (m)
        z: 多边形z坐标数组 (m)
        x_obs: 观测点x坐标数组 (m)
        z_obs: 观测点z坐标（常数，通常为0）(m)
        rho: 密度对比度 (kg/m³)
        check_vertex: 是否检查观测点是否与顶点重合
    
    Returns:
        重力异常数组 (mGal)
    """
    x_obs = np.asarray(x_obs)
    gravity = np.zeros_like(x_obs, dtype=float)
    
    for i, x0 in enumerate(x_obs):
        try:
            gravity[i] = talwani2d_gravity(x, z, x0, z_obs, rho, check_vertex=check_vertex)
        except ValueError as e:
            if check_vertex:
                warnings.warn(f"观测点 {i} ({x0:.2f}, {z_obs:.2f}): {e}")
                gravity[i] = np.nan
            else:
                raise
    
    return gravity


def talwani2d_gravity_multibody(
    bodies: List[Tuple[np.ndarray, np.ndarray, float]],
    x0: float,
    z0: float,
    check_vertex: bool = True
) -> float:
    """
    计算多个多边形体的总重力异常
    
    Args:
        bodies: 多边形体列表，每个元素为 (x, z, rho)
                x: x坐标数组 (m)
                z: z坐标数组 (m)
                rho: 密度对比度 (kg/m³)
        x0: 观测点x坐标 (m)
        z0: 观测点z坐标 (m)
        check_vertex: 是否检查观测点是否与顶点重合
    
    Returns:
        总重力异常 (mGal)
    """
    total_gravity = 0.0
    
    for x, z, rho in bodies:
        try:
            gravity = talwani2d_gravity(x, z, x0, z0, rho, check_vertex=check_vertex)
            
            # 检查多边形方向（顺时针/逆时针）
            # 通过计算多边形面积判断
            area = polygon_area(x, z)
            if area < 0:
                # 逆时针，需要取反
                gravity = -gravity
            
            total_gravity += gravity
        except ValueError as e:
            if check_vertex:
                warnings.warn(f"跳过多边形体: {e}")
            else:
                raise
    
    return total_gravity


def polygon_area(x: np.ndarray, z: np.ndarray) -> float:
    """
    计算多边形面积（带符号）
    
    正面积表示顺时针，负面积表示逆时针
    
    Args:
        x: x坐标数组
        z: z坐标数组
    
    Returns:
        多边形面积（带符号）
    """
    x = np.asarray(x)
    z = np.asarray(z)
    
    # 使用Shoelace公式
    n = len(x)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * z[j] - x[j] * z[i]
    
    return area / 2.0


def convert_density_units(density: float, from_unit: str = 'g/cm3', to_unit: str = 'kg/m3') -> float:
    """
    转换密度单位
    
    Args:
        density: 密度值
        from_unit: 源单位 ('g/cm3' 或 'kg/m3')
        to_unit: 目标单位 ('g/cm3' 或 'kg/m3')
    
    Returns:
        转换后的密度值
    """
    if from_unit == to_unit:
        return density
    
    if from_unit == 'g/cm3' and to_unit == 'kg/m3':
        return density * 1000.0
    elif from_unit == 'kg/m3' and to_unit == 'g/cm3':
        return density / 1000.0
    else:
        raise ValueError(f"不支持的单位转换: {from_unit} -> {to_unit}")


def calculate_gravity_from_model(
    model_x: np.ndarray,
    model_z: np.ndarray,
    model_density: np.ndarray,
    x_obs: Union[np.ndarray, List[float]],
    z_obs: float = 0.0,
    background_density: float = 2670.0,
    method: str = 'grid'
) -> np.ndarray:
    """
    从密度模型计算重力异常
    
    将密度模型转换为多边形体，然后计算重力异常
    
    Args:
        model_x: 模型x坐标数组 (m)
        model_z: 模型z坐标数组 (m)
        model_density: 模型密度数组 (kg/m³)
        x_obs: 观测点x坐标数组 (m)
        z_obs: 观测点z坐标（常数）(m)
        background_density: 背景密度 (kg/m³)，默认2670 kg/m³
        method: 计算方法
                'grid': 将每个网格单元作为矩形体
                'contour': 使用等密度线作为多边形
    
    Returns:
        重力异常数组 (mGal)
    """
    x_obs = np.asarray(x_obs)
    gravity = np.zeros_like(x_obs, dtype=float)
    
    if method == 'grid':
        # 将每个网格单元作为矩形体
        # 这是一个简化方法，实际应该使用更精确的多边形积分
        warnings.warn("Grid方法尚未完全实现，建议使用contour方法")
        # TODO: 实现网格方法
        pass
    elif method == 'contour':
        # 使用等密度线
        # TODO: 实现等密度线方法
        warnings.warn("Contour方法尚未完全实现")
        pass
    else:
        raise ValueError(f"不支持的方法: {method}")
    
    return gravity


# 便捷函数
def calculate_gravity_anomaly(
    polygon_x: Union[np.ndarray, List[float]],
    polygon_z: Union[np.ndarray, List[float]],
    observation_x: Union[np.ndarray, List[float], float],
    observation_z: float = 0.0,
    density_contrast: float = 100.0,
    density_unit: str = 'kg/m3'
) -> Union[float, np.ndarray]:
    """
    便捷函数：计算重力异常
    
    Args:
        polygon_x: 多边形x坐标 (m)
        polygon_z: 多边形z坐标 (m)
        observation_x: 观测点x坐标 (m)，可以是单个值或数组
        observation_z: 观测点z坐标 (m)，默认0（地表）
        density_contrast: 密度对比度
        density_unit: 密度单位 ('kg/m3' 或 'g/cm3')
    
    Returns:
        重力异常 (mGal)，如果observation_x是数组则返回数组
    """
    # 转换密度单位
    if density_unit == 'g/cm3':
        density_contrast = convert_density_units(density_contrast, 'g/cm3', 'kg/m3')
    
    observation_x = np.asarray(observation_x)
    
    if observation_x.ndim == 0:
        # 单个观测点
        return talwani2d_gravity(
            polygon_x, polygon_z,
            float(observation_x), observation_z,
            density_contrast
        )
    else:
        # 观测点数组
        return talwani2d_gravity_profile(
            polygon_x, polygon_z,
            observation_x, observation_z,
            density_contrast
        )
