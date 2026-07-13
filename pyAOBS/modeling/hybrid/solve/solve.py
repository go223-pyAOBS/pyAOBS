"""
测试ttcrpy的compute_L功能和小规模反演
"""

import numpy as np
import ttcrpy.rgrid as rg
import time
import logging
import matplotlib.pyplot as plt
from scipy.sparse.linalg import lsqr
from scipy.sparse import vstack
import numba
from numba import jit, prange
import multiprocessing as mp
from functools import partial
from line_profiler import LineProfiler
import psutil
import warnings
warnings.filterwarnings('ignore')

# 启用numba缓存以加快重复执行速度
@jit(nopython=True, cache=True, parallel=True)
def create_velocity_anomaly(velocity, nx, nz, x_center, z_center, radius, factor):
    """使用numba加速的速度异常体创建函数"""
    for i in prange(nx):
        for j in prange(nz):
            dist = np.sqrt((i - x_center)**2 + (j - z_center)**2)
            if dist < radius:
                velocity[i, j] *= factor
    return velocity

@jit(nopython=True, cache=True)
def create_base_velocity(nx, nz, v0, grad):
    """使用numba加速的基础速度场创建函数"""
    z = np.linspace(0, 1, nz)
    base_velocity = v0 + grad * z[:, np.newaxis]
    return np.tile(base_velocity, (1, nx)).T

def create_simple_model(nx: int, nz: int, add_anomaly: bool = True) -> np.ndarray:
    """创建简单的速度模型
    
    Args:
        nx: x方向网格数
        nz: z方向网格数
        add_anomaly: 是否添加速度异常体
        
    Returns:
        velocity: 速度模型(km/s)
    """
    # 创建基础速度场，速度随深度线性增加
    z = np.linspace(0, 1, nz)  # 归一化深度
    v0 = 1.5  # 表层速度(km/s)
    grad = 3.5  # 速度梯度(km/s per unit depth)
    base_velocity = v0 + grad * z[:, np.newaxis]  # 形状(nz, 1)
    velocity = np.tile(base_velocity, (1, nx)).T  # 形状(nx, nz)
    
    # 添加速度异常体
    if add_anomaly:
        # 添加中部高速异常体
        x_center1 = nx // 3
        z_center1 = nz // 4
        radius1 = min(nx, nz) // 6
        for i in range(nx):
            for j in range(nz):
                dist = np.sqrt((i - x_center1)**2 + (j - z_center1)**2)
                if dist < radius1:
                    velocity[i, j] *= 1.5  # 提高50%的速度
        # 添加右部低速异常体
        x_center2 = 2 * nx // 3
        z_center2 = nz // 4
        radius2 = min(nx, nz) // 6
        for i in range(nx):
            for j in range(nz):
                dist = np.sqrt((i - x_center2)**2 + (j - z_center2)**2)
                if dist < radius2:
                    velocity[i, j] *= 0.5  # 降低50%的速度
        # 添加右部低速异常体
        x_center3 = nx // 2
        z_center3 = nz // 2
        radius3 = min(nx, nz) // 4
        for i in range(nx):
            for j in range(nz):
                dist = np.sqrt((i - x_center3)**2 + (j - z_center3)**2)
                if dist < radius3:
                    velocity[i, j] *= 1.2  # 降低50%的速度    
    return velocity

def analyze_L_matrix(L, iteration: int, logger):
    """分析L矩阵的特征
    
    Args:
        L: Fréchet矩阵
        iteration: 当前迭代次数
        logger: 日志记录器
    """
    logger.info(f"\n迭代 {iteration + 1} 的L矩阵分析:")
    logger.info(f"L矩阵形状: {L.shape}")
    logger.info(f"L矩阵非零元素数量: {L.nnz}")
    logger.info(f"L矩阵稀疏度: {L.nnz / (L.shape[0] * L.shape[1]):.6f}")
    
    # 分析L矩阵的数值范围
    L_data = L.data
    logger.info(f"L矩阵非零元素值范围: [{L_data.min():.2e}, {L_data.max():.2e}]")
    logger.info(f"L矩阵非零元素平均值: {L_data.mean():.2e}")
    logger.info(f"L矩阵非零元素标准差: {L_data.std():.2e}")
    
    # 检查每行的非零元素数量
    row_nnz = np.diff(L.indptr)
    logger.info(f"每行非零元素数量范围: [{row_nnz.min()}, {row_nnz.max()}]")
    logger.info(f"每行平均非零元素数量: {row_nnz.mean():.2f}")

def build_covariance_matrices(nx: int, nz: int, tt_obs: np.ndarray, slowness: np.ndarray):
    """构建数据和模型协方差矩阵
    
    Args:
        nx: x方向网格数
        nz: z方向网格数
        tt_obs: 观测走时数据
        slowness: 当前慢度模型
        
    Returns:
        Cd: 数据协方差矩阵
        Cm: 模型协方差矩阵
    """
    # 构建数据协方差矩阵（使用固定误差0.05）
    data_errors = np.full_like(tt_obs, 0.05)  # 所有数据使用相同的误差0.05
    Cd = np.diag(data_errors**2)
    
    # 构建模型协方差矩阵（基于慢度的平方）
    model_variance = slowness.flatten()**2
    Cm = np.diag(model_variance)
    
    return Cd, Cm

def normalize_system(G: np.ndarray, d: np.ndarray, Cd: np.ndarray, Cm: np.ndarray):
    """标准化线性系统
    
    Args:
        G: 原始走时矩阵
        d: 走时残差
        Cd: 数据协方差矩阵
        Cm: 模型协方差矩阵
        
    Returns:
        G_norm: 标准化后的走时矩阵
        d_norm: 标准化后的残差
        Cm_sqrt: 模型协方差矩阵的平方根（用于恢复原始慢度更新）
    """
    # 计算协方差矩阵的平方根
    Cd_sqrt_inv = np.diag(1.0 / np.sqrt(np.diag(Cd)))
    Cm_sqrt = np.diag(np.sqrt(np.diag(Cm)))
    
    # 标准化系统
    G_norm = Cd_sqrt_inv @ G @ Cm_sqrt
    d_norm = Cd_sqrt_inv @ d
    
    return G_norm, d_norm, Cm_sqrt

def build_smoothing_operators(nx: int, nz: int, smesh, corr_lengths=None):
    """构建改进的平滑算子
    
    Args:
        nx: x方向网格数
        nz: z方向网格数
        smesh: 网格对象
        corr_lengths: 相关长度函数或元组(Lh, Lv)
            
    Returns:
        LH: 水平方向平滑算子
        LV: 垂直方向平滑算子
    """
    n_total = nx * nz
    
    # 如果没有提供相关长度函数，使用默认值
    if corr_lengths is None:
        Lh, Lv = 4.0, 2.0  # 默认相关长度：水平50km，垂直10km
    
    # 水平方向平滑
    LH = np.zeros((n_total, n_total))
    # 垂直方向平滑
    LV = np.zeros((n_total, n_total))
    
    # 构建网格点坐标
    x = np.linspace(0, 500, nx)  # 假设模型宽度为500km
    z = np.linspace(0, 30, nz)   # 假设模型深度为30km
    
    for i in range(nx):
        for k in range(nz):
            inode = k + i * nz  # 当前节点索引
            
            # 获取当前点的相关长度
            if callable(corr_lengths):
                Lh, Lv = corr_lengths(x[i], z[k])
            
            Lh2 = Lh * Lh
            Lv2 = Lv * Lv
            
            # 水平方向平滑
            beta_sum_h = 0.0
            for ii in range(nx):
                dx = x[ii] - x[i]
                if abs(dx) <= Lh:
                    jnode = k + ii * nz
                    if jnode != inode:
                        beta = np.exp(-dx*dx/Lh2)
                        LH[inode, jnode] = beta
                        beta_sum_h += beta
            LH[inode, inode] = -beta_sum_h
            
            # 垂直方向平滑
            beta_sum_v = 0.0
            for kk in range(nz):
                dz = z[kk] - z[k]
                if abs(dz) <= Lv:
                    jnode = kk + i * nz
                    if jnode != inode:
                        beta = np.exp(-dz*dz/Lv2)
                        LV[inode, jnode] = beta
                        beta_sum_v += beta
            LV[inode, inode] = -beta_sum_v
            
    return LH, LV

def build_depth_dependent_lengths(nx, nz):
    """构建深度相关的相关长度函数
    
    Args:
        nx: x方向网格数
        nz: z方向网格数
        
    Returns:
        function: 返回给定位置的相关长度(Lh, Lv)
    """
    def corr_lengths(x, z):
        # 水平相关长度随深度增加
        Lh = 2.0 + 4.0 * (z / 30.0)  # 从30km到70km
        # 垂直相关长度也随深度增加
        Lv = 1.0 + 1.0 * (z / 30.0)   # 从5km到20km
        return Lh, Lv
    
    return corr_lengths

def build_norm_kernel():
    """构建基础归一化核矩阵
    
    Returns:
        T_common: 基础归一化核矩阵
    """
    # 构建4x4的基础归一化核矩阵
    T_common = np.array([
        [4, 2, 1, 2],
        [2, 4, 2, 1],
        [1, 2, 4, 2],
        [2, 1, 2, 4]
    ]) * (1.0/36.0)
    
    return T_common

def build_cell_norm_kernel(nx: int, nz: int, dx: np.ndarray, dz: np.ndarray):
    """为每个网格单元构建归一化核矩阵
    
    Args:
        nx: x方向网格数
        nz: z方向网格数
        dx: x方向网格间距
        dz: z方向网格间距
        
    Returns:
        T: 所有单元的归一化核矩阵列表
    """
    T_common = build_norm_kernel()
    T_cells = []
    
    for i in range(nx-1):
        for k in range(nz-1):
            # 考虑网格单元面积的局部归一化核
            T_local = T_common * (dx[i] * dz[k])
            T_cells.append(T_local)
            
    return T_cells

def build_damping_matrix(nx: int, nz: int, dx: np.ndarray, dz: np.ndarray, 
                        depth_weights: np.ndarray = None):
    """构建改进的阻尼矩阵
    
    Args:
        nx: x方向网格数
        nz: z方向网格数
        dx: x方向网格间距
        dz: z方向网格间距
        depth_weights: 深度相关权重（可选）
        
    Returns:
        D: 阻尼矩阵
    """
    n_total = nx * nz
    D = np.zeros((n_total, n_total))
    
    # 获取所有单元的归一化核
    T_cells = build_cell_norm_kernel(nx, nz, dx, dz)
    
    # 构建完整的阻尼矩阵
    cell_idx = 0
    for i in range(nx-1):
        for k in range(nz-1):
            # 获取当前单元的四个节点索引
            idx = k + i * nz
            nodes = [idx, idx+1, idx+nz+1, idx+nz]
            
            # 获取当前单元的归一化核
            T_local = T_cells[cell_idx]
            
            # 将局部归一化核加入全局阻尼矩阵
            for ii, ni in enumerate(nodes):
                for jj, nj in enumerate(nodes):
                    D[ni, nj] += T_local[ii, jj]
            
            cell_idx += 1
    
    # 如果提供了深度权重，应用它们
    if depth_weights is not None:
        weights = depth_weights.flatten()
        D = np.diag(weights) @ D
    
    return D

def update_damping_weights(iteration: int, rms_history: list, 
                          current_damping: float, base_damping: float):
    """更新阻尼权重的策略
    
    Args:
        iteration: 当前迭代次数
        rms_history: RMS历史记录
        current_damping: 当前阻尼因子
        base_damping: 基础阻尼因子
        
    Returns:
        new_damping: 更新后的阻尼因子
    """
    if iteration == 0:
        return current_damping
        
    rms_ratio = rms_history[-1] / rms_history[-2]
    
    if rms_ratio > 1.0:  # RMS增加
        new_damping = min(current_damping * 1.5, base_damping * 2.0)
    elif rms_ratio < 0.95:  # RMS显著减小
        new_damping = max(current_damping * 0.8, base_damping * 0.1)
    else:  # RMS变化不大
        new_damping = current_damping
        
    return new_damping

def apply_post_smoothing(dm, L_cal, grid, lambda_v=0.2):
    """应用后验平滑
    
    Args:
        dm: 模型更新量
        L_cal: Fréchet矩阵
        grid: 网格对象
        lambda_v: 平滑强度
        
    Returns:
        smoothed_dm: 平滑后的更新量
    """
    nx, nz = dm.shape
    
    # 构建2D平滑算子
    def build_2d_smoothing_operator(nx, nz, sigma_x=2.0, sigma_z=1.0):
        """构建2D高斯平滑算子"""
        x = np.linspace(-3*sigma_x, 3*sigma_x, min(nx//2, 7))
        z = np.linspace(-3*sigma_z, 3*sigma_z, min(nz//2, 7))
        X, Z = np.meshgrid(x, z)
        kernel = np.exp(-(X**2/(2*sigma_x**2) + Z**2/(2*sigma_z**2)))
        kernel = kernel / kernel.sum()
        return kernel
    
    # 获取平滑核
    smooth_kernel = build_2d_smoothing_operator(nx, nz)
    
    # 应用平滑
    from scipy.signal import convolve2d
    smoothed_dm = convolve2d(dm, smooth_kernel, mode='same', boundary='symm')
    
    # 计算校正项 Bmc
    # 计算走时差异
    dt_smooth = L_cal @ smoothed_dm.flatten()
    dt_orig = L_cal @ dm.flatten()
    dt_diff = dt_smooth - dt_orig
    
    # 求解校正项
    Bmc = lsqr(L_cal, dt_diff, iter_lim=100)[0].reshape(nx, nz)
    
    # 应用最终的更新
    final_dm = smoothed_dm - lambda_v * Bmc
    
    return final_dm

def solve():
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 模型参数（使用小规模网格进行测试）
    nx, nz = 501, 31  # 减小模型尺寸
    xmin, xmax = 0, 500  # km
    zmin, zmax = 0, 30  # km
    
    # 创建真实模型
    logger.info("创建真实速度模型...")
    true_velocity = create_simple_model(nx-1, nz-1, add_anomaly=True)
    
    # 创建初始模型（不包含异常体）
    initial_velocity = create_simple_model(nx-1, nz-1, add_anomaly=False)
    
    # 转换坐标到米
    xmin_m, xmax_m = xmin , xmax
    zmin_m, zmax_m = zmin , zmax
    
    # 创建网格
    logger.info("初始化网格...")
    x = np.linspace(xmin_m, xmax_m, nx)
    z = np.linspace(zmin_m, zmax_m, nz)
    grid = rg.Grid2d(
        x=x,
        z=z,
        n_threads=5,  # 多线程
        cell_slowness=True,
        method='SPM',
        aniso='iso'
    )
    
    # 创建更多的源接收点配置以提高覆盖度
    n_sources = 20  # 增加震源数量
    n_receivers = 200  # 增加接收点数量
    sources = []
    receivers = []
    
    # 在表面均匀分布震源，略微扩大范围
    for src_x in np.linspace(0, xmax_m, n_sources):
        sources.append([src_x, 0.])
    sources = np.array(sources)
    
    # 在表面均匀分布接收点，扩大范围
    for rcv_x in np.linspace(5, 495, n_receivers):  # 扩大接收点范围
        receivers.append([rcv_x, 0.])
    receivers = np.array(receivers)
    
    # 使用真实模型计算观测数据
    logger.info("计算观测数据...")
    true_slowness = 1.0 / true_velocity
    grid.set_slowness(true_slowness)
    
    # 收集所有震源的观测数据
    all_tt_obs = []
    all_rays_obs = []
    all_L_obs = []
    
    for src in sources:
        tt, rays, L = grid.raytrace(src.reshape(1, 2), 
                                    receivers, 
                                    return_rays=True, 
                                    compute_L=True)
        all_tt_obs.extend(tt)
        all_rays_obs.extend(rays)
        all_L_obs.append(L)
    
    tt_obs = np.array(all_tt_obs)
    L_obs = vstack(all_L_obs)
    
    # 添加观测噪声
    noise_level = 0.002  # 进一步减小噪声水平到0.2%
    noise = np.random.normal(0, noise_level * np.mean(tt_obs), tt_obs.shape)
    tt_obs += noise
    
    # 使用初始模型进行反演
    logger.info("开始反演测试...")
    current_velocity = initial_velocity.copy()
    max_iterations = 8  # 增加最大迭代次数
    step_length = 0.5  # 增大初始步长
    min_step_length = 0.1  # 最小步长限制
    
    # 构建深度相关的相关长度函数
    corr_lengths = build_depth_dependent_lengths(nx-1, nz-1)
    
    # 构建改进的平滑算子
    LH, LV = build_smoothing_operators(nx-1, nz-1, grid, corr_lengths)
    logger.info(f"水平平滑算子维度: {LH.shape}")
    logger.info(f"垂直平滑算子维度: {LV.shape}")
    
    # 记录反演历史
    rms_history = []
    velocity_updates = []
    max_update_history = []
    
    # 初始化权重和参数（降低初始权重）
    lambda_h = 0.1  # 减小水平平滑权重
    lambda_v = 0.2  # 减小垂直平滑权重
    damping_factor = 0.02  # 减小初始阻尼因子
    
    # 设置正则化参数的范围（调整范围）
    lambda_h_range = (0.05, 0.3)  # 降低水平平滑权重范围
    lambda_v_range = (0.1, 0.6)   # 降低垂直平滑权重范围
    damping_range = (0.01, 0.08)  # 降低阻尼因子范围
    
    # 非均匀正则化权重
    def build_depth_weights(nx, nz):
        """构建深度相关的权重"""
        z = np.linspace(0, 1, nz)
        depth_weights = 1.0 + 1.0 * z  # 随深度增加权重
        return np.tile(depth_weights, nx).reshape(nx, nz)
    
    def build_sensitivity_weights(L_matrix):
        """基于L矩阵计算灵敏度权重"""
        # 计算每个模型参数的灵敏度
        sensitivity = np.array(L_matrix.power(2).sum(axis=0)).flatten()
        # 归一化灵敏度
        sensitivity = sensitivity / np.max(sensitivity)
        # 转换为权重：高灵敏度区域权重小，低灵敏度区域权重大
        weights = np.exp(-sensitivity / np.max(sensitivity))  # 添加小量避免除零
        return weights.reshape(nx-1, nz-1)
    
    def update_adaptive_weights(current_weights, dm, current_velocity, iteration):
        """基于更新量更新自适应权重"""
        if iteration == 0:
            return current_weights
        current_slowness = 1.0 / current_velocity
        # 计算相对更新量
        relative_update = np.abs(dm / current_slowness)
        
        # 更新量较大的区域降低权重，较小的区域增加权重
        update_scale = 1.0 / (1.0 + 0.7 * relative_update)
        
        # 平滑权重变化
        new_weights = current_weights * update_scale
        
        # 归一化权重
        new_weights = new_weights / np.mean(new_weights)
        
        # 限制权重范围，避免过度调整
        new_weights = np.clip(new_weights, 0.1, 10.0)
        
        return new_weights
    
    # 初始化权重
    depth_weights = build_depth_weights(nx-1, nz-1)
    adaptive_weights = np.ones((nx-1, nz-1))
    
    # 在反演循环开始前初始化
    dx = np.diff(np.linspace(0, 500, nx))  # x方向网格间距
    dz = np.diff(np.linspace(0, 30, nz))   # z方向网格间距
    base_damping = 0.01  # 基础阻尼因子
    current_damping = base_damping
    
    for iteration in range(max_iterations):
        # 设置当前模型
        current_slowness = 1.0 / current_velocity
        grid.set_slowness(current_slowness)
        
        # 构建协方差矩阵
        Cd, Cm = build_covariance_matrices(nx-1, nz-1, tt_obs, current_slowness)
        
        # 收集所有震源的计算数据
        all_tt_cal = []
        all_L_cal = []
        
        for src in sources:
            tt, rays, L = grid.raytrace(src.reshape(1, 2), receivers, return_rays=True, compute_L=True)
            all_tt_cal.extend(tt)
            all_L_cal.append(L)
        
        tt_cal = np.array(all_tt_cal)
        L_cal = vstack(all_L_cal)
        
        # 计算走时残差
        dt = tt_obs - tt_cal
        rms = np.sqrt(np.mean(dt**2))
        rms_history.append(rms)
        
        logger.info(f"\n迭代 {iteration + 1}:")
        logger.info(f"RMS = {rms:.6f} s")
        
        # 标准化系统
        G_norm, d_norm, Cm_sqrt = normalize_system(L_cal.toarray(), dt, Cd, Cm)
        
        # 更新灵敏度权重
        sensitivity_weights = build_sensitivity_weights(L_cal)

        # 组合所有权重
        combined_weights = depth_weights * sensitivity_weights * adaptive_weights

        # 构建阻尼矩阵
        D = build_damping_matrix(nx-1, nz-1, dx, dz, depth_weights)
        
        # 构建增广系统
        A = vstack([
            G_norm,  # 标准化的走时矩阵
            np.sqrt(lambda_h) * LH,  # 水平平滑约束
            np.sqrt(lambda_v) * LV,  # 垂直平滑约束
            np.sqrt(current_damping) * D  # 改进的阻尼约束
        ])
        
        b = np.concatenate([
            d_norm,  # 标准化的残差
            np.zeros(LH.shape[0]),  # 水平平滑约束右端项
            np.zeros(LV.shape[0]),  # 垂直平滑约束右端项
            np.zeros(combined_weights.size)  # 阻尼约束右端项
        ])
        
        # 求解标准化系统
        dm_norm = lsqr(A, b, iter_lim=1000, atol=1e-8, btol=1e-8)[0]
        
        # 恢复原始尺度的慢度更新量
        dm = (Cm_sqrt @ dm_norm).reshape(nx-1, nz-1)
        max_update = np.max(np.abs(dm))
        max_update_history.append(max_update)
        
        # 约束更新幅度
        max_relative_update = 0.8  # 最大相对更新量
        
        # 约束慢度更新的相对变化
        relative_slowness_update = dm / current_slowness
        relative_slowness_update = np.clip(relative_slowness_update, 
                                         -max_relative_update, 
                                         max_relative_update)
        
        # 应用更新
        new_slowness = current_slowness + step_length * (relative_slowness_update * current_slowness)
        
        # 应用后验平滑
        if iteration > 0:  # 从第二次迭代开始应用
            # 计算当前更新量
            current_dm = new_slowness - current_slowness
            
            # 应用后验平滑，平滑强度随迭代次数增加
            smooth_strength = min(0.1 + 0.05 * iteration, 0.3)
            smoothed_dm = apply_post_smoothing(
                current_dm, 
                L_cal,
                grid,
                lambda_v=smooth_strength
            )
            
            # 更新慢度模型
            new_slowness = current_slowness + smoothed_dm
            
            logger.info(f"应用后验平滑，强度: {smooth_strength:.3f}")
        
        # 转换回速度并限制范围
        current_velocity = 1.0 / new_slowness
        current_velocity = np.clip(current_velocity, 1.0, 8.0)
        
        velocity_updates.append(current_velocity.copy())
        
        # 修改自适应步长和正则化参数的更新策略
        if iteration > 0:
            rms_ratio = rms / rms_history[-2]
            
            if rms_ratio > 1.05:  # RMS显著增加
                step_length = max(step_length * 0.6, min_step_length)  # 更温和的步长减小
                lambda_h = min(lambda_h * 1.3, lambda_h_range[1])     # 更温和的权重增加
                lambda_v = min(lambda_v * 1.3, lambda_v_range[1])
                damping_factor = min(damping_factor * 1.3, damping_range[1])
                logger.info(f"适度减小步长到: {step_length:.4f}")
                logger.info(f"适度增加正则化参数: λh={lambda_h:.4f}, λv={lambda_v:.4f}, damping={damping_factor:.4f}")
            
            elif rms_ratio > 1.0:  # RMS轻微增加
                step_length = max(step_length * 0.8, min_step_length)
                lambda_h = min(lambda_h * 1.1, lambda_h_range[1])
                lambda_v = min(lambda_v * 1.1, lambda_v_range[1])
                damping_factor = min(damping_factor * 1.1, damping_range[1])
                logger.info(f"轻微减小步长到: {step_length:.4f}")
                logger.info(f"轻微增加正则化参数: λh={lambda_h:.4f}, λv={lambda_v:.4f}, damping={damping_factor:.4f}")
            
            elif rms_ratio < 0.95:  # RMS显著改善
                step_length = min(step_length * 1.2, 0.8)  # 允许更大的步长增加
                lambda_h = max(lambda_h * 0.8, lambda_h_range[0])    # 更大幅度减小权重
                lambda_v = max(lambda_v * 0.8, lambda_v_range[0])
                damping_factor = max(damping_factor * 0.8, damping_range[0])
                logger.info(f"显著增大步长到: {step_length:.4f}")
                logger.info(f"显著减小正则化参数: λh={lambda_h:.4f}, λv={lambda_v:.4f}, damping={damping_factor:.4f}")
        
        # 检查收敛条件
        if iteration >= 2:  # 至少进行3次迭代后才检查收敛
            rms_change = abs(rms - rms_history[-2]) / rms_history[-2]
            if rms_change < 0.005 and max_update < 0.001:  # 更严格的收敛条件
                logger.info("达到收敛条件：RMS变化和更新量都很小")
                break
            elif iteration > 5 and all(rms > min(rms_history[:-1]) for _ in range(5)):  # 连续5次未改善
                logger.info("达到收敛条件：连续多次迭代未能改善RMS")
                # 回退到最佳模型
                best_iteration = np.argmin(rms_history)
                current_velocity = velocity_updates[best_iteration]
                break
    
    # 绘制结果
    # 创建2x3的网格，但调整比例使左侧占据更多空间
    fig = plt.figure(figsize=(24, 16))  # 增加整体尺寸
    gs = plt.GridSpec(2, 2, width_ratios=[2, 1])  # 设置2x2的网格，左侧列宽度是右侧的2倍

    # 左侧两个模型图
    # 1. 真实模型
    ax1 = fig.add_subplot(gs[0, 0])
    im0 = ax1.imshow(true_velocity.T, 
                    extent=[xmin, xmax, zmax, zmin],
                    cmap='jet')
    ax1.set_aspect(5)  # 调整纵横比
    plt.colorbar(im0, ax=ax1, label='Velocity (km/s)')
    # 绘制射线路径
    for ray in all_rays_obs:
        if ray is not None and len(ray) >= 2:
            ray_km = ray  # 转换为千米
            ax1.plot(ray_km[:, 0], ray_km[:, 1], 'w-', linewidth=0.5, alpha=0.3)
    # 绘制震源和接收点
    ax1.plot(sources[:, 0], sources[:, 1], 'r^', label='Sources', markersize=8)
    ax1.plot(receivers[:, 0], receivers[:, 1], 'bv', label='Receivers', markersize=6)
    ax1.set_title('True Model with Ray Paths')
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Z (km)')
    ax1.legend()

    # 2. 反演结果
    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(current_velocity.T, 
                    extent=[xmin, xmax, zmax, zmin],
                    cmap='jet')
    ax2.set_aspect(5)  # 调整纵横比
    plt.colorbar(im2, ax=ax2, label='Velocity (km/s)')
    # 获取最后一次迭代的射线路径
    final_rays = []
    for src in sources:
        _, rays, _ = grid.raytrace(src.reshape(1, 2), receivers, return_rays=True, compute_L=True)
        final_rays.extend(rays)
    # 绘制最终射线路径
    for ray in final_rays:
        if ray is not None and len(ray) >= 2:
            ray_km = ray
            ax2.plot(ray_km[:, 0], ray_km[:, 1], 'w-', linewidth=0.5, alpha=0.3)
    # 绘制震源和接收点
    ax2.plot(sources[:, 0], sources[:, 1], 'r^', label='Sources', markersize=8)
    ax2.plot(receivers[:, 0], receivers[:, 1], 'bv', label='Receivers', markersize=6)
    ax2.set_title('Inverted Model with Ray Paths')
    ax2.set_xlabel('X (km)')
    ax2.set_ylabel('Z (km)')
    ax2.legend()

    # 右侧两个图
    # 1. RMS历史
    ax3 = fig.add_subplot(gs[0, 1])
    ax3.plot(range(1, len(rms_history) + 1), rms_history, 'b-o')
    ax3.set_title('RMS Misfit History')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('RMS (s)')
    ax3.grid(True)

    # 2. 最大更新量历史
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(range(1, len(max_update_history) + 1), max_update_history, 'r-o')
    ax4.set_title('Max Update History')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Max Update')
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig('test_compute_L_inversion.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 输出最终统计信息
    logger.info("\n反演结果统计:")
    logger.info(f"初始RMS: {rms_history[0]:.6f} s")
    logger.info(f"最终RMS: {rms_history[-1]:.6f} s")
    logger.info(f"迭代次数: {len(rms_history)}")
    logger.info(f"速度范围: {current_velocity.min():.2f} - {current_velocity.max():.2f} km/s")
    
    return current_velocity, rms_history

if __name__ == '__main__':
    solve() 