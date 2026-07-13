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
        # 添加中部低速异常体
        x_center1 = nx // 3
        z_center1 = nz // 4
        radius1 = min(nx, nz) // 6
        for i in range(nx):
            for j in range(nz):
                dist = np.sqrt((i - x_center1)**2 + (j - z_center1)**2)
                if dist < radius1:
                    velocity[i, j] *= 1.5  # 降低20%的速度

    
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

def test_compute_L():
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
        n_threads=5,  # 使用单线程以便于调试
        cell_slowness=True,
        method='SPM',
        aniso='iso'
    )
    
    # 创建更多的源接收点配置以提高覆盖度
    n_sources = 20  # 增加震源数量
    n_receivers = 200  # 增加接收点数量
    sources = []
    receivers = []
    
    # 在表面均匀分布震源
    for src_x in np.linspace(0, xmax_m, n_sources):
        sources.append([src_x, 0.])
    sources = np.array(sources)
    
    # 在表面均匀分布接收点,扩大范围
    for rcv_x in np.linspace(10, 480, n_receivers):
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
        tt, rays, L = grid.raytrace(src.reshape(1, 2), receivers, return_rays=True, compute_L=True)
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
    max_iterations = 5  # 增加最大迭代次数
    step_length = 0.5  # 增大初始步长
    min_step_length = 0.1  # 最小步长限制
    
    # 构建平滑算子
    def build_smoothing_operator(nx, nz):
        """构建平滑算子
        
        Args:
            nx: x方向网格数
            nz: z方向网格数
            
        Returns:
            D: 平滑算子矩阵
        """
        # 计算总的网格点数
        n_total = nx * nz
        
        # 计算水平和垂直方向差分的数量
        n_diff_x = (nx - 1) * nz      # 水平方向差分数量
        n_diff_z = nx * (nz - 1)      # 垂直方向差分数量
        n_diff_total = n_diff_x + n_diff_z  # 总差分数量
        
        # 初始化差分矩阵
        D = np.zeros((n_diff_total, n_total))
        
        # 填充水平方向差分
        k = 0
        for i in range(nx-1):
            for j in range(nz):
                idx = i * nz + j
                D[k, idx] = -1
                D[k, idx + nz] = 1
                k += 1
                
        # 填充垂直方向差分
        for i in range(nx):
            for j in range(nz-1):
                idx = i * nz + j
                D[k, idx] = -1
                D[k, idx + 1] = 1
                k += 1
                
        return D
    
    # 构建平滑正则化矩阵
    D = build_smoothing_operator(nx-1, nz-1)
    logger.info(f"平滑算子矩阵维度: {D.shape}")
    
    # 记录反演历史
    rms_history = []
    velocity_updates = []
    max_update_history = []
    
    # 初始化权重和参数
    lambda_reg_init = 0.05  # 初始正则化参数
    lambda_reg = lambda_reg_init
    damping_factor = 0.01  # 初始阻尼因子
    
    # 非均匀正则化权重
    def build_depth_weights(nx, nz):
        """构建深度相关的权重"""
        z = np.linspace(0, 1, nz)
        depth_weights = 1.0 + 2.0 * z  # 随深度增加权重
        return np.tile(depth_weights, nx).reshape(nx, nz)
    
    def build_sensitivity_weights(L_matrix):
        """基于L矩阵计算灵敏度权重"""
        # 计算每个模型参数的灵敏度
        sensitivity = np.array(L_matrix.power(2).sum(axis=0)).flatten()
        # 归一化灵敏度
        sensitivity = sensitivity / np.max(sensitivity)
        # 转换为权重：高灵敏度区域权重小，低灵敏度区域权重大
        weights = 1.0 / (sensitivity + 0.1)  # 添加小量避免除零
        return weights.reshape(nx-1, nz-1)
    
    def update_adaptive_weights(current_weights, dm, current_velocity, iteration):
        """基于更新量更新自适应权重"""
        if iteration == 0:
            return current_weights
            
        # 计算相对更新量
        relative_update = np.abs(dm / current_velocity)
        
        # 更新量较大的区域降低权重，较小的区域增加权重
        update_scale = 1.0 / (1.0 + 5.0 * relative_update)
        
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
    
    for iteration in range(max_iterations):
        # 设置当前模型
        current_slowness = 1.0 / current_velocity
        grid.set_slowness(current_slowness)
        
        # 收集所有震源的计算数据
        all_tt_cal = []
        all_L_cal = []
        
        for src in sources:
            tt, rays, L = grid.raytrace(src.reshape(1, 2), receivers, return_rays=True, compute_L=True)
            all_tt_cal.extend(tt)
            all_L_cal.append(L)
        
        tt_cal = np.array(all_tt_cal)
        L_cal = vstack(all_L_cal)
        
        # 更新灵敏度权重
        sensitivity_weights = build_sensitivity_weights(L_cal)
        
        # 分析L矩阵
        analyze_L_matrix(L_cal, iteration, logger)
        
        # 计算走时残差
        dt = tt_obs - tt_cal
        rms = np.sqrt(np.mean(dt**2))
        rms_history.append(rms)
        
        logger.info(f"\n迭代 {iteration + 1}:")
        logger.info(f"RMS = {rms:.6f} s")
        
        # 更新自适应权重
        if iteration > 0:
            adaptive_weights = update_adaptive_weights(
                adaptive_weights, 
                dm,  # 上一次的更新量
                current_velocity, 
                iteration
            )
            
        # 组合所有权重
        combined_weights = depth_weights * sensitivity_weights * adaptive_weights
        
        # 将权重转换为对角矩阵形式
        W = np.diag(combined_weights.flatten())
        
        # 构建增广系统，加入非均匀正则化和权重
        A = vstack([
            L_cal,  # 射线矩阵
            np.sqrt(lambda_reg) * (D @ W),  # 加权平滑约束
            np.sqrt(damping_factor) * np.diag(combined_weights.flatten())  # 深度加权阻尼
        ])
        
        b = np.concatenate([
            dt,  # 走时残差
            np.zeros(D.shape[0]),  # 平滑约束右端项
            np.zeros(combined_weights.size)  # 阻尼约束右端项
        ])
        
        # 求解
        dm, istop, itn, r1norm = lsqr(A, b, iter_lim=1000, atol=1e-8, btol=1e-8)[:4]
        
        # 更新模型
        dm = dm.reshape(nx-1, nz-1)
        max_update = np.max(np.abs(dm))
        max_update_history.append(max_update)
        logger.info(f"最大模型更新量: {max_update:.6f}")
        
        # 自适应步长和正则化参数
        if iteration > 0:
            if rms > rms_history[-2]:
                step_length = max(step_length * 0.7, min_step_length)
                lambda_reg = min(lambda_reg * 1.5, lambda_reg_init * 2.0)  # 增加正则化
                logger.info(f"减小步长到: {step_length}, 增加正则化参数到: {lambda_reg}")
            elif rms < rms_history[-2] * 0.95:  # 如果RMS显著改善
                step_length = min(step_length * 1.1, 0.8)
                lambda_reg = max(lambda_reg * 0.8, lambda_reg_init * 0.1)  # 减小正则化
                logger.info(f"增大步长到: {step_length}, 减小正则化参数到: {lambda_reg}")
        
        # 约束更新幅度
        max_relative_update = 0.7  # 最大相对更新量
        
        # 将慢度更新量转换为速度更新
        current_slowness = 1.0 / current_velocity
        dm_slowness = dm  # dm已经是慢度更新量
        
        # 约束慢度更新的相对变化
        relative_slowness_update = dm_slowness / current_slowness
        relative_slowness_update = np.clip(relative_slowness_update, -max_relative_update, max_relative_update)
        
        # 应用更新
        new_slowness = current_slowness + step_length * (relative_slowness_update * current_slowness)
        
        # 转换回速度并限制范围
        current_velocity = 1.0 / new_slowness
        current_velocity = np.clip(current_velocity, 1.0, 8.0)
        
        velocity_updates.append(current_velocity.copy())
        
        # 检查收敛
        if (rms < 0.002 and iteration > 5) or (max_update < 0.001 and iteration > 5):
            logger.info("达到收敛条件")
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
    test_compute_L() 