import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from initial_model import InitialModelGenerator
import logging
from scipy.ndimage import gaussian_filter
import multiprocessing as mp
import ttcrpy.rgrid as rg
from typing import List, Tuple

# 设置多进程启动方法为spawn
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

def create_training_model(nx: int, nz: int) -> np.ndarray:
    """创建用于训练的合成速度模型，只包含单个高速异常体
    
    Args:
        nx: x方向网格数
        nz: z方向网格数
        
    Returns:
        velocity: 速度模型 (km/s)
    """
    # 创建基础速度场（速度随深度增加）
    z = np.linspace(0, 1, nz)
    v0 = 2.0  # 固定表层速度
    grad = 3.0  # 固定速度梯度
    base_velocity = v0 + grad * z[:, np.newaxis]
    velocity = np.tile(base_velocity, (1, nx)).T
    
    # 添加单个高速异常体
    x_center = np.random.randint(nx//4, 3*nx//4)
    z_center = np.random.randint(nz//4, 3*nz//4)
    radius = min(nx, nz) // 8  # 固定大小的异常体
    velocity_factor = 1.3  # 固定30%的速度增加
    
    for i in range(nx):
        for j in range(nz):
            dist = np.sqrt((i - x_center)**2 + (j - z_center)**2)
            if dist < radius:
                velocity[i, j] *= velocity_factor
    
    return velocity

def create_test_model(nx: int, nz: int) -> np.ndarray:
    """创建用于测试的真实速度模型，只包含单个高速异常体
    
    Args:
        nx: x方向网格数
        nz: z方向网格数
        
    Returns:
        velocity: 速度模型 (km/s)
    """
    # 创建基础速度场（速度随深度增加）
    z = np.linspace(0, 1, nz)
    v0 = 2.0  # 固定表层速度
    grad = 3.0  # 固定速度梯度
    base_velocity = v0 + grad * z[:, np.newaxis]
    velocity = np.tile(base_velocity, (1, nx)).T
    
    # 添加单个高速异常体（位置固定）
    x_center = nx // 2
    z_center = nz // 3
    radius = min(nx, nz) // 8
    for i in range(nx):
        for j in range(nz):
            dist = np.sqrt((i - x_center)**2 + (j - z_center)**2)
            if dist < radius:
                velocity[i, j] *= 1.3  # 增加30%的速度
                
    return velocity

def generate_ray_paths(nx: int, nz: int, n_sources: int, n_receivers: int, velocity: np.ndarray) -> tuple:
    """使用raytrace生成射线路径配置
    
    Args:
        nx: x方向网格数（单元数）
        nz: z方向网格数（单元数）
        n_sources: 震源数量
        n_receivers: 每个震源对应的接收点数量
        velocity: 速度模型（定义在单元中心）
        
    Returns:
        sources: 震源位置数组
        receivers: 接收点位置数组
        ray_paths: 射线路径列表
        travel_times: 走时数组
    """
    # 创建网格（注意：网格点数比单元数多1）
    x = np.linspace(0, nx, nx+1)  # nx+1个节点
    z = np.linspace(0, nz, nz+1)  # nz+1个节点
    grid = rg.Grid2d(x=x, z=z, 
                     n_threads=6, 
                     cell_slowness=True,  # 慢度定义在单元中心
                     method='SPM')
    
    # 设置慢度场（确保大小与单元数匹配）
    slowness = 1.0 / velocity  # velocity的形状应该是(nx, nz)
    grid.set_slowness(slowness.flatten())  # ttcrpy期望一维数组
    
    # 在表面均匀分布震源（使用节点坐标）
    # 确保震源位置在网格范围内，并留出边界空间
    source_x = np.linspace(2, nx-2, n_sources)  # 留出2个网格点的边界保护
    sources = np.array([[x, 0.0] for x in source_x])
    
    # 为每个震源创建接收点
    all_receivers = []
    all_ray_paths = []
    all_travel_times = []
    
    # 设置边界保护区域
    border_protection = 2  # 距离边界的最小网格点数
    min_offset = 5  # 震源和接收点之间的最小距离
    
    for src in sources:
        # 计算接收点范围，确保在网格内部且避开边界
        # 默认在震源右侧布置接收点
        x_min = min(nx-border_protection, src[0] + min_offset)
        x_max = min(nx-border_protection, src[0] + nx/4)  # 限制最大偏移距为1/4网格
        
        # 如果右侧空间不足，则在左侧布置接收点
        if x_max - x_min < min_offset:
            x_max = max(border_protection, src[0] - min_offset)
            x_min = max(border_protection, src[0] - nx/4)
            rcv_x = np.linspace(x_max, x_min, n_receivers)  # 注意交换顺序
        else:
            rcv_x = np.linspace(x_min, x_max, n_receivers)
        
        # 创建接收点数组
        receivers = np.array([[x, 0.0] for x in rcv_x])
        
        # 确保源和接收点数组的形状和类型正确
        src_reshaped = src.reshape(1, 2).astype(np.float64)
        receivers = receivers.astype(np.float64)
        
        # 使用raytrace计算走时和射线路径
        tt, rays = grid.raytrace(src_reshaped, 
                                receivers, 
                                return_rays=True, 
                                compute_L=False)
        
        # 确保tt是一维数组
        if isinstance(tt, np.ndarray) and tt.ndim > 1:
            tt = tt.flatten()
        
        all_receivers.extend(receivers)
        all_ray_paths.extend(rays)
        all_travel_times.extend(tt)
    
    return (np.array(sources), 
            np.array(all_receivers), 
            all_ray_paths, 
            np.array(all_travel_times))

def evaluate_model(true_velocity: np.ndarray, 
                  predicted_velocity: np.ndarray,
                  output_dir: str = 'real_case_outputs'):
    """评估模型重建效果
    
    Args:
        true_velocity: 真实速度模型
        predicted_velocity: 预测的速度模型
        output_dir: 输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. 计算相对误差
    rel_error = np.abs(predicted_velocity - true_velocity) / true_velocity
    mean_rel_error = np.mean(rel_error) * 100  # 转换为百分比
    
    # 2. 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 真实模型
    im0 = axes[0, 0].imshow(true_velocity.T, cmap='jet', aspect='auto')
    axes[0, 0].set_title('True Velocity Model')
    plt.colorbar(im0, ax=axes[0, 0], label='Velocity (km/s)')
    
    # 预测模型
    im1 = axes[0, 1].imshow(predicted_velocity.T, cmap='jet', aspect='auto')
    axes[0, 1].set_title('Predicted Velocity Model')
    plt.colorbar(im1, ax=axes[0, 1], label='Velocity (km/s)')
    
    # 相对误差
    im2 = axes[1, 0].imshow(rel_error.T, cmap='hot', aspect='auto')
    axes[1, 0].set_title(f'Relative Error (Mean: {mean_rel_error:.2f}%)')
    plt.colorbar(im2, ax=axes[1, 0], label='Relative Error')
    
    # 速度剖面对比
    mid_x = true_velocity.shape[0] // 2
    axes[1, 1].plot(true_velocity[mid_x, :], 'b-', label='True')
    axes[1, 1].plot(predicted_velocity[mid_x, :], 'r--', label='Predicted')
    axes[1, 1].set_title('Velocity Profile at Mid-X')
    axes[1, 1].set_xlabel('Depth')
    axes[1, 1].set_ylabel('Velocity (km/s)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    plt.close()
    
    # 3. 输出统计信息
    stats = {
        'mean_relative_error': mean_rel_error,
        'max_relative_error': np.max(rel_error) * 100,
        'rmse': np.sqrt(np.mean((predicted_velocity - true_velocity)**2)),
        'velocity_range_true': [np.min(true_velocity), np.max(true_velocity)],
        'velocity_range_pred': [np.min(predicted_velocity), np.max(predicted_velocity)]
    }
    
    return stats

def generate_dataset(n_samples: int, nx: int, nz: int, n_sources: int, n_receivers_per_source: int):
    """生成训练或验证数据集
    
    Args:
        n_samples: 样本数量
        nx: x方向网格数
        nz: z方向网格数
        n_sources: 震源数量
        n_receivers_per_source: 每个震源对应的接收点数量
        
    Returns:
        dataset: 数据集列表，每个元素为(travel_times, ray_paths, sources, receivers, velocity)
    """
    dataset = []
    for i in range(n_samples):
        # 使用训练模型生成器创建随机的合成模型
        velocity = create_training_model(nx-1, nz-1)
        # 生成射线路径和走时
        sources, receivers, ray_paths, travel_times = generate_ray_paths(
            nx-1, nz-1, n_sources, n_receivers_per_source, velocity
        )
        
        # 添加随机噪声
        noise_level = np.random.uniform(0.001, 0.003)
        noise = np.random.normal(0, noise_level * np.mean(travel_times), travel_times.shape)
        travel_times += noise
        
        dataset.append((travel_times, ray_paths, sources, receivers, velocity))
        print("generate dataset success",i)
    return dataset

def save_dataset(dataset: List[Tuple], filename: str):
    """保存数据集到文件
    
    Args:
        dataset: 数据集列表
        filename: 保存的文件名
    """
    # 创建保存目录
    save_dir = 'saved_datasets'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 将数据转换为可保存的格式
    save_data = []
    for travel_times, ray_paths, sources, receivers, velocity in dataset:
        # 将射线路径转换为列表格式（因为numpy数组可能长度不同）
        ray_paths_list = [ray.tolist() if ray is not None else None for ray in ray_paths]
        data_item = {
            'travel_times': travel_times,
            'ray_paths': ray_paths_list,
            'sources': sources,
            'receivers': receivers,
            'velocity': velocity
        }
        save_data.append(data_item)
    
    # 保存数据
    save_path = os.path.join(save_dir, filename)
    np.save(save_path, save_data, allow_pickle=True)
    print(f"数据集已保存到: {save_path}")

def load_dataset(filename: str) -> List[Tuple]:
    """从文件加载数据集
    
    Args:
        filename: 数据文件名
        
    Returns:
        dataset: 数据集列表
    """
    # 构建文件路径
    save_dir = 'saved_datasets'
    load_path = os.path.join(save_dir, filename)
    
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"找不到数据文件: {load_path}")
    
    # 加载数据
    save_data = np.load(load_path, allow_pickle=True)
    
    # 转换回原始格式
    dataset = []
    for data_item in save_data:
        # 将射线路径转换回numpy数组
        ray_paths = [np.array(ray) if ray is not None else None 
                    for ray in data_item['ray_paths']]
        dataset.append((
            data_item['travel_times'],
            ray_paths,
            data_item['sources'],
            data_item['receivers'],
            data_item['velocity']
        ))
    
    print(f"已加载数据集: {load_path}")
    return dataset

def test_model_recovery(generator, nx: int, nz: int, n_sources: int, n_receivers: int):
    """测试模型是否能够恢复训练样本
    
    Args:
        generator: 训练好的模型生成器
        nx: x方向网格数
        nz: z方向网格数
        n_sources: 震源数量
        n_receivers: 接收点数量
    """
    logger = logging.getLogger(__name__)
    logger.info("\n开始测试模型恢复能力...")
    
    # 注意：create_training_model和generate_ray_paths使用的是nx-1和nz-1
    nx_model = nx - 1
    nz_model = nz - 1
    
    # 从训练数据生成一个样本
    test_velocity = create_training_model(nx_model, nz_model)
    
    # 生成射线路径和走时数据
    sources, receivers, ray_paths, travel_times = generate_ray_paths(
        nx_model, nz_model, n_sources, n_receivers, test_velocity
    )
    
    # 使用模型进行预测
    predicted_velocity = generator.generate(
        travel_times=travel_times,
        ray_paths=ray_paths,
        source_positions=sources,
        receiver_positions=receivers
    )
    
    # 调整预测结果的尺寸以匹配输入模型
    if predicted_velocity.shape != test_velocity.shape:
        # 如果预测结果是49x49，我们需要裁剪到48x48
        predicted_velocity = predicted_velocity[:-1, :-1]
    
    # 确保预测结果和真实模型具有相同的形状
    assert test_velocity.shape == predicted_velocity.shape, \
        f"Shape mismatch: true={test_velocity.shape}, pred={predicted_velocity.shape}"
    
    # 评估结果
    stats = evaluate_model(
        test_velocity, 
        predicted_velocity,
        output_dir='recovery_test_outputs'
    )
    
    # 输出评估结果
    logger.info("\n训练样本恢复测试结果:")
    logger.info(f"平均相对误差: {stats['mean_relative_error']:.2f}%")
    logger.info(f"最大相对误差: {stats['max_relative_error']:.2f}%")
    logger.info(f"均方根误差: {stats['rmse']:.4f} km/s")
    
    return stats

def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 设置CUDA线程数
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.set_num_threads(4)
    
    # 模型参数 - 注意这里的nx和nz是网格点数，实际单元数是nx-1和nz-1
    nx, nz = 49, 49  # 这样实际的速度场大小将是48x48
    n_sources = 40
    n_receivers_per_source = 80
    
    # 数据集参数
    n_training_samples = 1000  # 增加训练样本数
    n_validation_samples = 100  # 增加验证样本数
    
    # 检查是否存在保存的数据集
    train_data_file = 'train_data.npy'
    valid_data_file = 'valid_data.npy'
    
    try:
        # 尝试加载已保存的数据集
        logger.info("尝试加载已保存的数据集...")
        train_data = load_dataset(train_data_file)
        valid_data = load_dataset(valid_data_file)
        logger.info("成功加载已保存的数据集")
    except FileNotFoundError:
        # 如果找不到保存的数据集，则重新生成
        logger.info("未找到保存的数据集，开始生成新的数据集...")
        
        # 生成训练集和验证集
        train_data = generate_dataset(
            n_training_samples, nx, nz, n_sources, n_receivers_per_source
        )
        valid_data = generate_dataset(
            n_validation_samples, nx, nz, n_sources, n_receivers_per_source
        )
        
        # 保存数据集
        logger.info("保存数据集以供将来使用...")
        save_dataset(train_data, train_data_file)
        save_dataset(valid_data, valid_data_file)
    
    # 2. 初始化模型生成器
    logger.info("初始化模型生成器...")
    generator = InitialModelGenerator(nx, nz)
    
    # 3. 尝试加载预训练模型，如果失败则重新训练
    model_path = 'my_models/final_model.pth'  # 修改为final_model.pth
    try:
        logger.info(f"尝试加载预训练模型: {model_path}")
        generator = InitialModelGenerator(nx=nx, nz=nz, model_path=model_path)
        logger.info("成功加载预训练模型")
    except FileNotFoundError:
        logger.info("未找到预训练模型，开始训练新模型...")
        # 训练模型
        generator.train(
            train_data=train_data,
            valid_data=valid_data,
            n_epochs=300,  # 减少训练轮数
            batch_size=32,  # 增加批次大小
            learning_rate=1e-4,  # 略微增加学习率
            save_dir='my_models'
        )
        # 训练完成后，加载最终模型
        generator = InitialModelGenerator(nx=nx, nz=nz, model_path=model_path)
        logger.info("已加载训练完成的模型")

    # 加载或训练模型后，首先测试模型的恢复能力
    recovery_stats = test_model_recovery(
        generator, 
        nx, 
        nz, 
        n_sources, 
        n_receivers_per_source
    )
    
    # 如果恢复测试结果不理想，提前警告
    if recovery_stats['mean_relative_error'] > 5.0:  # 设置一个合理的阈值
        logger.warning("\n警告：模型无法很好地恢复训练样本，可能需要重新训练或调整参数")
        logger.warning("建议检查：1) 训练是否充分 2) 模型结构是否合适 3) 数据生成是否合理")
    
    # 4. 创建测试用的真实模型
    logger.info("创建测试用的真实速度模型...")
    true_velocity = create_test_model(nx-1, nz-1)  # 注意这里使用nx-1和nz-1
    
    # 5. 生成测试数据
    logger.info("生成射线路径配置和走时数据...")
    sources, receivers, ray_paths, travel_times = generate_ray_paths(
        nx-1, nz-1, n_sources, n_receivers_per_source, true_velocity  # 注意这里使用nx-1和nz-1
    )
    
    # 添加实际噪声水平的噪声
    noise = np.random.normal(0, 0.002 * np.mean(travel_times), travel_times.shape)
    travel_times += noise
    
    # 6. 使用加载的模型进行预测
    logger.info("生成速度模型...")
    predicted_velocity = generator.generate(
        travel_times=travel_times,
        ray_paths=ray_paths,
        source_positions=sources,
        receiver_positions=receivers
    )
    
    # 调整预测结果的尺寸以匹配输入模型
    if predicted_velocity.shape != true_velocity.shape:
        predicted_velocity = predicted_velocity[:-1, :-1]
    
    # 7. 评估结果
    logger.info("评估模型重建效果...")
    stats = evaluate_model(true_velocity, predicted_velocity)
    
    # 8. 输出评估结果
    logger.info("\n模型评估结果:")
    logger.info(f"平均相对误差: {stats['mean_relative_error']:.2f}%")
    logger.info(f"最大相对误差: {stats['max_relative_error']:.2f}%")
    logger.info(f"均方根误差: {stats['rmse']:.4f} km/s")
    logger.info(f"真实速度范围: {stats['velocity_range_true'][0]:.2f} - {stats['velocity_range_true'][1]:.2f} km/s")
    logger.info(f"预测速度范围: {stats['velocity_range_pred'][0]:.2f} - {stats['velocity_range_pred'][1]:.2f} km/s")

if __name__ == '__main__':
    main() 