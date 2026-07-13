import os
import numpy as np
import matplotlib.pyplot as plt
import pytest
from pyAOBS.modeling.rayinvr.models import VelocityModel, PhaseType
from pyAOBS.model_building.zeltform import ZeltVelocityModel2d

pytestmark = pytest.mark.integration

def test_velocity_model():
    """测试速度模型的基本功能"""
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # 创建输出目录
    output_dir = os.path.join(project_root, 'test_outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 从v.in文件初始化模型
    model_file = os.path.join(project_root, 'v.in')  # v.in应该在项目根目录
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"找不到速度模型文件: {model_file}")
        
    zelt_model = ZeltVelocityModel2d(model_file)
    velocity_model = VelocityModel(zelt_model)
    
    # 测试1: 获取模型边界
    x_range, z_range = velocity_model.get_model_bounds()
    print("模型边界:")
    print(f"X范围: {x_range}")
    print(f"Z范围: {z_range}")
    
    # 测试2: 获取速度值
    test_points = [
        (0.0, 0.0),  # 原点
        (x_range[0], z_range[0]),  # 左上角
        ((x_range[0] + x_range[1])/2, (z_range[0] + z_range[1])/2),  # 中心点
    ]
    
    print("\n速度值测试:")
    for x, z in test_points:
        try:
            v = velocity_model.get_velocity(x, z)
            print(f"位置 ({x:.2f}, {z:.2f}): {v:.2f} km/s")
        except ValueError as e:
            print(f"位置 ({x:.2f}, {z:.2f}): {str(e)}")
            
    # 测试3: 获取层索引
    print("\n层索引测试:")
    test_depths = [0.0, 5.0, 10.0, 15.0]
    for z in test_depths:
        try:
            layer_idx = velocity_model.get_layer_index(z)
            print(f"深度 {z:.2f} km: 第 {layer_idx} 层")
        except ValueError as e:
            print(f"深度 {z:.2f} km: {str(e)}")
            
    # 测试4: 绘制速度模型
    plt.figure(figsize=(12, 8))
    ax = velocity_model.plot_model()
    plt.title("Velocity Model")
    plt.savefig(os.path.join(output_dir, 'velocity_model.png'))
    plt.close()
    
    # 测试5: 生成速度剖面
    x_profile = np.linspace(x_range[0], x_range[1], 100)
    z_profile = np.linspace(z_range[0], z_range[1], 100)
    
    # 水平剖面
    plt.figure(figsize=(12, 6))
    z_test = 5.0  # 在5km深度
    v_profile = []
    for x in x_profile:
        try:
            v = velocity_model.get_velocity(x, z_test)
            v_profile.append(v)
        except ValueError:
            v_profile.append(np.nan)
    plt.plot(x_profile, v_profile)
    plt.title(f"Horizontal Velocity Profile at {z_test:.2f} km")
    plt.xlabel("Distance (km)")
    plt.ylabel("Velocity (km/s)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'horizontal_profile.png'))
    plt.close()
    
    # 垂直剖面
    plt.figure(figsize=(6, 12))
    x_test = (x_range[0] + x_range[1])/2  # 在中心位置
    v_profile = []
    for z in z_profile:
        try:
            v = velocity_model.get_velocity(x_test, z)
            v_profile.append(v)
        except ValueError:
            v_profile.append(np.nan)
    plt.plot(v_profile, z_profile)
    plt.title(f"Vertical Velocity Profile at {x_test:.2f} km")
    plt.xlabel("Velocity (km/s)")
    plt.ylabel("Depth (km)")
    plt.gca().invert_yaxis()  # 反转Y轴使深度向下增加
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'vertical_profile.png'))
    plt.close()
    
    # 测试6: 生成2D速度网格
    nx, nz = 100, 100
    x_grid = np.linspace(x_range[0], x_range[1], nx)
    z_grid = np.linspace(z_range[0], z_range[1], nz)
    X, Z = np.meshgrid(x_grid, z_grid)
    V = np.zeros_like(X)
    
    for i in range(nz):
        for j in range(nx):
            try:
                V[i,j] = velocity_model.get_velocity(X[i,j], Z[i,j])
            except ValueError:
                V[i,j] = np.nan
                
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(X, Z, V, shading='auto', cmap='jet')
    plt.colorbar(label='Velocity (km/s)')
    plt.title("2D Velocity Distribution")
    plt.xlabel("Distance (km)")
    plt.ylabel("Depth (km)")
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(output_dir, 'velocity_2d.png'))
    plt.close()

if __name__ == "__main__":
    test_velocity_model() 