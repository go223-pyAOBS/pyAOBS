#!/usr/bin/env python
"""
测试RAYINVR主程序和射线追踪功能
这个简单脚本用于测试rayinvr_wrapper中的简化功能
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path
import ctypes

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# 导入RAYINVR包装器
try:
    from rayinvr_wrapper import RayinvrWrapper
    # print("成功导入RayinvrWrapper")
except ImportError as e:
    print(f"导入失败: {e}")
    sys.exit(1)

def read_velocity_model(file_path):
    """读取并解析v.in文件，以了解模型结构"""
    # print(f"\n读取模型文件: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        lines = [line.strip() for line in lines if line.strip()]
        
        # 第一行包含模型的x范围
        x_info = lines[0].split()
        if len(x_info) >= 2:
            try:
                x_min = float(x_info[1])
                x_max = float(x_info[2])
                # print(f"模型X范围: {x_min} - {x_max}")
            except:
                print(f"无法解析X范围: {lines[0]}")
        
        # 打印模型的前几行，帮助理解结构
        # print("\n模型前10行:")
        # for i, line in enumerate(lines[:min(10, len(lines))]):
        #     print(f"{i+1}: {line}")
        
    except Exception as e:
        print(f"读取模型文件出错: {e}")

def create_simple_model():
    """创建一个简单的测试模型文件"""
    # print("创建简单的测试速度模型文件: v.in")
    model_content = """1 0.00 100.00
0 0.00   0.00
         0      0
1 0.00 100.00
0 5.00   5.00
         0      0
1 0.00 100.00
0 1.50   1.50
         0      0
1 0.00 100.00
0 6.00   6.00
         0      0
"""
    with open("v.in", "w") as f:
        f.write(model_content)
    
    # 创建简单的r.in文件
    # print("创建简单的射线追踪参数文件: r.in")
    r_in_content = """&pltpar iplot=0, imod=1, iray=1, &end
&axepar xmin=0., xmax=100., zmin=0., zmax=20., &end
&trapar ishot=1, xshot=10., zshot=0., ray=1.0, nray=5, &end
&invpar &end
"""
    with open("r.in", "w") as f:
        f.write(r_in_content)
    
    # print("测试文件已创建")

def test_combined_functions():
    """综合测试：在同一个rayinvr实例中运行所有功能"""
    # print("\n" + "="*60)
    # print("开始综合测试：在同一个实例中运行并获取射线路径")
    # print("="*60)
    
    # 检查并创建必需的输入文件
    required_files = ['v.in', 'r.in']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"警告: 缺少以下必需文件: {', '.join(missing_files)}")
        # print("创建简单的测试文件...")
        create_simple_model()
    
    # 创建并保持单个RAYINVR实例
    rayinvr = RayinvrWrapper()
    
    # 1. 运行射线追踪
    # print("\n[步骤1] 运行RAYINVR主程序...")
    success = rayinvr.run_rayinvr()
    if not success:
        print("RAYINVR主程序执行失败")
        return None, None
    # print("RAYINVR主程序成功执行完成")
    
    # 检查输出文件
    for f in ['r1.out', 'p.out', 'tx.out']:
        if os.path.exists(f):
            # print(f"输出文件 {f} 已生成, 大小: {os.path.getsize(f)} 字节")
            pass

    # 3. 获取存储的射线
    # print("\n[步骤3] 获取存储的射线...")
    n_rays = rayinvr.get_ray_count()
    print(f"存储的射线数量: {n_rays}")
    if n_rays == 0:
        print("警告：没有找到存储的射线")
        return None, None
        
    rays = rayinvr.get_all_rays()
    obs_data = rayinvr.get_observed_data()    
    return rays, obs_data
    
    # print("\n综合测试完成")

def plot_multiple_rays(rays, obs_data):
    """绘制多条射线路径和走时曲线"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 设置颜色循环
    colors = plt.cm.jet(np.linspace(0, 1, len(rays)))
    
    # 绘制射线路径
    for i, ray in enumerate(rays):
        ax1.plot(ray['x'], ray['z'], color=colors[i], linewidth=1.5, 
                alpha=0.8, label=f"Ray #{i+1}")
        # 标记起点和终点
        ax1.scatter(ray['x'][0], ray['z'][0], color=colors[i], s=30, marker='o')
        ax1.scatter(ray['x'][-1], ray['z'][-1], color=colors[i], s=30, marker='x')
    
    ax1.invert_yaxis()  # 反转y轴使深度向下为正
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_title('Ray Path')
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Depth (km)')
    
    # 绘制走时曲线
    distances = [ray['x'][-1] for ray in rays]
    times = [ray['total_time'] for ray in rays]
    
    # 按距离排序
    idx = np.argsort(distances)
    sorted_distances = [distances[i] for i in idx]
    sorted_times = [times[i] for i in idx]
    
    # 绘制计算走时
    ax2.plot(sorted_distances, sorted_times, 'b-', linewidth=2, label='Calculated')
    ax2.scatter(sorted_distances, sorted_times, color='blue', s=40)
    
    # 绘制观测走时
    # 按相位分组绘制
    unique_phases = np.unique(obs_data['phase'])
    for phase in unique_phases:
        mask = obs_data['phase'] == phase
        ax2.errorbar(obs_data['x'][mask], obs_data['t'][mask], 
                    yerr=obs_data['u'][mask], fmt='o', color='red',
                    label=f'Observed (Phase {phase})', alpha=0.7,
                    markersize=5, capsize=3)
    
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_title('Traveltime Curve')
    ax2.set_xlabel('Distance (km)')
    ax2.set_ylabel('Traveltime (s)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('multiple_rays.png')
    plt.show()

if __name__ == "__main__":
    # 运行综合测试 - 所有功能在同一个rayinvr实例中执行
    print("第一次运行lib.rayinvr主程序")    
    rays, obs_data = test_combined_functions()

    print("第二次运行lib.rayinvr主程序")
    rays1, obs_data1 = test_combined_functions()
    print("第三次运行lib.rayinvr主程序")
    rays2, obs_data2 = test_combined_functions()   
    #plot_multiple_rays(rays1, obs_data1)
