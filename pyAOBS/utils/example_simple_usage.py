"""
简单使用示例 - 演示如何使用简化的岩石分类接口

这个示例展示了最简单的使用方式，只需提供速度数据即可完成岩石分类。
"""

import numpy as np
from pyAOBS.utils import classify_velocity_model, SimpleRockClassifier

def example_1_single_point():
    """示例1: 分类单个点"""
    print("=" * 60)
    print("示例1: 分类单个点")
    print("=" * 60)
    
    # 最简单的方式：只提供P波速度
    rock_type = classify_velocity_model(vp=6.1)
    print(f"只提供Vp=6.1 km/s: {rock_type}")
    
    # 提供更多信息，提高精度
    rock_type = classify_velocity_model(vp=6.1, vs=3.5, density=2.65)
    print(f"提供Vp=6.1, Vs=3.5, density=2.65: {rock_type}")
    
    print()


def example_2_velocity_model():
    """示例2: 分类速度模型"""
    print("=" * 60)
    print("示例2: 分类速度模型")
    print("=" * 60)
    
    # 创建速度模型数据
    vp = np.array([6.1, 6.5, 7.5, 8.1])  # P波速度 (km/s)
    vs = np.array([3.5, 3.7, 4.1, 4.5])  # S波速度 (km/s)
    depth = np.array([0, 10, 20, 30])    # 深度 (km)
    
    # 一键分类
    results = classify_velocity_model(
        vp=vp,
        vs=vs,
        depth=depth
    )
    
    print("分类结果:")
    print(results[['depth', 'vp', 'vs', 'rock_type', 'probability']])
    print()


def example_3_auto_estimation():
    """示例3: 自动估算缺失参数"""
    print("=" * 60)
    print("示例3: 只提供P波速度，自动估算其他参数")
    print("=" * 60)
    
    # 只提供P波速度和深度，系统会自动：
    # 1. 估算S波速度（使用Brocher公式）
    # 2. 估算密度（使用Gardner公式）
    # 3. 根据深度估算压力和温度
    # 4. 进行温度压力校正
    # 5. 进行岩石分类
    
    results = classify_velocity_model(
        vp=[6.1, 6.5, 7.5, 8.1],
        depth=[0, 10, 20, 30]
    )
    
    print("自动估算的结果:")
    print(results[['depth', 'vp', 'vs', 'density', 'rock_type']])
    print()


def example_4_with_correction():
    """示例4: 带温度压力校正的分类"""
    print("=" * 60)
    print("示例4: 带温度压力校正的分类")
    print("=" * 60)
    
    classifier = SimpleRockClassifier(auto_correct=True)
    
    # 提供温度和压力信息
    results = classifier.classify_model(
        vp=[6.1, 6.5, 7.5],
        vs=[3.5, 3.7, 4.1],
        density=[2.65, 2.9, 3.0],
        pressure=[100, 200, 300],      # 压力 (MPa)
        temperature=[20, 50, 100]      # 温度 (°C)
    )
    
    print("带校正的分类结果:")
    print(results[['vp', 'pressure', 'temperature', 'rock_type', 'probability']])
    print()


def example_5_from_file():
    """示例5: 从文件加载并分类"""
    print("=" * 60)
    print("示例5: 从文件加载并分类")
    print("=" * 60)
    
    # 创建示例数据文件
    import pandas as pd
    sample_data = pd.DataFrame({
        'depth': [0, 10, 20, 30],
        'Vp': [6.1, 6.5, 7.5, 8.1],
        'Vs': [3.5, 3.7, 4.1, 4.5],
        'density': [2.65, 2.9, 3.0, 3.3]
    })
    sample_data.to_csv('sample_velocity_model.csv', index=False)
    print("已创建示例文件: sample_velocity_model.csv")
    
    # 从文件分类
    from pyAOBS.utils import classify_from_file
    results = classify_from_file(
        'sample_velocity_model.csv',
        vp_column='Vp',
        vs_column='Vs',
        density_column='density',
        depth_column='depth',
        output_file='classified_results.csv'
    )
    
    print("\n分类结果:")
    print(results[['depth', 'Vp', 'rock_type', 'rock_probability']])
    print("结果已保存到: classified_results.csv")
    print()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("简化岩石分类接口使用示例")
    print("=" * 60 + "\n")
    
    # 运行所有示例
    example_1_single_point()
    example_2_velocity_model()
    example_3_auto_estimation()
    example_4_with_correction()
    example_5_from_file()
    
    print("=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
