"""
Differential Effective Medium (DEM) - 对比empirical_formulas.py和dem_jenny_v2.py
==================================================================================

对比使用调整后的empirical_formulas.py和直接使用dem_jenny_v2.py得到的结果
"""

import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
# 使用默认字体，避免字体找不到的问题
# plt.rcParams['font.family'] = 'arial'  # 注释掉，使用系统默认字体

# 导入两种实现
import sys
from pathlib import Path
import importlib.util

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# 方法1: 直接导入Jenny_v2的实现
jenny_v2_dir = current_dir / 'ToHaibo_Fig6b' / 'DEM_code_for_Jenny_v2'
jenny_v2_file = jenny_v2_dir / 'dem_jenny_v2.py'

spec = importlib.util.spec_from_file_location("dem_jenny_v2", jenny_v2_file)
dem_jenny_v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dem_jenny_v2)

# 使用Jenny_v2的函数
dem_jenny_v2_func = dem_jenny_v2.dem
moduli_to_velocities_jenny = dem_jenny_v2.moduli_to_velocities

# 方法2: 使用调整后的empirical_formulas.py
from .empirical_formulas import (
    calculate_dem_effective_moduli,
    calculate_velocity_from_moduli
)

print("\n" + "="*80)
print("DEM模型计算 - 对比empirical_formulas.py和dem_jenny_v2.py")
print("="*80)
print("\n使用Unaltered Peridotite参数")
print("方法1: 直接使用dem_jenny_v2.py")
print("方法2: 使用调整后的empirical_formulas.py")
print("="*80 + "\n")

# DEM建模参数（使用Jenny_v2的unaltered peridotite参数）
# 从job_dem_main.m中的注释部分获取
k1 = 122.73  # 背景材料的体积模量 (GPa) - unaltered peridotite
mu1 = 76.40  # 背景材料的剪切模量 (GPa) - unaltered peridotite
k2 = 2.32  # 包含物的体积模量 (GPa，水)
mu2 = 0.0  # 包含物的剪切模量 (GPa，水为0)

phic = 0.65  # Critical Porosity（与Jenny_v2一致）
rho1 = 3.316  # 背景密度 (g/cm³) - unaltered peridotite
rho2 = 1.03  # 包含物密度 (g/cm³，水)

# 从模量计算Vp和Vs（用于显示）
rho_kg_m3 = rho1 * 1000  # kg/m³
vp0 = np.sqrt((k1 * 1e9 + 4 * mu1 * 1e9 / 3) / rho_kg_m3) / 1000  # km/s
vs0 = np.sqrt(mu1 * 1e9 / rho_kg_m3) / 1000  # km/s

# 选择与Jenny_v2 peridotite示例相同的纵横比
# 注意：MATLAB代码中asp是[100,10,5,1]或[10000,1000,200,100,50,20]，但传入dem函数时使用1/asp(j)
# 所以实际纵横比alpha = 1/asp
# 使用更完整的纵横比列表以便更好的对比
asp_values = [10000, 1000, 500, 200, 150, 100, 75, 50, 33, 20, 10, 5, 1]
aspect_ratios = [1.0 / a for a in asp_values]  # 实际纵横比
aspect_ratio_labels = [f'α={alpha:.4f}' for alpha in aspect_ratios]

print(f"背景材料参数 (Unaltered Peridotite，与Jenny_v2一致):")
print(f"  K1 = {k1} GPa")
print(f"  μ1 = {mu1} GPa")
print(f"  密度 = {rho1} g/cm³")
print(f"  计算得到的 Vp = {vp0:.4f} km/s")
print(f"  计算得到的 Vs = {vs0:.4f} km/s")
print(f"  Vp/Vs = {vp0 / vs0:.4f}")
print(f"  包含物: K2 = {k2} GPa, μ2 = {mu2} GPa (水)")
print(f"  包含物密度: ρ2 = {rho2} g/cm³")
print(f"  临界孔隙度: phic = {phic}")
print(f"  纵横比数量: {len(aspect_ratios)}")
print(f"  纵横比: {aspect_ratios}\n")

# 存储对比结果
results_jenny_v2 = {}
results_empirical = {}
comparison_stats = {}

# 孔隙度范围用于对比
porosities = np.linspace(0, 0.4, 41)

# 对每个纵横比进行计算和对比
print("开始计算和对比...")
for j, asp in enumerate(asp_values):
    alpha = 1.0 / asp  # 实际纵横比
    label = f'α={alpha:.4f}'
    
    print(f"\n  计算纵横比 {label}...")
    
    # 方法1: 直接使用Jenny_v2的dem函数
    print(f"    方法1: 使用dem_jenny_v2.py...")
    k_jenny_array, mu_jenny_array, por_jenny = dem_jenny_v2_func(k1, mu1, k2, mu2, alpha, phic)
    
    # 方法2: 使用empirical_formulas.py
    print(f"    方法2: 使用empirical_formulas.py...")
    K_empirical = []
    mu_empirical = []
    
    for phi in porosities:
        K_eff, mu_eff = calculate_dem_effective_moduli(
            porosity=phi,
            aspect_ratio=alpha,
            host_vp=vp0,
            host_vs=vs0,
            host_density=rho1,
            inclusion_k=k2,
            inclusion_mu=mu2,
            n_steps=100,
            critical_porosity=phic
        )
        K_empirical.append(K_eff)
        mu_empirical.append(mu_eff)
    
    K_empirical = np.array(K_empirical)
    mu_empirical = np.array(mu_empirical)
    
    # 插值Jenny_v2的结果到相同的孔隙度数组
    from scipy.interpolate import interp1d
    if len(por_jenny) > 1 and np.all(np.diff(por_jenny) > 0):
        K_jenny_interp = interp1d(por_jenny, k_jenny_array, kind='linear', 
                                  fill_value=(k_jenny_array[0], k_jenny_array[-1]), 
                                  bounds_error=False)
        mu_jenny_interp = interp1d(por_jenny, mu_jenny_array, kind='linear', 
                                   fill_value=(mu_jenny_array[0], mu_jenny_array[-1]), 
                                   bounds_error=False)
        K_jenny = K_jenny_interp(porosities)
        mu_jenny = mu_jenny_interp(porosities)
    else:
        K_jenny = np.interp(porosities, por_jenny, k_jenny_array)
        mu_jenny = np.interp(porosities, por_jenny, mu_jenny_array)
    
    # 计算速度
    rho_array = (1.0 - porosities) * rho1 + porosities * rho2
    
    Vp_jenny = np.zeros(len(porosities))
    Vs_jenny = np.zeros(len(porosities))
    VpVs_jenny = np.zeros(len(porosities))
    
    Vp_empirical = np.zeros(len(porosities))
    Vs_empirical = np.zeros(len(porosities))
    VpVs_empirical = np.zeros(len(porosities))
    
    for i in range(len(porosities)):
        # Jenny_v2方法
        Vp_jenny[i], Vs_jenny[i], VpVs_jenny[i] = moduli_to_velocities_jenny(
            K_jenny[i], mu_jenny[i], rho_array[i]
        )
        # empirical_formulas方法
        Vp_empirical[i], Vs_empirical[i] = calculate_velocity_from_moduli(
            K_empirical[i], mu_empirical[i], rho_array[i]
        )
        if Vs_empirical[i] > 0.0001:
            VpVs_empirical[i] = Vp_empirical[i] / Vs_empirical[i]
        else:
            VpVs_empirical[i] = np.nan
    
    # 存储结果
    results_jenny_v2[alpha] = {
        'K': K_jenny,
        'mu': mu_jenny,
        'Vp': Vp_jenny,
        'Vs': Vs_jenny,
        'VpVs': VpVs_jenny,
        'por': por_jenny,
        'k': k_jenny_array,
        'mu_orig': mu_jenny_array
    }
    
    results_empirical[alpha] = {
        'K': K_empirical,
        'mu': mu_empirical,
        'Vp': Vp_empirical,
        'Vs': Vs_empirical,
        'VpVs': VpVs_empirical
    }
    
    # 计算差异统计
    K_diff = np.abs(K_jenny - K_empirical)
    mu_diff = np.abs(mu_jenny - mu_empirical)
    Vp_diff = np.abs(Vp_jenny - Vp_empirical)
    Vs_diff = np.abs(Vs_jenny - Vs_empirical)
    
    comparison_stats[alpha] = {
        'K_max_diff': np.max(K_diff),
        'K_mean_diff': np.mean(K_diff),
        'K_relative_max': np.max(K_diff / (K_jenny + 1e-10)) * 100,
        'mu_max_diff': np.max(mu_diff),
        'mu_mean_diff': np.mean(mu_diff),
        'mu_relative_max': np.max(mu_diff / (mu_jenny + 1e-10)) * 100,
        'Vp_max_diff': np.max(Vp_diff),
        'Vp_mean_diff': np.mean(Vp_diff),
        'Vs_max_diff': np.max(Vs_diff),
        'Vs_mean_diff': np.mean(Vs_diff)
    }
    
    # 输出对比统计
    print(f"\n    ===== 对比统计 ({label}) =====")
    print(f"    K模量差异: 最大={comparison_stats[alpha]['K_max_diff']:.6f} GPa, "
          f"平均={comparison_stats[alpha]['K_mean_diff']:.6f} GPa, "
          f"最大相对误差={comparison_stats[alpha]['K_relative_max']:.4f}%")
    print(f"    μ模量差异: 最大={comparison_stats[alpha]['mu_max_diff']:.6f} GPa, "
          f"平均={comparison_stats[alpha]['mu_mean_diff']:.6f} GPa, "
          f"最大相对误差={comparison_stats[alpha]['mu_relative_max']:.4f}%")
    print(f"    Vp差异: 最大={comparison_stats[alpha]['Vp_max_diff']:.6f} km/s, "
          f"平均={comparison_stats[alpha]['Vp_mean_diff']:.6f} km/s")
    print(f"    Vs差异: 最大={comparison_stats[alpha]['Vs_max_diff']:.6f} km/s, "
          f"平均={comparison_stats[alpha]['Vs_mean_diff']:.6f} km/s")
    print(f"    ============================================\n")

# 创建对比图形
print("\n绘制对比结果...")

# 创建图形（与Jenny_v2一致）
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))

# 对每个纵横比进行绘图
for j, asp in enumerate(asp_values):
    alpha = 1.0 / asp  # 实际纵横比
    label = f'α={alpha:.4f}'
    
    # 获取结果
    K_jenny = results_jenny_v2[alpha]['K']
    mu_jenny = results_jenny_v2[alpha]['mu']
    Vp_jenny = results_jenny_v2[alpha]['Vp']
    Vs_jenny = results_jenny_v2[alpha]['Vs']
    VpVs_jenny = results_jenny_v2[alpha]['VpVs']
    
    K_empirical = results_empirical[alpha]['K']
    mu_empirical = results_empirical[alpha]['mu']
    Vp_empirical = results_empirical[alpha]['Vp']
    Vs_empirical = results_empirical[alpha]['Vs']
    VpVs_empirical = results_empirical[alpha]['VpVs']
    
    # 计算密度
    rho_array = (1.0 - porosities) * rho1 + porosities * rho2
    
    # 绘制模量 vs 孔隙度（两种方法）
    axes1[0].plot(porosities, K_jenny, '-', label=f'K_jenny (α={alpha:.4f})', linewidth=2)
    axes1[0].plot(porosities, K_empirical, '--', label=f'K_empirical (α={alpha:.4f})', linewidth=1.5)
    axes1[0].plot(porosities, mu_jenny, '-', label=f'μ_jenny (α={alpha:.4f})', linewidth=2)
    axes1[0].plot(porosities, mu_empirical, '--', label=f'μ_empirical (α={alpha:.4f})', linewidth=1.5)
    axes1[1].plot(porosities, rho_array, '-', label=f'rho (α={alpha:.4f})')
    
    # 计算速度（仅对por < 0.4）
    por_cal = 0.4
    mask = porosities < por_cal
    
    if np.any(mask):
        # 绘制速度 vs 孔隙度（两种方法）
        axes2[0].plot(porosities[mask], Vp_jenny[mask], '-', label=f'Vp_jenny (α={alpha:.4f})', linewidth=2)
        axes2[0].plot(porosities[mask], Vp_empirical[mask], '--', label=f'Vp_empirical (α={alpha:.4f})', linewidth=1.5)
        axes2[1].plot(porosities[mask], Vs_jenny[mask], '-', label=f'Vs_jenny (α={alpha:.4f})', linewidth=2)
        axes2[1].plot(porosities[mask], Vs_empirical[mask], '--', label=f'Vs_empirical (α={alpha:.4f})', linewidth=1.5)
        axes2[2].plot(porosities[mask], VpVs_jenny[mask], '-', label=f'VpVs_jenny (α={alpha:.4f})', linewidth=2)
        axes2[2].plot(porosities[mask], VpVs_empirical[mask], '--', label=f'VpVs_empirical (α={alpha:.4f})', linewidth=1.5)
        
        # 绘制Vp vs Vp/Vs（两种方法）
        ax3.plot(Vp_jenny[mask], VpVs_jenny[mask], 'o', label=f'jenny (α={alpha:.4f})', markersize=4)
        ax3.plot(Vp_empirical[mask], VpVs_empirical[mask], 'x', label=f'empirical (α={alpha:.4f})', markersize=3)

# 设置图形1（模量和密度）
axes1[0].set_xlabel('Fractional Porosity')
axes1[0].set_ylabel('Effective Moduli (GPa)')
axes1[0].set_title('Elastic moduli vs Porosity')
axes1[0].legend(loc='best', fontsize=8)
axes1[0].grid(True, alpha=0.3)

axes1[1].set_xlabel('Fractional Porosity')
axes1[1].set_ylabel('Density (g/cm³)')
axes1[1].set_title('Density vs Porosity')
axes1[1].set_xticks(np.arange(0, 0.45, 0.05))
axes1[1].set_yticks(np.arange(2.0, 3.6, 0.1))
axes1[1].set_ylim([2.0, 3.5])
axes1[1].set_xlim([0, 0.4])
axes1[1].legend(loc='best', fontsize=8)
axes1[1].grid(True, alpha=0.3)

# 设置图形2（速度）
axes2[0].set_xlabel('Fractional Porosity')
axes2[0].set_ylabel('Vp (km/s)')
axes2[0].set_title('Vp vs Porosity')
axes2[0].set_xticks(np.arange(0, 0.45, 0.05))
axes2[0].set_yticks(np.arange(3.0, 8.7, 0.2))
axes2[0].set_ylim([3.0, 8.6])
axes2[0].legend(loc='best', fontsize=8)
axes2[0].grid(True, alpha=0.3)

axes2[1].set_xlabel('Fractional Porosity')
axes2[1].set_ylabel('Vs (km/s)')
axes2[1].set_title('Vs vs Porosity')
axes2[1].legend(loc='best', fontsize=8)
axes2[1].grid(True, alpha=0.3)

axes2[2].set_xlabel('Fractional Porosity')
axes2[2].set_ylabel('Vp/Vs ratio')
axes2[2].set_title('Vp/Vs vs Porosity')
axes2[2].set_ylim([1.6, 2.15])
axes2[2].legend(loc='best', fontsize=8)
axes2[2].grid(True, alpha=0.3)

# 设置图形3（Vp vs Vp/Vs）
ax3.set_xlabel('Vp (km/s)')
ax3.set_ylabel('Vp/Vs ratio')
ax3.set_title('Vp vs Vp/Vs Ratio')
ax3.set_xlim([3, 8.5])
ax3.set_ylim([1.6, 2.15])
ax3.legend(loc='best', fontsize=8)
ax3.grid(True, alpha=0.3)

plt.tight_layout()

# 保存图片
output_file1 = current_dir / 'dem_comparison_fig1_moduli_density.png'
output_file2 = current_dir / 'dem_comparison_fig2_velocities.png'
output_file3 = current_dir / 'dem_comparison_fig3_vp_vpvs.png'
fig1.savefig(output_file1, dpi=150, bbox_inches='tight')
fig2.savefig(output_file2, dpi=150, bbox_inches='tight')
fig3.savefig(output_file3, dpi=150, bbox_inches='tight')
print(f"\n图片已保存到:")
print(f"  图1 (模量和密度): {output_file1}")
print(f"  图2 (速度): {output_file2}")
print(f"  图3 (Vp vs Vp/Vs): {output_file3}")

# 输出总体对比统计
print("\n" + "="*80)
print("总体对比统计")
print("="*80)
all_K_max_diff = [stats['K_max_diff'] for stats in comparison_stats.values()]
all_mu_max_diff = [stats['mu_max_diff'] for stats in comparison_stats.values()]
all_Vp_max_diff = [stats['Vp_max_diff'] for stats in comparison_stats.values()]
all_Vs_max_diff = [stats['Vs_max_diff'] for stats in comparison_stats.values()]

print(f"所有纵横比的K模量最大差异: {np.max(all_K_max_diff):.6f} GPa")
print(f"所有纵横比的μ模量最大差异: {np.max(all_mu_max_diff):.6f} GPa")
print(f"所有纵横比的Vp最大差异: {np.max(all_Vp_max_diff):.6f} km/s")
print(f"所有纵横比的Vs最大差异: {np.max(all_Vs_max_diff):.6f} km/s")
print("="*80)

print("\n" + "="*80)
print("计算完成！")
print("="*80)
print("DEM模型 - 对比empirical_formulas.py和dem_jenny_v2.py")
print("="*80)
print(f"背景材料: K1={k1} GPa, μ1={mu1} GPa, ρ1={rho1} g/cm³")
print(f"包含物: K2={k2} GPa, μ2={mu2} GPa, ρ2={rho2} g/cm³")
print(f"临界孔隙度: phic={phic}")
print(f"纵横比数量: {len(asp_values)}")
print("="*80)

# 显示图片（如果可能）
try:
    plt.show()
except:
    print("\n注意: 无法显示交互式窗口，图片已保存到文件")
