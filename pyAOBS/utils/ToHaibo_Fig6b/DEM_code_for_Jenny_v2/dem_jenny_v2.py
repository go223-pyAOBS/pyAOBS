"""
DEM (Differential Effective Medium) Model - Jenny_v2 Python Implementation
============================================================================

完全复刻Jenny_v2的MATLAB实现，独立于其他Python代码。

基于Berryman (1980)的DEM理论，使用Critical Porosity缩放。

作者: 基于T. Mukerji的MATLAB代码转换
"""

import numpy as np
from scipy.integrate import ode
import warnings


def dem(k1, mu1, k2, mu2, asp, phic):
    """
    DEM - Effective elastic moduli using Differential Effective Medium formulation.
    
    参数:
        k1, mu1: 背景材料的体积模量和剪切模量 (GPa)
        k2, mu2: 包含物的体积模量和剪切模量 (GPa)
        asp: 纵横比（椭球短轴/长轴）
            <1 for oblate spheroids (扁椭球，如裂缝)
            >1 for prolate spheroids (长椭球)
        phic: 临界孔隙度（Critical Porosity），用于修改的DEM模型
            =1 for usual DEM (标准DEM)
    
    返回:
        k: 有效体积模量数组 (GPa)
        mu: 有效剪切模量数组 (GPa)
        por: 实际孔隙度数组（por = phic * t，其中t是积分变量）
    
    注意:
        - 积分变量t的范围是0到0.99999
        - 实际孔隙度por = phic * t
        - 对于修改的DEM，phase two是临界相，POR是实际孔隙度
    """
    # 创建ODE求解器
    # 使用Runge-Kutta 4/5阶方法（dopri5），匹配MATLAB的ode45
    solver = ode(demyprime)
    solver.set_integrator('dopri5', atol=1e-10, rtol=1e-10)
    
    # 设置初始条件
    y0 = np.array([k1, mu1])
    solver.set_initial_value(y0, 0.0)
    
    # 设置参数（通过f_params传递）
    solver.set_f_params(k1, mu1, k2, mu2, asp, phic)
    
    # 积分范围：从0到0.99999
    t_final = 0.99999
    dt = 0.001  # 输出步长
    
    # 存储结果
    t_list = [0.0]
    k_list = [k1]
    mu_list = [mu1]
    
    # 积分
    while solver.successful() and solver.t < t_final:
        solver.integrate(min(solver.t + dt, t_final))
        if solver.successful():
            t_list.append(solver.t)
            k_list.append(solver.y[0])
            mu_list.append(solver.y[1])
    
    # 转换为numpy数组
    t_array = np.array(t_list)
    k_array = np.real(np.array(k_list))  # 取实部
    mu_array = np.real(np.array(mu_list))  # 取实部
    
    # 计算实际孔隙度：por = phic * t
    por_array = phic * t_array
    
    return k_array, mu_array, por_array


def demyprime(t, y, k1, mu1, k2, mu2, asp, phic):
    """
    微分方程定义，用于DEM模型
    
    参数:
        t: 积分变量（0到0.99999）
        y: [k, mu] - 当前的有效体积模量和剪切模量
        k1, mu1: 背景材料的体积模量和剪切模量
        k2, mu2: 包含物的体积模量和剪切模量
        asp: 纵横比
        phic: 临界孔隙度
    
    返回:
        yprime: [dk/dt, dmu/dt] - 模量对积分变量t的导数
    """
    # 计算临界相的模量（虽然计算了，但实际不使用）
    krc = k1 * k2 / ((1 - phic) * k2 + phic * k1)
    murc = mu1 * mu2 / ((1 - phic) * mu2 + phic * mu1)
    
    # 实际使用的包含物模量：直接使用k2, mu2
    ka = k2
    mua = mu2
    # 注意：MATLAB代码中注释掉了 ka=krc; mua=murc;
    
    k = y[0]
    mu = y[1]
    
    # 如果asp=1，强制设为0.99避免奇点
    if abs(asp - 1.0) < 1e-10:
        asp = 0.99
    
    # 计算theta和fn（形状因子）
    if asp < 1.0:
        # 扁椭球（oblate spheroid）
        sqrt_term = np.sqrt(1 - asp**2)
        theta = (asp / ((1 - asp**2)**(3/2))) * (np.arccos(asp) - asp * sqrt_term)
        fn = (asp**2 / (1 - asp**2)) * (3 * theta - 2)
    elif asp > 1.0:
        # 长椭球（prolate spheroid）
        sqrt_term = np.sqrt(asp**2 - 1)
        theta = (asp / ((asp**2 - 1)**(3/2))) * (asp * sqrt_term - np.arccosh(asp))
        fn = (asp**2 / (asp**2 - 1)) * (2 - 3 * theta)
    else:
        # asp == 1.0 (应该已经被设为0.99)
        theta = 1.0
        fn = 0.0
    
    # 计算泊松比和r
    nu = (3 * k - 2 * mu) / (2 * (3 * k + mu))
    r = (1 - 2 * nu) / (2 * (1 - nu))
    
    # 计算a和b
    a = mua / mu - 1
    b = (1.0 / 3.0) * (ka / k - mua / mu)
    
    # 计算F1-F9（完整的Eshelby张量）
    f1a = 1 + a * ((3.0/2.0) * (fn + theta) - r * ((3.0/2.0) * fn + (5.0/2.0) * theta - (4.0/3.0)))
    
    # F2的计算：分两步（与MATLAB代码完全一致）
    f2a = 1 + a * (1 + (3.0/2.0) * (fn + theta) - (r/2.0) * (3 * fn + 5 * theta)) + b * (3 - 4 * r)
    f2a = f2a + (a/2.0) * (a + 3 * b) * (3 - 4 * r) * (fn + theta - r * (fn - theta + 2 * theta**2))
    
    f3a = 1 + a * (1 - (fn + (3.0/2.0) * theta) + r * (fn + theta))
    
    f4a = 1 + (a/4.0) * (fn + 3 * theta - r * (fn - theta))
    
    f5a = a * (-fn + r * (fn + theta - (4.0/3.0))) + b * theta * (3 - 4 * r)
    
    f6a = 1 + a * (1 + fn - r * (fn + theta)) + b * (1 - theta) * (3 - 4 * r)
    
    f7a = 2 + (a/4.0) * (3 * fn + 9 * theta - r * (3 * fn + 5 * theta)) + b * theta * (3 - 4 * r)
    
    f8a = a * (1 - 2 * r + (fn/2.0) * (r - 1) + (theta/2.0) * (5 * r - 3)) + b * (1 - theta) * (3 - 4 * r)
    
    f9a = a * ((r - 1) * fn - r * theta) + b * theta * (3 - 4 * r)
    
    # 计算P和Q（几何因子）
    pa = 3 * f1a / f2a
    qa = (2.0 / f3a) + (1.0 / f4a) + ((f4a * f5a + f6a * f7a - f8a * f9a) / (f2a * f4a))
    pa = pa / 3.0
    qa = qa / 5.0
    
    # 计算导数
    krhs = (ka - k) * pa
    murhs = (mua - mu) * qa
    
    # 微分方程：分母是(1-t)，不是(1-por)
    yprime = np.array([
        krhs / (1 - t),
        murhs / (1 - t)
    ])
    
    return yprime


def moduli_to_velocities(K, mu, rho):
    """
    从弹性模量计算速度
    
    参数:
        K, mu: 体积模量和剪切模量 (GPa)
        rho: 密度 (g/cm³)
    
    返回:
        Vp: P波速度 (km/s)
        Vs: S波速度 (km/s)
        VpVs_ratio: Vp/Vs比值
    """
    # 转换为SI单位
    rho_kg_m3 = rho * 1000  # g/cm³ → kg/m³
    K_Pa = K * 1e9  # GPa → Pa
    mu_Pa = mu * 1e9  # GPa → Pa
    
    # 计算速度 (m/s)
    Vp = np.sqrt((K_Pa + (4.0/3.0) * mu_Pa) / rho_kg_m3)
    Vs = np.sqrt(mu_Pa / rho_kg_m3)
    
    # 转换为 km/s
    Vp = Vp / 1000
    Vs = Vs / 1000
    
    # 计算Vp/Vs比值（避免除以0）
    if Vs > 0.0001:
        VpVs_ratio = Vp / Vs
    else:
        VpVs_ratio = np.nan
        warnings.warn('Division by zero attempted in Vp/Vs calculation')
    
    return Vp, Vs, VpVs_ratio


def vpvs_to_moduli(Vp, Vs, rho):
    """
    从速度计算弹性模量
    
    参数:
        Vp: P波速度 (km/s)
        Vs: S波速度 (km/s)
        rho: 密度 (g/cm³)
    
    返回:
        K: 体积模量 (GPa)
        mu: 剪切模量 (GPa)
    """
    # 转换为SI单位
    rho_kg_m3 = rho * 1000  # g/cm³ → kg/m³
    Vp_ms = Vp * 1000  # km/s → m/s
    Vs_ms = Vs * 1000  # km/s → m/s
    
    # 计算剪切模量
    mu_Pa = rho_kg_m3 * Vs_ms**2
    mu = mu_Pa * 1e-9  # Pa → GPa
    
    # 计算体积模量
    K_Pa = rho_kg_m3 * (Vp_ms**2 - (4.0/3.0) * Vs_ms**2)
    K = K_Pa * 1e-9  # Pa → GPa
    
    return K, mu


if __name__ == "__main__":
    """
    主程序 - 复刻job_dem_main.m
    """
    import matplotlib.pyplot as plt
    
    # 参数设置（basalt示例）
    k1 = 79.23  # 背景材料的体积模量 (GPa)
    mu1 = 47.61  # 背景材料的剪切模量 (GPa)
    k2 = 2.32  # 包含物的体积模量 (GPa，水)
    mu2 = 0.0  # 包含物的剪切模量 (GPa，水为0)
    
    # 注意：MATLAB代码中asp是[10000,1000,...]，但传入dem函数时使用1/asp(j)
    # 所以实际纵横比alpha = 1/asp
    asp_values = [10000, 1000, 500, 200, 150, 100, 75, 50, 33, 20]
    alpha_values = [1.0 / a for a in asp_values]  # 实际纵横比
    
    phic = 0.65  # 临界孔隙度
    rho1 = 2.946  # 背景密度 (g/cm³)
    rho2 = 1.03  # 包含物密度 (g/cm³，水)
    
    # 创建图形
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))
    
    # 对每个纵横比进行计算
    for j, asp in enumerate(asp_values):
        alpha = 1.0 / asp  # 实际纵横比
        
        # 调用DEM函数
        k, mu, por = dem(k1, mu1, k2, mu2, alpha, phic)
        
        # 计算密度
        rho = (1 - por) * rho1 + por * rho2
        
        # 绘制模量 vs 孔隙度
        axes1[0].plot(por, k, '-', label=f'k (α={alpha:.4f})')
        axes1[0].plot(por, mu, '--', label=f'mu (α={alpha:.4f})')
        axes1[1].plot(por, rho, '-', label=f'rho (α={alpha:.4f})')
        
        # 计算速度（仅对por < 0.4）
        por_cal = 0.4
        mask = por < por_cal
        
        if np.any(mask):
            Vp = np.zeros(len(por[mask]))
            Vs = np.zeros(len(por[mask]))
            VpVs_ratio = np.zeros(len(por[mask]))
            
            for i in range(len(por[mask])):
                Vp[i], Vs[i], VpVs_ratio[i] = moduli_to_velocities(
                    k[mask][i], mu[mask][i], rho[mask][i]
                )
            
            # 绘制速度 vs 孔隙度
            axes2[0].plot(por[mask], Vp, '-', label=f'α={alpha:.4f}')
            axes2[1].plot(por[mask], Vs, '-', label=f'α={alpha:.4f}')
            axes2[2].plot(por[mask], VpVs_ratio, '-', label=f'α={alpha:.4f}')
            
            # 绘制Vp vs Vp/Vs
            ax3.plot(Vp, VpVs_ratio, 'o', label=f'α={alpha:.4f}')
    
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
    plt.show()
    
    print("\n计算完成！")
    print("=" * 80)
    print("Jenny_v2 DEM模型 - Python实现")
    print("=" * 80)
    print(f"背景材料: K1={k1} GPa, μ1={mu1} GPa, ρ1={rho1} g/cm³")
    print(f"包含物: K2={k2} GPa, μ2={mu2} GPa, ρ2={rho2} g/cm³")
    print(f"临界孔隙度: phic={phic}")
    print(f"纵横比数量: {len(asp_values)}")
    print("=" * 80)
