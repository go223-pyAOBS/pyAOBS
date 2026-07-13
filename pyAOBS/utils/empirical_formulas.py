"""
empirical_formulas.py - Empirical formulas library for rock physics calculations

This module provides a unified collection of empirical formulas used in rock physics,
including density calculation, velocity conversion, pressure/temperature correction,
and elastic moduli calculation.

All formulas are based on published literature and have been verified for consistency
across the pyAOBS codebase.

Author: pyAOBS Team
Date: 2025
"""

import numpy as np
from typing import Optional, Union, Dict, Tuple
from dataclasses import dataclass


# ============================================================================
# Correction Parameters (Pressure and Temperature)
# ============================================================================

@dataclass
class CorrectionParameters:
    """温压校正参数
    
    这些参数与isrock.py中的CorrectionParameters保持一致。
    注意：温度系数alpha定义为负数，公式中使用减号。
    """
    # P波速度校正系数
    vp_pressure_beta: float = 0.0002  # 压力系数 (1/MPa)
    vp_temp_alpha: float = -0.50e-4   # 温度系数 (1/°C)，负数
    
    # S波速度校正系数
    vs_pressure_beta: float = 0.00015  # 压力系数 (1/MPa)
    vs_temp_alpha: float = -0.40e-4    # 温度系数 (1/°C)，负数
    
    # 密度校正系数（如果需要）
    density_pressure_beta: float = 0.0001  # 压力系数 (1/MPa)
    density_temp_alpha: float = -0.10e-4   # 温度系数 (1/°C)，负数


# 默认校正参数实例
DEFAULT_CORRECTION = CorrectionParameters()


# ============================================================================
# Density Calculation Formulas
# ============================================================================

def calculate_density_gardner(vp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Gardner公式计算密度 (Gardner et al., 1974)
    
    适用于沉积岩，特别是未固结和弱固结的沉积物。
    
    公式: ρ = 0.31 × (Vp in m/s)^0.25
    
    Args:
        vp: P波速度 (km/s)
        
    Returns:
        密度 (g/cm³)
        
    Reference:
        Gardner, G.H.F., Gardner, L.W., & Gregory, A.R. (1974). 
        Formation velocity and density—the diagnostic basics for stratigraphic traps. 
        Geophysics, 39(6), 770-780.
    """
    vp = np.asarray(vp)
    # 转换为m/s后计算
    density = 0.31 * (vp * 1000) ** 0.25
    return density if isinstance(vp, np.ndarray) else float(density)


def calculate_density_nafe_drake(vp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Nafe-Drake公式计算密度 (Nafe & Drake, 1963)
    
    适用于海洋沉积物，Vp范围: 1.5-8.5 km/s
    
    公式: ρ = 1.6612×Vp - 0.4721×Vp² + 0.0671×Vp³ - 0.0043×Vp⁴ + 0.000106×Vp⁵
    
    Args:
        vp: P波速度 (km/s)
        
    Returns:
        密度 (g/cm³)
        
    Reference:
        Nafe, J.E., & Drake, C.L. (1963). Physical properties of marine sediments. 
        In The Sea (Vol. 3, pp. 794-815).
    """
    vp = np.asarray(vp)
    density = (1.6612 * vp - 0.4721 * vp**2 + 0.0671 * vp**3 - 
               0.0043 * vp**4 + 0.000106 * vp**5)
    return density if isinstance(vp, np.ndarray) else float(density)


def calculate_density_brocher(vp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Brocher公式计算密度 (Brocher, 2005)
    
    基于Nafe-Drake曲线的多项式拟合，适用于地壳岩石。
    注意：此公式与Nafe-Drake公式相同，因为Brocher是基于Nafe-Drake曲线拟合的。
    
    公式: ρ = 1.6612×Vp - 0.4721×Vp² + 0.0671×Vp³ - 0.0043×Vp⁴ + 0.000106×Vp⁵
    
    Args:
        vp: P波速度 (km/s)
        
    Returns:
        密度 (g/cm³)
        
    Reference:
        Brocher, T.M. (2005). Empirical relations between elastic wavespeeds and 
        density in the Earth's crust. Bulletin of the Seismological Society of 
        America, 95(6), 2081-2092.
    """
    # 与Nafe-Drake公式相同
    return calculate_density_nafe_drake(vp)


def calculate_density_castagna(vp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Castagna公式计算密度
    
    公式: ρ = 1.66 × Vp^0.261
    
    Args:
        vp: P波速度 (km/s)
        
    Returns:
        密度 (g/cm³)
    """
    vp = np.asarray(vp)
    density = 1.66 * (vp ** 0.261)
    return density if isinstance(vp, np.ndarray) else float(density)


def calculate_density_lindseth(vp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Lindseth公式计算密度
    
    公式: ρ = 0.31 × Vp + 1.7
    
    Args:
        vp: P波速度 (km/s)
        
    Returns:
        密度 (g/cm³)
    """
    vp = np.asarray(vp)
    density = 0.31 * vp + 1.7
    return density if isinstance(vp, np.ndarray) else float(density)


def calculate_density(vp: Union[float, np.ndarray], 
                     method: str = 'gardner') -> Union[float, np.ndarray]:
    """
    统一的密度计算接口
    
    Args:
        vp: P波速度 (km/s)
        method: 计算方法，可选：
               'gardner' - Gardner公式 (适用于沉积岩)
               'nafe_drake' - Nafe-Drake公式 (适用于海洋沉积物)
               'brocher' - Brocher公式 (适用于地壳岩石，与Nafe-Drake相同)
               'castagna' - Castagna公式
               'lindseth' - Lindseth公式
               
    Returns:
        密度 (g/cm³)
        
    Raises:
        ValueError: 如果方法不支持
    """
    method = method.lower()
    
    if method == 'gardner':
        return calculate_density_gardner(vp)
    elif method == 'nafe_drake':
        return calculate_density_nafe_drake(vp)
    elif method == 'brocher':
        return calculate_density_brocher(vp)
    elif method == 'castagna':
        return calculate_density_castagna(vp)
    elif method == 'lindseth':
        return calculate_density_lindseth(vp)
    else:
        raise ValueError(f"不支持的密度计算方法: {method}. "
                         f"可选方法: 'gardner', 'nafe_drake', 'brocher', 'castagna', 'lindseth'")


# ============================================================================
# S-wave Velocity Calculation Formulas
# ============================================================================

def calculate_vs_brocher(vp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Brocher公式计算S波速度 (Brocher, 2005)
    
    适用于地壳岩石，Vp范围: 1.5-8.5 km/s
    
    公式: Vs = 0.7858 - 1.2344×Vp + 0.7949×Vp² - 0.1238×Vp³ + 0.0064×Vp⁴
    
    Args:
        vp: P波速度 (km/s)
        
    Returns:
        S波速度 (km/s)
        
    Reference:
        Brocher, T.M. (2005). Empirical relations between elastic wavespeeds and 
        density in the Earth's crust. Bulletin of the Seismological Society of 
        America, 95(6), 2081-2092.
    """
    vp = np.asarray(vp)
    vs = (0.7858 - 1.2344 * vp + 0.7949 * vp**2 - 
          0.1238 * vp**3 + 0.0064 * vp**4)
    return vs if isinstance(vp, np.ndarray) else float(vs)


def calculate_vs_castagna(vp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Castagna公式计算S波速度 (Castagna et al., 1985)
    
    适用于泥岩和页岩。
    
    公式: Vs = (Vp - 1.36) / 1.16
    
    Args:
        vp: P波速度 (km/s)
        
    Returns:
        S波速度 (km/s)
        
    Reference:
        Castagna, J.P., Batzle, M.L., & Eastwood, R.L. (1985). 
        Relationships between compressional-wave and shear-wave velocities in 
        clastic silicate rocks. Geophysics, 50(4), 571-581.
    """
    vp = np.asarray(vp)
    vs = (vp - 1.36) / 1.16
    return vs if isinstance(vp, np.ndarray) else float(vs)


def calculate_vs(vp: Union[float, np.ndarray], 
                 method: str = 'brocher') -> Union[float, np.ndarray]:
    """
    统一的S波速度计算接口
    
    Args:
        vp: P波速度 (km/s)
        method: 计算方法，可选：
               'brocher' - Brocher公式 (适用于地壳岩石)
               'castagna' - Castagna公式 (适用于泥岩)
               
    Returns:
        S波速度 (km/s)
        
    Raises:
        ValueError: 如果方法不支持
    """
    method = method.lower()
    
    if method == 'brocher':
        return calculate_vs_brocher(vp)
    elif method == 'castagna':
        return calculate_vs_castagna(vp)
    else:
        raise ValueError(f"不支持的S波速度计算方法: {method}. "
                         f"可选方法: 'brocher', 'castagna'")


# ============================================================================
# Inverse Calculation: Vp from Vs (Newton-Raphson Method)
# ============================================================================

def calculate_vp_from_vs_brocher(vs: Union[float, np.ndarray],
                                 max_iterations: int = 10,
                                 tolerance: float = 1e-6) -> Union[float, np.ndarray]:
    """
    从S波速度反算P波速度（使用Brocher公式的逆变换）
    
    使用Newton-Raphson迭代方法求解Brocher公式的逆变换。
    Brocher公式: Vs = 0.7858 - 1.2344×Vp + 0.7949×Vp² - 0.1238×Vp³ + 0.0064×Vp⁴
    
    Args:
        vs: S波速度 (km/s)
        max_iterations: 最大迭代次数，默认10
        tolerance: 收敛容差，默认1e-6
        
    Returns:
        P波速度 (km/s)
    """
    vs = np.asarray(vs)
    # 检查是否为标量或0维数组
    scalar_input = vs.ndim == 0 or (not isinstance(vs, np.ndarray))
    if scalar_input:
        # 如果是标量或0维数组，提取标量值并转换为1维数组
        if vs.ndim == 0:
            vs_val_scalar = float(vs.item())
        else:
            vs_val_scalar = float(vs)
        vs = np.array([vs_val_scalar])
    
    vp_result = np.zeros_like(vs)
    
    for i, vs_val in enumerate(vs):
        # 初始猜测：假设Vp/Vs = 1.732 (√3)
        vp_guess = vs_val * 1.732
        
        for iteration in range(max_iterations):
            # 计算当前Vp对应的Vs
            vs_calc = (0.7858 - 1.2344 * vp_guess + 0.7949 * vp_guess**2 - 
                      0.1238 * vp_guess**3 + 0.0064 * vp_guess**4)
            
            # 计算导数 dVs/dVp
            dvs_dvp = (-1.2344 + 2 * 0.7949 * vp_guess - 
                      3 * 0.1238 * vp_guess**2 + 4 * 0.0064 * vp_guess**3)
            
            # 检查导数是否太小（避免除零）
            if abs(dvs_dvp) < tolerance:
                break
            
            # Newton-Raphson迭代
            vp_guess = vp_guess - (vs_calc - vs_val) / dvs_dvp
            
            # 确保Vp > Vs（物理约束）
            if vp_guess < 0:
                vp_guess = vs_val * 1.732
            elif vp_guess < vs_val * 1.5:
                vp_guess = vs_val * 1.5
        
        vp_result[i] = max(vp_guess, vs_val * 1.5)  # 确保Vp > 1.5×Vs
    
    if scalar_input:
        return float(vp_result[0])
    return vp_result


# ============================================================================
# Pressure and Temperature Correction
# ============================================================================

def correct_velocity_pressure(velocity: Union[float, np.ndarray],
                              pressure: Union[float, np.ndarray],
                              target_pressure: float = 200.0,
                              is_s_wave: bool = False,
                              correction_params: Optional[CorrectionParameters] = None) -> Union[float, np.ndarray]:
    """
    速度的压力校正
    
    公式: V_corrected = V_original × (1 + β × ΔP)
    其中: ΔP = target_pressure - pressure
         β = 0.0002/MPa (P波) 或 0.00015/MPa (S波)
    
    Args:
        velocity: 原始速度值 (km/s)
        pressure: 原始压力值 (MPa)
        target_pressure: 目标压力值 (MPa)，默认200.0
        is_s_wave: 是否为S波速度
        correction_params: 校正参数，如果为None则使用默认值
        
    Returns:
        校正后的速度值 (km/s)
    """
    if correction_params is None:
        correction_params = DEFAULT_CORRECTION
    
    beta = correction_params.vs_pressure_beta if is_s_wave else correction_params.vp_pressure_beta
    delta_p = target_pressure - pressure
    
    velocity = np.asarray(velocity)
    pressure = np.asarray(pressure)
    delta_p = np.asarray(delta_p)
    
    v_corrected = velocity * (1 + beta * delta_p)
    
    if not isinstance(velocity, np.ndarray) and not isinstance(pressure, np.ndarray):
        return float(v_corrected)
    return v_corrected


def correct_velocity_temperature(velocity: Union[float, np.ndarray],
                                temperature: Union[float, np.ndarray],
                                target_temperature: float = 25.0,
                                is_s_wave: bool = False,
                                correction_params: Optional[CorrectionParameters] = None) -> Union[float, np.ndarray]:
    """
    速度的温度校正
    
    公式: V_corrected = V_original × (1 - α × ΔT)
    其中: ΔT = temperature - target_temperature
         α = -0.50e-4/°C (P波) 或 -0.40e-4/°C (S波) [负数]
    
    注意：alpha定义为负数，公式中使用减号。
    实际效果：1 - (-0.50e-4) × ΔT = 1 + 0.50e-4 × ΔT
    
    Args:
        velocity: 原始速度值 (km/s)
        temperature: 原始温度值 (°C)
        target_temperature: 目标温度值 (°C)，默认25.0
        is_s_wave: 是否为S波速度
        correction_params: 校正参数，如果为None则使用默认值
        
    Returns:
        校正后的速度值 (km/s)
    """
    if correction_params is None:
        correction_params = DEFAULT_CORRECTION
    
    alpha = correction_params.vs_temp_alpha if is_s_wave else correction_params.vp_temp_alpha
    delta_t = temperature - target_temperature
    
    velocity = np.asarray(velocity)
    temperature = np.asarray(temperature)
    delta_t = np.asarray(delta_t)
    
    v_corrected = velocity * (1 - alpha * delta_t)
    
    if not isinstance(velocity, np.ndarray) and not isinstance(temperature, np.ndarray):
        return float(v_corrected)
    return v_corrected


def correct_velocity(velocity: Union[float, np.ndarray],
                    pressure: Optional[Union[float, np.ndarray]] = None,
                    temperature: Optional[Union[float, np.ndarray]] = None,
                    target_pressure: float = 200.0,
                    target_temperature: float = 25.0,
                    is_s_wave: bool = False,
                    correction_params: Optional[CorrectionParameters] = None) -> Union[float, np.ndarray]:
    """
    速度的温压校正（组合校正）
    
    先进行压力校正，再进行温度校正。
    
    Args:
        velocity: 原始速度值 (km/s)
        pressure: 原始压力值 (MPa)，如果为None则跳过压力校正
        temperature: 原始温度值 (°C)，如果为None则跳过温度校正
        target_pressure: 目标压力值 (MPa)，默认200.0
        target_temperature: 目标温度值 (°C)，默认25.0
        is_s_wave: 是否为S波速度
        correction_params: 校正参数，如果为None则使用默认值
        
    Returns:
        校正后的速度值 (km/s)
    """
    v_corrected = velocity
    
    # 压力校正
    if pressure is not None:
        v_corrected = correct_velocity_pressure(
            v_corrected, pressure, target_pressure, is_s_wave, correction_params
        )
    
    # 温度校正
    if temperature is not None:
        v_corrected = correct_velocity_temperature(
            v_corrected, temperature, target_temperature, is_s_wave, correction_params
        )
    
    return v_corrected


# ============================================================================
# Elastic Moduli Calculation
# ============================================================================

def calculate_elastic_moduli(vp: Union[float, np.ndarray],
                            vs: Union[float, np.ndarray],
                            density: Union[float, np.ndarray]) -> Dict[str, Union[float, np.ndarray]]:
    """
    计算弹性模量
    
    公式:
    - 剪切模量: μ = ρ × Vs²
    - 拉梅常数λ: λ = ρ × (Vp² - 2×Vs²)
    - 体积模量: K = λ + 2×μ/3
    - 杨氏模量: E = μ × (3×λ + 2×μ) / (λ + μ)
    
    Args:
        vp: P波速度 (km/s)
        vs: S波速度 (km/s)
        density: 密度 (g/cm³)
        
    Returns:
        包含以下键的字典：
        - 'bulk_modulus': 体积模量 (GPa)
        - 'shear_modulus': 剪切模量 (GPa)
        - 'young_modulus': 杨氏模量 (GPa)
        - 'lame_lambda': 拉梅常数λ (GPa)
    """
    vp = np.asarray(vp) * 1000  # 转换为m/s
    vs = np.asarray(vs) * 1000
    rho = np.asarray(density) * 1000  # 转换为kg/m³
    
    # 计算弹性模量
    mu = rho * vs**2 * 1e-9  # 剪切模量 (GPa)
    lambda_ = rho * (vp**2 - 2 * vs**2) * 1e-9  # 拉梅常数λ (GPa)
    K = lambda_ + 2 * mu / 3  # 体积模量 (GPa)
    E = mu * (3 * lambda_ + 2 * mu) / (lambda_ + mu)  # 杨氏模量 (GPa)
    
    # 如果是标量输入，转换为Python标量
    if not isinstance(vp, np.ndarray):
        return {
            'bulk_modulus': float(K),
            'shear_modulus': float(mu),
            'young_modulus': float(E),
            'lame_lambda': float(lambda_)
        }
    
    return {
        'bulk_modulus': K,
        'shear_modulus': mu,
        'young_modulus': E,
        'lame_lambda': lambda_
    }


# ============================================================================
# Utility Functions
# ============================================================================

def calculate_vp_vs_ratio(vp: Union[float, np.ndarray],
                        vs: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    计算Vp/Vs比值
    
    Args:
        vp: P波速度 (km/s)
        vs: S波速度 (km/s)
        
    Returns:
        Vp/Vs比值
    """
    vp = np.asarray(vp)
    vs = np.asarray(vs)
    
    ratio = vp / vs
    return ratio if isinstance(vp, np.ndarray) or isinstance(vs, np.ndarray) else float(ratio)


def calculate_poisson_ratio(vp: Union[float, np.ndarray],
                           vs: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    计算泊松比
    
    公式: ν = ((Vp/Vs)² - 2) / (2 × ((Vp/Vs)² - 1))
    
    这是从Vp/Vs比值计算泊松比的标准公式。
    
    Args:
        vp: P波速度 (km/s)
        vs: S波速度 (km/s)
        
    Returns:
        泊松比（无量纲）
    """
    vp_vs_ratio = calculate_vp_vs_ratio(vp, vs)
    vp_vs_squared = vp_vs_ratio ** 2
    
    poisson = (vp_vs_squared - 2) / (2 * (vp_vs_squared - 1))
    return poisson if isinstance(vp, np.ndarray) or isinstance(vs, np.ndarray) else float(poisson)


def calculate_vp_vs_ratio_from_poisson(poisson: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    从泊松比计算Vp/Vs比值（泊松比和波速比的关系）
    
    公式: R = sqrt((2ν - 2) / (2ν - 1))
    其中 R = Vp/Vs, ν = 泊松比
    
    这是泊松比公式的逆变换：
    ν = (R² - 2) / (2(R² - 1))
    
    Args:
        poisson: 泊松比（无量纲）
        
    Returns:
        Vp/Vs比值
        
    Reference:
        标准弹性理论关系
    """
    poisson = np.asarray(poisson)
    
    # 检查物理约束：泊松比通常在 -1 到 0.5 之间
    # 对于大多数岩石，泊松比在 0.1 到 0.4 之间
    # 当 ν = 0.5 时，材料是不可压缩的，R → ∞
    # 当 ν < 0.5 时，公式有效
    
    # 避免除零和负数开方
    denominator = 2 * poisson - 1
    numerator = 2 * poisson - 2
    
    # 检查分母是否为零或负数
    if isinstance(poisson, np.ndarray):
        mask = (denominator <= 0) | (numerator <= 0)
        if mask.any():
            import warnings
            warnings.warn("Some Poisson ratio values are out of valid range (should be < 0.5)")
        ratio = np.sqrt(numerator / denominator)
        ratio[mask] = np.nan
    else:
        if denominator <= 0 or numerator <= 0:
            import warnings
            warnings.warn(f"Poisson ratio {poisson} is out of valid range (should be < 0.5)")
            return np.nan
        ratio = np.sqrt(numerator / denominator)
    
    return ratio if isinstance(poisson, np.ndarray) else float(ratio)


# ============================================================================
# Serpentinization Relationships
# ============================================================================

def calculate_serpentinization_from_water_content(water_content: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    从含水量计算蛇纹石化程度
    
    公式: β = 100 × w / 13
    其中: β 是蛇纹石化程度的百分数 (%)
          w 是含水量百分数 (H2O wt%)
    
    Args:
        water_content: 含水量百分数 (H2O wt%)
        
    Returns:
        蛇纹石化程度百分数 (%)
    """
    water_content = np.asarray(water_content)
    serpentinization = 100 * water_content / 13.0
    return serpentinization if isinstance(water_content, np.ndarray) else float(serpentinization)


def calculate_water_content_from_serpentinization(serpentinization: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    从蛇纹石化程度计算含水量
    
    公式: w = β × 13 / 100
    其中: β 是蛇纹石化程度的百分数 (%)
          w 是含水量百分数 (H2O wt%)
    
    Args:
        serpentinization: 蛇纹石化程度百分数 (%)
        
    Returns:
        含水量百分数 (H2O wt%)
    """
    serpentinization = np.asarray(serpentinization)
    water_content = serpentinization * 13.0 / 100.0
    return water_content if isinstance(serpentinization, np.ndarray) else float(water_content)


def calculate_vp_from_serpentinization(serpentinization: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    从蛇纹石化程度计算P波速度
    
    公式: Vp = 7922 - 32.5 × β
    其中: Vp 单位是 m/s
          β 是蛇纹石化程度的百分数 (%)
    
    Args:
        serpentinization: 蛇纹石化程度百分数 (%)
        
    Returns:
        P波速度 (m/s)
        
    Note:
        返回值的单位是 m/s，如果需要 km/s，需要除以 1000
    """
    serpentinization = np.asarray(serpentinization)
    vp = 7922.0 - 32.5 * serpentinization
    return vp if isinstance(serpentinization, np.ndarray) else float(vp)


def calculate_vs_from_serpentinization(serpentinization: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    从蛇纹石化程度计算S波速度
    
    公式: Vs = 4371 - 21.8 × β
    其中: Vs 单位是 m/s
          β 是蛇纹石化程度的百分数 (%)
    
    Args:
        serpentinization: 蛇纹石化程度百分数 (%)
        
    Returns:
        S波速度 (m/s)
        
    Note:
        返回值的单位是 m/s，如果需要 km/s，需要除以 1000
    """
    serpentinization = np.asarray(serpentinization)
    vs = 4371.0 - 21.8 * serpentinization
    return vs if isinstance(serpentinization, np.ndarray) else float(vs)


def calculate_serpentinization_from_vp(vp: Union[float, np.ndarray], 
                                       vp_unit: str = 'm/s') -> Union[float, np.ndarray]:
    """
    从P波速度反算蛇纹石化程度
    
    公式: β = (7922 - Vp) / 32.5
    其中: Vp 单位是 m/s
          β 是蛇纹石化程度的百分数 (%)
    
    Args:
        vp: P波速度
        vp_unit: 速度单位，'m/s' 或 'km/s'，默认 'm/s'
        
    Returns:
        蛇纹石化程度百分数 (%)
    """
    vp = np.asarray(vp)
    
    # 转换为 m/s
    if vp_unit.lower() == 'km/s':
        vp = vp * 1000.0
    
    serpentinization = (7922.0 - vp) / 32.5
    return serpentinization if isinstance(vp, np.ndarray) else float(serpentinization)


def calculate_serpentinization_from_vs(vs: Union[float, np.ndarray],
                                       vs_unit: str = 'm/s') -> Union[float, np.ndarray]:
    """
    从S波速度反算蛇纹石化程度
    
    公式: β = (4371 - Vs) / 21.8
    其中: Vs 单位是 m/s
          β 是蛇纹石化程度的百分数 (%)
    
    Args:
        vs: S波速度
        vs_unit: 速度单位，'m/s' 或 'km/s'，默认 'm/s'
        
    Returns:
        蛇纹石化程度百分数 (%)
    """
    vs = np.asarray(vs)
    
    # 转换为 m/s
    if vs_unit.lower() == 'km/s':
        vs = vs * 1000.0
    
    serpentinization = (4371.0 - vs) / 21.8
    return serpentinization if isinstance(vs, np.ndarray) else float(serpentinization)


# ============================================================================
# Water Content and Porosity Relationships
# ============================================================================

def calculate_water_content_from_porosity(porosity: Union[float, np.ndarray],
                                          rock_type: Optional[str] = None,
                                          rock_density: Optional[float] = None,
                                          water_density: float = 1.03) -> Union[float, np.ndarray]:
    """
    从孔隙度计算含水量
    
    公式: w (wt%) = 100% × (ρ_w × φ / (ρ_r × (1 - φ) + ρ_w × φ))
    其中: w 是含水量百分数 (H2O wt%)
          φ 是孔隙度 (0-1)
          ρ_w 是水密度 (g/cm³)，默认1.03（海水密度）
          ρ_r 是岩石密度 (g/cm³)
    
    支持的岩石类型（预定义）：
    - 'dunite': 纯橄岩，密度 3.31 g/cm³
    - 'gabbro': 辉长岩，密度 2.968 g/cm³
    
    Args:
        porosity: 孔隙度 (0-1)
        rock_type: 岩石类型，'dunite' 或 'gabbro'（可选，如果提供则忽略rock_density）
        rock_density: 岩石密度 (g/cm³)（可选，如果未提供且rock_type也未提供，则使用默认值3.31）
        water_density: 水密度 (g/cm³)，默认 1.03（海水密度）
        
    Returns:
        含水量百分数 (H2O wt%)
        
    Raises:
        ValueError: 如果既未提供rock_type也未提供rock_density，或rock_type不支持
    """
    porosity = np.asarray(porosity)
    
    # 定义预定义岩石密度
    predefined_rock_densities = {
        'dunite': 3.31,
        'gabbro': 2.968
    }
    
    # 确定使用的岩石密度
    if rock_type is not None:
        rock_type_lower = rock_type.lower()
        if rock_type_lower not in predefined_rock_densities:
            raise ValueError(f"不支持的岩石类型: {rock_type}. 可选类型: {list(predefined_rock_densities.keys())}")
        rock_density = predefined_rock_densities[rock_type_lower]
    elif rock_density is not None:
        # 使用用户提供的密度值
        if rock_density <= 0:
            raise ValueError(f"岩石密度必须大于0，当前值: {rock_density}")
        rock_density = rock_density
    else:
        # 默认使用dunite的密度
        rock_density = predefined_rock_densities['dunite']
    
    # 计算含水量
    # w = 100 * (ρ_w * φ / (ρ_r * (1 - φ) + ρ_w * φ))
    numerator = water_density * porosity
    denominator = rock_density * (1 - porosity) + water_density * porosity
    water_content = 100.0 * numerator / denominator
    
    return water_content if isinstance(porosity, np.ndarray) else float(water_content)


def calculate_porosity_from_water_content(water_content: Union[float, np.ndarray],
                                          rock_type: Optional[str] = None,
                                          rock_density: Optional[float] = None,
                                          water_density: float = 1.03) -> Union[float, np.ndarray]:
    """
    从含水量反算孔隙度
    
    公式: φ = (w/100 × ρ_r) / (ρ_w + w/100 × (ρ_r - ρ_w))
    其中: w 是含水量百分数 (H2O wt%)
          φ 是孔隙度 (0-1)
          ρ_w 是水密度 (g/cm³)，默认1.03（海水密度）
          ρ_r 是岩石密度 (g/cm³)
    
    这是含水量公式的逆变换。
    
    支持的岩石类型（预定义）：
    - 'dunite': 纯橄岩，密度 3.31 g/cm³
    - 'gabbro': 辉长岩，密度 2.968 g/cm³
    
    Args:
        water_content: 含水量百分数 (H2O wt%)
        rock_type: 岩石类型，'dunite' 或 'gabbro'（可选，如果提供则忽略rock_density）
        rock_density: 岩石密度 (g/cm³)（可选，如果未提供且rock_type也未提供，则使用默认值3.31）
        water_density: 水密度 (g/cm³)，默认 1.03（海水密度）
        
    Returns:
        孔隙度 (0-1)
        
    Raises:
        ValueError: 如果既未提供rock_type也未提供rock_density，或rock_type不支持
    """
    water_content = np.asarray(water_content)
    
    # 定义预定义岩石密度
    predefined_rock_densities = {
        'dunite': 3.31,
        'gabbro': 2.968
    }
    
    # 确定使用的岩石密度
    if rock_type is not None:
        rock_type_lower = rock_type.lower()
        if rock_type_lower not in predefined_rock_densities:
            raise ValueError(f"不支持的岩石类型: {rock_type}. 可选类型: {list(predefined_rock_densities.keys())}")
        rock_density = predefined_rock_densities[rock_type_lower]
    elif rock_density is not None:
        # 使用用户提供的密度值
        if rock_density <= 0:
            raise ValueError(f"岩石密度必须大于0，当前值: {rock_density}")
        rock_density = rock_density
    else:
        # 默认使用dunite的密度
        rock_density = predefined_rock_densities['dunite']
    
    # 从含水量反算孔隙度
    # 原始公式: w = 100 * (ρ_w * φ / (ρ_r * (1 - φ) + ρ_w * φ))
    # 解出 φ:
    # w/100 = (ρ_w * φ) / (ρ_r * (1 - φ) + ρ_w * φ)
    # w/100 * (ρ_r * (1 - φ) + ρ_w * φ) = ρ_w * φ
    # w/100 * ρ_r - w/100 * ρ_r * φ + w/100 * ρ_w * φ = ρ_w * φ
    # w/100 * ρ_r = φ * (ρ_w + w/100 * ρ_r - w/100 * ρ_w)
    # w/100 * ρ_r = φ * (ρ_w + w/100 * (ρ_r - ρ_w))
    # φ = (w/100 * ρ_r) / (ρ_w + w/100 * (ρ_r - ρ_w))
    
    w_frac = water_content / 100.0
    numerator = w_frac * rock_density
    denominator = water_density + w_frac * (rock_density - water_density)
    porosity = numerator / denominator
    
    return porosity if isinstance(water_content, np.ndarray) else float(porosity)


# ============================================================================
# Differential Effective Medium (DEM) Model
# ============================================================================
# 使用Critical Porosity DEM实现（基于Berryman 1980, 1992）

def _dem_critical_porosity(k1, mu1, k2, mu2, asp, phic):
    """
    DEM - Effective elastic moduli using Differential Effective Medium formulation
    with Critical Porosity scaling.
    
    使用临界孔隙度缩放的微分等效介质模型计算有效弹性模量。
    基于Berryman (1980, 1992)的DEM理论，采用Critical Porosity方法。
    
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
        - 使用Runge-Kutta 4/5阶方法（dopri5）进行数值积分
    """
    from scipy.integrate import ode
    
    # 创建ODE求解器
    # 使用Runge-Kutta 4/5阶方法（dopri5），匹配MATLAB的ode45
    solver = ode(_dem_ode_rhs)
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


def _dem_ode_rhs(t, y, k1, mu1, k2, mu2, asp, phic):
    """
    DEM模型的微分方程右端项（ODE Right-Hand Side）
    
    定义Critical Porosity DEM模型的微分方程系统。
    基于Berryman (1980)的完整Eshelby张量理论。
    
    参数:
        t: 积分变量（0到0.99999）
        y: [k, mu] - 当前的有效体积模量和剪切模量
        k1, mu1: 背景材料的体积模量和剪切模量
        k2, mu2: 包含物的体积模量和剪切模量
        asp: 纵横比
        phic: 临界孔隙度
    
    返回:
        yprime: [dk/dt, dmu/dt] - 模量对积分变量t的导数
    
    注意:
        - 使用完整的Eshelby张量（F1-F9）计算几何因子P和Q
        - 微分方程分母使用(1-t)，对应Critical Porosity缩放
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


def calculate_dem_effective_moduli(
    porosity: float,
    aspect_ratio: float,
    host_vp: float,
    host_vs: float,
    host_density: float,
    inclusion_k: float = 2.32,  # 水的体积模量 (GPa)
    inclusion_mu: float = 0.0,  # 水的剪切模量 (GPa)，为0
    n_steps: int = 100,
    critical_porosity: float = 1.0  # 临界孔隙度，默认1.0（标准DEM）
) -> Tuple[float, float]:
    """
    使用微分等效介质（DEM）模型计算含裂缝岩石的等效弹性模量
    
    基于Berryman (1980, 1992)的DEM理论，使用Critical Porosity方法实现。
    通过逐步添加包含物（水填充裂缝）到背景材料中来计算等效模量。
    
    Args:
        porosity: 孔隙度（0-1）。如果critical_porosity < 1.0，这是实际孔隙度
        aspect_ratio: 裂缝纵横比（椭球短轴/长轴，通常0.0001-0.05）
        host_vp: 背景材料P波速度 (km/s)
        host_vs: 背景材料S波速度 (km/s)
        host_density: 背景材料密度 (g/cm³)
        inclusion_k: 包含物体积模量 (GPa)，默认2.32（水）
        inclusion_mu: 包含物剪切模量 (GPa)，默认0.0（水）
        n_steps: 数值积分步数（已弃用，保留用于兼容性）
        critical_porosity: 临界孔隙度（0-1），默认1.0（标准DEM）
            - 如果critical_porosity = 1.0：使用标准DEM方法（phic=1.0）
            - 如果critical_porosity < 1.0：使用Critical Porosity DEM方法（Jenny_v2方式）
        
    Returns:
        (K_eff, mu_eff): 等效体积模量和剪切模量 (GPa)
    """
    # 1. 计算背景材料的弹性模量
    rho_kg_m3 = host_density * 1000  # 转换为kg/m³
    vp_ms = host_vp * 1000  # 转换为m/s
    vs_ms = host_vs * 1000  # 转换为m/s
    
    mu1 = rho_kg_m3 * vs_ms**2 * 1e-9  # 剪切模量 (GPa)
    lambda1 = rho_kg_m3 * (vp_ms**2 - 2 * vs_ms**2) * 1e-9  # 拉梅常数 (GPa)
    K1 = lambda1 + 2 * mu1 / 3  # 体积模量 (GPa)
    
    # 2. 使用Critical Porosity DEM实现
    # 调用_dem_critical_porosity获取完整的模量数组
    k_array, mu_array, por_array = _dem_critical_porosity(
        K1, mu1, inclusion_k, inclusion_mu, aspect_ratio, critical_porosity
    )
    
    # 3. 插值到目标孔隙度
    from scipy.interpolate import interp1d
    
    # 确保por_array是单调递增的
    if len(por_array) > 1 and np.all(np.diff(por_array) > 0):
        # 插值K和mu到目标孔隙度
        K_interp = interp1d(por_array, k_array, kind='linear', 
                           fill_value=(k_array[0], k_array[-1]), 
                           bounds_error=False)
        mu_interp = interp1d(por_array, mu_array, kind='linear', 
                            fill_value=(mu_array[0], mu_array[-1]), 
                            bounds_error=False)
        
        K_eff = float(K_interp(porosity))
        mu_eff = float(mu_interp(porosity))
    else:
        # 如果插值失败，使用最近邻插值
        K_eff = float(np.interp(porosity, por_array, k_array))
        mu_eff = float(np.interp(porosity, por_array, mu_array))
    
    # 4. 确保结果合理
    K_eff = max(0.1, K_eff)
    mu_eff = max(0.01, mu_eff)
    
    return float(K_eff), float(mu_eff)


def calculate_velocity_from_moduli(
    K: float,
    mu: float,
    density: float
) -> Tuple[float, float]:
    """
    从弹性模量计算P波和S波速度
    
    公式：
    Vs = sqrt(μ / ρ)
    Vp = sqrt((K + 4μ/3) / ρ)
    
    Args:
        K: 体积模量 (GPa)
        mu: 剪切模量 (GPa)
        density: 密度 (g/cm³)
        
    Returns:
        (vp, vs): P波和S波速度 (km/s)
    """
    rho_kg_m3 = density * 1000  # 转换为kg/m³
    
    # 计算速度（m/s）
    vs_ms = np.sqrt(mu * 1e9 / rho_kg_m3)  # GPa转换为Pa
    vp_ms = np.sqrt((K + 4*mu/3) * 1e9 / rho_kg_m3)
    
    # 转换为km/s
    vp = vp_ms / 1000.0
    vs = vs_ms / 1000.0
    
    return float(vp), float(vs)


def calculate_dem_velocity(
    porosity: float,
    aspect_ratio: float,
    host_vp: float,
    host_vs: float,
    host_density: float,
    inclusion_k: float = 2.32,
    inclusion_mu: float = 0.0,
    inclusion_density: float = 1.03,  # 水的密度 (g/cm³)
    n_steps: int = 100,
    critical_porosity: float = 1.0  # 临界孔隙度，默认1.0（标准DEM）
) -> Tuple[float, float, float]:
    """
    使用DEM模型计算含裂缝岩石的等效速度
    
    这是calculate_dem_effective_moduli和calculate_velocity_from_moduli的组合函数。
    基于Berryman (1980, 1992)的DEM理论，使用Critical Porosity方法实现。
    
    Args:
        porosity: 孔隙度（0-1）
        aspect_ratio: 裂缝纵横比
        host_vp: 背景材料P波速度 (km/s)
        host_vs: 背景材料S波速度 (km/s)
        host_density: 背景材料密度 (g/cm³)
        inclusion_k: 包含物体积模量 (GPa)，默认2.32（水）
        inclusion_mu: 包含物剪切模量 (GPa)，默认0.0（水）
        inclusion_density: 包含物密度 (g/cm³)，默认1.03（水）
        n_steps: 数值积分步数（已弃用，保留用于兼容性）
        critical_porosity: 临界孔隙度（0-1），默认1.0（标准DEM）
            - 如果critical_porosity = 1.0：使用标准DEM方法
            - 如果critical_porosity < 1.0：使用Critical Porosity DEM方法（推荐0.65用于页岩/细粒岩石）
        
    Returns:
        (vp, vs, vp_vs_ratio): 等效P波速度、S波速度和Vp/Vs比值
    """
    # 计算等效模量
    K_eff, mu_eff = calculate_dem_effective_moduli(
        porosity, aspect_ratio, host_vp, host_vs, host_density,
        inclusion_k, inclusion_mu, n_steps, critical_porosity
    )
    
    # 计算有效密度（考虑孔隙度）
    # 使用体积平均：ρ_eff = (1 - φ) * ρ_host + φ * ρ_inclusion
    effective_density = (1 - porosity) * host_density + porosity * inclusion_density
    
    # 计算等效速度
    vp, vs = calculate_velocity_from_moduli(K_eff, mu_eff, effective_density)
    
    # 计算Vp/Vs比值
    vp_vs_ratio = vp / vs if vs > 0 else 0.0
    
    return float(vp), float(vs), float(vp_vs_ratio)


# ============================================================================
# Depth-based Pressure and Temperature Calculation
# ============================================================================

def calculate_pressure_from_depth(
    depth: Union[float, np.ndarray],
    pressure_gradient: float = 30.0,
    seafloor_depth: Optional[Union[float, np.ndarray]] = None,
) -> Union[float, np.ndarray]:
    """从深度计算压力（从海底面开始）
    
    Args:
        depth: 深度 (km)
        pressure_gradient: 压力梯度 (MPa/km)，默认30.0
        seafloor_depth: 海底面深度 (km)，如果提供，则从海底面开始计算；否则从0开始
        
    Returns:
        压力值 (MPa)
    """
    depth_arr = np.asarray(depth, dtype=float)
    
    # 如果提供了海底面深度，计算相对于海底面的深度
    if seafloor_depth is not None:
        seafloor_arr = np.asarray(seafloor_depth, dtype=float)
        # 确保形状匹配
        if depth_arr.shape != seafloor_arr.shape:
            if seafloor_arr.ndim == 0:
                # 标量海底面深度，广播到所有深度
                seafloor_arr = np.full_like(depth_arr, float(seafloor_arr))
            else:
                raise ValueError(f"Shape mismatch: depth {depth_arr.shape} vs seafloor_depth {seafloor_arr.shape}")
        depth_below_seafloor = np.maximum(0.0, depth_arr - seafloor_arr)
    else:
        # 没有提供海底面深度，从0开始计算
        depth_below_seafloor = depth_arr
    
    pressure = pressure_gradient * depth_below_seafloor
    if isinstance(depth, np.ndarray):
        return pressure
    return float(pressure)


def calculate_temperature_from_depth(
    depth: Union[float, np.ndarray],
    temperature_gradient: float = 30.0,
    surface_temperature: float = 25.0,
    seafloor_depth: Optional[Union[float, np.ndarray]] = None,
    detect_seawater_from_vp: Optional[Union[float, np.ndarray]] = None,
    seawater_vp_threshold: float = 1.6,
) -> Union[float, np.ndarray]:
    """从深度计算温度（从海底面开始）
    
    Args:
        depth: 深度 (km)
        temperature_gradient: 温度梯度 (°C/km)，默认30.0
        surface_temperature: 表面温度 (°C)，默认25.0
        seafloor_depth: 海底面深度 (km)，如果提供，则从海底面开始计算；否则尝试自动检测或从0开始
        detect_seawater_from_vp: 用于检测海水层的Vp值（可选，仅在seafloor_depth未提供时使用）
        seawater_vp_threshold: 海水Vp阈值 (km/s)，默认1.6
        
    Returns:
        温度值 (°C)
    """
    depth_arr = np.asarray(depth, dtype=float)
    seafloor_depth_val = 0.0
    
    # 如果提供了海底面深度，直接使用
    if seafloor_depth is not None:
        seafloor_depth_val = float(np.asarray(seafloor_depth, dtype=float).item())
    # 否则，如果提供了Vp值，尝试检测海水层
    elif detect_seawater_from_vp is not None:
        vp_arr = np.asarray(detect_seawater_from_vp, dtype=float)
        if depth_arr.shape == vp_arr.shape and depth_arr.ndim == 1:
            # 查找第一个非海水点（vp >= threshold）
            rock_mask = vp_arr >= seawater_vp_threshold
            if np.any(rock_mask):
                seafloor_depth_val = float(depth_arr[rock_mask][0])
        elif depth_arr.ndim == 0 and vp_arr.ndim == 0:
            # 标量情况
            if vp_arr >= seawater_vp_threshold:
                seafloor_depth_val = float(depth_arr)
    
    # 计算相对于海底面的深度
    depth_below_seafloor = np.maximum(0.0, depth_arr - seafloor_depth_val)
    temperature = surface_temperature + temperature_gradient * depth_below_seafloor
    
    if isinstance(depth, np.ndarray):
        return temperature
    return float(temperature)


