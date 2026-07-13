# 经验公式库使用说明

## 概述

`empirical_formulas.py` 是一个统一的经验公式库，包含了pyAOBS项目中使用的所有岩石物理经验公式。该库提供了统一的接口，确保整个项目中公式使用的一致性。

## 导入方式

```python
# 方式1：从utils模块导入（推荐）
from pyAOBS.utils import (
    calculate_density,
    calculate_vs,
    correct_velocity,
    calculate_elastic_moduli
)

# 方式2：直接从empirical_formulas模块导入
from pyAOBS.utils.empirical_formulas import (
    calculate_density_gardner,
    calculate_vs_brocher,
    correct_velocity
)
```

## 主要功能

### 1. 密度计算

支持多种经验公式：

```python
import numpy as np
from pyAOBS.utils import calculate_density

vp = 5.0  # P波速度 (km/s)

# Gardner公式（适用于沉积岩）
density_gardner = calculate_density(vp, method='gardner')

# Nafe-Drake公式（适用于海洋沉积物）
density_nafe = calculate_density(vp, method='nafe_drake')

# Brocher公式（适用于地壳岩石，与Nafe-Drake相同）
density_brocher = calculate_density(vp, method='brocher')

# Castagna公式
density_castagna = calculate_density(vp, method='castagna')

# Lindseth公式
density_lindseth = calculate_density(vp, method='lindseth')
```

**公式详情：**

- **Gardner (1974)**: ρ = 0.31 × (Vp in m/s)^0.25
- **Nafe-Drake (1963)**: ρ = 1.6612×Vp - 0.4721×Vp² + 0.0671×Vp³ - 0.0043×Vp⁴ + 0.000106×Vp⁵
- **Brocher (2005)**: 与Nafe-Drake相同
- **Castagna**: ρ = 1.66 × Vp^0.261
- **Lindseth**: ρ = 0.31 × Vp + 1.7

### 2. S波速度计算

```python
from pyAOBS.utils import calculate_vs

vp = 5.0  # P波速度 (km/s)

# Brocher公式（适用于地壳岩石）
vs_brocher = calculate_vs(vp, method='brocher')
# 公式: Vs = 0.7858 - 1.2344×Vp + 0.7949×Vp² - 0.1238×Vp³ + 0.0064×Vp⁴

# Castagna公式（适用于泥岩）
vs_castagna = calculate_vs(vp, method='castagna')
# 公式: Vs = (Vp - 1.36) / 1.16
```

### 3. 从S波速度反算P波速度

```python
from pyAOBS.utils import calculate_vp_from_vs_brocher

vs = 3.0  # S波速度 (km/s)

# 使用Newton-Raphson迭代方法求解
vp = calculate_vp_from_vs_brocher(vs)
```

### 4. 温压校正

```python
from pyAOBS.utils import correct_velocity

# 原始数据
vp_original = 5.0  # km/s
pressure = 100.0   # MPa
temperature = 50.0  # °C

# 校正到标准条件（200 MPa, 25°C）
vp_corrected = correct_velocity(
    vp_original,
    pressure=pressure,
    temperature=temperature,
    target_pressure=200.0,
    target_temperature=25.0,
    is_s_wave=False  # False表示P波，True表示S波
)

# 只进行压力校正
from pyAOBS.utils import correct_velocity_pressure
vp_pressure_corrected = correct_velocity_pressure(
    vp_original,
    pressure=pressure,
    target_pressure=200.0,
    is_s_wave=False
)

# 只进行温度校正
from pyAOBS.utils import correct_velocity_temperature
vp_temp_corrected = correct_velocity_temperature(
    vp_original,
    temperature=temperature,
    target_temperature=25.0,
    is_s_wave=False
)
```

**校正参数：**

- P波压力系数: β = 0.0002/MPa
- S波压力系数: β = 0.00015/MPa
- P波温度系数: α = -0.50e-4/°C（负数）
- S波温度系数: α = -0.40e-4/°C（负数）

**校正公式：**
- 压力校正: V_corrected = V_original × (1 + β × ΔP)
- 温度校正: V_corrected = V_original × (1 - α × ΔT)

### 5. 弹性模量计算

```python
from pyAOBS.utils import calculate_elastic_moduli

vp = 5.0      # P波速度 (km/s)
vs = 3.0      # S波速度 (km/s)
density = 2.7  # 密度 (g/cm³)

moduli = calculate_elastic_moduli(vp, vs, density)

print(f"体积模量: {moduli['bulk_modulus']:.2f} GPa")
print(f"剪切模量: {moduli['shear_modulus']:.2f} GPa")
print(f"杨氏模量: {moduli['young_modulus']:.2f} GPa")
print(f"拉梅常数λ: {moduli['lame_lambda']:.2f} GPa")
```

**计算公式：**
- 剪切模量: μ = ρ × Vs²
- 拉梅常数λ: λ = ρ × (Vp² - 2×Vs²)
- 体积模量: K = λ + 2×μ/3
- 杨氏模量: E = μ × (3×λ + 2×μ) / (λ + μ)

### 6. 辅助函数

```python
from pyAOBS.utils import calculate_vp_vs_ratio, calculate_poisson_ratio

vp = 5.0
vs = 3.0

# 计算Vp/Vs比值
vp_vs_ratio = calculate_vp_vs_ratio(vp, vs)

# 计算泊松比
poisson = calculate_poisson_ratio(vp, vs)
# 公式: ν = ((Vp/Vs)² - 2) / (2 × ((Vp/Vs)² - 1))
```

## 数组支持

所有函数都支持numpy数组输入：

```python
import numpy as np
from pyAOBS.utils import calculate_density, calculate_vs

vp_array = np.array([4.0, 5.0, 6.0, 7.0])

# 批量计算密度
density_array = calculate_density(vp_array, method='gardner')

# 批量计算S波速度
vs_array = calculate_vs(vp_array, method='brocher')
```

## 自定义校正参数

如果需要使用自定义的校正参数：

```python
from pyAOBS.utils.empirical_formulas import CorrectionParameters, correct_velocity

# 创建自定义校正参数
custom_params = CorrectionParameters(
    vp_pressure_beta=0.00025,  # 自定义P波压力系数
    vp_temp_alpha=-0.60e-4,    # 自定义P波温度系数
    vs_pressure_beta=0.00018,  # 自定义S波压力系数
    vs_temp_alpha=-0.45e-4      # 自定义S波温度系数
)

# 使用自定义参数进行校正
vp_corrected = correct_velocity(
    vp_original,
    pressure=pressure,
    temperature=temperature,
    correction_params=custom_params
)
```

## 参考文献

1. **Gardner, G.H.F., Gardner, L.W., & Gregory, A.R. (1974)**. Formation velocity and density—the diagnostic basics for stratigraphic traps. *Geophysics*, 39(6), 770-780.

2. **Brocher, T.M. (2005)**. Empirical relations between elastic wavespeeds and density in the Earth's crust. *Bulletin of the Seismological Society of America*, 95(6), 2081-2092.

3. **Nafe, J.E., & Drake, C.L. (1963)**. Physical properties of marine sediments. In *The Sea* (Vol. 3, pp. 794-815).

4. **Castagna, J.P., Batzle, M.L., & Eastwood, R.L. (1985)**. Relationships between compressional-wave and shear-wave velocities in clastic silicate rocks. *Geophysics*, 50(4), 571-581.

## 注意事项

1. **单位一致性**：
   - 速度单位：km/s
   - 密度单位：g/cm³
   - 压力单位：MPa
   - 温度单位：°C

2. **公式适用范围**：
   - Gardner公式适用于沉积岩，特别是未固结和弱固结的沉积物
   - Nafe-Drake和Brocher公式适用于Vp范围1.5-8.5 km/s
   - Castagna公式适用于泥岩和页岩

3. **温压校正**：
   - 温度系数alpha定义为负数，公式中使用减号
   - 校正顺序：先压力校正，再温度校正

4. **数值稳定性**：
   - 从Vs反算Vp时使用Newton-Raphson迭代，确保收敛
   - 所有函数都经过数值稳定性测试

## 迁移指南

如果您的代码中直接使用了经验公式，建议迁移到统一库：

**旧代码：**
```python
# 直接使用公式
density = 0.31 * (vp * 1000) ** 0.25
vs = 0.7858 - 1.2344 * vp + 0.7949 * vp**2 - 0.1238 * vp**3 + 0.0064 * vp**4
```

**新代码：**
```python
from pyAOBS.utils import calculate_density, calculate_vs

density = calculate_density(vp, method='gardner')
vs = calculate_vs(vp, method='brocher')
```

这样可以确保：
- 公式的一致性
- 代码的可维护性
- 未来公式更新的便利性
