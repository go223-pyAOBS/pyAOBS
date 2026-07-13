# Talwani 2D 重力异常计算代码分析

## 概述

Talwani方法用于计算2D多边形体的重力异常，基于Talwani et al. (1959)的经典算法。代码实现了多种重力场分量的计算。

## 核心文件

1. **talwani.h** - 定义常量和辅助函数
2. **talwani2d.c** - 主要实现文件
3. **talwani2d_inc.h** - GMT模块选项定义

## 关键常量定义 (talwani.h)

```c
#define TOL        1.0e-7      // 计算容差
#define SI_TO_MGAL 1.0e5       // m/s² 转换为 mGal
#define SI_TO_EOTVOS 1.0e9     // (m/s²)/m 转换为 Eotvos
#define NEWTON_G   6.67430e-11 // 万有引力常数 (m³/(kg·s²))
```

## 核心计算函数

### 1. 2D重力异常计算 (talwani2d_get_grav2d)

**函数签名：**
```c
double talwani2d_get_grav2d(
    struct GMT_CTRL *GMT,
    double x[],      // 多边形x坐标数组
    double z[],      // 多边形z坐标数组
    unsigned int n,  // 顶点数量（最后一个点=第一个点）
    double x0,       // 观测点x坐标
    double z0,       // 观测点z坐标
    double rho       // 密度对比度 (kg/m³)
)
```

**算法原理：**
- 基于Talwani (1959)的2D多边形重力公式
- 对多边形的每条边进行积分
- 公式：对每条边计算 `(xi*zi1 - zi*xi1) * ((xi1-xi)*(φi-φi1) + (zi1-zi)*log(ri1/ri)) / (dx²+dz²)`
- 最终结果：`sum * 2.0 * G * rho * SI_TO_MGAL` (单位：mGal)

**关键步骤：**
1. 计算每个顶点相对于观测点的坐标差 (xi, zi)
2. 计算极角 φ = atan2(zi, xi)
3. 计算距离 r = hypot(xi, zi)
4. 对每条边进行积分求和
5. 乘以万有引力常数和密度，转换为mGal

### 2. 2.5D重力异常计算 (talwani2d_grav_2_5D)

**函数签名：**
```c
double talwani2d_grav_2_5D(
    struct GMT_CTRL *GMT,
    double x[], double z[], unsigned int n,
    double x0, double z0,
    double rho,
    double ymin,  // y方向最小范围
    double ymax   // y方向最大范围
)
```

**算法原理：**
- 基于Rasmussen & Pedersen (1979)的2.5D方法
- 用于有限长度的2D体（在y方向有有限延伸）
- 使用积分I1计算y方向的贡献

### 3. 垂直重力梯度 (talwani2d_get_vgg2d)

**函数签名：**
```c
double talwani2d_get_vgg2d(
    struct GMT_CTRL *GMT,
    double *x, double *z, unsigned int n,
    double x0, double z0,
    double rho
)
```

**算法原理：**
- 基于Kim & Wessel (2016)的解析解
- 计算垂直重力梯度（VGG）
- 单位：Eotvos (1 Eotvos = 0.1 mGal/km)

### 4. 大地水准面异常 (talwani2d_get_geoid2d)

**函数签名：**
```c
double talwani2d_get_geoid2d(
    struct GMT_CTRL *GMT,
    double y[], double z[], unsigned int n,
    double y0, double z0,
    double rho,
    double G0  // 正常重力值
)
```

**算法原理：**
- 基于Chapman (1979)的方法
- 计算大地水准面异常（单位：米）

## 数据结构

```c
struct TALWANI2D_BODY2D {
    int n;          // 顶点数量
    double rho;     // 密度对比度 (kg/m³)
    double *x;      // x坐标数组
    double *z;      // z坐标数组
};
```

## 输入要求

1. **多边形定义：**
   - 顶点坐标 (x, z) 数组
   - 最后一个点必须等于第一个点（闭合多边形）
   - 坐标单位：米（或通过-M选项指定为千米）

2. **密度：**
   - 单位：kg/m³ 或 g/cm³（如果<10，自动转换为kg/m³）
   - 可以从文件头读取，或通过-D选项指定

3. **观测点：**
   - 观测点坐标 (x0, z0)
   - z0通常为0（地表）或负值（地下）

## 输出

- **FAA (Free Air Anomaly)**: 自由空气重力异常，单位 mGal
- **VGG (Vertical Gravity Gradient)**: 垂直重力梯度，单位 Eotvos
- **Geoid**: 大地水准面异常，单位 米

## Python实现要点

### 1. 核心算法转换

将C代码转换为Python时需要注意：

```python
import numpy as np

# 常量定义
NEWTON_G = 6.67430e-11  # m³/(kg·s²)
SI_TO_MGAL = 1.0e5      # m/s² to mGal
SI_TO_EOTVOS = 1.0e9    # (m/s²)/m to Eotvos

def talwani2d_gravity(x, z, x0, z0, rho):
    """
    计算2D多边形体的重力异常
    
    Args:
        x: 多边形x坐标数组 (m)
        z: 多边形z坐标数组 (m)
        x0: 观测点x坐标 (m)
        z0: 观测点z坐标 (m)
        rho: 密度对比度 (kg/m³)
    
    Returns:
        重力异常 (mGal)
    """
    n = len(x)
    if x[0] != x[-1] or z[0] != z[-1]:
        # 确保多边形闭合
        x = np.append(x, x[0])
        z = np.append(z, z[0])
        n = len(x)
    
    sum_val = 0.0
    xi = x[0] - x0
    zi = z[0] - z0
    phi_i = np.arctan2(zi, xi)
    ri = np.hypot(xi, zi)
    
    if ri == 0:
        raise ValueError("观测点与多边形顶点重合！")
    
    for i in range(n - 1):
        i1 = i + 1
        xi1 = x[i1] - x0
        zi1 = z[i1] - z0
        phi_i1 = np.arctan2(zi1, xi1)
        ri1 = np.hypot(xi1, zi1)
        
        if ri1 == 0:
            raise ValueError("观测点与多边形顶点重合！")
        
        # Talwani公式的核心计算
        numerator = (xi * zi1 - zi * xi1) * (
            (xi1 - xi) * (phi_i - phi_i1) + 
            (zi1 - zi) * np.log(ri1 / ri)
        )
        denominator = (xi1 - xi)**2 + (zi1 - zi)**2
        
        if abs(denominator) > 1e-10:
            sum_val += numerator / denominator
        
        xi = xi1
        zi = zi1
        ri = ri1
        phi_i = phi_i1
    
    # 转换为mGal
    gravity = 2.0 * NEWTON_G * rho * SI_TO_MGAL * sum_val
    return gravity
```

### 2. 在imodel_gui.py中集成

**建议的实现步骤：**

1. **创建Python版本的Talwani计算模块**
   - 文件：`pyAOBS/utils/gravity_talwani.py`
   - 实现2D和2.5D重力计算函数

2. **扩展GravityCalculator类**
   - 添加talwani2d方法
   - 支持多边形体计算
   - 支持多个多边形体叠加

3. **在GUI中添加重力模拟功能**
   - 添加"Gravity Simulation"按钮/菜单
   - 允许用户选择多边形区域
   - 输入密度对比度
   - 选择观测点或观测线
   - 显示重力异常曲线或2D图

4. **数据流程：**
   ```
   速度模型 → 密度模型（通过经验公式）→ 多边形体定义 → Talwani计算 → 重力异常
   ```

### 3. 关键注意事项

1. **坐标系统：**
   - Talwani代码使用米作为单位
   - 如果模型使用千米，需要转换
   - z轴方向：向下为正（或通过-A选项反转）

2. **多边形方向：**
   - 顺时针：正密度对比度产生正重力异常
   - 逆时针：需要取反

3. **观测点位置：**
   - 不能与多边形顶点重合
   - 通常在地表（z=0）或海面

4. **密度对比度：**
   - 相对于背景密度的差值
   - 正密度对比度产生正重力异常

## 参考文献

1. Talwani, M., Worzel, J. L., & Landisman, M. (1959). Rapid gravity computations for two-dimensional bodies with application to the Mendocino submarine fracture zone. *J. Geophys. Res.*, 64, 49-59.

2. Rasmussen, R., & Pedersen, L. B. (1979). End corrections in potential field modeling. *Geophys. Prospect.*, 27, 749-760.

3. Kim, S.-S., & Wessel, P. (2016). New analytic solutions for modeling vertical gravity gradient anomalies. *Geochem. Geophys. Geosyst.*, 17, doi:10.1002/2016GC006263.

4. Chapman, M. E. (1979). Techniques for interpretation of geoid anomalies. *J. Geophys. Res.*, 84(B8), 3793-3801.
