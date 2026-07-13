# DEM模型实现对比分析：我们的代码 vs Rockphypy

## 1. Rockphypy的DEM实现分析

### 1.1 PQ函数（几何因子计算）

**位置**：EM.py 第724-776行

**关键实现**：
```python
@staticmethod
def PQ(Km, Gm, Ki, Gi, alpha):
    if alpha==1:
        # 球形包含物
        P = (Km+4*Gm/3)/(Ki+4*Gm/3)
        kesai = Gm/6 * (9*Km+8*Gm)/(Km+2*Gm)
        Q = (Gm+kesai)/(Gi+kesai)
    else:
        # 椭球包含物
        if alpha<1:
            # 扁椭球
            theta = alpha/(1.0 - alpha**2)**(3.0/2.0) * (np.arccos(alpha) - alpha*np.sqrt(1.0 - alpha**2))
        else:
            # 长椭球
            theta = alpha/(alpha**2-1)**(3.0/2.0) * (alpha*(alpha**2-1)**0.5 - np.cosh(alpha)**-1)
        
        f = alpha**2*(3.0*theta - 2.0)/(1.0 - alpha**2)
        
        A = Gi/Gm - 1.0
        B = (Ki/Km - Gi/Gm)/3.0
        R = Gm/(Km + (4.0/3.0)*Gm)
        
        # 计算F1-F9（Eshelby张量相关）
        F1 = 1.0 + A*(1.5*(f + theta) - R*(1.5*f + 2.5*theta - 4.0/3.0))
        F2 = 1.0 + A*(1.0 + 1.5*(f + theta) - R*(1.5*f + 2.5*theta)) + B*(3.0 - 4.0*R) + A*(A + 3.0*B)*(1.5 - 2.0*R)*(f + theta - R*(f - theta + 2.0*theta**2))
        F3 = 1.0 + A*(1.0 - f - 1.5*theta + R*(f + theta))
        F4 = 1.0 + (A/4.0)*(f + 3.0*theta - R*(f - theta))
        F5 = A*(-f + R*(f + theta - 4.0/3.0)) + B*theta*(3.0 - 4.0*R)
        F6 = 1.0 + A*(1.0 + f - R*(f + theta)) + B*(1.0 - theta)*(3.0 - 4.0*R)
        F7 = 2.0 + (A/4.0)*(3.0*f + 9.0*theta - R*(3.0*f + 5.0*theta)) + B*theta*(3.0 - 4.0*R)
        F8 = A*(1.0 - 2.0*R + (f/2.0)*(R - 1.0) + (theta/2.0)*(5.0*R - 3.0)) + B*(1.0 - theta)*(3.0 - 4.0*R)
        F9 = A*((R - 1.0)*f - R*theta) + B*theta*(3.0 - 4.0*R)
        
        Tiijj = 3*F1/F2
        Tijij = Tiijj/3 + 2/F3 + 1/F4 + (F4*F5 + F6*F7 - F8*F9)/(F2*F4)
        P = Tiijj/3
        Q = (Tijij - P)/5
    return P, Q
```

**关键特点**：
1. 使用完整的Eshelby张量计算（F1-F9）
2. 通过Tiijj和Tijij计算P和Q
3. 这是标准的Berryman (1980)公式

### 1.2 DEM函数（微分方程）

**位置**：EM.py 第779-787行

```python
@staticmethod
def DEM(y, t, params):
    K_eff, G_eff = y  # 当前有效模量
    Gi, Ki, alpha = params  # 参数
    P, Q = EM.PQ(G_eff, K_eff, Gi, Ki, alpha)  # 注意：参数顺序是G, K, Gi, Ki
    derivs = [1/(1-t) * (Ki-K_eff) * P,  1/(1-t) * (Gi-G_eff) * Q]
    return derivs
```

**注意**：
- 参数顺序：`PQ(G_eff, K_eff, Gi, Ki, alpha)` - **G在前，K在后！**
- 微分方程：`dK/dt = (Ki-K) * P / (1-t)`
- 微分方程：`dG/dt = (Gi-G) * Q / (1-t)`

### 1.3 Berryman_DEM函数（主函数）

**位置**：EM.py 第790-819行

```python
@staticmethod
def Berryman_DEM(Km, Gm, Ki, Gi, alpha, phi):
    params = [Gi, Ki, alpha]  # 注意顺序：Gi, Ki, alpha
    y0 = [Km, Gm]  # 初始条件：[K, G]
    tStop = phi
    tInc = 0.01
    t = np.arange(0, tStop+tInc, tInc)
    psoln = odeint(EM.DEM, y0, t, args=(params,))
    K_dry_dem = psoln[:,0]
    G_dry_dem = psoln[:,1]
    return K_dry_dem, G_dry_dem, t
```

## 2. 我们的实现分析

### 2.1 几何因子计算

**当前实现**（修正后）：
- 对于球形：使用简化公式（正确）
- 对于椭球：使用简化的Berryman公式，但**不是完整的Eshelby张量计算**

**问题**：
1. **缺少完整的Eshelby张量计算**：我们的实现使用了简化的公式，而rockphypy使用了完整的F1-F9计算
2. **Q的计算不正确**：我们使用的公式 `Q = (1/5) * [1 + (3/2)*f + (5/2)*A*(B + (3/2)*f)]` 不是标准的Berryman公式

### 2.2 微分方程

**我们的实现**：
```python
dK_dphi = (inclusion_k - K_prime) * P / (1 - phi)
dmu_dphi = (inclusion_mu - mu_prime) * Q / (1 - phi)
```

**对比rockphypy**：
```python
dK_dt = (Ki - K_eff) * P / (1 - t)
dG_dt = (Gi - G_eff) * Q / (1 - t)
```

**结论**：微分方程形式一致，这是正确的。

## 3. 关键差异和问题

### 问题1：几何因子P和Q的计算方法

**Rockphypy（标准）**：
- 使用完整的Eshelby张量（F1-F9）
- 通过Tiijj和Tijij计算
- 这是Berryman (1980)的完整公式

**我们的实现（简化）**：
- 使用简化的公式
- 对于椭球，公式可能不准确

### 问题2：参数顺序

**Rockphypy的PQ函数**：
```python
PQ(Km, Gm, Ki, Gi, alpha)  # K, G, Ki, Gi, alpha
```

但在DEM函数中调用时：
```python
P, Q = EM.PQ(G_eff, K_eff, Gi, Ki, alpha)  # G在前，K在后！
```

**注意**：这可能是rockphypy的一个bug，或者有特殊原因。需要验证。

### 问题3：theta和f的计算

**Rockphypy**：
```python
if alpha<1:
    theta = alpha/(1.0 - alpha**2)**(3.0/2.0) * (np.arccos(alpha) - alpha*np.sqrt(1.0 - alpha**2))
else:
    theta = alpha/(alpha**2-1)**(3.0/2.0) * (alpha*(alpha**2-1)**0.5 - np.cosh(alpha)**-1)
f = alpha**2*(3.0*theta - 2.0)/(1.0 - alpha**2)
```

**我们的实现**：
```python
theta = alpha_safe / ((1.0 - alpha_safe**2)**1.5 + 1e-10) * (arccos_alpha - alpha_safe * sqrt_term)
f = (alpha_safe**2 * (3*theta - 2)) / (1.0 - alpha_safe**2 + 1e-10)
```

**结论**：theta和f的计算基本一致，但我们的实现只处理了α<1的情况。

## 4. 建议的修正

### 修正1：使用完整的Eshelby张量计算

应该使用rockphypy的完整公式来计算P和Q，包括F1-F9的计算。

### 修正2：处理长椭球（α>1）的情况

当前实现只处理了扁椭球（α<1），应该添加长椭球（α>1）的处理。

### 修正3：验证参数顺序

需要确认PQ函数的参数顺序是否正确。

## 5. 测试建议

1. **对比球形情况**：设置α=1，对比结果
2. **对比椭球情况**：使用相同的输入参数，对比P和Q的值
3. **对比最终结果**：使用相同的输入，对比最终的K_eff和G_eff
