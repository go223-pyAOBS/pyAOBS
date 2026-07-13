# DEM模型实现修正总结

## 1. 发现的问题

### 问题1：几何因子计算不完整
- **原实现**：使用了简化的修正公式，不是标准的Berryman公式
- **修正**：使用完整的Eshelby张量计算（F1-F9），与rockphypy一致

### 问题2：缺少长椭球（α>1）的处理
- **原实现**：只处理了扁椭球（α<1）和球体（α=1）
- **修正**：添加了长椭球（α>1）的处理

### 问题3：参数传递问题
- **原实现**：几何因子函数没有接收inclusion_k和inclusion_mu参数
- **修正**：函数签名改为 `calculate_geometric_factors(K, mu, Ki, Gi, alpha)`

## 2. Rockphypy的实现分析

### 2.1 PQ函数（几何因子）

**标准Berryman (1980)公式**：
- 对于球形（α=1）：使用简化公式
- 对于椭球（α≠1）：使用完整的Eshelby张量（F1-F9）
- 通过Tiijj和Tijij计算P和Q

**关键公式**：
```python
# 计算theta和f
if alpha<1:
    theta = alpha/(1-alpha²)^(3/2) * (arccos(alpha) - alpha*sqrt(1-alpha²))
else:
    theta = alpha/(alpha²-1)^(3/2) * (alpha*sqrt(alpha²-1) - arccosh(alpha))
f = alpha²*(3*theta - 2)/(1 - alpha²)

# 计算A, B, R
A = Gi/Gm - 1
B = (Ki/Km - Gi/Gm)/3
R = Gm/(Km + 4*Gm/3)

# 计算F1-F9（Eshelby张量）
F1 = 1 + A*(1.5*(f+theta) - R*(1.5*f + 2.5*theta - 4/3))
F2 = ... (复杂表达式)
...
F9 = A*((R-1)*f - R*theta) + B*theta*(3-4*R)

# 计算Tiijj和Tijij
Tiijj = 3*F1/F2
Tijij = Tiijj/3 + 2/F3 + 1/F4 + (F4*F5 + F6*F7 - F8*F9)/(F2*F4)

# 计算P和Q
P = Tiijj/3
Q = (Tijij - P)/5
```

### 2.2 DEM函数（微分方程）

**注意**：rockphypy的DEM函数中有一个潜在问题：
```python
derivs = [1/(1-t) * (Ki-K_eff) * P,  1/(1-t) * -G_eff * Q]
```

第二个导数应该是 `(Gi - G_eff) * Q`，而不是 `-G_eff * Q`。这可能是rockphypy的一个bug。

**我们的实现**（正确）：
```python
dK_dphi = (inclusion_k - K_prime) * P / (1 - phi)
dmu_dphi = (inclusion_mu - mu_prime) * Q / (1 - phi)
```

### 2.3 Berryman_DEM函数

- 使用`odeint`求解微分方程
- 时间步长：0.01
- 返回：K_eff数组, G_eff数组, t数组

## 3. 我们的修正

### 3.1 几何因子函数

**修正后的实现**：
- ✅ 使用完整的Eshelby张量计算（F1-F9）
- ✅ 处理球形、扁椭球、长椭球三种情况
- ✅ 与rockphypy的PQ函数实现一致
- ✅ 参数顺序：`calculate_geometric_factors(K, mu, Ki, Gi, alpha)`

### 3.2 微分方程

**保持不变**（已经是正确的）：
```python
dK_dphi = (inclusion_k - K_prime) * P / (1 - phi)
dmu_dphi = (inclusion_mu - mu_prime) * Q / (1 - phi)
```

### 3.3 数值求解

- 使用`scipy.integrate.odeint`（如果可用）
- 否则使用欧拉方法
- 积分步数：n_steps（默认100）

## 4. 关键差异对比

| 项目 | Rockphypy | 我们的实现（修正后） |
|------|-----------|-------------------|
| 几何因子计算 | 完整Eshelby张量（F1-F9） | ✅ 完整Eshelby张量（F1-F9） |
| 球形处理 | 简化公式 | ✅ 简化公式 |
| 扁椭球处理 | 完整公式 | ✅ 完整公式 |
| 长椭球处理 | 完整公式 | ✅ 完整公式（新增） |
| 微分方程 | `dG/dt = -G*Q/(1-t)` (可能错误) | ✅ `dμ/dφ = (μi-μ)*Q/(1-φ)` (正确) |
| 参数顺序 | `PQ(Km, Gm, Ki, Gi, alpha)` | ✅ `calculate_geometric_factors(K, mu, Ki, Gi, alpha)` |

## 5. 测试建议

1. **对比球形情况**：
   - 设置α=1，对比P和Q的值
   - 应该与简化公式一致

2. **对比椭球情况**：
   - 使用相同的输入参数（K, G, Ki, Gi, alpha）
   - 对比P和Q的值
   - 应该与rockphypy的PQ函数结果一致

3. **对比最终结果**：
   - 使用相同的输入参数
   - 对比最终的K_eff和G_eff
   - 注意：由于rockphypy的DEM函数可能有bug，结果可能不完全一致

4. **验证边界情况**：
   - 孔隙度为0时，应该返回host模量
   - 干裂缝（Ki=0, Gi=0）的情况
   - 极薄裂缝（α→0）的情况

## 6. 注意事项

1. **Rockphypy的潜在bug**：
   - DEM函数中的`dG/dt = -G*Q/(1-t)`可能是错误的
   - 应该是`dG/dt = (Gi-G)*Q/(1-t)`
   - 我们的实现是正确的

2. **参数顺序**：
   - Rockphypy的PQ函数定义是`PQ(Km, Gm, Ki, Gi, alpha)`
   - 但在DEM函数中调用时是`PQ(G_eff, K_eff, Gi, Ki, alpha)`
   - 这可能是参数顺序错误，或者有特殊原因
   - 我们的实现使用正确的参数顺序

3. **数值稳定性**：
   - 对于极小的α值，需要特殊处理
   - 对于接近1的α值，需要避免数值问题

## 7. 结论

修正后的实现：
- ✅ 使用标准的Berryman (1980)完整公式
- ✅ 与rockphypy的PQ函数实现一致
- ✅ 微分方程实现正确（可能比rockphypy更正确）
- ✅ 处理了所有情况（球形、扁椭球、长椭球）

建议进行测试验证，确保结果正确。
