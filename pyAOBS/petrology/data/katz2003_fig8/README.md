# Katz (2003) Fig.8 — eq. (16) ΔT(X) 与 eq. (17) X_sat(P) 标定

Katz et al. (2003) Fig.8 两 panel：

| Panel | 内容 | Katz 公式 |
|-------|------|-----------|
| **(a)** | 实验 ΔT vs 熔体 H₂O (wt%) | eq. (16): ΔT = **K** X^**γ**；Table 2: K=43, γ=0.75；轴 **0–60 wt%**, **0–900°C** |
| **(b)** | 饱和熔体含水量 vs P | eq. (17): X_sat = χ₁ P^λ + χ₂ P；χ₁=12, χ₂=1, λ=0.6；轴 **0–8 GPa**, **0–50 wt%** |

## 实验来源（图注）

- **(a)** Hirose & Kawamoto (1995); Kawamoto & Holloway (1997); Gaetani & Grove (1998); Grove (2001)
- **(b)** Dixon et al. (1995) basalt ≤0.5 GPa；Mysen & Wheeler (2000) Al-free haploandesite (79% SiO₂)

## 文件

| 文件 | 用途 |
|------|------|
| `delta_t_calibration.csv` | Panel (a) 散点 |
| `x_sat_calibration.csv` | Panel (b) 散点 |

当前 CSV 为 **种子点**（围绕 Table 2 拟合曲线），`verified=partial`；与 Katz ESI 逐点核对后改为 `yes`。

## 复现

```bash
py -3.11 petrology/validation/reproduce_katz2003_fig8.py --show
```
