# 时差（$\Delta t$）公式与代码口径说明

本文档给出 `iphase` 模块中 **PPS−PPP**、**PSS−PSP** 的观测与理论计算方式。显示公式统一用 **Pandoc 兼容的 `$$` 块**（避免 `\[` 在列表缩进下被误解析）。

---

## 1. 符号约定

- $\Delta t$：时差  
- $V_p, V_s$：末段介质中的 P/S 波速度  
- $h$：末段垂向厚度（或等效厚度）  
- $L$：末段路径长度  
- $p = \mathrm{d}t/\mathrm{d}x$：由走时曲线斜率估计的慢度量  
- $x_{\mathrm{conv}}$：转换点位置  
- $x_{\mathrm{rec}}$：接收点位置  

---

## 2. 观测时差定义（符号方向）

### 2.1 PPS−PPP

- 严格配对：

$$
\Delta t_{\mathrm{PPS-PPP}} = t_{\mathrm{PPS}} - t_{\mathrm{PPP}}
$$

- 拟合模式（默认）：

$$
\Delta t_{\mathrm{PPS-PPP}} = t_{\mathrm{PPS}} - t_{\mathrm{PPP,fit}}(x)
$$

代码位置：

- `phase_combine.py` 中 `compute_ppp_pps_diff_pairs`（`t_diff = t2 - t1`）  
- `iphase_gui.py` 中 `_compute_file_result`

### 2.2 PSS−PSP

- 严格配对：

$$
\Delta t_{\mathrm{PSS-PSP}} = t_{\mathrm{PSS}} - t_{\mathrm{PSP}}
$$

- 拟合模式（默认）：

$$
\Delta t_{\mathrm{PSS-PSP}} = t_{\mathrm{PSS}} - t_{\mathrm{PSP,fit}}(x)
$$

代码位置：

- `iphase_gui.py` 中 `_compute_pss_psp_pairs_observed`

结论：观测侧统一为“后相位减前相位”，$\Delta t > 0$ 表示后相位更晚到时。

---

## 3. 理论核心公式（共用）

当只比较末段 C$\rightarrow$R 的 P/S 速度差异时，两类时差都采用

$$
\Delta t = L\left(\frac{1}{V_s} - \frac{1}{V_p}\right).
$$

在常见沉积层条件 $V_s < V_p$ 下，有

$$
\left(\frac{1}{V_s} - \frac{1}{V_p}\right) > 0.
$$

---

## 4. 主程序中 $L$ 的两种来源

### 4.1 PPS−PPP 主理论：`pps_minus_ppp_from_ppp_slope`

1) 由 PPP 走时曲线估计

$$
p = \frac{\mathrm{d}t}{\mathrm{d}x}.
$$

2) 定义末段 P 波角度 $\theta_p$：

$$
\sin\theta_p = |p|V_p.
$$

3) 路径长度取

$$
L = \frac{h}{\cos\theta_p}
  = \frac{h}{\sqrt{1-(|p|V_p)^2}}.
$$

4) 代入核心式：

$$
\Delta t_{\mathrm{PPS-PPP}} = L\left(\frac{1}{V_s} - \frac{1}{V_p}\right).
$$

### 4.2 PSS−PSP 主理论：`pss_minus_psp_from_pss_slope_with_profile`

1) 由 PSS 走时曲线估计 $p$。  
2) 结合 $h(x)$、$V_p/V_s(x)$ 迭代 $x_{\mathrm{conv}}$。  
3) 几何段长取

$$
L = \sqrt{h^2 + \left(x_{\mathrm{conv}} - x_{\mathrm{rec}}\right)^2}.
$$

4) 代入核心式：

$$
\Delta t_{\mathrm{PSS-PSP}} = L\left(\frac{1}{V_s} - \frac{1}{V_p}\right).
$$

说明：两者都用同一 $\Delta t$ 核心式，但 $L$ 的构造不同（有意保留为两种近似）。

---

## 5. 1D 诊断弹窗中的“统一口径”

在 `show_1d_diagnostics` 中，为了对照两条曲线的形态，PSS 侧也采用

$$
L = \frac{h}{\sqrt{1-(|p|V_p)^2}},\qquad
\Delta t = L\left(\frac{1}{V_s} - \frac{1}{V_p}\right).
$$

其中 $p$ 来自 PSS 斜率，因此这是一条“诊断展示曲线”，不等同于主算法中的几何迭代 $L$。

为避免近临界发散，代码增加了：

- $|p|V_p < 0.95$ 过滤  
- $L$ 上限截断  

---

## 6. 结论

- 观测定义一致且正确：

$$
\Delta t_{\mathrm{PPS-PPP}} = t_{\mathrm{PPS}} - t_{\mathrm{PPP}},\quad
\Delta t_{\mathrm{PSS-PSP}} = t_{\mathrm{PSS}} - t_{\mathrm{PSP}}.
$$

- 理论核心统一为：

$$
\Delta t = L\left(\frac{1}{V_s} - \frac{1}{V_p}\right).
$$

- 差异仅在 $L$ 的求法：  
  PPS−PPP 主算法用 $h/\cos\theta_p$；  
  PSS−PSP 主算法用 $\sqrt{h^2 + (x_{\mathrm{conv}}-x_{\mathrm{rec}})^2}$（含 $x_{\mathrm{conv}}$ 迭代）。
