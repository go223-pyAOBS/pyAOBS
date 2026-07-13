# OBS 海底定位与地震摆方位确定：实现策略

## 一、目标与背景

海底地震仪（OBS）自由落体投放后，其**海底位置**和**水平分量方位**均未知。本策略参考 **ppol** 的思路，提出**同时利用直达波走时和地震波形**的联合反演方案：

| 信息来源     | 物理量       | 反演目标           | 典型精度              |
|--------------|--------------|--------------------|------------------------|
| 直达波走时   | 炮-OBS 走时  | OBS 位置 (x, y, z) | 水平 ~20–35 m，深度 ~10 m |
| P 波偏振     | 水平分量波形 | 水平分量方位角     | ~5–15°                  |

---

## 二、方法论概述

### 2.1 直达波走时定位（Position Relocation）

**原理**：利用主动源（气枪/电火花）直达水波或直达 P 波走时，建立炮点坐标、OBS 坐标与海水速度的约束方程，通过反演得到 OBS 位置。

**常见实现方式**：

- **多维扫描**：在 (x, y, z_obs, v_water) 空间中搜索，使观测走时与理论走时的残差最小
- **最小二乘反演**：线性化走时方程，迭代求解位置修正量
- **蒙特卡洛**：模拟 OBS 落底过程，结合最小二乘获得位置估计及不确定性

**数据流**：
```
炮点坐标 (x_shot, y_shot, z_shot) + 直达波拾取时间 t_obs
    → 走时方程: t_obs = dist(shot, OBS) / v_water (+ 海底地形修正)
    → 反演 OBS (x, y, z)
```

### 2.2 地震摆方位确定（Orientation，参考 ppol）

**原理**（ppol，Scholz et al. 2017）：比较**理论反方位角**与**P 波初至偏振方向**，通过统计或拟合得到水平分量的旋转角（方位角偏移）。

**核心步骤**：

1. **事件与台站**：使用地震事件或已知位置的主动源
2. **理论反方位角**：由事件坐标与台站坐标计算 backazimuth
3. **P 波偏振**：对 P 波初至窗口做 2D/3D 主成分分析（PCA），得到质点运动方向
4. **方位偏移**：理论反方位角 − 观测偏振方向 = 水平分量旋转角
5. **统计与筛选**：对多事件结果做加权平均、离群值剔除，可选考虑各向异性等

**数据流**：
```
事件坐标 + OBS 坐标 → 理论反方位角
P 波波形 (E, N 或 H1, H2, Z) → PCA 偏振分析 → 观测偏振方向
两者相减 → 方位角偏移 → 统计得到最终方位
```

### 2.3 位置误差与方位差的耦合（用于联合校正）

**关键点**：若 OBS **位置有误**，炮点→接收点的几何方向会随之改变，从而**理论反方位角**也会偏。此时“理论反方位角 − 观测偏振”的差值**不再只是地震摆方位偏移**，而是：

- **真实方位偏移**（传感器安装偏差，与炮点方向无关）
- **位置误差引起的“视方位差”**（随炮点方位变化）

因此：

1. **方位残差中蕴含位置信息**：不同炮点方向上的方位残差若呈现**系统性随方位角变化**（例如呈正弦形态），说明很大一部分来自位置误差，可用于反推 OBS 位置修正量。
2. **可用于校正**：在得到初步位置和初步方位后，用**方位残差随炮点方向的变化**约束位置更新；再用更新后的位置重算理论反方位角、重新估计方位。如此迭代或联合反演，可同时改进位置与方位。

下文第四节、第五节给出如何把该耦合纳入实现（方位残差反演位置、联合/迭代工作流）。

---

## 三、与 ppol 的对应关系

| ppol 功能/概念        | 本策略对应实现                         |
|------------------------|----------------------------------------|
| 理论反方位角           | 由事件+OBS 坐标计算，OBS 坐标可来自走时定位 |
| P 波偏振 (2D/3D PCA)   | ObsPy `polarization_analysis` 或自实现 |
| config.yml 配置        | 本模块 `relocation_config.yaml`        |
| results_retained / all | 保留 / 全部 测量结果的输出与评估       |
| 统计与筛选             | 离群值剔除、加权平均、可选各向异性拟合 |

**差异与扩展**：

- ppol 假定 OBS 位置已知，本策略**联合反演位置+方位**
- 本策略支持**主动源直达波**作为事件（炮点坐标已知），而 ppol 主要面向天然地震
- 位置反演需炮点坐标、拾取走时，可与 pyAOBS 现有拾取、记录文件对接

---

## 四、实现策略（分阶段）

### 阶段 1：数据结构与接口

**目标**：定义统一输入输出，与现有 pyAOBS 模块对接。

**1.1 输入数据结构**

```python
# 定位输入
@dataclass
class PositionInput:
    shot_coords: List[Tuple[float, float, float]]  # (x, y, z) km
    obs_travel_times: List[float]                   # 直达波走时 s
    water_velocity: float = 1.5                     # 海水速度 km/s
    obs_initial: Tuple[float, float, float]         # OBS 初始猜测 (x, y, z)

# 定向输入（ppol 风格）
@dataclass
class OrientationInput:
    event_coords: List[Tuple[float, float, float]]  # 事件 (x, y, z)
    obs_coords: Tuple[float, float, float]          # OBS 位置（可用阶段1结果）
    waveforms: List[ObsPy Stream]                   # 三分量波形
    p_arrival_times: List[float]                    # P 波到时
    # 或使用 SAC/MSEED 路径 + 事件目录
```

**1.2 与现有模块对接**

- `data_loader`：Z 格式 → 道头、炮检距、记录文件（炮点坐标）
- `pick_manager`：拾取字 1 常为直达波 → 拾取时间
- `theoretical_traveltime`：提供理论走时接口，用于残差计算与对比

**1.3 输出**

```python
@dataclass
class RelocationResult:
    obs_x: float           # OBS 东向位置 km
    obs_y: float           # OBS 北向位置 km
    obs_z: float           # OBS 深度 km（海底面下或水深）
    orientation: float     # 水平分量方位偏移（度，北为0，顺时针）
    position_uncertainty: Optional[Tuple[float, float, float]]  # (σx, σy, σz)
    orientation_uncertainty: Optional[float]
```

---

### 阶段 2：直达波走时定位

**2.1 走时模型**

- 直达水波：`t = d / v_water`，`d = sqrt((x_s-x_o)^2 + (y_s-y_o)^2 + (z_s-z_o)^2)`
- 可选：海底起伏、等效海水速度随深度变化

**2.2 反演算法**

1. **L2 最小二乘**：线性化走时方程，迭代更新 OBS 位置
2. **网格/随机搜索**：在 (x, y, z) 空间扫描，最小化 `sum((t_obs - t_pred)^2)`
3. **可选**：贝叶斯/蒙特卡洛，输出位置不确定性

**2.3 与拾取系统对接**

- 从 `pick_manager` 或头文件读取拾取字 1（或指定字）作为直达波到时
- 从 `records` / `rfile` 或记录文件读取炮点 (xmod, ymod) 及炮点深度
- 调用 `TheoreticalTravelTimeCalculator` 或等价接口计算理论走时（若用射线追踪）

---

### 阶段 3：P 波偏振定向（ppol 风格）

**3.1 理论反方位角**

```python
def backazimuth(event_lon, event_lat, obs_lon, obs_lat) -> float:
    # 或 UTM 坐标系下的等效计算
```

**3.2 偏振分析**

- 使用 ObsPy `obspy.signal.polarization.polarization_analysis`
- 在 P 波到时前后取短窗（如 -0.1 s 到 +0.3 s）
- 提取方位角、倾角，得到观测偏振方向

**3.3 方位偏移与统计**

- 对每个事件：`offset = backazimuth_theory - polarization_observed`
- 离群值剔除（如 3σ、MAD）
- 加权平均（可按 SNR、射线路径等加权）
- 输出 `orientation` 及不确定性

**3.4 配置（类似 ppol config.yml）**

```yaml
# relocation_config.yaml
orientation:
  filter: [1.0, 20.0]        # 带通 Hz
  window_before: 0.1         # P 波前 s
  window_after: 0.3          # P 波后 s
  method: "2D"               # 2D or 3D PCA
  culling_sigma: 3.0         # 离群值剔除
```

---

### 阶段 4：联合工作流与“方位残差→位置校正”

**4.1 顺序流程（基线）**

```
1. 加载数据 (DataLoader) + 拾取 (PickManager)
2. 提取炮点坐标、直达波走时
3. 位置反演 → OBS (x, y, z)
4. 使用 (x, y, z) 计算理论反方位角
5. 对三分量波形做 P 波偏振分析
6. 统计得到 orientation
7. 输出 RelocationResult
```

**4.2 利用方位残差做位置校正（推荐）**

位置不对会导致炮点→接收点方向变化，从而理论反方位角与“计划/真值”有差异，该差异会混入“理论−观测”的方位残差。**该信息可用于校正位置**，两种实现方式：

**(A) 方位残差反演位置修正量**

- 设当前 OBS 位置为 (x₀, y₀)，真实方位偏移为 φ（未知）。
- 对每个炮 i：理论反方位角 θᵢ = f(炮点坐标, x₀, y₀)；观测偏振方向 ψᵢ 由波形得到。
- 若仅存在方位偏移：残差 Δᵢ = θᵢ − ψᵢ 应近似为常数 φ（与炮点方向无关）。
- 若存在位置误差：θᵢ 会随炮点相对 OBS 的方位系统性偏转，Δᵢ 会随炮点方位角 αᵢ 变化（近似为 Δᵢ ≈ φ + g(αᵢ, Δx, Δy)，其中 g 由位置误差 (Δx, Δy) 引起）。
- **做法**：构造目标函数，例如最小化 Σᵢ (Δᵢ − φ − g(αᵢ, Δx, Δy))²，联合求解 (Δx, Δy, φ)，或先拟合 Δᵢ(αᵢ) 的形态得到 (Δx, Δy)，再估计 φ。
- 得到 (Δx, Δy) 后更新 OBS 位置，再重算理论反方位角、重新统计 orientation。

**(B) 迭代联合**

1. 用直达波走时得到初始位置 (x, y, z)。
2. 用该位置计算各炮的理论反方位角，做 P 波偏振定向，得到每个炮的方位残差 Δᵢ。
3. 若方位残差随炮点方向有明显系统性变化：用 (A) 反演位置修正 (Δx, Δy)，更新 (x, y)，回到步骤 2；否则进入步骤 4。
4. 用最终位置再算一次理论反方位角，对 Δᵢ 做统计（离群剔除、加权平均）得到最终 orientation。

**实现要点**：

- 需要**多方位炮点**（不同 αᵢ 分布均匀更好），否则位置在某一方向上的分量难以从方位残差中分辨。
- 方位残差对位置偏导数：∂θ/∂x_obs、∂θ/∂y_obs 可由几何关系解析或数值求导，用于最小二乘或高斯–牛顿迭代。
- 与走时反演可联合：目标函数 = 走时残差加权平方和 + 方位残差加权平方和，同时反演 (x, y, z) 与 φ。

**4.3 小结：为何要纳入校正**

| 现象 | 原因 | 用途 |
|------|------|------|
| 方位残差随炮点方向变化 | 位置误差导致理论反方位角随几何变化 | 用残差形态反推 (Δx, Δy) |
| 方位残差近似常数 | 以方位偏移为主，位置误差小 | 直接统计得到 orientation |
| 走时+方位联合目标函数 | 两类观测共同约束位置与方位 | 一次反演同时得到位置与方位 |

---

### 阶段 5：GUI 与可视化

**5.1 集成到 zplotpy**

- 菜单或工具栏："OBS 定位与定向"
- 选择 OBS 号、拾取字、记录范围
- 配置海水速度、滤波参数
- 显示位置残差、方位角分布、不确定性

**5.2 可视化内容**

- 炮点–OBS 平面分布与走时残差
- 各事件偏振方向与理论反方位角（玫瑰图或极坐标）
- 定位前后的理论走时 vs 观测走时对比

---

## 五、模块结构建议

```
pyAOBS/visualization/zplotpy/relocation/
├── __init__.py
├── IMPLEMENTATION_STRATEGY.md    # 本文档
├── config.py                     # 配置加载
├── position.py                   # 直达波走时定位
│   ├── travel_time_residual()
│   ├── invert_position()
│   └── scan_position()
├── orientation.py                # P 波偏振定向（ppol 风格）
│   ├── compute_backazimuth()
│   ├── polarization_analysis()
│   └── compute_orientation()
├── joint.py                      # 联合工作流与方位→位置反馈
│   ├── run_relocation()
│   ├── position_correction_from_azimuth_residuals()  # 用方位残差反演 (Δx, Δy)
│   └── run_relocation_iterative()                    # 迭代：位置↔方位联合
└── gui.py                        # 可选 GUI 对话框
```

---

## 六、依赖与参考

**依赖**：

- ObsPy（波形、偏振分析、坐标计算）
- NumPy / SciPy（优化、线性代数）
- 现有 pyAOBS：`data_loader`, `pick_manager`, `theoretical_traveltime`

**参考**：

- **ppol**：https://ppol.readthedocs.io, https://gitlab.com/johnrobertscholz/ppol  
  Scholz et al. (2017), GJI, doi:10.1093/gji/ggw426
- **OBS 定位**：直达水波反演，多维扫描，误差 ~20–35 m（如南海 OBS 位置校正相关文献）
- **OrientPy / orient**：Rayleigh 波偏振定向，可作为补充方法

---

## 七、实施优先级

| 优先级 | 任务                                         | 预计工作量 |
|--------|----------------------------------------------|------------|
| P0     | 阶段 1 数据结构、与拾取/记录对接             | 1–2 天     |
| P0     | 阶段 2 直达波定位（最小二乘）                 | 2–3 天     |
| P1     | 阶段 3 P 波偏振定向                          | 2–3 天     |
| P2     | 阶段 4 联合工作流 + **方位残差→位置校正**     | 2–3 天     |
| P3     | 阶段 5 GUI 与可视化                          | 2–3 天     |

建议先完成 P0，验证定位与定向后再实现“方位残差反演位置修正”与迭代联合（阶段 4.2），这样位置误差会反映在方位差中，并用于校正。
