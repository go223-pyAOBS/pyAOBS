# iphase 工程实现策略（拾取震相处理与可视化）

## 一、工程定位

`iphase` 是专门面向 **拾取震相（tx.in 格式）** 的处理与可视化子工程，目标：

- 提供结构化的 `tx.in` 读写与数据模型；
- 重写并扩展现有 Fortran 工具 `txphase.f` / `txconv.f` 的功能；
- 支持交互式的相位筛选、组合与质量控制可视化；
- 与现有 `pyAOBS` / `zplotpy` 流程集成，为走时反演和建模提供更现代的前处理工具链。

现有 Fortran 程序只覆盖两个典型操作：

- `txphase.f`：从单个 `tx.in` 中 **筛选特定相位**，输出新的 `tx.in`；
- `txconv.f`：在三个 `tx.in` 中 **匹配同一测点的多相位（PPP/PPS/PSS）**，构造差时（PPS-PPP）与修正 PSP 等新相位并输出。

`iphase` 将在保持与这些工具兼容的前提下，给出 Python 化、可组合、可视化的上层封装。

---

## 二、tx.in 数据模型抽象（参考 vedit 现有实现）

从 `txphase.f` / `txconv.f` 可知，`tx.in` 每行格式为：

- `x, t, u, i`（Fortran 格式 `3f10.3, i10`）

推断语义：

- `x`：偏移或接收点位置；
- `t`：到时；
- `u`：不确定度或权重；
- `i`：相位编号或特殊标记：
  - `i == 0`：炮点头（shot header）；
  - `i > 0`：具体相位拾取；
  - `i == -1`：文件结束标记。

文件结构是一个「多炮点 tx 表」，按顺序排布：

> `shot(i=0)` → 若干 `pick(i>0)` → `shot(i=0)` → … → `i=-1`

在 `iphase` 中建议抽象为，并 **尽量复用/对齐当前 `vedit` 系统中对 `tx.in` 的读取与使用方式**：

- `Shot`：包含炮点 id、`xshot,tshot,ushot` 以及该炮所有拾取 `picks: List[Pick]`；
- `Pick`：包含 `x,t,u,phase_id` 等字段；
- `PhaseDataset`：封装所有炮点，支持按相位、按炮点索引与切片。

---

## 三、模块划分

建议在 `pyAOBS/visualization/iphase/` 目录下组织如下模块：

- `io_tx.py`
  - `read_tx(path) -> PhaseDataset`：解析 tx.in 文件；
  - `write_tx(dataset, path)`：写回 tx.in，保持与 Fortran 工具兼容的格式。

- `models.py`
  - 定义 `Shot`、`Pick`、`PhaseDataset` 等核心数据结构。

- `phase_filter.py`（对应并扩展 `txphase.f`）
  - `select_phases(dataset, phase_ids) -> PhaseDataset`：保留指定相位；
  - `exclude_phases(dataset, phase_ids) -> PhaseDataset`：剔除指定相位；
  - 保证仅当某炮至少有一条保留拾取时才输出对应炮头。

- `phase_combine.py`（对应并扩展 `txconv.f`）
  - 泛化的多相位组合框架，支持：
    - 多个输入 `PhaseDataset` 或同一 dataset 内多相位；
    - 炮点匹配（基于 `xshot,tshot`，带容差）；
    - 接收点位置匹配（基于 `x`，带容差）；
    - 可选时间容差。
  - 提供典型算子：
    - `combine_ppp_pps_pss_to_psp(...)`：复刻 `txconv.f` 的 PPP/PPS/PSS → PPS-PPP 与修正 PSP 的逻辑。

- `qc_metrics.py`
  - 统计/质量控制函数，如：
    - 每相位拾取数、每炮拾取数；
    - 偏移–到时分布；
    - 相位对（如 PPS-PPP）的差时统计。

- `plot_tx.py`
  - 基于 matplotlib 或现有 zplot 风格的静态可视化：
    - `plot_offset_time(dataset, phase_ids=None)`：偏移–到时散点图；
    - `plot_phase_coverage(dataset)`：相位覆盖图。

- `iphase_gui.py`（或与 `zplotpy` 集成）
  - 提供交互式界面：加载 单个或者多个tx.in、选择相位、应用筛选/组合、查看结果。

---

## 四、核心功能需求

### 4.1 tx.in 文件管理

- 读写 tx.in，支持批量处理；
- 做基础校验：
  - 炮头/拾取顺序合法；
  - `i` 值范围合理；
  - 文件中是否存在 `i=-1` 结束标记等。

### 4.2 相位筛选与编辑（Python 版 txphase）

- **筛选特定相位**：
  - 与 `txphase.f` 行为一致：输入待保留相位号列表，输出只包含这些相位的 tx.in；
  - 若某炮没有任何被选相位，则该炮点整体被丢弃（不输出炮头）。

- **相位剔除**：
  - 可选反向操作：剔除指定相位，仅保留其余相位。

- **统计信息**：
  - 返回每个相位的拾取计数、炮点计数等，便于做 QC。

### 4.3 多相位组合与派生相位（Python 版 txconv）

- **匹配逻辑说明**：
  - 匹配成功与否主要取决于 **PPP 与 PPS**（同一炮点、同一接收点 x）；
  - PSS 的偏移可能不在 PPP/PPS 范围内，若无一一对应偏移，可后续通过 PPP 与 PPS 的差值矫正得到 PSP'。

- **多文件/多相位匹配**：
  - 基于炮点头（`xshot,tshot`）和接收点位置 `x` 的双重匹配；
  - 支持设置 `dx_tol, dt_tol` 容差，避免因数值微差导致匹配失败。

- **组合相位构造**：
  - 复刻 `txconv.f` 核心逻辑：
    - 在 PPP / PPS / PSS 同一测点都存在时：
      - 构造新相位 1：`t_new1 = t_PPS - t_PPP`；
      - 构造新相位 2：`t_new2 = t_PSS - (t_PPS - t_PPP)`（示意，依原代码）；
      - 不确定度 `u` 按和或其他规则组合；
      - 位置 `x_new` 为 `x` 的平均或采用某一相位的值。

- **可扩展的组合规则**：
  - 抽象为通用框架，允许用户定义：
    - 输入相位集合；
    - 匹配条件；
    - 到时与不确定度的组合公式；
    - 输出相位编号。

### 4.4 可视化与交互

- **偏移–到时散点图**：
  - 每个拾取画为 `(x, t)` 点，按相位/不确定度着色；
  - 支持按相位过滤。

- **覆盖与 QC 图**：
  - 显示相位在偏移/深度上的分布；
  - 显示不同相位对之间的差时分布。

- **图形界面**（与 `zplotpy` 或单独 GUI 结合）：
  - 左侧：文件/相位/规则选择；
  - 右侧：图形显示与输出日志；
  - 支持导出新的 tx.in 文件，并显示统计信息。

---

## 五、实现步骤（推荐路线）

### 步骤 1：I/O 与数据模型

1. 在 `pyAOBS/visualization/iphase/` 创建：
   - `io_tx.py`：实现 `read_tx` / `write_tx`；
   - `models.py`：定义 `Shot` / `Pick` / `PhaseDataset`。
2. 用一个简单脚本测试：
   - `dataset = read_tx('tx.in')`；
   - `write_tx(dataset, 'tx_copy.in')`；
   - 用 diff 或 Fortran 工具检查两者一致性。

### 步骤 2：功能对齐 Fortran 版本

1. 在 `phase_filter.py` 中实现 Python 版 `select_phases`：
   - 严格对齐 `txphase.f` 的行为（炮头写出时机、统计逻辑等）。
2. 在 `phase_combine.py` 中实现 PPP/PPS/PSS 组合逻辑：
   - 复刻 `txconv.f` 的匹配与组合规则；
   - 支持简单配置（相位号、容差）。
3. 编写测试：
   - 以相同 tx.in 输入，比较 Fortran 与 Python 输出的差异（允许极小浮点误差）。

### 步骤 3：可视化与 CLI / GUI

1. 在 `plot_tx.py` 中实现基本绘图函数；
2. 新建简单命令行工具：
   - 如 `iphase-select` / `iphase-combine` 等，用 argparse 封装；
3. 根据需要在 `visualization/iphase` 中增加 GUI：
   - 加载/保存 tx.in；
   - 相位列表勾选；
   - 一键应用筛选/组合并在图上更新结果。

### 步骤 4：与现有流程集成

1. 在 `pyAOBS.visualization.iphase` 中导出关键 API：
   - `pyAOBS.visualization.iphase.read_tx` / `write_tx`；
   - `pyAOBS.visualization.iphase.select_phases` / `combine_phases` 等。
2. 在走时反演/建模前处理脚本中，统一使用 `iphase` 完成 tx 预处理。

---

## 六、理论 PPS-PPP 与观测对比策略

### 6.1 目标

1. **理论 PPS-PPP**：已知沉积层 P 波速度 \(V_p\)、S 波速度 \(V_s\) 和厚度 \(h\) 时，计算各偏移距下的理论 PPS-PPP 走时差；
2. **观测 PPS-PPP**：从真实拾取的 PPP/PPS 得到各偏移距的 PPS-PPP，并插值到任意偏移距；
3. **对比与残差**：理论 vs 观测，统计偏离误差，用于 QC 或沉积层参数约束。

### 6.2 理论 PPS-PPP 计算

#### 6.2.1 单层水平沉积层解析公式（近似 / 初值）

对单一水平沉积层（厚度 \(h\)，\(V_p\)、\(V_s\)），PPP 与 PPS 在基底反射后上行路径的差异为：

- PPP 上行：P 波，走时 \(t_P = h/(V_p \cos i_p)\)
- PPS 上行：S 波，走时 \(t_S = h/(V_s \cos i_s)\)

由 Snell 定律 \(p = \sin i_p/V_p = \sin i_s/V_s\)（\(p\) 为射线参数），得：

\[
\Delta t_{\text{PPS-PPP}} = \frac{h}{V_s \cos i_s} - \frac{h}{V_p \cos i_p}, \quad
\cos i_p = \sqrt{1 - (p V_p)^2}, \quad \cos i_s = \sqrt{1 - (p V_s)^2}
\]

- **任意偏移距**：从偏移距 \(x\) 反推射线参数 \(p\)。单层时 \(i_p = \arctan(x/(2h))\)，\(p = \sin i_p/V_p\)；有水层时用 Newton 迭代求解。理论 PPS-PPP 随 offset 变化。

**实现（近似模型）**：在 `iphase/theoretical_ppp_pps.py` 中提供：

- `pps_minus_ppp_1d_layer(h, vp, vs, offsets, h_water=0, v_water=1.5)`  
  单层水平沉积层（含可选水层）的理论 PPS-PPP，基于射线参数 \(p(\text{offset})\) 与 Snell 定律，**PPS-PPP 随偏移距变化**；
- `fit_sediment_params(pairs, h0, vp0, vs0, ...)`  
  反演沉积层参数，使理论值与观测拾取在各自偏移距处尽量一致；
- `pps_minus_ppp_from_rayinvr(working_dir, phase_ppp, phase_pps)`  
  从 RAYINVR 的 tx.out 构造 2D 模型下的理论 PPS-PPP 插值函数。

#### 6.2.2 基于 RAYINVR 的理论走时（推荐，用于广角 PPP/PPS）

对于你关心的**广角折射 PPP + 基底转换 PPS**，应使用完整的射线追踪：
PPP、PPS 在壳/幔高速层中的折射路径、转换位置都由速度模型决定，
简单 1D 近似（6.2.1）只能提供趋势，不足以精确描述。

当存在 2D 速度模型（v.in）时，可复用 `pyAOBS.visualization.zplotpy.theoretical_traveltime.TheoreticalTravelTimeCalculator` 与 `pyAOBS.modeling.rayinvr.RayinvrWrapper`：

1. **分离震相**：在 tx.in 中，PPP 与 PPS 使用不同相位号（如 5=PPP、14=PPS）；
2. **运行 RAYINVR**：对每个炮点，分别计算 PPP、PPS 的理论走时，得到 \((x, t_{\text{PPP}})\)、\((x, t_{\text{PPS}})\)；
3. **差值**：\(\Delta t_{\text{theory}}(x) = t_{\text{PPS}}(x) - t_{\text{PPP}}(x)\)。

参考路径：`visualization/zplotpy/theoretical_traveltime.py`，其中已实现：

- `TheoreticalTravelTimeCalculator`：基于 RAYINVR 计算理论走时；
- `get_theoretical_times()` / 从射线数据提取走时；
- 按相位（ipf）区分观测点。

需确保 RAYINVR 的 v.in / r.in 中定义了 PPP、PPS 对应的反射路径，并在 tx.in 中正确设置相位号。

### 6.3 观测 PPS-PPP（离散拾取点）

拾取数据仅提供离散的 \((offset, t_{\text{PPS-PPP}})\) 点，**无需插值**。现有接口已支持：

- `compute_ppp_pps_diff_pairs(ds_ppp, ds_pps)` → `(model_dist, true_offset, t_diff)` 列表；
- `stats_ppp_pps_diff_by_offset(pairs, degree=2)` → 正/负偏移距的拟合曲线、残差、RMS（用于 QC）。

阶段二的核心是：**在观测偏移距处，使理论 PPS-PPP 与拾取观测值尽量一致**，通过调整沉积层参数实现。

### 6.4 理论 vs 观测对比流程

1. **输入**：沉积层初值 \((h, V_p, V_s)\)；拾取数据 `ds_ppp`、`ds_pps`；
2. **观测**：`compute_ppp_pps_diff_pairs` → 得到离散点 `(offset_i, t_obs_i)`；
3. **理论**：调用 `pps_minus_ppp_1d_layer` 在相同 `offset_i` 处计算 `t_theory_i`；
4. **残差**：`residual_i = t_obs_i - t_theory_i`，统计 mean、std、RMS；
5. **反演/调整**：调整 \(h\)、\(V_p\)、\(V_s\)，使残差最小（如 L2 范数最小化），得到与观测匹配的沉积层参数；
6. **可视化**：在同一幅图中绘制理论曲线、观测点及残差分布。

### 6.5 参考代码

| 功能 | 模块路径 | 说明 |
|------|----------|------|
| **理论 PPS-PPP** | `visualization/iphase/theoretical_ppp_pps.py` | `pps_minus_ppp_1d_layer`、`fit_sediment_params`、`pps_minus_ppp_from_rayinvr` |
| 理论走时（RAYINVR） | `visualization/zplotpy/theoretical_traveltime.py` | `TheoreticalTravelTimeCalculator`、`get_theoretical_times()` |
| RAYINVR 封装 | `modeling/rayinvr/rayinvr_wrapper.py` | `RayinvrWrapper`、射线路径、走时、相位 |
| 2D 射线追踪 | `modeling/rayinvr/ray_tracer.py` | `RayTracer`、`VelocityModel`（需 Zelt 模型） |
| 观测 PPS-PPP | `visualization/iphase/phase_combine.py` | `compute_ppp_pps_diff_pairs` |
| 拟合与统计 | `visualization/iphase/qc_metrics.py` | `stats_ppp_pps_diff_by_offset`、`DiffFitStats` |

### 6.6 实现优先级

1. **阶段 1**：实现 `pps_minus_ppp_1d_layer`（射线参数版，含可选水层），并与现有 `plot_difference_offset_with_fit` 叠加显示理论曲线；
2. **阶段 2**：在观测偏移距处计算理论 PPS-PPP，与拾取值对比；实现沉积层参数 \((h, V_p, V_s)\) 的调整/反演，使理论值与观测值尽量一致；
3. **阶段 3**：在具备 v.in 等条件下，对接 `TheoreticalTravelTimeCalculator`，支持 2D 模型的理论 PPS-PPP。

---

## 七、后续可扩展方向

- 支持更多输入格式（如 OBS 专用格式）并转换到内部 `PhaseDataset`；
- 在可视化界面中加入交互式拾取编辑（添加/删除/拖动点）；
- 加入与速度模型/射线路径叠加显示的功能，进行联合 QC；
- 与 denoise 模块联动，对去噪前后拾取得到的差异做分析。

