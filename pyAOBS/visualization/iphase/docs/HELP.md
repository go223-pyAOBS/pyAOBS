# iphase 帮助

拾取震相（`tx.in`）分析工区：多文件可视化、PPS−PPP / PSS−PSP 时差、1D/2D/2Dequi 理论对比、PSP 导出与 r.in 联动。  
工具栏 **帮助**（或 `F1`）打开本页；窗口为**非模态**，可不关窗继续操作主界面。

```
[工具栏] 新建工区 | 打开工区 | 保存工区 | 帮助 | 退出
[阶段]   1 输入 | 2 走时图 | 3 输出
```

---

## 1. 推荐工作流

1. **新建工区**（或 **打开工区**）→ 目录含 `meta/iphase_project.json`、`inputs/`、`outputs/`、`cache/`
2. **1 输入**：添加走时 `tx.in`、海底地形、OBS/炮点深度、可选 `r.in` → **加载到走时图**
3. **2 走时图**：调参数条（时差模式 / 厚度·Vp·Vs / pois…），运行 2D 正演、1D 反演/诊断
4. **3 输出**：保存走时图、按模式导出 PSP、打开 `outputs/`
5. **保存工区**（`Ctrl+Shift+S`）写入工程状态（参数 + 文件路径）

下次：直接 **打开工区** → 自动恢复参数与输入路径；有走时文件时进入走时图页。

---

## 2. 工区文件

| 路径 | 内容 |
|------|------|
| `meta/iphase_project.json` | 工程主文件（名称、分析参数、工作流路径） |
| `inputs/` | 建议放置或引用走时、地形、炮深等输入 |
| `outputs/` | PSP 导出、图像等产出（建议） |
| `cache/` | 反演 / 2D 理论缓存（运行时也可写系统临时目录） |

工程 JSON 主要字段：

- `analysis`：时差模式、窗口、PSP 相位号、h/Vp/Vs、pois、开关等
- `workflow.tx_files` / `seafloor_path` / `shot_depth_path` / `rin_path`：相对工区或绝对路径

**保存工区** 只存工程元数据与路径，**不会**自动改写原始 `tx.in`。  
**导出PSP** / **保存图像** 才写出数据或图件。

未保存更改时切换工区或退出，会弹出短时 Yes/No（或 Save/Discard/Cancel）确认。

---

## 3. tx.in 与默认相位号

### 3.1 文件格式（与 RAYINVR / Fortran 工具兼容）

每行近似：`x  t  u  i`（Fortran 风格 `3f10.3,i10`）

| `i` | 含义 |
|-----|------|
| `0` | 炮点头（shot header）：`xshot,tshot,ushot` |
| `>0` | 该相位拾取：`x,t,u,phase_id` |
| `-1` | 文件结束 |

结构：`shot → picks… → shot → picks… → -1`。

OBS 模型距离：取 `tshot ≈ -1` 的炮头第一列众数（兼容异常文件时再退回全部炮头）。

**左右支**：炮头 `t=±1` 表示 L/R 支，**同 xshot 合并为一个 OBS**；列表标签 `L3/R5` 为左右侧拾取数。  
实现与 vedit / tomo2d 共用：`pyAOBS.modeling.rayinvr.tx_obs_catalog`。

### 3.2 GUI 默认相位（可在参数条改 PSP 号）

| 角色 | 默认相位号 |
|------|------------|
| PPP | **5** |
| PPS | **14** |
| PSS | **24** |
| PSP（输入/导出） | **40**（`PSP相位号`） |

CLI `combine` 的默认 ip1–ip5 与历史 `txconv.f` 示例不同；**GUI 以上表为准**。若你的拾取编号不同，请改 `PSP相位号`，并保证 PPP/PPS/PSS 与文件一致（当前 GUI 固定 5/14/24；若需改 PPP/PPS/PSS 号请在数据侧统一或用 CLI）。

### 3.3 命令行（无 GUI）

```bash
python -m pyAOBS.visualization.iphase info tx.in
python -m pyAOBS.visualization.iphase select tx.in -o tx.out --phases 5 14 24
python -m pyAOBS.visualization.iphase combine tx1.in tx2.in tx3.in -o1 tx1.out -o2 tx2.out
python -m pyAOBS.visualization.iphase rin-gui
```

---

## 4. 主图 2×2 说明

绘图区为 **PySide6** 主窗内的 `FigureCanvasQTAgg`（matplotlib Qt 后端），与 imodel / idata 科学图一致。

| 位置 | 内容 |
|------|------|
| **左上** | 折合走时（Vred≈7）：PPP / PPS / PSS（及 PSP）；2Dequi 可叠等效相；2D 可叠 `tx.out` 理论点；OBS 标注 |
| **右上** | 观测 **PPS−PPP**（及 1D 下 **PSS−PSP**）+ LocalLinear 拟合与误差统计 |
| **左下** | **理论 vs 观测** 时差（随 1D / 2D / 2Dequi）；理论−观测误差统计 |
| **右下** | **校正 PSP** 与原始 PSS（折合，Vred≈4）；多种 PSP 来源叠绘 |

- **单文件**：可混用偏移 / model distance（实现上以绘制逻辑为准）。  
- **多文件**：横轴统一 **model distance**；可选 **共享 y 轴**（右上/左下）。  
- 状态栏会提示理论模式、缓存命中、2Dequi 缺等效相等摘要。

---

## 5. 时差模式：1D / 2D / 2Dequi

| 模式 | 含义 | 典型输入 |
|------|------|----------|
| **1D** | 用厚度 `h`、`Vp`、`Vs` 与走时斜率估路径因子 \(L\)，算理论 PPS−PPP / PSS−PSP | 参数条厚度·Vp·Vs |
| **2D** | 读 RAYINVR `tx.out`（及 `r.in`/`v.in`）构理论时差；可 **运行2D正演** 刷新 | 工作目录中的 r.in、v.in、tx.out |
| **2Dequi** | 写/用 `tx_2Dequiv.in` 等等效构造；可选「2Dequi写等效PSP」后自动正演 | `PPS/PSS` 比值、等效开关 |

相关开关：

- **2D失败回退**：2D 理论失败时回退 1D（避免空白）。  
- **强制重算**：忽略部分缓存，强制重算。  
- **按r.in保留组过滤**：仅保留 `r.in` 中 `ivray` 且 `nray>0` 的相位组对应拾取（第二阶段过滤）。  
- **严格配对**：PPS−PPP / PSS−PSP 要求同道严格匹配；未勾选时在后相位点处用前相位拟合曲线取值（默认）。  
- **校正策略**（插值 / 严格 / 宽松）：控制受控插值差分对的构造策略。

公式口径详见同目录旁文档 `TIME_DIFF_FORMULAS.md`（观测均为「后相位 − 前相位」）。

---

## 6. 阶段页说明

### 6.1 输入

| 项 | 作用 |
|------|------|
| 走时文件 tx.in | 列表添加/移除多个 `*.in`；**合并多台站文件会按炮头 xshot 拆成多个 OBS 分析单元** |
| 海底地形 | 两列 `x depth`（km）；反演剖面用 |
| OBS/炮点深度 | 同步 r.in 炮点深度。**两列** `xshot zshot`，或 **三列** `station.lis`（站号 x z，取后两列） |
| r.in | 可选；供保留组过滤与相位组编辑 |
| 加载到走时图 | 校验路径 → 加载数据 → 切到「走时图」 |
| 打开 r.in 相位组编辑器 | 独立进程；保存后主窗自动重载 |
| **工区剖面预览** | 海底曲线 + OBS 三角；**横轴范围按 OBS 分布留边**（约 8% / 最少 5 km） |

指定海底/OBS 深度路径后自动刷新预览（不必先点「加载到走时图」）。`station.lis` 三列时标注站号。

OBS/炮点深度示例（两列，与 `examples/offset_depth_OBS.txt` 同型）：

```text
30.818    2.72
36.848    2.58
...
```

`tomo2d/.../station.lis` 为三列 `站号 x z`，现已兼容（不会把站号当成 x）。

### 6.2 走时图

| 项 | 作用 |
|------|------|
| 运行2D正演 / 1D反演 / 1D诊断 / r.in相位组 | 分析动作 |
| 预览/筛选 tx.in… | 对齐 tomo2d「预览 tx.in」：勾选 OBS/震相即时预览折合走时；可导出 `tx_OBS*_sel.in` |
| 参数条 | 时差模式、开关、厚度·Vp·Vs、窗口、pois… |
| 主图 2×2 | 见第 4 节；滚轮/左键拖/右键拖交互见第 9 节 |

### 6.3 输出

| 项 | 作用 |
|------|------|
| PSP导出模式 | `picked` / `theory2d` / `theory_pss` / `theory2Dequi` |
| 保存走时图 | 导出当前主图 |
| 导出PSP文件 | 按模式写出派生 tx |
| 打开 outputs 文件夹 | 打开工区 `outputs/` |

---

## 7. 工具栏（工区级）

| 按钮 | 作用 |
|------|------|
| 新建工区 | 选目录 → 建布局 + `meta/iphase_project.json` |
| 打开工区 | 选 `iphase_project.json` → 恢复参数与文件 |
| 保存工区 | 写入当前分析参数与路径（`Ctrl+Shift+S`） |
| 帮助 | 打开本说明（`F1`） |
| 退出 | 关主窗（`Ctrl+Q`）；有未保存更改时询问 |

---

## 8. 参数条速查（走时图页）

参数条按当前**时差模式**分行显示；鼠标悬停控件可看说明。

| 分组 | 何时显示 | 控件 |
|------|----------|------|
| **通用** | 始终 | 时差模式、共享y轴、严格配对、强制重算、按r.in保留组过滤、校正策略、PSP相位号、OBS标注Y |
| **1D** | 模式=1D | 厚度·Vp·Vs、线性窗口、平滑半窗、二维参数场 |
| **2D** | 模式=2D | 2D失败回退、pois左支/右支（单行） |
| **2Dequi** | 模式=2Dequi | 上项 + PPS/PSS、2Dequi写等效PSP（合并为单行，标题为【2Dequi】） |

| 控件 | 说明 |
|------|------|
| 时差模式 | `1D` / `2D` / `2Dequi` |
| 共享y轴 | 多文件时统一右上/左下 y 范围 |
| 严格配对 / 强制重算 / 2D失败回退 / 按r.in保留组过滤 | 见第 5 节 |
| 校正策略 | 插值 / 严格 / 宽松 |
| 厚度 · Vp · Vs | 1D 理论与反演初值 |
| 线性窗口 | LocalLinear 拟合窗口点数（7–15） |
| 平滑半窗 | 密点平滑半窗（0 关闭） |
| 二维参数场 | `point` / `eff`（反演剖面场模式） |
| PSP相位号 | 输入/导出 PSP 的 phase id |
| OBS标注Y(s) | 主图 OBS 三角标注纵坐标 |
| pois左支 / 右支 | 分别写回并正演；**只更新 OBS 对应侧**理论点/时差，另一侧保留缓存 |
| 保存 pois | 【2D】/【2Dequi】参数行末尾按钮；写入 `pois_branches.json` 与工区 analysis |
| PPS/PSS | 2Dequi 比值 |
| 2Dequi写等效PSP | 是否写等效 PSP 并触发正演 |

参数条可折叠（「参数」按钮），以最大化绘图区。PSP 导出模式在「输出」页。

---

## 8. r.in 相位组编辑器

- 入口：工具栏 **r.in相位组**，或  
  `python -m pyAOBS.visualization.iphase.rin_phase_groups_gui [r.in]`  
  / `python -m pyAOBS.visualization.iphase rin-gui`
- 编辑 `ray / nrbnd / rbnd / ncbnd / cbnd / nray / ivray` 等相位组数组。
- 与主窗 **分进程** 运行，避免双 GUI 抢事件循环。
- 主窗轮询 `r.in` 修改时间：**保存即自动重载** 走时过滤/显示，无需先关编辑器。
- 若当前上下文猜不到 `r.in`，会弹出文件选择框。

---

## 9. 快捷键与窗口行为

| 操作 | 快捷键 |
|------|--------|
| 添加走时文件 | `Ctrl+O`（切到「1 输入」并弹出文件选择） |
| 保存走时图 | `Ctrl+S`（等同输出页「保存走时图…」） |
| 保存工区 | `Ctrl+Shift+S` |
| 帮助 | `F1` / 工具栏「帮助」 |
| 退出 | `Ctrl+Q` / 工具栏「退出」 / 关窗 |

说明：快捷键挂在主窗 `QAction` 上；焦点在参数输入框时通常仍可用。  
工具窗、结果提示、帮助均为 **非模态**。  
仅 `QFileDialog` 与关闭/丢弃确认保持短时模态。  
参数条下拉框在重绘前会延迟，避免弹层卡住（`connect_combo_deferred`）。

绘图导航（手势对齐 idata / zplotpy / relocation 的 pyqtgraph ViewBox；**非键盘快捷键**）：

| 操作 | 效果 |
|------|------|
| 左键拖拽 | 平移当前子图 |
| 右键拖拽 | 连续缩放（向右/上放大，非框选） |
| 滚轮 | 以光标为中心缩放 |
| 双击子图 | 该子图复位到数据范围 |

说明：波形工区底层是 **pyqtgraph**；iphase 主图因 2×2 科学曲线仍用 **matplotlib**，鼠标操作已按同一套 ViewBox 习惯实现。  
换一批走时文件会重置视图；参数微调重绘时会尽量保留当前缩放。

---

## 10. 启动与关于

```bash
python -m pyAOBS.visualization.iphase.gui
```

- Linux/WSL 若默认 Wayland：启动时未设置 `QT_QPA_PLATFORM` 会自动优先 `xcb`，避免最大化主窗弹出结果框时 `xdg_wm_base` 协议崩溃；需要原生 Wayland 时可 `export QT_QPA_PLATFORM=wayland`。
- 包：`pyAOBS.visualization.iphase`
- GUI：`pyAOBS.visualization.iphase.gui`（PySide6）
- 工程文件：`meta/iphase_project.json`；目录 `inputs/`、`outputs/`、`cache/`
- 文档：`visualization/iphase/docs/HELP.md`（本文件）
- 时差公式：`visualization/iphase/TIME_DIFF_FORMULAS.md`（TeX/PDF 见 `_archive/formula_build/`）
- 大批量本地样例：`visualization/iphase/_archive/datasets/txin/`（演示数据用 `examples/`）
- 核心库：`io_tx` / `phase_filter` / `phase_combine` / `theory2d_service` / `equi2d` / `theoretical_ppp_pps`
- 对应历史 Fortran：`txphase.f`（筛选）、`txconv.f`（PPP/PPS/PSS 组合）
- Workbench 插件：`iphase.gui`（审计包装启动同一 GUI 入口）

上游波形拾取见 zplotpy；姿态校正见 relocation；速度模型见 imodel；数据转换见 idata。
