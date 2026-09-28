# imodel 帮助

交互式速度模型分析工区：加载 Vp/Vs（Zelt `v.in`、TOMO2D `smesh` 或 Grid）、剖面与物性、重力、岩石散点、LIP Petrology 桥接。  
工具栏 / 菜单 **帮助**（或 `F1`）打开本页；窗口为**非模态**，可不关窗继续操作主界面。

```
[工具栏] 新建工区 | 打开工区 | 保存工区 | Load Vp/Vs | Save Figure | …
[主区]   速度剖面图（全宽）
[右侧]   可折叠参数：界面 · Profile · 物性 · Petrology · Rocks …
[底栏]   工区徽章 · 状态
[日志]   主图下方信息区
```

---

## 1. 推荐工作流

1. **新建工区**（或 **打开工区**）→ 目录含 `meta/imodel_project.json`、`inputs/`、`outputs/`、`cache/`
2. **Load Vp**：默认过滤器为 Zelt `v.in`；也可选 TOMO2D `smesh`（`*.smesh` / `*smesh*`）或 Grid（`.grd` / `.nc`）
3. （可选）**Load Vs**、加载/显示界面、设定海底 / 基底 / 莫霍角色
4. 侧栏 **Profile**：单点 `At X` 或范围平均 → 预览窗导出 `V_1D_*.txt`
5. 工具：点/多边形选区、物性剖面、速度异常、重力、岩石散点、Petrology 导出
6. **保存工区**（`Ctrl+Shift+S`）写入路径与界面/重力/petrology 状态；**Save Figure** 存当前主图

下次：**打开工区** → 自动恢复模型与界面角色等。  
也可直接 `Load Vp` 不建工区；Workbench 启动时仍可写会话 JSON。

---

## 2. 工区文件

| 路径 | 内容 |
|------|------|
| `meta/imodel_project.json` | 工程主文件（名称、分析参数、工作流路径） |
| `inputs/` | 建议放置或引用 Vp/Vs、界面文件、重力格网等 |
| `outputs/` | 剖面 txt、物性导出、存图、petrology JSON/CSV（建议） |
| `cache/` | 可选缓存 |

工程 JSON 主要字段：

- `analysis`：是否显示界面、海底/基底/莫霍下拉选择
- `workflow.vp_model` / `vs_model` / `interface_files`
- `workflow` 重力观测目录·文件名·叠加开关·剖面经纬
- `workflow` petrology 观测 JSON / 沿迹 windows / `f_lower` 等

路径尽量相对工区根；Workbench 会话中会写成绝对路径便于 run 追踪。

**保存工区** 只存元数据与路径，**不会**改写原始 `v.in`。  
未保存更改时切换工区或退出，会弹出短时 Save / Discard / Cancel。

启动也可带工区：

```bash
python -m pyAOBS.visualization.imodel.gui path/to/meta/imodel_project.json
# 或目录；环境变量 PYAOBS_IMODEL_PROJECT=工区根
```

---

## 3. 加载模型与界面

| 操作 | 说明 |
|------|------|
| Load Vp | 默认 **Zelt v.in**；亦支持 **TOMO2D smesh**、Grid / NetCDF |
| Load Vs | Grid / smesh（`.grd` / `.nc` / `*.smesh`），不可为 v.in |
| 显示界面 | 侧栏勾选；v.in 层界面叠绘 |
| **色标** | 侧栏 Model → **色标**：tomo2d 内置 `vp` / `vs` / `vpvs` / `water`（`scale_*.cpt`），clim 用 CPT 域 |
| **速度等值线** | 侧栏 Model → **速度等值线**：叠加 tomo2d 内置等值线表（随色标切换；A 标注 / C 细线） |
| 海底 / 基底 / 莫霍 | 下拉指定角色（物性、剖面 Datum、Petrology H 等依赖） |
| 加载界面文件 | 外部界面曲线；可保存当前界面；网格模型也可从指定速度等值线提取界面 |

主图：**滚轮缩放 · 左拖平移 · 右拖连续缩放 · 双击 / Reset View 复位**。  
子图窗同样交互；均有 **Save Figure**（png/jpg/pdf/ps/eps/tif/svg；未写后缀时按所选类型补全）。

---

## 4. 一维剖面（侧栏 Profile）

| 项 | 说明 |
|----|------|
| Datum | `bm` 相对基底 · `sf` 相对海底 · `z0` 模型顶 |
| At X | 单点垂直剖面（同时自动加入对比袋） |
| Average | X 范围平均 + `vp_min`/`vp_max` 包络；主图标注采样 X（同时自动加入对比袋） |
| 预览窗 | 竖长横窄；**Save V_1D…** / **Save all samples…**；另有 **Save Figure**（单独看本段） |
| **对比图** | 多段 Vp–depth 同窗叠绘（不同颜色 + 图例）；混用 Datum 时标题警告 |
| **Clr 对比** | 清空对比袋 |

导出命名示例：`V_1D_from_{sf\|bm\|z0}_….txt`（列：depth, vp）；包络为 `*_envelope.txt`（depth, vp_min, vp_max）。相对基准时首行 depth=0。

**多段同图（1D）**：对不同 X / 不同 Range 多次 At X 或 Average → 侧栏点 **对比图** 叠绘；需要清空时用 **Clr 对比**。对比窗可 **Save Figure** / **Export txt…**。

---

## 5. 交互选区与物性

| 工具 | 操作 |
|------|------|
| Point Selection | 左键加点 · 右键删点 · `D` 删末点 · `C` 清空；点选后日志输出物性 |
| Polygon Selection | 左键加点 · 右键闭合 |
| Clear Selections | 清除点/多边形 |
| Density / T / P Profile | 二维物性剖面子窗 |
| Velocity Anomaly ΔV | V / Vref / ΔV；参考可用 layer average（推荐，需 v.in） |
| Rocks | Vp–Vs / Vp/Vs–Vp 散点；可投点、多边形采样、DEM 曲线 |

---

## 6. 重力与 Petrology

### 重力（Gravity Toolbox）

- Full Model / 多边形体 / 方法对比；可叠加世界重力 `.grd`
- **平面图 lon×lat**：轨迹与站位；需 Profile Lon/Lat 或模型轴含经纬
- 参数与观测路径写入工区 / Workbench 会话

### Petrology 桥接

- 侧栏或菜单 **Export to LIP Petrology…**：按界面几何导出地壳观测（每次成功导出 **累加** 到 H–Vp 投点列表；LIP 仍用最近一点）
- **H–Vp图**：Fig.12a / Fig.15 投图；多段累加点以不同颜色/标记叠绘（非模态）
- **Clr 投点**：清空已累加观测（状态显示「已累加 N 点」）
- 沿迹滑窗导出 → 可启动 LIP Petrology GUI
- 工区/会话：`petrology_observations` 列表（兼容旧字段 `petrology_observation`）
- Workbench：插件 `petrology.lip.gui` 可读最近 `imodel.gui` 的 `gui_state`

**多段同图（H–Vp）**：在不同段/不同 X 多次 Export（或 Quick）→ **H–Vp图** 同图叠投 → 需要重来时 **Clr 投点**。

---

## 7. Workbench 联动

| 项 | 说明 |
|----|------|
| 插件 | `imodel.gui` |
| work_dir | 填工区目录 →「应用 GUI 表单」传入路径并设 `PYAOBS_IMODEL_PROJECT` |
| model_file / aux_file | 登记到运行 inputs（追踪用） |
| 会话 | `PYAOBS_GUI_STATE_FILE` 键名 `imodel_gui`（含 `imodel_project`、`vs_model_file` 等） |

独立启动未设会话文件时，仅工区 JSON 持久化（若已打开并保存工区）。

---

## 8. 快捷键与窗口规则

| 操作 | 键 / 入口 |
|------|-----------|
| 加载 Vp | `Ctrl+O` |
| 保存工区 | `Ctrl+Shift+S` |
| 导出 Petrology | `Ctrl+Shift+P` |
| H–Vp 预览 | `Ctrl+Shift+2` |
| 退出 | `Ctrl+Q` |
| 帮助 | `F1` / 菜单「帮助」 |

工具窗、结果提示、帮助、User Guide 类长文均为 **非模态**（不冻结主窗）。  
短交互例外：文件选择框、未保存工区时的 Save/Discard/Cancel。

---

## 9. 启动与关于

```bash
python -m pyAOBS.visualization.imodel.gui
python -m pyAOBS.visualization.imodel.gui path/to/imodel_project.json
# Workbench 插件 imodel.gui
```

依赖：`PySide6`、`matplotlib`、`numpy` 等；推荐 `pip install 'pyAOBS[gui-qt]'`。

| | |
|--|--|
| 名称 | Interactive Velocity Model Viewer（imodel） |
| 作者 | Haibo Huang |
| 文档 | `visualization/imodel/docs/HELP.md`（本文件） |
| 包说明 | [`README.md`](../README.md) |

旧 Tk 说明已归档于 `_archive/docs/`（不维护）。核心算法亦可无 GUI：`ProfileExtractor`、`PropertyCalculator` 等见包 API。
