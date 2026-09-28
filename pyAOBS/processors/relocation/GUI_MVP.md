# OBS 姿态校正 GUI — 工程化工区

## 入口

```bash
python -m pyAOBS.processors.relocation.gui
```

主窗：`gui/project_window.py` → `RelocationProjectWindow`（对齐 OBS RTM / zplotpy 工区模式）。  
波形底座：嵌入的 `RelocationViewer`（`zplotpy.gui.qt_fast_viewer.QtFastViewer`）。

用户文档：工具栏 **帮助**（`F1`）→ [`docs/HELP.md`](docs/HELP.md)（含快捷键与关于，单按钮）。

## 布局

```
[工具栏] 新建工区 | 打开工区 | 保存工区 | 帮助 | 退出
[阶段]   1输入 | 2波形/拾取 | 3姿态校正 | 4输出
[主区]   当前阶段面板（波形阶段嵌入 RelocationViewer）
[底栏]   页签：输出 | V段（V 段列表从参数条迁出）
```

波形台自身：

- 顶部工具栏：打开/重绘/保存参数、数据信息、位置图、导出图等（**无**独立帮助/退出；由工程主窗工具栏提供）
- 参数条：矮高度横向面板（基础/增益/去噪/拾取/对齐/波形操作按钮/高级校正/走时模板）
- 独立启动时：剖面下有「V段」薄条；嵌入工区后列表挂到主窗底栏「V段」页签

## 工区文件

| 路径 | 内容 |
|------|------|
| `meta/relocation_project.json` | 工程主文件 |
| `outputs/waveop.json` | V 选波 + 校正基准 |
| `outputs/picks.out` | 拾取 |
| `outputs/attitude_solution.json` | 姿态解 + UI 参数 |
| `outputs/viewer_params.json` | 查看器显示参数 |

JSON 字段：`inputs`（dfile/hfile/rfile/terrain/geom）、`workflow`、`attitude_ui`、`attitude_solution`、可选内嵌 `waveform_selections`。

几何默认 **`geom=obs`**（sx=OBS，gx/rx=炮），与 RTM / `geometry_roles` 一致。

## 推荐流程

1. **新建工区** → 选空目录  
2. **1 输入**：填 Z/HDR/水深 →「加载到波形工作台」  
3. **2 波形/拾取**：分量 / 拾取 / V 选波（列表见底栏「V段」）/ 增益滤波  
4. **3 姿态校正**：汇总全部 V 段运行。震相约定：
   - `apick=1`：直达水波 → 走时/位置 + 姿态（倾角可选，`INC_th=atan(x/h)`）
   - 其它 `apick`：折射/反射 → 默认仅姿态（极化/ORI）；可与直达一起估方位
   - 仅有次生相时：强制 `w_tt=0`、不搜位置/倾角几何  
   校正输入默认 `原始截窗 + rmean + rtrend +（可选）主图带通`，**不含增益**。水深优先用输入页文件。  
5. 结果窗为统一页签：迭代诊断 | 三分量波形 | 极化 | OBS漂移 | 方位对比 | ppol每道分布  
6. **4 输出** / **保存工区**：写出 JSON + outputs/*

下次：**打开工区** → 自动读入 `outputs/attitude_solution.json` → 姿态页显示当前解（作反演初值）。  
「预览当前解」打开结果图窗；整剖面主图预览需在结果窗点「应用到主图预览」。  
未勾选校正倾角时 `tilt=0`（Z 不旋）；全局走时 shift 会叠到主图预览显示。  
「保存姿态结果」只写工区 JSON；「接受为当前修正」会把解写入内存波形与 OBS 道头，并可覆盖/另存 `.z`、可选写 `.hdr`。  
也可在输出页单独「导出全部」写出该 JSON。

快捷键：嵌入工区时波形台 **Q 退出已禁用**；退出请用工具栏 **退出**。  
帮助：工具栏单按钮 / `F1` → `docs/HELP.md`（非模态）。

## 与旧入口关系

- `RelocationViewer` 仍可用作独立工作台（`gui/main_window.py`），供嵌入或对照。  
- 服务层 `services/`、位置对比窗保持不变。  
- 早期 `section_canvas` 仍为实验遗留。
