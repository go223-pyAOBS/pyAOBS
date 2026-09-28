# OBS 姿态校正工区 — 帮助

工具栏 **帮助**（或 `F1`）打开本页；窗口为**非模态**，可不关窗继续操作主界面。

纯波形/拾取（无姿态阶段）请用：`python -m pyAOBS.visualization.zplotpy.gui`。

---

## 1. 入口与布局

```bash
python -m pyAOBS.processors.relocation.gui
```

主窗：`gui/project_window.py` → `RelocationProjectWindow`。  
波形底座：嵌入的 `RelocationViewer`（`zplotpy.gui.qt_fast_viewer.QtFastViewer` + 姿态混入）。

```
[工具栏] 新建工区 | 打开工区 | 保存工区 | 帮助 | 退出
[阶段]   1 输入 | 2 波形/拾取 | 3 姿态校正 | 4 输出
[主区]   当前阶段面板（波形阶段嵌入 RelocationViewer）
[底栏]   页签：输出 | V段
```

嵌入工区时波形台 **Q 退出已禁用**；请用工具栏 **退出** 或关主窗。

---

## 2. 工区文件

| 路径 | 内容 |
|------|------|
| `meta/relocation_project.json` | 工程主文件 |
| `outputs/waveop.json` | V 选波 + 校正基准 |
| `outputs/picks.out` | 拾取 |
| `outputs/attitude_solution.json` | 姿态解 + UI 参数 |
| `outputs/viewer_params.json` | 查看器显示参数 |

几何默认 **`geom=obs`**（sx=OBS，gx/rx=炮），与 RTM / zplotpy 一致。

---

## 3. 推荐流程

1. **新建工区** → 选空目录
2. **1 输入**：填 Z/HDR/水深 →「加载到波形工作台」
3. **2 波形/拾取**：分量 / 拾取 / V 选波（底栏「V段」）/ 增益滤波
4. **3 姿态校正**：汇总全部 V 段运行。震相约定：
   - `apick=1`：直达水波 → 走时/位置 + 姿态（倾角可选，`INC_th=atan(x/h)`）
   - 其它 `apick`：折射/反射 → 默认仅姿态（极化/ORI）；可与直达一起估方位
   - 仅有次生相时：强制 `w_tt=0`、不搜位置/倾角几何  
   校正输入默认「原始截窗 + rmean + rtrend +（可选）主图带通」，**不含增益**。水深优先用输入页文件。
5. 结果窗页签：迭代诊断 | 三分量波形 | 极化 | OBS漂移 | 方位对比 | ppol每道分布
6. **4 输出** / **保存工区**：写出 JSON + `outputs/*`

下次：**打开工区** → 自动读入 `outputs/attitude_solution.json` → 姿态页显示当前解（作反演初值）。  
「预览当前解」打开结果图窗；整剖面主图预览需在结果窗点「应用到主图预览」。  
未勾选校正倾角时 `tilt=0`（Z 不旋）；全局走时 shift 会叠到主图预览。  
「保存姿态结果」只写工区 JSON；「接受为当前修正」会把解写入内存波形与 OBS 道头，并可覆盖/另存 `.z`、可选写 `.hdr`。

---

## 4. 快捷键

工区工具栏：新建 / 打开 / 保存工区；**帮助**；退出。  
`F1` / 工具栏「帮助」打开本说明。波形台焦点下 `H` 也可打开帮助（转发本文档或波形台摘要）。

### 波形台常用

| 键 | 说明 |
|----|------|
| P | 拾取模式；左键加点，右键删当前字 |
| Shift+移动 | 连续拾取（每道一次） |
| 1/2/3、[/] | 切换拾取字 |
| S / Ctrl+S | 写 HDR / 保存拾取 |
| V / Shift+V | 加 V 段 / 删当前字最近一段 |
| A / F | 临时对齐 / 自适应更新拾取时间 |
| C | 插值相关拾取 |
| Ctrl+Z / Ctrl+Y | 撤销 / 重做 |
| Z / O | 放大 / 缩小 |
| ← / → | 上一炮 / 下一炮 |
| H / F1 | 帮助 |

嵌入时 **Q 退出已禁用**。完整波形交互以加载工作台后波形台行为为准。

---

## 5. 关于

- 包：`pyAOBS.processors.relocation`
- 工程文件：`meta/relocation_project.json`
- 文档：`processors/relocation/docs/HELP.md`（本文件）
- 入口：`python -m pyAOBS.processors.relocation.gui`
- 波形底座：`RelocationViewer`（zplotpy QtFastViewer + 姿态混入）
