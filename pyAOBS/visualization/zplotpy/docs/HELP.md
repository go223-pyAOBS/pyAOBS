# zplotpy 波形工区 — 使用说明

OBS / 炮集波形显示、拾取与 V 选波工区。工具栏 **帮助**（或 `F1` / `H`）打开本页；窗口为**非模态**，可不关窗继续操作主界面。

姿态联合反演不在本工区，请另启独立入口：

```bash
python -m pyAOBS.processors.relocation.gui
```

---

## 1. 入口与布局

```bash
python -m pyAOBS.visualization.zplotpy.gui
```

主窗：`gui/project_window.py` → `ZplotProjectWindow`（对齐 relocation / RTM / idata 工区模式）。  
波形底座：`gui/qt_fast_viewer.py` → 嵌入的 `QtFastViewer`。

```
[工具栏] 新建工区 | 打开工区 | 保存工区 | 帮助 | 退出
[阶段]   1 输入 | 2 波形/拾取 | 3 输出
[主区]   当前阶段面板（波形阶段嵌入 QtFastViewer）
[底栏]   页签：输出 | V段（V 段列表从参数条迁出）
```

纯查看器（调试）：`python -m pyAOBS.visualization.zplotpy.gui.qt_fast_viewer`

包分层：`gui/`（界面）· `core/`（领域）· `gui/legacy_tk/`（Tk 遗留）· `tests/`（脚本）；详见 [`README.md`](../README.md)。

波形台自身（嵌入后）：

- 顶部工具栏：打开 / 重绘 / 保存.z、数据信息、位置 Map、导出图等（**无**独立「保存/加载参数」；参数随工区存取）
- 参数条：横向面板（基础 / 增益滤波 / 去噪 / 拾取 / 对齐叠加 / 波形操作 / 高级校正 / 走时模板）；高度可拖分割条；各面板边框高度统一
- 独立启动时：剖面下有「V段」薄条；嵌入工区后列表挂到主窗底栏「V段」页签

嵌入工区时波形台 **Q 退出已禁用**；请用工具栏 **退出** 或关主窗。

---

## 2. 工区文件

| 路径 | 内容 |
|------|------|
| `meta/zplotpy_project.json` | 工程主文件（路径与轻量状态） |
| `outputs/waveop.json` | V 选波 + 校正基准 |
| `outputs/picks.out` | 拾取（zplot 格式） |
| `outputs/viewer_params.json` | 查看器显示/处理参数 |

目录约定还有 `inputs/`、`cache/`（可选拷贝或仅记绝对路径）。

JSON 字段概要：`inputs`（dfile/hfile/rfile/terrain_path/geom）、`workflow`（picks/waveop/viewer_params 路径、current_apick、stage_index）、可选内嵌 `waveform_selections`。

几何默认 **`geom=obs`**（与波形/RTM 习惯一致：sx=OBS，gx/rx=炮）；输入页可选 `segy` / `auto`。位置 Map 等共用该约定。

---

## 3. 推荐流程

1. **新建工区** → 选择空目录（自动建 `meta/` `inputs/` `outputs/` `cache/`）
2. **1 输入**：填 `.z` / `.hdr`（可选 `.rec`）/ 水深 →「加载到波形工作台」
3. **2 波形/拾取**：分量、增益滤波、去噪、P 拾取、V 选波（列表见底栏「V段」）、叠加、tx.in 等
4. **保存工区**：写出 `meta/zplotpy_project.json`，并同步 `outputs/*`（含 **viewer_params.json**，下次打开并加载数据时自动恢复参数）
5. **3 输出**：可改相对路径、立即导出、打开 outputs 文件夹

下次：**打开工区**（选 `zplotpy_project.json`）→ 可选立即加载 Z → 参数/拾取/V 段尽量从 outputs 恢复。

---

## 4. 阶段说明

### 4.1 输入

- 指定数据路径；相对路径相对工区根解析。
- **geom**：`obs` / `segy` / `auto`。
- 水深文件供位置 Map 共用，无需在波形台重复加载。

### 4.2 波形/拾取

参数面板：标题栏拖拽可重排；左右边可调宽；纵向上以最高面板为准统一边框高度。「隐藏面板」可临时腾出剖面高度。

「数据信息」：数据概览与道头参数合并为页签。「波形操作」单列：叠加 / 清除V / 存V / 载V。  
tx.in：**写出**用工具栏「写入HDR」→「写入tx.in」；**读入/映射**在「走时模板」面板。

### 4.3 输出

- 默认相对路径：`outputs/waveop.json`、`picks.out`、`viewer_params.json`
- 「立即导出全部产物」与「保存工区」都会写 sidecar；保存工区另写工程 JSON

---

## 5. 快捷键

工区工具栏：新建 / 打开 / 保存工区；**帮助**；退出。  
`F1` / `H` / 工具栏「帮助」均打开本说明。

### 文件与界面

| 键 | 说明 |
|----|------|
| Ctrl+O | 打开数据 |
| Ctrl+R | 重绘 |
| Ctrl+S | 保存拾取 |
| Q | 退出（嵌入工区时已禁用，请用工具栏「退出」） |
| H / F1 | 本帮助 |

「数据信息」按钮：数据概览 \| 道头参数。  
V 段列表：工区底栏「V段」页签，或独立窗剖面下薄条。

### 拾取

| 键 | 说明 |
|----|------|
| P | 拾取模式；左键加点，右键删当前字 |
| Shift+移动 | 连续拾取（每道一次） |
| 1 / 2 / 3、[ / ] | 切换拾取字 |
| Ctrl+Z / Ctrl+Y | 撤销 / 重做最近一次拾取修改 |
| S | 写 HDR（随后可用工具栏「写入tx.in」） |
| C | 插值相关拾取（需 ≥2 个种子） |
| A | 临时对齐（再按 A 清除） |
| F | 自适应更新拾取时间（不平移波形） |
| V / Shift+V | 加 V 段 / 删当前字最近一段 |
| M / Shift+M | 多边形 mute（可拖顶点）/ 反选；拾取面板 Mute 旁有「反选」勾选；Delete 删当前顶点 |
| D-D | 两次按键按偏移范围批量删除当前拾取字 |
| X | 移除 / 恢复最近道 |

### 波形操作面板（无快捷键，点按钮）

叠加 / 清除V（仅当前 apick）/ 存V / 载V

### 视图与高级

| 键 | 说明 |
|----|------|
| Z / O | 放大 / 缩小（以鼠标为中心） |
| ← / → | 上一炮 / 下一炮 |
| I | 最近道详细信息 |
| T / Shift+T | 理论走时 / 清除 |
| W / Shift+W | 水层校正（需先有理论走时）/ 校正曲线 |

走时模板面板：读取 / 清除 / 预览映射 / 映射 tx.in（只负责导入侧）。  
写出 tx.in：工具栏 **写入HDR** → **写入tx.in**（转换对话框可为各 OBS 填 xmod；有 `.rec` 时按炮号预填）。

### 4.3 记录文件 `.rec` / `.rsp`（可选）

ASCII，典型一行：`ishnum  xmod  ymod  az  [title]`。

| 用途 | 说明 |
|------|------|
| 模型位置表 | 炮号 → 模型坐标 `(xmod,ymod)` 与方位 `az` |
| 写 tx.in | 「写入tx.in」对话框按 `ishnum` 预填 `xmod` |
| 理论走时 | 生成/推断炮点位置时可作参考（RAYINVR / r.in） |
| **非必须** | 浏览、上一炮/下一炮、拾取只靠 `.z`/`.hdr` 道头里的 `ishoti` |

无 `.rec` 时：换炮与拾取照常；写 tx.in 时 `xmod` 默认 0，需手填。

---

## 6. 与姿态工区（relocation）的关系

| 项目 | zplotpy 本工区 | relocation 姿态工区 |
|------|----------------|---------------------|
| 入口 | `…zplotpy.gui` | `…relocation.gui` |
| 阶段 | 输入 → 波形/拾取 → 输出 | 输入 → 波形/拾取 → **姿态校正** → 输出 |
| 工程 JSON | `meta/zplotpy_project.json` | `meta/relocation_project.json` |
| 姿态反演 | 无 | 有（工区内嵌波形台「姿态」按钮） |

两边独立启动、工程文件不共用；需要姿态时请直接打开 relocation 工区。

relocation 侧约定摘要（详见该工区帮助）：

- `apick=1`：直达水波 → 走时/位置 + 姿态
- 其它 `apick`：折射/反射 → 默认仅姿态；可与直达一起估方位
- 校正输入默认「原始截窗 + rmean + rtrend +（可选）主图带通」，**不含增益**

---

## 7. 关于

- 包：`pyAOBS.visualization.zplotpy`（`gui/` + `core/`）
- 工程文件：`meta/zplotpy_project.json`
- 文档：`visualization/zplotpy/docs/HELP.md`（本文件）
- 入口：
  - 工区（默认）：`python -m pyAOBS.visualization.zplotpy.gui`
  - 纯查看器：`python -m pyAOBS.visualization.zplotpy.gui.qt_fast_viewer`
  - 姿态工区：`python -m pyAOBS.processors.relocation.gui`
- 核心能力：大文件快速渲染、分量过滤、增益/滤波/去噪、拾取与自动/插值相关、V 选波与叠加、tx.in 走时模板、理论走时/水层校正/静校正
- 姿态联合反演请用独立 relocation 工区（本波形台不再提供跳转按钮）
