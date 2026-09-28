# zplotpy 波形工区 — 工程化说明

## 入口

```bash
python -m pyAOBS.visualization.zplotpy.gui
# 或
python pyAOBS/visualization/zplotpy/run.py
```

主窗：`gui/project_window.py` → `ZplotProjectWindow`（对齐 relocation / RTM / idata）。  
波形底座：`gui/qt_fast_viewer.py` → `QtFastViewer`。

用户文档：工具栏 **帮助**（`F1` / `H`）→ [`docs/HELP.md`](docs/HELP.md)。

## 包分层（对齐 idata）

| 目录 | 职责 |
|------|------|
| 根 / `services/` | `project.py`、`run.py`、工区目录约定 |
| `gui/` | 工程主窗、阶段面板、`qt_fast_viewer`（壳 + `mixins/` 功能混入）、帮助 |
| `gui/mixins/` | `QtFastViewer` 按主题拆分的方法包；索引见该目录 `__init__.py` |
| `gui/legacy_tk/` | Tk 旧 GUI（deprecated） |
| `core/` | 加载 / 处理 / 拾取 / 叠加 / 走时等无 UI 逻辑 |
| `tests/` | check/verify/test 脚本与样例数据 |
| `src/` | Fortran 内核源（路径由 `core/src_kernel_bridge` 锚定包根） |

对外请直接导入 `…zplotpy.gui.…` / `…zplotpy.core.…`（根目录兼容 shim 已移除）。

波形查看器：`qt_fast_viewer.py` 只保留状态初始化与窗口生命周期；业务在 `gui/mixins/`（见该目录 `__init__.py` 索引）。

## 布局

```
[工具栏] 新建工区 | 打开工区 | 保存工区 | 帮助 | 退出
[阶段]   1输入 | 2波形/拾取 | 3输出
[主区]   当前阶段面板（波形阶段嵌入 QtFastViewer）
[底栏]   页签：输出 | V段
```

## 工区文件

| 路径 | 内容 |
|------|------|
| `meta/zplotpy_project.json` | 工程主文件 |
| `outputs/waveop.json` | V 选波 + 校正基准 |
| `outputs/picks.out` | 拾取 |
| `outputs/viewer_params.json` | 查看器参数（保存工区自动写；打开加载自动恢复） |

几何默认 **`geom=obs`**。姿态联合反演请另启 `python -m pyAOBS.processors.relocation.gui`。

## 推荐流程

1. **新建工区** → 选空目录  
2. **1 输入**：填 Z/HDR/水深 →「加载到波形工作台」  
3. **2 波形/拾取**：分量 / 拾取 / V 选波（底栏「V段」）/ 增益滤波 / 叠加；写出 tx.in：工具栏「写入HDR」→「写入tx.in」  
4. **保存工区** / **3 输出**：写出 JSON + `outputs/*`

嵌入时波形台 **Q 退出已禁用**；退出用工具栏「退出」。
