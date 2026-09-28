# TOMO2D（pyAOBS）

二维走时层析成像工作流：Python 封装（`TomoAnd`）+ **Qt GUI**。

## 启动 GUI

```bash
python -m pyAOBS.modeling.tomo2d
# 或
python -m pyAOBS.modeling.tomo2d.gui
# 或
from pyAOBS.modeling.tomo2d import launch_tomo2d_gui
launch_tomo2d_gui()
```

Workbench 插件：`tomo2d.gui`（审计启动器 `pyAOBS.workbench.gui_audit_launchers.tomo2d_gui`）。

## 可执行文件路径

按优先级：

1. GUI 顶栏 `bin_path`
2. 环境变量 `PYAOBS_TOMO2D_BIN` 或 `TOMO2D_BIN`
3. 系统 `PATH` 中的 `tomo2d` 系命令

## 工区（推荐）

目录约定（与 vedit / idata 对齐）::

```text
<meta/tomo2d_project.json>
inputs/      # 输入：geom、ttimes、边界等
outputs/     # 产出：smesh、反演结果等
runs/        # tt_inverse 可复现运行包（ttinv_*）与 QC
cache/       # 临时文件
```

`runs/ttinv_*/outputs/` 运行结束后按类型归入 `models/` `residuals/` `rays/` `dws/` `logs/`（**不**自动选定 final.smesh）。

GUI：**新建工区** / **打开工区** / **保存工区**（顶栏常显按钮，无菜单栏）。

示例：打开 `examples/meta/tomo2d_project.json`（workdir 由 `meta/` 旁推断）。

仍支持「保存/加载配置」导出独立 JSON；工区是带目录布局的工程文件。

## 工作目录与配置

- 打开工区后，表单 `work_dir` 与工区根一致；相对路径相对该根解析
- 运行摘要可写 `work_dir/tomo2d_gui.log`
- Workbench 会话：`PYAOBS_GUI_STATE_FILE` 可记住上次工区路径与表单
- **并行**：顶栏「并行 / 策略环境变量」对应 `OMP_NUM_THREADS`、`TOMO2D_INV_*`、`TOMO2D_FWD_OMP`、`TOMO2D_GRAPH_FS_ENUM`（见 `src/README_OMP_BUILD.md`），随工区/配置 JSON 的 `env.*` 键保存

## 包结构

```text
tomo2d/
  tomand.py              # CLI 封装
  help_docs.py           # 程序内帮助文案
  tt_inverse_*.py / tx2tomo2d.py
  gui/                   # Qt（PySide6）
    app.py / main_window.py
    state/ FormState
    services/            # 无 UI：路径、收集参数、审计、workflow
    panels/ dialogs/ plots/ workers/
```

## 测试

```bash
pytest pyAOBS/tests/test_tomo2d.py pyAOBS/tests/test_tomo2d_gui_smoke.py \
       pyAOBS/tests/test_tomo2d_gui_services.py pyAOBS/tests/test_tomo2d_project.py -q
```

## 交互约定

- 绘图：pyqtgraph 式滚轮缩放 / 左拖平移 / 右拖连续缩放 / 双击复位
- 分析图 / 集合统计：右键**点一下**出菜单（保存或日志操作），右键**拖**仍为缩放
- 子窗口：非模态（短交互如文件框、Yes/No 除外）
- GUI 操作说明：F1 或顶栏「帮助」（`docs/HELP.md`）
- 水层多次波正演（未实现，策略草稿）：`docs/WATER_MULTIPLES.md`
