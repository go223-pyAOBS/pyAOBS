# pyAOBS Workbench 使用说明

`pyAOBS.workbench` 提供一个项目化 GUI 工作台，用于把 `tomo2d` 等流程运行记录统一沉淀到项目目录，支持复现、筛选、批量复跑与结果导出。

## 1. 快速启动

**默认启动 Qt 版 Workbench**（PySide6，与 imodel / LIP Petrology 风格一致）：

```python
from pyAOBS.workbench.app import launch_workbench

launch_workbench()
```

或：

```bash
python -m pyAOBS.workbench.app
```

依赖：`pip install -e ".[gui-qt]"`

**旧版 Tk 界面**（仅作 fallback）：

```bash
set PYAOBS_WORKBENCH_UI=tk
python -m pyAOBS.workbench.app
```

## 2. 项目结构

创建项目后，会在项目根目录生成标准结构（示意）：

- `project.yaml`：项目元信息
- `runs/`：每次运行的目录与 `manifest.json`
- `state/ui_state.json`：界面状态（筛选条件、页签、选中项等）
- 其他业务目录（`datasets/`、`models/`、`picks/` 等）

## 3. 运行节点（tomo2d）

右侧“运行节点”页提供两种方式：

- **模板化运行**：`tt_inverse_standard` / `tt_forward_basic`
- **自定义运行**：直接填写 `executable + args + env + inputs`

已内置 GUI 插件：

- `tomo2d.shell`
- `tomo2d.gui`
- `zplotpy.gui`
- `imodel.gui`
- `iphase.gui`
- `data.gui`（统一数据转换 UI：idata）

非 tomo2d 插件支持“GUI 启动参数（插件专用）”小表单：

- `zplotpy.gui`：`itype/data/header/record/irec`
- `imodel.gui`：`work_dir/node_id/model_file/aux_file/extra_args`
- `iphase.gui`：`work_dir/node_id/tx_files/rin_file/txout_file/extra_args`

说明：`imodel/iphase` 当前主要通过 GUI 内部打开数据，上述文件字段会写入运行记录的 `input_files`，
用于追踪与复跑（不一定映射为命令行参数）。其中 **`imodel.gui` 工作台入口为 Qt 版**
（`pyAOBS.visualization.imodel_qt`，需 PySide6）；若需旧版 Tk 界面，请直接执行
`python -m pyAOBS.visualization.imodel_gui`。

`processor.*` 三个数据转换插件默认采用“自定义运行”模式（`executable + args + inputs`），
可通过“填入插件启动示例”一键生成参数骨架后再按项目路径修改。

`tomo2d.gui` 用于直接启动 TOMO2D 图形界面（命令模式并行保留 `tomo2d.shell`）。

`data.gui` 用于直接启动统一转换界面 `idata`（已整合原先三个独立转换节点），在一个窗口中管理：
- `obem_tsm_to_sac_obspy`
- `sac2y_v2_1_obspy`
- `raw2sac_v1_1_obspy`

GUI 审计增强：
- Workbench 运行每个节点时会注入 `PYAOBS_RUN_ID/PYAOBS_PROJECT_ROOT/PYAOBS_RUN_DIR/PYAOBS_AUDIT_LOG`
- `manifest.json` 新增 `audit.session_log` 字段，指向 `state/gui_sessions/<run_id>.jsonl`
- `data.gui(idata)` 会记录关键交互事件（启动、浏览、执行、完成、关闭）
- `tomo2d.gui` 当前记录会话级事件（启动/关闭/异常）
- `zplotpy.gui/imodel.gui/iphase.gui` 已通过审计包装启动器记录会话级事件（启动/关闭/异常；`imodel` 记录的 `frontend=qt`）
- `zplotpy.gui/imodel.gui/iphase.gui/tomo2d.gui` 额外启用运行时钩子，记录常见过程级事件：
  - `tk_filedialog`（打开/保存文件对话框；**主要面向 Tk GUI**，Qt 版 imodel 通常无此类审计）
  - `tk_messagebox`（确认/警告/错误弹窗）
  - `subprocess_run_* / subprocess_popen_*`（GUI 内部外部命令执行）

运行节点插件下拉按推荐流程排序：
`data.gui -> zplotpy.gui -> tomo2d.gui -> tomo2d.shell -> imodel.gui -> iphase.gui`

运行节点命令区支持“目录类型一键填入”：
- 快速设置 `cwd`：`datasets/raw`、`datasets/processed`、`picks`、`models`、`interpretation`
- `CWD=项目树选中目录`：以左侧项目树当前选中项（若为文件则取其父目录）设置 `cwd`
- `选中项加入 inputs`：将左侧项目树当前选中目录或文件追加到 `inputs`

其中 `iphase tx_files` 支持两种导入方式：

- 文件对话框多选导入
- 从左侧项目树当前选中节点导入（选目录时会批量扫描常见 `tx` 文件）
- 支持 `name_filter` 关键词过滤，仅导入文件名匹配项：
  - 单关键词：`OBS22`
  - 多关键词 OR（逗号分隔）：`OBS22,OBS23`
  - 通配符：`OBS2*`、`*lineA*`

运行节点支持“保存节点预设 / 加载节点预设”：

- 保存路径：`project/state/node_presets/*.json`
- 会保存：插件类型、模板类型、通用命令区内容、GUI 专用表单状态
- 便于快速切换不同工作流配置（例如 tomo2d 参数组、iphase 批次筛选规则）

并提供“管理节点预设”面板：

- 列表查看预设（名称 / 插件 / 修改时间）
- 一键加载
- 重命名
- 复制
- 删除
- 编辑标签（tags）
- 关键词筛选（按预设名、插件名、标签匹配）

关闭应用时若存在运行中任务，会弹出退出保护：

- 等待任务完成（保持应用打开）
- 请求取消任务并退出
- 取消关闭，返回工作台

底部状态区包含任务状态仪表条：

- 单任务状态（空闲/运行中/取消中）
- 批量任务状态（进度、success/failed/cancelled、是否取消中）

运行历史页新增“批量任务队列”面板：

- 待提交任务（pending）
- 运行中任务（running）
- 最近完成任务（recent_done，支持双击定位到对应 run 详情与实时日志）
- 最近完成任务支持右键菜单：定位详情 / 打开 run 目录 / 复制 `new_run_id` / 打开实时日志

主要能力：

- 模板字段校验（数值范围、文件存在性、工作目录检查）
- 项目树一键选 `mesh/data` 输入
- 参数分组折叠显示（基础输入 / 约束参数 / 运行并行）
- 悬停帮助提示（联动 `modeling/tomo2d/help_docs.py`）
- 单节点任务后台执行（GUI 不阻塞，完成后自动刷新运行历史）
- 单节点任务支持“取消运行”（状态写入 `cancelled`）

## 4. 运行前检查（可编辑）

点击“运行前检查与预览”或直接运行时，会弹出预检窗口：

- 显示命令、环境变量、输入文件、风险提示
- 可在窗口内直接编辑后回填到主表单
- 常见风险提示包括：
  - `-DQ` 与 `-DV` 联动检查
  - OpenMP 变量缺失提醒
  - 可执行程序路径检查

## 5. 运行历史与详情

“运行历史”页支持：

- 记录筛选：`status` / `node` / `关键词` / `仅失败`
- 选中记录查看详情（`manifest` 关键信息、`stdout/stderr` 尾部）
- 实时日志查看（`stdout/stderr` 切换，支持自动跟随）
- 日志关键词过滤与高亮（可选“仅匹配行”）
- 打开 run 目录
- 打开 GUI 审计日志（若该 run 配置了 `audit.session_log`）
- 删除选中 run（含二次确认，删除对应 `runs/<run_id>` 目录）
- 一键清理非成功（failed/cancelled）
- 回填到运行节点（用于修改后重跑）
- 失败一键复跑（复用历史参数，仍走预检流程）

## 6. 批量复跑失败任务

基于“当前筛选结果”执行批量复跑：

- 可选并发：`1/2/4`
- 批量前先出“预检查报告”：
  - 可执行任务列表
  - 将跳过任务及原因
- 支持“取消批量”：停止提交新任务，并尝试中止已启动任务
- 批量过程中状态栏显示进度与成功/失败统计

## 7. 导出能力

### 7.1 导出当前筛选运行历史

- `导出筛选结果 CSV`
- `导出筛选结果 JSON`

字段包含：`run_id/node_id/status/return_code/created_at/finished_at/elapsed_s/command/error/run_dir`。

### 7.2 导出最近一次批量复跑结果

- `导出最近批量结果 CSV`
- `导出最近批量结果 JSON`

字段包含：`source_run_id/new_run_id/node_id/status/return_code/elapsed_s/cwd/command/error`。  
其中未执行项会以 `status=skipped` 输出并附跳过原因。

## 8. 最近批量结果面板

运行历史页底部提供“最近批量复跑结果”面板：

- 显示总计统计与逐条明细
- 支持状态筛选：`全部/success/failed/skipped`
- 支持一键复制到剪贴板

## 9. 状态持久化与自动恢复

以下状态会保存到 `state/ui_state.json` 并在打开项目后自动恢复：

- 窗口大小与项目树选中项
- 右侧页签（运行历史/运行节点）
- 运行历史筛选条件
- 批量并发配置
- 最近批量结果面板筛选
- 上次选中的 `run_id`（可自动定位）

## 10. 备注

- 目前内置插件以 `tomo2d.shell` 为主，结构上已支持扩展到 `iphase` / `zplotpy` / `imodel`。
- 批量复跑会生成新的 run 目录，不覆盖历史记录，便于对比与追溯。

## 11. 常见问题（FAQ）

### Q1: 为什么我设置了 `-DQ`，但看起来没有生效？

`tt_inverse` 中 `-DQ`（空间阻尼文件）通常需要与正的 `-DV` 联动使用。  
如果 `-DV` 缺失或 `<=0`，预检会提示风险，且反演效果可能异常。

### Q2: 并发怎么设置更稳妥？

推荐优先设置：

- `OMP_NUM_THREADS`（常从源数量附近起步）
- `OMP_PLACES=threads`
- `OMP_PROC_BIND=spread`
- 反演时开启 `TOMO2D_INV_OMP=1`，正演时开启 `TOMO2D_FWD_OMP=1`

实际最佳值与数据规模、CPU 拓扑相关，建议通过短跑对比确定。

### Q3: 为什么批量复跑预检查里有“将跳过任务”？

常见原因：

- 记录缺少 `run_dir`
- `manifest.json` 不存在或损坏
- 从历史记录重建命令失败（参数不完整、路径无效等）

预检查报告会逐条给出跳过原因，便于你快速修复后再批量执行。

### Q4: 预检窗口里改了参数，最终会不会真的按修改值运行？

会。  
点击“确认并回填”后，修改内容会写回主表单，并按回填后的参数执行。

### Q5: 导出的“最近批量结果”里 `skipped` 是什么？

`skipped` 表示该条任务在批量预检查阶段被判定不可执行，因此未进入实际运行。  
导出里会在 `error` 字段保留跳过原因。

### Q6: 打开项目后没有恢复到上次筛选/选中项怎么办？

请确认项目目录下存在 `state/ui_state.json`，且可正常读写。  
如果你手动修改过项目结构或 run 目录，部分历史项（如上次 `run_id`）可能无法定位，会回退到当前可见的默认选择。
