# imodel — Interactive velocity-model analysis

Package layout (aligned with iphase / zplotpy):

| Path | Role |
|------|------|
| `engine.py` | Core logic: profiles, properties, gravity (no Qt) |
| `velocity_anomaly.py` | Velocity anomaly vs reference models |
| `gravity_obs_grid.py` | Observed gravity grid I/O and lon–lat helpers |
| `talwani_optional.py` | Optional Talwani 2D gravity integration |
| `gui/` | PySide6 interactive application |
| `docs/HELP.md` | GUI 帮助（工具栏「帮助」/ `F1`） |
| `_archive/` | Legacy Tk GUI and old docs (unsupported) |

## Launch

```bash
python -m pyAOBS.visualization.imodel.gui
python -m pyAOBS.visualization.imodel.gui path/to/meta/imodel_project.json
```

Requires PySide6 (`pip install 'pyAOBS[gui-qt]'`).

(`python -m pyAOBS.visualization.imodel` still redirects to the GUI with a hint.)

## Help

GUI 内 **帮助** 或 `F1` 打开 [`docs/HELP.md`](docs/HELP.md)（非模态）。内容含工区、剖面、物性、重力、Petrology、Workbench 与快捷键。

## Python API

```python
from pyAOBS.visualization.imodel import ProfileExtractor, PropertyCalculator
from pyAOBS.visualization.imodel.velocity_anomaly import depthwise_horizontal_mean_velocity_anomaly
```

## Notes

- Workbench JSON 会话键仍可能写作 `imodel_gui`（历史字段名），与包路径无关。
- **工区 MVP**：`meta/imodel_project.json` + 工具栏「新建/打开/保存工区」。
  - 布局：`inputs/`、`outputs/`、`cache/`、`meta/`
  - Workbench：`imodel.gui` 表单可填 `work_dir`，应用后传入工区路径并设置 `PYAOBS_IMODEL_PROJECT`；运行会话仍写入 `PYAOBS_GUI_STATE_FILE`（含 `imodel_project` 字段）。
- Root shims `visualization.velocity_anomaly` and `visualization.gravity_obs_grid` re-export from `imodel.*`.
