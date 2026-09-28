# zplotpy — 波形工区（工程化）

OBS / 炮集波形显示、拾取与 V 选波。布局对齐 `idata` / `relocation`。

**使用说明**：[`docs/HELP.md`](docs/HELP.md)（工具栏 **帮助** / `F1`）。

## 启动

```bash
python -m pyAOBS.visualization.zplotpy.gui
# 或
python pyAOBS/visualization/zplotpy/run.py
```

纯查看器调试：

```bash
python -m pyAOBS.visualization.zplotpy.gui.qt_fast_viewer
```

姿态联合反演：`python -m pyAOBS.processors.relocation.gui`（独立工区，与本波形工区工程文件不共用）。

## 包结构（对齐 idata）

```
zplotpy/
├── project.py, run.py, services/     # 工程壳
├── gui/                              # PySide6 工区 + QtFastViewer
│   ├── project_window.py, panels/, help_dialog.py
│   ├── qt_fast_viewer.py
│   ├── mixins/                       # 功能混入索引：gui/mixins/__init__.py
│   └── legacy_tk/                    # Tk 遗留（deprecated）
├── core/                             # 无 UI 领域逻辑
├── tests/                            # 诊断脚本与样例数据
├── docs/HELP.md
└── src/                              # Fortran 内核源（本档未搬）
```

## 工区文件

| 路径 | 内容 |
|------|------|
| `meta/zplotpy_project.json` | 工程主文件 |
| `outputs/waveop.json` | V 选波 |
| `outputs/picks.out` | 拾取 |
| `outputs/viewer_params.json` | 查看器参数 |

几何默认 **`geom=obs`**。
