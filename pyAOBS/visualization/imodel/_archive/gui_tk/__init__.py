"""imodel Tkinter GUI 包。

职责划分：见同目录下 `*_mixin` 风格模块（Workbench / 模型表面 / 交互 / 剖面 /
物性 UI / 下方物性面板 / 岩石散点 / 重力）。

`viewer.py` 仅保留主类多继承、`__init__`、`create_widgets`、`create_menu`、`run`/`main`。
共用第三方与 pyAOBS 导入集中在 `deps.py`；会话状态模块 `workbench_state.py` 为轻量依赖。
"""

from .viewer import InteractiveModelViewerGUI, main

__all__ = ["InteractiveModelViewerGUI", "main"]
