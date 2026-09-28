# -*- coding: utf-8 -*-
"""已迁移：请使用 ``pyAOBS.processors.idata.gui`` / ``processors/idata/run.py``。"""

from __future__ import annotations

import warnings

warnings.warn(
    "processors.raw2sac.idata_gui 已迁移至 processors.idata.gui；请改用新路径。",
    DeprecationWarning,
    stacklevel=2,
)

raise ImportError(
    "idata_gui 已迁至 processors/idata/gui。"
    "请运行: python processors/idata/run.py"
    " 或: python -m pyAOBS.processors.idata.gui"
)
