# -*- coding: utf-8 -*-
"""python -m pyAOBS.processors.idata.gui"""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_if_needed() -> None:
    """若经 -m 启动且已进入本包，补充 raw2sac 到 path 供 segy_trace_header。"""
    gui_dir = Path(__file__).resolve().parent
    processors_dir = gui_dir.parents[1]
    raw2sac_dir = processors_dir / "raw2sac"
    repo_root = processors_dir.parent.parent
    for p in (str(repo_root), str(raw2sac_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)


_bootstrap_if_needed()

from .main_window import run_idata_app  # noqa: E402

raise SystemExit(run_idata_app())
