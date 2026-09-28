# -*- coding: utf-8 -*-
"""idata 启动脚本（Workbench ``data.gui`` 默认入口）。

刻意不 ``import pyAOBS.processors``，避免 ``processors/__init__.py`` 拉取 pygmt 等重依赖。
通过将 ``processors/`` 加入 path，以 ``import idata...`` 加载本包。
"""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap() -> None:
    idata_dir = Path(__file__).resolve().parent
    processors_dir = idata_dir.parent
    pyaobs_pkg = processors_dir.parent
    repo_root = pyaobs_pkg.parent
    raw2sac_dir = processors_dir / "raw2sac"
    for p in (repo_root, raw2sac_dir, processors_dir):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def main() -> int:
    _bootstrap()
    from idata.gui.main_window import run_idata_app

    return int(run_idata_app() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
