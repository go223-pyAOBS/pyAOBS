# -*- coding: utf-8 -*-
"""兼容入口：转发到独立工区 ``processors/idata/run.py``。

请优先使用::

    python pyAOBS/processors/idata/run.py
    python -m pyAOBS.processors.idata.gui
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def main() -> int:
    run_py = Path(__file__).resolve().parents[1] / "idata" / "run.py"
    if not run_py.is_file():
        print(f"idata run.py not found: {run_py}", file=sys.stderr)
        return 1
    spec = importlib.util.spec_from_file_location("pyaobs_idata_run", run_py)
    if spec is None or spec.loader is None:
        print(f"cannot load: {run_py}", file=sys.stderr)
        return 1
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return int(mod.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
