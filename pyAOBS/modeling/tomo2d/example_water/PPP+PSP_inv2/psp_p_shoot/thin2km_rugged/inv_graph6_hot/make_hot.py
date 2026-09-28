#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单场公平对照：只准备网格，走时由本目录 tt_forward 生成。"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from make_fair_grids import prepare  # noqa: E402


def main() -> int:
    prepare(HERE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
