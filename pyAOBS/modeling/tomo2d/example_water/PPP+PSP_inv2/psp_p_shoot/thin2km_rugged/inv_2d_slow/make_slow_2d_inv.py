#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""崎岖面下低速：二维转折当观测，图论两步反演。真模型 = rugged slow（Vs0≈2.89 < 盖层底 Vp）。"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "inv_2d"))
import make_rugged_2d_inv as mk  # noqa: E402

mk.HERE = HERE
mk.MODEL_NAME = "slow"
mk.CRUST_TRUE = 5.00
mk.CRUST_START = 4.82


def main() -> int:
    return mk.main()


if __name__ == "__main__":
    raise SystemExit(main())
