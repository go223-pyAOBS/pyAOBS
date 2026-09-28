#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""崎岖转换面：图论与二维射击共用同一条 conv.refl 折线。

在倾斜面 z=4+0.03(x-50) 上叠加两档正弦起伏，再圆整到 0.2 km 网格
（与 incline 一样，避免 -X 取路径成环）。射击按折线当地斜率做 Snell，不共 p。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "thin2km"))
sys.path.insert(0, str(HERE.parent / "thin2km_incline"))
sys.path.insert(0, str(HERE.parent.parent.parent / "ps_inv"))
sys.path.insert(0, str(HERE.parent.parent.parent / "ps_fwd"))

import run_thin2km_incline as inc  # noqa: E402

inc.HERE = HERE

AMP1, LEN1 = 0.22, 12.0
AMP2, LEN2 = 0.10, 7.0


def z_conv_smooth(x: float) -> float:
    return (
        inc.ZC0
        + inc.SLOPE * (x - inc.XREF)
        + AMP1 * math.sin(2.0 * math.pi * (x - inc.XREF) / LEN1)
        + AMP2 * math.sin(2.0 * math.pi * (x - 20.0) / LEN2)
    )


def z_conv(x: float) -> float:
    z = z_conv_smooth(x)
    k = int(round(z / inc.DZ_IFACE))
    z = k * inc.DZ_IFACE
    return max(inc.H + inc.DZ_IFACE, min(inc.ZMAX - 2.0, z))


inc.z_conv_smooth = z_conv_smooth
inc.z_conv = z_conv


def main() -> int:
    xs = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    print(
        "rugged iface  "
        + "  ".join(f"zc({x:.0f})={z_conv(x):.2f}" for x in xs)
    )
    return inc.main()


if __name__ == "__main__":
    raise SystemExit(main())
