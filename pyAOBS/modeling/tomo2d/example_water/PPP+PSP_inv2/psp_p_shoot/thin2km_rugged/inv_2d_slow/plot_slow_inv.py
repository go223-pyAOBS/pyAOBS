#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""面下低速反演出图（复用 inv_2d 画法，真值射击用 slow）。"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "inv_2d"))
import plot_rugged_inv as p  # noqa: E402

p.HERE = HERE


def _shoot_true_psp():
    import make_rugged_2d_inv as mk  # noqa: WPS433

    mk.CRUST_TRUE = 5.00
    md = mk.inc.Model("slow", 5.00)
    rays = mk.shoot_codes(md, (6,))
    return [(r.xs, r.zs) for r in rays]


p._shoot_true_psp = _shoot_true_psp


if __name__ == "__main__":
    raise SystemExit(p.main())
