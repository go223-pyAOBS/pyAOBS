#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""图论观测两步反演出图。"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "inv_2d"))
import plot_rugged_inv as p  # noqa: E402

p.HERE = HERE
p.TRUE_RAY_LABEL = "图论初至"
p.REC_RAY_LABEL = "图论初至（PSP 当 0）"
p.SUPTITLE_MODELS = "崎岖面  观测=图论初至  反演=图论初至两步"
p.SUPTITLE_RAYS = "崎岖面射线：收回与真值都是图论初至"


def _shoot_true_psp():
    rp = HERE / "rays_true.dat"
    if not rp.is_file():
        return []
    if (HERE / "syn_true.dat").is_file() and (HERE / "syn_inv.dat").is_file():
        return p.rays_used_in_inv(rp, HERE / "syn_true.dat", HERE / "syn_inv.dat")
    return p.parse_rays(rp)


p._shoot_true_psp = _shoot_true_psp


if __name__ == "__main__":
    raise SystemExit(p.main())
