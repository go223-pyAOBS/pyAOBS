#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 inv_2d 拷贝真/初网格与几何。观测走时由 WSL tt_forward 在真模型上算。"""

from __future__ import annotations

import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "inv_2d"

NEED = (
    "true_vp.smesh",
    "start_vp.smesh",
    "true_mixed.smesh",
    "seafloor.refl",
    "conv.refl",
    "damp_lid.dat",
    "vcorr.dat",
    "geom_ppp.dat",
    "geom_psp0.dat",
)


def main() -> int:
    for name in NEED:
        shutil.copyfile(SRC / name, HERE / name)
    print("copied grids/geom from inv_2d  (syn_*.dat 由 tt_forward 真模型生成)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
