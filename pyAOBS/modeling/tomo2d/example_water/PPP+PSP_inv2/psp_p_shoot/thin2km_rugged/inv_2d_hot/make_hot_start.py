#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 inv_2d 的 start_mixed 面下 Vs 整体 +0.35，初值偏快，逼反演减速。"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "inv_2d"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE.parents[2]))
import inv_grid as g  # noqa: E402
import make_ppp_psp_inv_case as m2  # noqa: E402

DV = 0.35


def main() -> int:
    for name in (
        "syn_inv.dat", "geom_psp0.dat", "seafloor.refl", "conv.refl",
        "damp_lid.dat", "vcorr.dat", "true_mixed.smesh",
    ):
        shutil.copyfile(SRC / name, HERE / name)
    xs, zs, vel = m2.parse_smesh(SRC / "start_mixed.smesh")
    lines = [
        f"{len(xs)} {len(zs)} {g.V_WATER:.4f} {g.V_AIR}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    n = 0
    for i, x in enumerate(xs):
        zi = g.z_conv(x)
        row = []
        for k, z in enumerate(zs):
            v = vel[i][k]
            if z >= zi - 1e-9:
                v = v + DV
                n += 1
            row.append(f"{v:.4f}")
        lines.append(" ".join(row))
    (HERE / "start_hot.smesh").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"start_hot.smesh  面下 +{DV:g} km/s  n={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
