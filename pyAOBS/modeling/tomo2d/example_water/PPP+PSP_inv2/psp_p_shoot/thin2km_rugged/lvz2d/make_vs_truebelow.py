#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""盖层保持 PPS/PPP 收回，面下换成真 Vs。"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inv_2d"))
import inv_grid as g  # noqa: E402
import make_ppp_psp_inv_case as m2  # noqa: E402
from make_lvz import write_smesh  # noqa: E402


def graft_true_below(lid_vs: Path, true_vs: Path, out: Path) -> tuple[int, int]:
    xs, zs, lid = m2.parse_smesh(lid_vs)
    xs2, zs2, tru = m2.parse_smesh(true_vs)
    if xs2 != xs or zs2 != zs:
        raise SystemExit("grid mismatch")
    n_lid = n_bel = 0
    out_v = []
    for i, x in enumerate(xs):
        zi = g.z_conv(x)
        col = []
        for k, z in enumerate(zs):
            if z < zi - 1e-9:
                col.append(lid[i][k])
                if z > g.H + 1e-9:
                    n_lid += 1
            else:
                col.append(tru[i][k])
                n_bel += 1
        out_v.append(col)
    write_smesh(out, xs, zs, out_v)
    return n_lid, n_bel


def main() -> int:
    lid = HERE / "path_a" / "rec_vs_lid.smesh"
    tru = HERE / "true_vs.smesh"
    out = HERE / "vs_lid_truebelow.smesh"
    n_lid, n_bel = graft_true_below(lid, tru, out)
    print(f"{out.name}: lid from rec_vs_lid n={n_lid}  below from true_vs n={n_bel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
