#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PPP+PSS 同一次联合：geom_joint 同时含 raytype 0 和 8。

无 -k 的传统 P 反演、以及只 8 冻 Vp 的 -k，仍走原来的单场路径。
同一次联合需要 -k，且数据里同时有 0/1 和 6/7/8。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ps_inv"))
import make_ps_inv_case as base  # noqa: E402

V_WATER = base.V_WATER
H = base.H
Z_CONV = base.Z_CONV
XMIN, XMAX, DX = base.XMIN, base.XMAX, base.DX
ZMAX, DZ = base.ZMAX, base.DZ
KAPPA_TRUE = base.KAPPA_TRUE
KAPPA_START = base.KAPPA_START
VP_TRUE = base.VP_TRUE
VP_START = base.VP_START
OBS_XS = base.OBS_XS
OBS_Z = base.OBS_Z
SHOT_Z = base.SHOT_Z
SHOT_DX = base.SHOT_DX
SHOT_XS = base.SHOT_XS
LH, LV = base.LH, base.LV
WSV, WTV = base.WSV, base.WTV
PICK_U = base.PICK_U
CODES_PPP = base.CODES_PPP
CODES = (8,)
CODES_JOINT = (0, 8)
# 以后一次性 PPP+PPS+PSS+PSP：CODES_ALL = (0, 7, 8, 6)
CODES_ALL = (0, 7, 8, 6)

write_vp = base.write_vp
write_vs = base.write_vs
write_seafloor = base.write_seafloor
write_conv = base.write_conv
write_vcorr = base.write_vcorr
write_geom = base.write_geom
write_vs_from_smesh = base.write_vs_from_smesh
parse_smesh = base.parse_smesh
node_stats = base.node_stats
illum_x_range = base.illum_x_range
vs_at = base.vs_at


def main() -> None:
    here = Path(__file__).resolve().parent
    write_vp(here / "true_vp.smesh", **VP_TRUE)
    write_vp(here / "start_vp.smesh", **VP_START)
    write_vp(here / "vp.smesh", **VP_START)
    write_vs(here / "true_vs.smesh", KAPPA_TRUE, **VP_TRUE)
    write_vs(here / "start_vs.smesh", KAPPA_START, **VP_START)
    write_seafloor(here / "seafloor.refl")
    write_conv(here / "conv.refl")
    write_geom(here / "geom_ppp.dat", codes=CODES_PPP)
    write_geom(here / "geom_inv.dat", codes=CODES)
    write_geom(here / "geom_joint.dat", codes=CODES_JOINT)
    write_vcorr(here / "vcorr.dat")
    print(f"wrote {here}")
    print(f"  geom_joint.dat  codes {list(CODES_JOINT)}  （同一次 PPP+PSS）")
    print(f"  geom_ppp.dat    codes {list(CODES_PPP)}  （可选两步的 PPP）")
    print(f"  geom_inv.dat    codes {list(CODES)}  （可选两步的 PSS）")
    print(f"  start_vs.smesh  start_vp / {KAPPA_START:g}")
    print(f"  vcorr.dat     Lh={LH:g} Lv={LV:g}")


if __name__ == "__main__":
    main()
