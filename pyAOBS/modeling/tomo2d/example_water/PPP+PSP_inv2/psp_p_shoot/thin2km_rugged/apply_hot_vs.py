#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""冻 PPP 收回的 Vp；Vs = rec_vp/κ + HOT_DV。默认只加面下；--lid 连盖层一起加（PSS）。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "inv_2d"))
import inv_grid as g  # noqa: E402


def apply(
    folder: Path,
    dv: float = g.HOT_DV,
    *,
    lid: bool = False,
    lid_only: bool = False,
    kappa_only: bool = False,
    true_lid_vp: bool = False,
) -> None:
    rec_vp = folder / "rec_vp.smesh"
    if not rec_vp.is_file():
        raise SystemExit(f"missing {rec_vp} — 先跑公平两步 PPP")
    if true_lid_vp:
        true_vp = folder / "true_vp.smesh"
        nlid = g.graft_lid_vp(true_vp, rec_vp, rec_vp)
        ppp = folder / "ppp_vp.smesh"
        if ppp.is_file():
            g.graft_lid_vp(true_vp, ppp, ppp)
        print(f"{folder.name}: rec_vp lid <- true_vp  n={nlid}  below stays PPP")
    start_vs = folder / "start_vs.smesh"
    g.write_vs_from_vp(rec_vp, start_vs, g.KAPPA)
    if kappa_only:
        n = 0
        where = "no bump"
    elif lid_only:
        n = g.bump_lid_only(start_vs, start_vs, dv)
        where = "lid only"
    elif lid:
        n = g.bump_below_seafloor(start_vs, start_vs, dv)
        where = "below seafloor (lid+below)"
    else:
        n = g.bump_below_conv(start_vs, start_vs, dv)
        where = "below conv"
    g.write_psp_speed(rec_vp, start_vs, folder / "start_mixed.smesh")
    extra = " " if kappa_only else f" + {dv:g} "
    print(f"{folder.name}: start_vs = rec_vp/{g.KAPPA:g}{extra}{where}  n={n}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("dest", nargs="?", default=".")
    p.add_argument("--lid", action="store_true", help="盖层也 +dv（PSS）")
    p.add_argument("--lid-only", action="store_true", help="只加盖层（PPS）")
    p.add_argument("--kappa", action="store_true", help="只写 rec_vp/κ，不加扰动")
    p.add_argument("--true-lid", action="store_true", help="盖层 Vp 换成真值，面下仍用 PPP 收回")
    args = p.parse_args()
    apply(
        Path(args.dest).resolve(),
        lid=args.lid,
        lid_only=args.lid_only,
        kappa_only=args.kappa,
        true_lid_vp=args.true_lid,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
