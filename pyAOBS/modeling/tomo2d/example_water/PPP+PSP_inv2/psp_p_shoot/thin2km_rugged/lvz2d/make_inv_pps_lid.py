#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""盖层 Vs 对照工区：PPS(7) vs PPS 盖层 SS 多次(10) vs 7+10。冻真 Vp、冻面下。不改 cmp_0178。"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "inv_2d"))
import inv_grid as g  # noqa: E402
import make_fwd_all as fwd  # noqa: E402
import make_inv_612 as case612  # noqa: E402
import make_pmp as pmp  # noqa: E402

SRC = HERE / "cmp_0178"
DEST = HERE / "inv_pps_lid"
JOBS = (
    ("geom_7.dat", (7,)),
    ("geom_10.dat", (10,)),
    ("geom_710.dat", (7, 10)),
)


def _write_geom(path: Path, codes: tuple[int, ...]) -> int:
    recs_by_obs: dict[float, list[str]] = {}
    nrec = 0
    for ox in case612.OBS_XS:
        recs = []
        for code in codes:
            for x in case612.SHOT_XS:
                if abs(x - ox) <= case612.DX_MIN:
                    continue
                recs.append(f"r  {x:8.3f}     0.010 {code:4d}     0.000     0.050")
        recs_by_obs[ox] = recs
        nrec += len(recs)
    lines = [str(len(case612.OBS_XS))]
    for ox in case612.OBS_XS:
        recs = recs_by_obs[ox]
        lines.append(f"s  {ox:8.3f}     2.000 {len(recs):4d}")
        lines.extend(recs)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return nrec


def main() -> int:
    if not (SRC / "true_vp.smesh").is_file():
        raise SystemExit(f"missing {SRC} (run make_lvz.py / make_pmp.py first)")
    DEST.mkdir(parents=True, exist_ok=True)
    for name in (
        "true_vp.smesh",
        "seafloor.refl",
        "conv.refl",
        "vcorr.dat",
        "start_vp.smesh",
    ):
        shutil.copyfile(SRC / name, DEST / name)
    shutil.copyfile(SRC / "moho_true.refl", DEST / "moho_true.refl")
    shutil.copyfile(SRC / "moho.refl", DEST / "moho.refl")
    n_x = case612.extend_smesh(DEST / "true_vp.smesh")
    case612.extend_smesh(DEST / "start_vp.smesh")
    case612.extend_refl(DEST / "seafloor.refl", lambda _x: g.H)
    xs = case612._xs_of(DEST / "seafloor.refl")
    case612.write_refl(DEST / "conv.refl", xs, case612.z_conv)
    case612.extend_refl(DEST / "moho_true.refl", pmp.z_moho_true)
    case612.extend_refl(DEST / "moho.refl", pmp.z_moho_start)
    case612.write_vcorr(DEST / "vcorr.dat", case612.VCORR_LH, case612.VCORR_LV)
    g.z_conv = case612.z_conv
    fwd.reshape_crust_mantle(
        DEST / "true_vp.smesh",
        DEST / "moho_true.refl",
        z_conv=case612.z_conv,
        z_conv_old=case612.z_conv_uncapped,
    )
    fwd.reshape_crust_mantle(
        DEST / "start_vp.smesh",
        DEST / "moho.refl",
        keep_anom=False,
        bg=fwd.START_1D,
        z_conv=case612.z_conv,
    )
    # 初值盖层 = START_1D（无 LID_LVZ，整体偏快），面下 = 真 Vs（FREEZE_BELOW）。
    n_lid = g.graft_lid_vp(DEST / "start_vs.smesh", DEST / "true_vs.smesh", DEST / "start_vs.smesh")
    s1 = fwd.START_1D
    print(
        f"  网格 0–{case612.XMAX:.0f} km  +{n_x} 列  OBS={case612.OBS_XS}  "
        f"start 盖层=START_1D {s1.lid0:g}→{s1.lid_iface:g} n={n_lid}  面下=真 Vs"
    )
    for name, codes in JOBS:
        nrec = _write_geom(DEST / name, codes)
        print(f"  {name}  nsrc={len(case612.OBS_XS)} nrec_total={nrec}  codes={codes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
