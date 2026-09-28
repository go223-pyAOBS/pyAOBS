#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vs 段工区：只反 12/13。冻真 Vp，初 Vs=真盖层 + 面下 1D，初莫霍无凸起。不改 cmp_0178。"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "inv_2d"))
import inv_grid as g  # noqa: E402
import make_fwd_all as fwd  # noqa: E402
import make_pmp as pmp  # noqa: E402

SRC = HERE / "cmp_0178"
DEST = HERE / "inv_1213"
DX_MIN = 10.0
CODES = (12, 13)


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
    fwd.reshape_crust_mantle(DEST / "true_vp.smesh", DEST / "moho_true.refl")
    g.write_vs_from_vp(DEST / "start_vp.smesh", DEST / "start_vs.smesh", g.KAPPA)
    n_lid = g.graft_lid_vp(DEST / "true_vs.smesh", DEST / "start_vs.smesh")
    print(f"  start_vs 盖层 <- 真 Vs  n={n_lid}  面下仍是 start_vp/{g.KAPPA:g}")

    recs_by_obs: dict[float, list[str]] = {}
    nrec = 0
    for ox in fwd.OBS_XS:
        recs = []
        for code in CODES:
            for x in fwd.SHOT_XS:
                if abs(x - ox) <= DX_MIN:
                    continue
                recs.append(f"r  {x:8.3f}     0.010 {code:4d}     0.000     0.050")
        recs_by_obs[ox] = recs
        nrec += len(recs)
    lines = [str(len(fwd.OBS_XS))]
    for ox in fwd.OBS_XS:
        recs = recs_by_obs[ox]
        lines.append(f"s  {ox:8.3f}     2.000 {len(recs):4d}")
        lines.extend(recs)
    (DEST / "geom_1213.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")
    zt = pmp.z_moho_true
    zs = pmp.z_moho_start
    print(
        f"wrote {DEST}  nsrc={len(fwd.OBS_XS)} nrec_total={nrec}  "
        f"|Δx|>{DX_MIN:.0f}  codes={CODES}"
    )
    print(f"  start Moho (x=50) {zs(50):.2f} km  true {zt(50):.2f} km")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
