#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""正演工区：6/7/8 一次波对照盖层 SS（10/11）与水柱 S→P（9/14/15）。"""

from __future__ import annotations

import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "cmp_0178"
DEST = HERE / "fwd_91011"
OBS_XS = (30.0, 50.0)
SHOT_XS = (20.0, 28.0, 36.0, 44.0, 52.0, 60.0, 68.0, 76.0)
CODES = (6, 9, 7, 10, 14, 8, 11, 15)
COPY = (
    "true_vp.smesh",
    "true_vs.smesh",
    "seafloor.refl",
    "conv.refl",
    "moho_true.refl",
)


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in COPY:
        sp = SRC / name
        if not sp.is_file():
            raise SystemExit(f"missing {sp} (run make_pmp.py / cmp_0178 first)")
        shutil.copyfile(sp, DEST / name)
    recs = []
    for code in CODES:
        for x in SHOT_XS:
            recs.append(f"r  {x:8.3f}     0.010 {code:4d}     0.000     0.050")
    lines = [str(len(OBS_XS))]
    for ox in OBS_XS:
        lines.append(f"s  {ox:8.3f}     2.000 {len(recs):4d}")
        lines.extend(recs)
    (DEST / "geom_91011.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {DEST}  nsrc={len(OBS_XS)} nrec/src={len(recs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
