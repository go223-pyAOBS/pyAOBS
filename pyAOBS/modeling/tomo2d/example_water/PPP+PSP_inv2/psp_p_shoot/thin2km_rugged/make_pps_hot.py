#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PPS 热初值工区：拷 PPP 收回的 rec_vp，几何改成 raytype 7。"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_fair_grids import prepare, write_geom_code  # noqa: E402

PPP_SRC = HERE / "inv_graph6k_hot"


def prepare_pps(dest: Path) -> None:
    dest = dest.resolve()
    prepare(dest)
    write_geom_code(dest / "geom_ppp.dat", dest / "geom_pps7.dat", 7)
    rec = PPP_SRC / "rec_vp.smesh"
    if not rec.is_file():
        raise SystemExit(f"missing {rec} — 先跑 PSP 公平两步 PPP")
    shutil.copyfile(rec, dest / "rec_vp.smesh")
    print(f"{dest.name}: geom_pps7.dat  rec_vp <- {PPP_SRC.name}")


def main() -> int:
    dest = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    prepare_pps(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
