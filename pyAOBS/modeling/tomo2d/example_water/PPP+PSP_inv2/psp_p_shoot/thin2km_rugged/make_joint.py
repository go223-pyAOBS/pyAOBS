#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P1 四震相联合：geom 0+6+7+8，Vp 初值=PPP 收回，不拷走时。"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
from make_fair_grids import prepare, write_geom_code, write_geom_codes  # noqa: E402

PPP_SRC = HERE / "inv_graph6k_hot"
CODES_ALL = (0, 6, 7, 8)
CODES_678 = (6, 7, 8)
CODES_078 = (0, 7, 8)
CODES_07 = (0, 7)


def prepare_joint(dest: Path) -> None:
    dest = dest.resolve()
    prepare(dest)
    write_geom_codes(dest / "geom_ppp.dat", dest / "geom_all.dat", CODES_ALL)
    write_geom_codes(dest / "geom_ppp.dat", dest / "geom_678.dat", CODES_678)
    write_geom_codes(dest / "geom_ppp.dat", dest / "geom_078.dat", CODES_078)
    write_geom_codes(dest / "geom_ppp.dat", dest / "geom_07.dat", CODES_07)
    write_geom_code(dest / "geom_ppp.dat", dest / "geom_psp6.dat", 6)
    write_geom_code(dest / "geom_ppp.dat", dest / "geom_pps7.dat", 7)
    write_geom_code(dest / "geom_ppp.dat", dest / "geom_pss8.dat", 8)
    rec = PPP_SRC / "rec_vp.smesh"
    if not rec.is_file():
        raise SystemExit(f"missing {rec} — 先跑 PSP 公平两步 PPP")
    shutil.copyfile(rec, dest / "ppp_vp.smesh")
    shutil.copyfile(rec, dest / "rec_vp.smesh")
    print(
        f"{dest.name}: geom_678.dat codes {list(CODES_678)}  "
        f"geom_078.dat {list(CODES_078)}  "
        f"geom_all.dat {list(CODES_ALL)}  ppp_vp <- {PPP_SRC.name}"
    )


def main() -> int:
    dest = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    prepare_joint(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
