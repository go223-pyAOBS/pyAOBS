#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 inv_2d 拷网格/几何（不拷走时）。写出 true_vs、true_mixed、geom_psp6。"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "inv_2d"))
import inv_grid as g  # noqa: E402

SRC = HERE / "inv_2d"
COPY = (
    "true_vp.smesh",
    "start_vp.smesh",
    "seafloor.refl",
    "conv.refl",
    "damp_lid.dat",
    "vcorr.dat",
    "geom_ppp.dat",
)


def write_geom_code(src: Path, dst: Path, code: int) -> None:
    write_geom_codes(src, dst, (code,))


def write_geom_codes(src: Path, dst: Path, codes: tuple[int, ...]) -> None:
    """每个炮–台对展开为多震相；s 行接收数 = 原接收数 × len(codes)。"""
    lines = src.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    i = 0
    if lines and lines[0].split()[:1] != ["s"]:
        out.append(lines[0])
        i = 1
    while i < len(lines):
        parts = lines[i].split()
        if parts[:1] == ["s"]:
            recs: list[str] = []
            i += 1
            while i < len(lines) and lines[i].split()[:1] == ["r"]:
                recs.append(lines[i])
                i += 1
            n = len(recs) * len(codes)
            out.append(f"s {float(parts[1]):8.3f} {float(parts[2]):9.3f} {n:4d}")
            for code in codes:
                for ln in recs:
                    p = ln.split()
                    out.append(
                        f"r {float(p[1]):8.3f} {float(p[2]):9.3f} "
                        f"{code:4d} {float(p[4]):9.3f} {float(p[5]):9.3f}"
                    )
        else:
            out.append(lines[i])
            i += 1
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")


def prepare(dest: Path) -> None:
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    for name in COPY:
        sp = SRC / name
        if not sp.is_file():
            raise SystemExit(f"missing {sp} — 先跑 inv_2d/make_rugged_2d_inv.py")
        shutil.copyfile(sp, dest / name)
    write_geom_code(dest / "geom_ppp.dat", dest / "geom_psp6.dat", 6)
    g.write_vs_from_vp(dest / "true_vp.smesh", dest / "true_vs.smesh", g.KAPPA)
    g.write_psp_speed(dest / "true_vp.smesh", dest / "true_vs.smesh", dest / "true_mixed.smesh")
    print(f"{dest.name}: grids from inv_2d, true_vs=true_vp/{g.KAPPA:g}, no syn copied")


def main() -> int:
    dest = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    prepare(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
