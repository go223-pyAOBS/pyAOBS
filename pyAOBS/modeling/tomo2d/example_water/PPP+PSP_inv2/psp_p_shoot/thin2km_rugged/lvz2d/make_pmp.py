#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在 lvz2d 上加起伏 Moho（type 1），写出 cmp_0178。

真值：中间缓凸（变浅）；反演初值：同区域斜率的光滑面（传统 PPP+PmP 可动深度）。
"""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inv_2d"))
from make_fair_grids import write_geom_codes  # noqa: E402
import inv_grid as g  # noqa: E402

DEST = HERE / "cmp_0178"
COPY = (
    "true_vp.smesh",
    "true_vs.smesh",
    "start_vp.smesh",
    "seafloor.refl",
    "conv.refl",
    "vcorr.dat",
    "geom_ppp.dat",
)


def _snap(z: float) -> float:
    z = int(round(z / g.DZ_IFACE)) * g.DZ_IFACE
    return z


def z_moho_true(x: float) -> float:
    """真 Moho：长波长中间上凸（变浅），幅度约 1.2 km，始终在转换面以下。"""
    z = (
        11.0
        + 0.015 * (x - g.XREF)
        - 1.20 * math.exp(-((x - g.XREF) / 26.0) ** 2)
    )
    return max(g.z_conv(x) + 4.0, min(g.ZMAX - 1.0, _snap(z)))


def z_moho_start(x: float) -> float:
    """建议初值：只保留区域斜率，无短波长起伏。"""
    z = 11.0 + 0.015 * (x - g.XREF)
    return max(g.z_conv(x) + 4.0, min(g.ZMAX - 1.0, _snap(z)))


def write_moho(src_conv: Path, dst: Path, zfun) -> tuple[float, float]:
    xs = [float(ln.split()[0]) for ln in src_conv.read_text(encoding="utf-8").splitlines() if ln.strip()]
    zs = [zfun(x) for x in xs]
    dst.write_text("".join(f"{x:.4f} {z:.4f}\n" for x, z in zip(xs, zs)), encoding="utf-8")
    return min(zs), max(zs)


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in COPY:
        sp = HERE / name
        if not sp.is_file():
            raise SystemExit(f"missing {sp}")
        shutil.copyfile(sp, DEST / name)
    ztlo, zthi = write_moho(DEST / "conv.refl", DEST / "moho_true.refl", z_moho_true)
    zslo, zshi = write_moho(DEST / "conv.refl", DEST / "moho.refl", z_moho_start)
    write_geom_codes(DEST / "geom_ppp.dat", DEST / "geom_holdout.dat", (0, 1, 6, 7, 8))
    write_geom_codes(DEST / "geom_ppp.dat", DEST / "geom_inv.dat", (0, 1, 7, 8))
    g.write_vs_from_vp(DEST / "start_vp.smesh", DEST / "start_vs.smesh", g.KAPPA)
    for sub in ("joint", "strat"):
        d = DEST / sub
        d.mkdir(parents=True, exist_ok=True)
        for name in (
            "true_vp.smesh",
            "true_vs.smesh",
            "start_vp.smesh",
            "start_vs.smesh",
            "seafloor.refl",
            "conv.refl",
            "moho.refl",
            "moho_true.refl",
            "vcorr.dat",
        ):
            shutil.copyfile(DEST / name, d / name)
    print(
        f"cmp_0178: true Moho z={ztlo:.2f}–{zthi:.2f} km  "
        f"start Moho z={zslo:.2f}–{zshi:.2f} km  geom 0+1+7+8 + holdout 6"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
