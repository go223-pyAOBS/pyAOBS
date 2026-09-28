#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""孤立实验：把真莫霍整体下移到 z(50)=15 km，网格加深到 20 km。

不改 inv_612 / inv_dd20 / cmp_0178。速度按新界面重铺。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inv_2d"))
import inv_grid as g  # noqa: E402
import make_ppp_psp_inv_case as m2  # noqa: E402
from make_fwd_all import TRUE_1D, reshape_crust_mantle  # noqa: E402
from make_lvz import apply_lvz, write_smesh  # noqa: E402

SRC = HERE / "inv_612"
WAVE = HERE / "wave_fwd"
DEST = WAVE / "moho15"
ZMAX = 20.0
DZ = 0.2
Z50 = 15.0


def _extend_vp(src: Path, dst: Path, zmax: float, dz: float) -> None:
    xs, zs, vel = m2.parse_smesh(src)
    n = int(round(zmax / dz)) + 1
    new_zs = [i * dz for i in range(n)]
    new_vel = []
    for col in vel:
        row = []
        for z in new_zs:
            if z <= zs[-1] + 1e-9:
                k = min(range(len(zs)), key=lambda j: abs(zs[j] - z))
                row.append(col[k])
            else:
                row.append(col[-1])
        new_vel.append(row)
    write_smesh(dst, xs, new_zs, new_vel)


def _shift_moho(src: Path, dst: Path, z50: float) -> tuple[float, float, float]:
    xs, zs = [], []
    for raw in src.read_text(encoding="utf-8").splitlines():
        a = raw.split()
        if len(a) >= 2:
            xs.append(float(a[0]))
            zs.append(float(a[1]))
    z_mid = zs[min(range(len(xs)), key=lambda i: abs(xs[i] - 50.0))]
    dz = z50 - z_mid
    zmax_ok = ZMAX - 1.0
    lines = []
    newz = []
    for x, z in zip(xs, zs):
        zz = z + dz
        k = int(round(zz / DZ))
        zz = k * DZ
        zz = max(g.H + 2.0, min(zmax_ok, zz))
        newz.append(zz)
        lines.append(f" {x:.4f} {zz:.4f}")
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return z_mid, min(newz), max(newz)


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in ("seafloor.refl", "conv.refl"):
        shutil.copyfile(SRC / name, DEST / name)
    geom = WAVE / "geom_obs50.dat"
    if not geom.is_file():
        raise SystemExit(f"missing {geom}")
    shutil.copyfile(geom, DEST / "geom_obs50.dat")
    z_old, zmin, zmax = _shift_moho(SRC / "moho_true.refl", DEST / "moho_true.refl", Z50)
    _extend_vp(SRC / "true_vp.smesh", DEST / "true_vp.smesh", ZMAX, DZ)
    # 旧幔节点会变成新地壳，不能 keep_anom；铺干净 1D 后再贴 LVZ。
    reshape_crust_mantle(
        DEST / "true_vp.smesh", DEST / "moho_true.refl", keep_anom=False, bg=TRUE_1D
    )
    apply_lvz(DEST / "true_vp.smesh")
    g.write_vs_from_vp(DEST / "true_vp.smesh", DEST / "true_vs.smesh", g.KAPPA)
    print(f"  moho shift z(50) {z_old:.2f} → {Z50:.2f}  range {zmin:.2f}–{zmax:.2f}  zmax_grid={ZMAX}")
    print(f"  wrote {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
