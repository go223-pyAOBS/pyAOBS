#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""water_checkboard：水柱棋盘格，对照 2/3 与仅 2 能收回多少异常。

几何同 water_inv：5 台、炮 30–70 km、间隔 0.2 km。
棋盘只在水里：10 km × 1 km，±0.10 km/s。反演 ``-Y -y -SV200``，
Lh=6、Lv=0.4（小于格子，避免正则先把棋盘抹平）。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "water_inv"))
import make_water_inv_case as inv  # noqa: E402

V_WATER = 1.50
V_START = 1.50
V_TRUE_BG = 1.50
V_SEDIMENT = 1.8
V_AIR = 0.33
H = inv.H
XMIN, XMAX, DX = inv.XMIN, inv.XMAX, inv.DX
ZMAX, DZ = inv.ZMAX, inv.DZ
OBS_X = inv.OBS_X
OBS_XS = inv.OBS_XS
SHOT_Z = inv.SHOT_Z
SHOT_DX = inv.SHOT_DX
SHOT_XS = inv.SHOT_XS
# 棋盘：与台距对齐，照明区 30–70 正好 4×2 格
CX, CZ = 10.0, 1.0
X0 = 30.0
DV = 0.10
LH, LV = 6.0, 0.4
WSV = 200.0

parse_smesh = inv.parse_smesh
node_stats = inv.node_stats
illum_x_range = inv.illum_x_range
write_seafloor = inv.write_seafloor


def checker_index(x: float, z: float) -> tuple[int, int]:
    ix = int(math.floor((x - X0) / CX + 1e-12))
    z_clip = min(max(z, 0.0), H - 1e-9)
    iz = int(math.floor(z_clip / CZ + 1e-12))
    return ix, iz


def checker_sign(x: float, z: float) -> int:
    ix, iz = checker_index(x, z)
    return 1 if (ix + iz) % 2 == 0 else -1


def v_at(x: float, z: float, *, checker: bool) -> float:
    if z > H + 1e-9:
        return V_SEDIMENT
    if not checker:
        return V_WATER
    return V_WATER + checker_sign(x, z) * DV


def _xs() -> list[float]:
    n = int(round((XMAX - XMIN) / DX)) + 1
    return [XMIN + i * DX for i in range(n)]


def _zs() -> list[float]:
    n = int(round(ZMAX / DZ)) + 1
    return [i * DZ for i in range(n)]


def write_smesh(path: Path, *, checker: bool) -> None:
    xs, zs = _xs(), _zs()
    nx, nz = len(xs), len(zs)
    lines = [
        f"{nx} {nz} {V_WATER:.4f} {V_AIR}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    for x in xs:
        col = [f"{v_at(x, z, checker=checker):.4f}" for z in zs]
        lines.append(" ".join(col))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_true(path: Path) -> None:
    write_smesh(path, checker=True)


def write_start(path: Path) -> None:
    write_smesh(path, checker=False)


def write_geom(path: Path, *, codes: tuple[int, ...] = (2, 3)) -> None:
    if not codes:
        raise ValueError("codes must not be empty")
    nrcv = len(SHOT_XS) * len(codes)
    lines = [str(len(OBS_XS))]
    for ox in OBS_XS:
        lines.append(inv._fmt_s_line(ox, H, nrcv))
        for kind in codes:
            for x in SHOT_XS:
                lines.append(inv._fmt_r_line(x, SHOT_Z, kind, 0.0, 0.0))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_vcorr(path: Path, *, lh: float = LH, lv: float = LV) -> None:
    inv.write_vcorr(path, lh=lh, lv=lv)


def water_field_compare(
    xs: list[float],
    zs: list[float],
    va: list[list[float]],
    vb: list[list[float]],
    *,
    x_lo: float,
    x_hi: float,
) -> tuple[float, float, int]:
    da: list[float] = []
    db: list[float] = []
    for i, x in enumerate(xs):
        if x < x_lo - 1e-9 or x > x_hi + 1e-9:
            continue
        for k, z in enumerate(zs):
            if z > H + 1e-9:
                continue
            da.append(va[i][k])
            db.append(vb[i][k])
    n = len(da)
    if n < 2:
        raise ValueError("too few water nodes")
    rms = math.sqrt(sum((a - b) ** 2 for a, b in zip(da, db)) / n)
    ma = sum(da) / n
    mb = sum(db) / n
    cov = sum((a - ma) * (b - mb) for a, b in zip(da, db))
    sa = math.sqrt(sum((a - ma) ** 2 for a in da))
    sb = math.sqrt(sum((b - mb) ** 2 for b in db))
    corr = cov / (sa * sb) if sa > 0.0 and sb > 0.0 else 0.0
    return rms, corr, n


def checker_recovery(
    xs: list[float],
    zs: list[float],
    vrec: list[list[float]],
    vtrue: list[list[float]],
    *,
    x_lo: float,
    x_hi: float,
) -> tuple[float, float, float, int]:
    """照明区水：极性一致率、收回 |Δv| / 真 |Δv|、扰动相关。"""
    pr: list[float] = []
    pt: list[float] = []
    hit = 0
    for i, x in enumerate(xs):
        if x < x_lo - 1e-9 or x > x_hi + 1e-9:
            continue
        for k, z in enumerate(zs):
            if z > H + 1e-9:
                continue
            dr = vrec[i][k] - V_WATER
            dt = vtrue[i][k] - V_WATER
            pr.append(dr)
            pt.append(dt)
            if dr * dt > 0.0:
                hit += 1
    n = len(pt)
    if n < 2:
        raise ValueError("too few water nodes")
    amp_t = sum(abs(v) for v in pt) / n
    amp_r = sum(abs(v) for v in pr) / n
    ratio = amp_r / amp_t if amp_t > 0.0 else 0.0
    mr = sum(pr) / n
    mt = sum(pt) / n
    cov = sum((a - mr) * (b - mt) for a, b in zip(pr, pt))
    sr = math.sqrt(sum((a - mr) ** 2 for a in pr))
    st = math.sqrt(sum((b - mt) ** 2 for b in pt))
    corr = cov / (sr * st) if sr > 0.0 and st > 0.0 else 0.0
    return hit / n, ratio, corr, n


def checker_x_lines() -> list[float]:
    x_lo, x_hi = illum_x_range()
    xs: list[float] = []
    x = X0
    while x < x_hi + 1e-9:
        if x > x_lo - 1e-9:
            xs.append(x)
        x += CX
    return xs


def checker_z_lines() -> list[float]:
    zs = [i * CZ for i in range(int(round(H / CZ)) + 1)]
    return [z for z in zs if 0.0 - 1e-9 <= z <= H + 1e-9]


def main() -> None:
    here = HERE
    write_true(here / "true.smesh")
    write_start(here / "start.smesh")
    write_seafloor(here / "seafloor.refl")
    write_geom(here / "geom_inv.dat", codes=(2, 3))
    write_geom(here / "geom_inv_c2.dat", codes=(2,))
    write_vcorr(here / "vcorr.dat")
    xs, zs, vt = parse_smesh(here / "true.smesh")
    x_lo, x_hi = illum_x_range()
    t_m, t_lo, t_hi, n_w = node_stats(xs, zs, vt, x_lo=x_lo, x_hi=x_hi, water=True)
    print(f"wrote {here}")
    print(
        f"  true.smesh   棋盘 {CX:g}×{CZ:g} km  ±{DV:g}  "
        f"illum water mean={t_m:.4f} [{t_lo:.4f},{t_hi:.4f}] n={n_w}"
    )
    print(f"  start.smesh  均匀水 {V_START:g}  沉积 {V_SEDIMENT:g}")
    print(
        f"  geom_inv.dat  {len(OBS_XS)} 台 {list(OBS_XS)}  "
        f"shots {min(SHOT_XS):g}..{max(SHOT_XS):g} step {SHOT_DX:g}  codes 2+3"
    )
    print(f"  geom_inv_c2.dat  同上，仅 code 2")
    print(f"  vcorr.dat     Lh={LH:g} Lv={LV:g}  （-CV，配 -SV{WSV:g}）")


if __name__ == "__main__":
    main()
