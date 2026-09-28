#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""water_inv2：真模型用 water_fwd 扰动场，初值均匀 1.5；4 台、炮距同 water_inv。

反演：``-Y -y -SV200``、Lh=8、Lv=1.0。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "water_inv"))
sys.path.insert(0, str(ROOT / "water_fwd"))

import make_water_fwd_case as fwd  # noqa: E402
import make_water_inv_case as inv  # noqa: E402

H = inv.H
OBS_X = inv.OBS_X
OBS_XS = (30.0, 43.3, 56.7, 70.0)
SHOT_Z = inv.SHOT_Z
SHOT_DX = inv.SHOT_DX
OFFSETS = inv.OFFSETS
SHOT_XS = inv.SHOT_XS
LH = 8.0
LV = 1.0
WSV = 200.0
V_START = fwd.V_WATER  # 均匀水 1.5
V_TRUE_BG = fwd.V_WATER
V_SEDIMENT = fwd.V_SEDIMENT
V_AIR = fwd.V_AIR
DX = inv.DX

parse_smesh = inv.parse_smesh
node_stats = inv.node_stats
illum_x_range = inv.illum_x_range
write_seafloor = inv.write_seafloor


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


def write_true(path: Path) -> None:
    """与 water_fwd/water.smesh 同一套扰动（seed/amp/相关长度）。"""
    fwd.write_smesh(path)


def write_start(path: Path) -> None:
    """均匀水 1.5，沉积 1.8，网格与真模型相同。"""
    fwd.write_smesh(path, noise_amp=0.0)


def water_field_compare(
    xs: list[float],
    zs: list[float],
    va: list[list[float]],
    vb: list[list[float]],
    *,
    x_lo: float,
    x_hi: float,
) -> tuple[float, float, int]:
    """照明区水结点：RMS(va−vb) 与相关系数。"""
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
        f"  true.smesh   water_fwd 背景 {V_TRUE_BG:g} + 扰动  "
        f"illum water mean={t_m:.4f} [{t_lo:.4f},{t_hi:.4f}] n={n_w}"
    )
    print(f"  start.smesh  均匀水 {V_START:g}  沉积 {V_SEDIMENT:g}")
    print("  seafloor.refl  (-Y 反演 / 正演也可用 -B 或 -F)")
    print(
        f"  geom_inv.dat  OBS x={list(OBS_XS)} z={H}  "
        f"shots {min(SHOT_XS):g}..{max(SHOT_XS):g} step {SHOT_DX:g}  codes 2+3"
        f"  （4 台）"
    )
    print(
        f"  geom_inv_c2.dat  同上，仅 code 2  "
        f"nrcv={len(SHOT_XS)}/台"
    )
    print(f"  vcorr.dat     Lh={LH:g} Lv={LV:g}  （-CV，配 -SV{WSV:g}）")


if __name__ == "__main__":
    main()
