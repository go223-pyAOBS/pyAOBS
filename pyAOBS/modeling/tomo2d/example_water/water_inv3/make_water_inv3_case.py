#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""water_inv3：0–200 km、水深 3 km、水中 30×1 km 高速异常。

10 台、台距 10 km；炮间隔 0.2 km。直达 2 偏移 ≤20 km，多次 3 偏移 ≤40 km。
反演：``-Y -y -SV200``、Lh=8、Lv=1.0。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "water_inv"))
import make_water_inv_case as inv  # noqa: E402

V_WATER = 1.50
V_ANOM = 1.65
V_START = 1.50
V_SEDIMENT = 1.8
V_AIR = 0.33
H = 3.0
XMIN, XMAX, DX = 0.0, 200.0, 2.0
ZMAX, DZ = 5.0, 0.1
# 高速异常：30 km 宽、1 km 高，放在测线中部、水柱中间
AX0, AX1 = 85.0, 115.0
AZ0, AZ1 = 1.0, 2.0
SHOT_Z = 0.01
SHOT_DX = 0.2
RANGE_2 = 20.0
RANGE_3 = 40.0
OBS_XS = tuple(55.0 + 10.0 * i for i in range(10))
OBS_X = 100.0
SHOT_XS = tuple(round(i * SHOT_DX, 1) for i in range(int(round(XMAX / SHOT_DX)) + 1))
LH, LV = 8.0, 1.0
WSV = 200.0
V_TRUE_BG = V_WATER

parse_smesh = inv.parse_smesh


def in_anomaly(x: float, z: float) -> bool:
    return AX0 - 1e-9 <= x <= AX1 + 1e-9 and AZ0 - 1e-9 <= z <= AZ1 + 1e-9


def v_at(x: float, z: float, *, anomaly: bool) -> float:
    if z > H + 1e-9:
        return V_SEDIMENT
    if anomaly and in_anomaly(x, z):
        return V_ANOM
    return V_WATER


def _xs() -> list[float]:
    n = int(round((XMAX - XMIN) / DX)) + 1
    return [XMIN + i * DX for i in range(n)]


def _zs() -> list[float]:
    n = int(round(ZMAX / DZ)) + 1
    return [i * DZ for i in range(n)]


def write_smesh(path: Path, *, anomaly: bool) -> None:
    xs, zs = _xs(), _zs()
    nx, nz = len(xs), len(zs)
    lines = [
        f"{nx} {nz} {V_WATER:.4f} {V_AIR}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    for x in xs:
        col = [f"{v_at(x, z, anomaly=anomaly):.4f}" for z in zs]
        lines.append(" ".join(col))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_true(path: Path) -> None:
    write_smesh(path, anomaly=True)


def write_start(path: Path) -> None:
    write_smesh(path, anomaly=False)


def write_seafloor(path: Path) -> None:
    xs = _xs()
    path.write_text("".join(f"{x:.4f} {H:.4f}\n" for x in xs), encoding="utf-8")


def write_vcorr(path: Path, *, lh: float = LH, lv: float = LV) -> None:
    path.write_text(
        "2 2\n"
        f"{XMIN:.0f} {XMAX:.0f}\n"
        "0.0 0.0\n"
        f"0.0 {ZMAX:.1f}\n"
        f"{lh:.1f} {lh:.1f}\n"
        f"{lh:.1f} {lh:.1f}\n"
        f"{lv:.1f} {lv:.1f}\n"
        f"{lv:.1f} {lv:.1f}\n",
        encoding="utf-8",
    )


def shots_for(ox: float, rng: float) -> list[float]:
    return [x for x in SHOT_XS if abs(x - ox) <= rng + 1e-9]


def write_geom(path: Path, *, codes: tuple[int, ...] = (2, 3)) -> None:
    if not codes:
        raise ValueError("codes must not be empty")
    ranges = {2: RANGE_2, 3: RANGE_3}
    lines = [str(len(OBS_XS))]
    for ox in OBS_XS:
        blocks = [(kind, shots_for(ox, ranges[kind])) for kind in codes]
        nrcv = sum(len(sx) for _k, sx in blocks)
        lines.append(inv._fmt_s_line(ox, H, nrcv))
        for kind, sx in blocks:
            for x in sx:
                lines.append(inv._fmt_r_line(x, SHOT_Z, kind, 0.0, 0.0))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def node_stats(
    xs: list[float],
    zs: list[float],
    vel: list[list[float]],
    *,
    x_lo: float,
    x_hi: float,
    water: bool,
) -> tuple[float, float, float, int]:
    vals: list[float] = []
    for i, x in enumerate(xs):
        if x < x_lo - 1e-9 or x > x_hi + 1e-9:
            continue
        for k, z in enumerate(zs):
            is_water = z <= H + 1e-9
            if is_water != water:
                continue
            vals.append(vel[i][k])
    if not vals:
        raise ValueError("no nodes in requested window")
    return sum(vals) / len(vals), min(vals), max(vals), len(vals)


def box_stats(
    xs: list[float],
    zs: list[float],
    vel: list[list[float]],
    *,
    x_lo: float,
    x_hi: float,
    z_lo: float,
    z_hi: float,
) -> tuple[float, float, float, int]:
    vals: list[float] = []
    for i, x in enumerate(xs):
        if x < x_lo - 1e-9 or x > x_hi + 1e-9:
            continue
        for k, z in enumerate(zs):
            if z < z_lo - 1e-9 or z > z_hi + 1e-9:
                continue
            vals.append(vel[i][k])
    if not vals:
        raise ValueError("no nodes in box")
    return sum(vals) / len(vals), min(vals), max(vals), len(vals)


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


def illum_x_range() -> tuple[float, float]:
    return min(OBS_XS) - RANGE_3 - DX, max(OBS_XS) + RANGE_3 + DX


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
    a_m, a_lo, a_hi, n_a = box_stats(
        xs, zs, vt, x_lo=AX0, x_hi=AX1, z_lo=AZ0, z_hi=AZ1
    )
    n2 = sum(len(shots_for(ox, RANGE_2)) for ox in OBS_XS)
    n3 = sum(len(shots_for(ox, RANGE_3)) for ox in OBS_XS)
    print(f"wrote {here}")
    print(
        f"  true.smesh   0–{XMAX:g} km  H={H:g}  水 {V_WATER:g}  "
        f"异常 {AX0:g}–{AX1:g} × {AZ0:g}–{AZ1:g} v={V_ANOM:g}  "
        f"box mean={a_m:.4f} n={n_a}"
    )
    print(
        f"  illum water mean={t_m:.4f} [{t_lo:.4f},{t_hi:.4f}] n={n_w}  "
        f"x=[{x_lo:g},{x_hi:g}]"
    )
    print(f"  start.smesh  均匀水 {V_START:g}  沉积 {V_SEDIMENT:g}")
    print(
        f"  geom_inv.dat  {len(OBS_XS)} 台 {list(OBS_XS)}  "
        f"shot {SHOT_DX:g} km  code2 |dx|<={RANGE_2:g} n={n2}  "
        f"code3 |dx|<={RANGE_3:g} n={n3}"
    )
    print(f"  vcorr.dat     Lh={LH:g} Lv={LV:g}  （-CV，配 -SV{WSV:g}）")


if __name__ == "__main__":
    main()
