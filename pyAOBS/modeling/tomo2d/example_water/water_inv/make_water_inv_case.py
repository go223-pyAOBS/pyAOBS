#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 tt_inverse 水核验收工区：topo=0、冻壳只反水。

真模型均匀水 1.45 km/s；初值 1.55；海底以下沉积 1.8 冻结。
炮点不变（海面 30–70 km、间隔 0.2 km）；多个 OBS 在海底。
raytype 2/3。正演造观测后再 ``tt_inverse -Y -y``。
"""

from __future__ import annotations

import math
from pathlib import Path

V_TRUE = 1.45
V_START = 1.55
V_SEDIMENT = 1.8
V_AIR = 0.33
H = 2.0
XMIN, XMAX, DX = 0.0, 100.0, 2.0
ZMAX, DZ = 4.0, 0.1
OBS_X = 50.0
OBS_XS = (30.0, 40.0, 50.0, 60.0, 70.0)
SHOT_Z = 0.01
SHOT_DX = 0.2  # 200 m 一炮
_OFF_N = int(round(20.0 / SHOT_DX))
OFFSETS = tuple(round(-20.0 + i * SHOT_DX, 1) for i in range(2 * _OFF_N + 1))
SHOT_XS = tuple(OBS_X + dx for dx in OFFSETS)
# 速度平滑相关长度（-CV）：水平约数个炮距，垂向约整个水柱；不跨海底（-y 已隔开）。
LH, LV = 8.0, 2.0
WSV = 200.0  # tt_inverse -SV


def v_at(z: float, v_water: float) -> float:
    if z > H + 1e-9:
        return V_SEDIMENT
    return v_water


def t_direct(dx: float, h: float = H, v: float = V_TRUE, z_shot: float = SHOT_Z) -> float:
    return math.hypot(dx, h - z_shot) / v


def t_mult(dx: float, h: float = H, v: float = V_TRUE, z_shot: float = SHOT_Z) -> float:
    return math.hypot(dx, 3.0 * h - z_shot) / v


def _xs() -> list[float]:
    n = int(round((XMAX - XMIN) / DX)) + 1
    return [XMIN + i * DX for i in range(n)]


def _zs() -> list[float]:
    n = int(round(ZMAX / DZ)) + 1
    return [i * DZ for i in range(n)]


def write_smesh(path: Path, v_water: float) -> None:
    xs, zs = _xs(), _zs()
    nx, nz = len(xs), len(zs)
    lines = [
        f"{nx} {nz} {v_water:.4f} {V_AIR}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    for x in xs:
        _ = x
        col = [f"{v_at(z, v_water):.4f}" for z in zs]
        lines.append(" ".join(col))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_seafloor(path: Path) -> None:
    xs = _xs()
    path.write_text("".join(f"{x:.4f} {H:.4f}\n" for x in xs), encoding="utf-8")


def write_vcorr(path: Path, *, lh: float = LH, lv: float = LV) -> None:
    """2×2 CorrelationLength2d：沿 x 不变，顶底相关长度相同。"""
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


def _fmt_s_line(x: float, z: float, npick: int) -> str:
    return f"s{x:10.3f}{z:10.3f}{npick:5d}"


def _fmt_r_line(x: float, z: float, kind: int, t: float, u: float) -> str:
    return f"r{x:10.3f}{z:10.3f}{kind:5d}{t:10.3f}{u:10.3f}"


def write_geom(path: Path) -> None:
    nrcv = len(SHOT_XS) * 2
    lines = [str(len(OBS_XS))]
    for ox in OBS_XS:
        lines.append(_fmt_s_line(ox, H, nrcv))
        for x in SHOT_XS:
            lines.append(_fmt_r_line(x, SHOT_Z, 2, 0.0, 0.0))
        for x in SHOT_XS:
            lines.append(_fmt_r_line(x, SHOT_Z, 3, 0.0, 0.0))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_smesh(path: Path) -> tuple[list[float], list[float], list[list[float]]]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    nx, nz, _vw, _va = lines[0].split()
    xs = [float(x) for x in lines[1].split()]
    zs = [float(z) for z in lines[3].split()]
    vel: list[list[float]] = []
    for i in range(int(nx)):
        vel.append([float(v) for v in lines[4 + i].split()])
        if len(vel[-1]) != int(nz):
            raise ValueError(f"{path}: column {i} has {len(vel[-1])} z values, expected {nz}")
    if len(xs) != int(nx) or len(zs) != int(nz):
        raise ValueError(f"{path}: nx/nz mismatch")
    return xs, zs, vel


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
    mean = sum(vals) / len(vals)
    vmin, vmax = min(vals), max(vals)
    return mean, vmin, vmax, len(vals)


def illum_x_range() -> tuple[float, float]:
    return min(SHOT_XS) - DX, max(SHOT_XS) + DX


def main() -> None:
    here = Path(__file__).resolve().parent
    write_smesh(here / "true.smesh", V_TRUE)
    write_smesh(here / "start.smesh", V_START)
    write_seafloor(here / "seafloor.refl")
    write_geom(here / "geom_inv.dat")
    write_vcorr(here / "vcorr.dat")
    print(f"wrote {here}")
    print(f"  true.smesh   topo=0  H={H} km  v_water={V_TRUE}  v_sed={V_SEDIMENT}")
    print(f"  start.smesh  v_water={V_START} (crust same 1.8)")
    print("  seafloor.refl  (-Y 反演 / 正演也可用 -B 或 -F)")
    print(
        f"  geom_inv.dat  OBS x={list(OBS_XS)} z={H}  "
        f"shots {min(SHOT_XS):g}..{max(SHOT_XS):g} step {SHOT_DX:g}  codes 2+3"
    )
    print(f"  vcorr.dat     Lh={LH:g} Lv={LV:g}  （-CV，配 -SV{WSV:g}）")
    print(f"  analytic  zero-offset t2={t_direct(0):.4f} t3={t_mult(0):.4f}")


if __name__ == "__main__":
    main()
