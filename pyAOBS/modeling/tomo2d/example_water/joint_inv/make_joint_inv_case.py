#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 tt_inverse 水+壳联合反演工区：topo=0，0/1+2/3 同时进核。

真模型水 1.45、沉积 1.8；初值水 1.55、沉积 2.0。
2/3 用海面炮；0/1 用海底炮（避免 code 0 图论初至走水柱）。
``tt_inverse -Y -F -u``（无 -y/-w）。
"""

from __future__ import annotations

from pathlib import Path

V_WATER_TRUE = 1.45
V_WATER_START = 1.55
V_SED_TRUE = 1.8
V_SED_START = 2.0
V_AIR = 0.33
H = 2.0
H_REFL = 3.2
SHOT_Z = 0.01
XMIN, XMAX, DX = 0.0, 100.0, 2.0
ZMAX, DZ = 4.0, 0.1
OBS_X = 50.0
OFFSETS = tuple(float(i) for i in range(-20, 21, 2))
LH, LV = 8.0, 0.5
WSV = 20.0


def v_at(z: float, v_water: float, v_sed: float) -> float:
    if z > H + 1e-9:
        return v_sed
    return v_water


def _xs() -> list[float]:
    n = int(round((XMAX - XMIN) / DX)) + 1
    return [XMIN + i * DX for i in range(n)]


def _zs() -> list[float]:
    n = int(round(ZMAX / DZ)) + 1
    return [i * DZ for i in range(n)]


def write_smesh(path: Path, v_water: float, v_sed: float) -> None:
    xs, zs = _xs(), _zs()
    nx, nz = len(xs), len(zs)
    lines = [
        f"{nx} {nz} {v_water:.4f} {V_AIR}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    for _x in xs:
        col = [f"{v_at(z, v_water, v_sed):.4f}" for z in zs]
        lines.append(" ".join(col))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_iface(path: Path, z: float) -> None:
    xs = _xs()
    path.write_text("".join(f"{x:.4f} {z:.4f}\n" for x in xs), encoding="utf-8")


def write_seafloor(path: Path) -> None:
    write_iface(path, H)


def write_basement(path: Path) -> None:
    write_iface(path, H_REFL)


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


def _fmt_s_line(x: float, z: float, npick: int) -> str:
    return f"s{x:10.3f}{z:10.3f}{npick:5d}"


def _fmt_r_line(x: float, z: float, kind: int, t: float, u: float) -> str:
    return f"r{x:10.3f}{z:10.3f}{kind:5d}{t:10.3f}{u:10.3f}"


def write_geom(path: Path) -> None:
    shots = [OBS_X + dx for dx in OFFSETS]
    nrcv = len(shots) * 4
    lines = ["1", _fmt_s_line(OBS_X, H, nrcv)]
    for x in shots:
        lines.append(_fmt_r_line(x, H, 0, 0.0, 0.0))
    for x in shots:
        lines.append(_fmt_r_line(x, H, 1, 0.0, 0.0))
    for x in shots:
        lines.append(_fmt_r_line(x, SHOT_Z, 2, 0.0, 0.0))
    for x in shots:
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
    strict: bool = False,
) -> tuple[float, float, float, int]:
    vals: list[float] = []
    for i, x in enumerate(xs):
        if x < x_lo - 1e-9 or x > x_hi + 1e-9:
            continue
        for k, z in enumerate(zs):
            if water:
                is_sel = z < H - 1e-9 if strict else z <= H + 1e-9
            else:
                is_sel = z > H + 1e-9
            if not is_sel:
                continue
            vals.append(vel[i][k])
    if not vals:
        raise ValueError("no nodes in requested window")
    mean = sum(vals) / len(vals)
    return mean, min(vals), max(vals), len(vals)


def illum_x_range() -> tuple[float, float]:
    return OBS_X + min(OFFSETS) - DX, OBS_X + max(OFFSETS) + DX


def main() -> None:
    here = Path(__file__).resolve().parent
    write_smesh(here / "true.smesh", V_WATER_TRUE, V_SED_TRUE)
    write_smesh(here / "start.smesh", V_WATER_START, V_SED_START)
    write_seafloor(here / "seafloor.refl")
    write_basement(here / "basement.refl")
    write_geom(here / "geom_inv.dat")
    write_vcorr(here / "vcorr.dat")
    print(f"wrote {here}")
    print(f"  true.smesh    水 {V_WATER_TRUE}  沉积 {V_SED_TRUE}")
    print(f"  start.smesh   水 {V_WATER_START}  沉积 {V_SED_START}")
    print(f"  seafloor.refl z={H}  （正演 -B / 反演 -Y）")
    print(f"  basement.refl z={H_REFL}  （-F，code 1；反演加 -u 冻界面）")
    print(
        f"  geom_inv.dat  OBS z={H}  0/1 海底炮  2/3 海面炮  "
        f"offsets {min(OFFSETS):g}..{max(OFFSETS):g}"
    )
    print(f"  vcorr.dat     Lh={LH:g} Lv={LV:g}  （-CV，配 -SV{WSV:g}）")


if __name__ == "__main__":
    main()
