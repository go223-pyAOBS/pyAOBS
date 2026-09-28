#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""两步反演工区：① PPP 反 Vp；② 冻收回 Vp，PSP 双场（-M Vp / -U Vs）只反面下。

真 / 初 Vp 都是两段梯度：转换面以上沉积、以下壳幔。初值梯度故意偏真值。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ps_fwd"))
import make_ps_fwd_case as fwd  # noqa: E402

V_WATER = fwd.V_WATER
H = fwd.H
Z_CONV = fwd.Z_CONV
XMIN, XMAX, DX = fwd.XMIN, fwd.XMAX, fwd.DX
ZMAX, DZ = fwd.ZMAX, fwd.DZ
KAPPA_TRUE = 1.73
KAPPA_START = 2.00
# 初始 P：同样「沉积 / 壳幔」两段梯度，截距和斜率都偏真值。
VP_START_SED0 = 2.10
VP_START_SED_GRAD = 0.30
VP_START_CRUST0 = 5.70
VP_START_CRUST_GRAD = 0.10
VP_TRUE = dict(
    sed0=fwd.VP_SED0,
    sed_grad=fwd.VP_SED_GRAD,
    crust0=fwd.VP_CRUST0,
    crust_grad=fwd.VP_CRUST_GRAD,
)
VP_START = dict(
    sed0=VP_START_SED0,
    sed_grad=VP_START_SED_GRAD,
    crust0=VP_START_CRUST0,
    crust_grad=VP_START_CRUST_GRAD,
)
OBS_XS = (30.0, 40.0, 50.0, 60.0, 70.0)
OBS_Z = H
SHOT_Z = fwd.SHOT_Z
SHOT_DX = 2.0
SHOT_XS = tuple(round(x, 1) for x in [20.0 + i * SHOT_DX for i in range(31)])
LH, LV = 8.0, 2.0
WSV = 200.0
WTV = 5.0
PICK_U = 0.05
CODES_PPP = (0,)
CODES = (6,)


def vs_at(z: float, kappa: float, **vp_kw) -> float:
    vp = fwd.vp_at(z, **vp_kw)
    if z <= H + 1e-9:
        return V_WATER
    return vp / kappa


def write_vp(path: Path, **vp_kw) -> None:
    fwd.write_smesh(path, **vp_kw)


def write_vs(path: Path, kappa: float, **vp_kw) -> None:
    xs, zs = fwd._xs(), fwd._zs()
    lines = [
        f"{len(xs)} {len(zs)} {V_WATER:.4f} {fwd.V_AIR}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    for _x in xs:
        lines.append(" ".join(f"{vs_at(z, kappa, **vp_kw):.4f}" for z in zs))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_seafloor(path: Path) -> None:
    fwd.write_seafloor(path)


def write_conv(path: Path) -> None:
    fwd.write_conv(path)


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


def write_geom(path: Path, *, codes: tuple[int, ...] = CODES) -> None:
    nrcv = len(SHOT_XS) * len(codes)
    lines = [str(len(OBS_XS))]
    for ox in OBS_XS:
        lines.append(fwd._fmt_s_line(ox, OBS_Z, nrcv))
        for kind in codes:
            for x in SHOT_XS:
                lines.append(fwd._fmt_r_line(x, SHOT_Z, kind, 0.0, PICK_U))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_vs_from_smesh(vp_path: Path, out_path: Path, kappa: float) -> None:
    """由一份 Vp 网格按 κ 写 Vs（水保持水速）。第二步冷启动用收回的 Vp。"""
    xs, zs, vp = parse_smesh(vp_path)
    lines = [
        f"{len(xs)} {len(zs)} {V_WATER:.4f} {fwd.V_AIR}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    for col in vp:
        vs = []
        for z, v in zip(zs, col):
            vs.append(f"{(V_WATER if z <= H + 1e-9 else v / kappa):.4f}")
        lines.append(" ".join(vs))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    z_lo: float,
    z_hi: float,
    z_hi_inclusive: bool = False,
) -> tuple[float, float, float, int]:
    vals: list[float] = []
    for i, x in enumerate(xs):
        if x < x_lo - 1e-9 or x > x_hi + 1e-9:
            continue
        for k, z in enumerate(zs):
            if z < z_lo - 1e-9:
                continue
            if z_hi_inclusive:
                if z > z_hi + 1e-9:
                    continue
            elif z >= z_hi - 1e-9:
                continue
            vals.append(vel[i][k])
    if not vals:
        raise ValueError("no nodes in requested window")
    mean = sum(vals) / len(vals)
    return mean, min(vals), max(vals), len(vals)


def illum_x_range() -> tuple[float, float]:
    return min(SHOT_XS) - DX, max(SHOT_XS) + DX


def main() -> None:
    here = Path(__file__).resolve().parent
    write_vp(here / "true_vp.smesh", **VP_TRUE)
    write_vp(here / "start_vp.smesh", **VP_START)
    write_vp(here / "vp.smesh", **VP_START)
    write_vs(here / "true_vs.smesh", KAPPA_TRUE, **VP_TRUE)
    write_vs(here / "start_vs.smesh", KAPPA_START, **VP_START)
    write_seafloor(here / "seafloor.refl")
    write_conv(here / "conv.refl")
    write_geom(here / "geom_ppp.dat", codes=CODES_PPP)
    write_geom(here / "geom_inv.dat", codes=CODES)
    write_vcorr(here / "vcorr.dat")
    print(f"wrote {here}")
    print(
        f"  true_vp.smesh   真 Vp  沉积 {fwd.VP_SED0}+{fwd.VP_SED_GRAD}(z-H)  "
        f"壳幔 {fwd.VP_CRUST0}+{fwd.VP_CRUST_GRAD}(z-Zc)"
    )
    print(
        f"  start_vp.smesh  初 Vp  沉积 {VP_START_SED0}+{VP_START_SED_GRAD}(z-H)  "
        f"壳幔 {VP_START_CRUST0}+{VP_START_CRUST_GRAD}(z-Zc)  （第一步 -M）"
    )
    print(f"  vp.smesh        同 start_vp（兼容旧命令）")
    print(f"  true_vs.smesh   Vs=真Vp/{KAPPA_TRUE:g}")
    print(f"  start_vs.smesh  Vs=初Vp/{KAPPA_START:g}（第二步改写成收回Vp/κ）")
    print("  conv.refl / seafloor.refl")
    print(
        f"  geom_ppp.dat  OBS x={list(OBS_XS)} z={OBS_Z}  "
        f"shots {min(SHOT_XS):g}..{max(SHOT_XS):g} step {SHOT_DX:g}  codes {list(CODES_PPP)}"
    )
    print(
        f"  geom_inv.dat  同上  codes {list(CODES)}"
    )
    print(f"  vcorr.dat     Lh={LH:g} Lv={LV:g}  （-CV；PPP 配 -SV，Vs 步再加 -TV）")


if __name__ == "__main__":
    main()
