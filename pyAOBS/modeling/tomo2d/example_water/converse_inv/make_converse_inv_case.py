#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 tt_inverse 折合 PSP 验收工区：冻盖层、只反转换面以下 Vs。

真模型 Vs0=3.20；初值 Vs0=3.55；梯度相同。水与盖层 Vp 两边一样。
炮在海面、OBS 在海底；只用 raytype 6。正演造观测后再 ``tt_inverse -B``。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "converse_fwd"))
import make_converse_fwd_case as fwd  # noqa: E402

V_WATER = fwd.V_WATER
V_P_CRUST = fwd.V_P_CRUST
VP_SED0 = fwd.VP_SED0
expected_lid_mean = fwd.expected_lid_mean
vp_sed = fwd.vp_sed
VS_TRUE = 3.20
VS_START = 3.55
VS_GRAD = fwd.VS_GRAD
# 双场盖层 Vs 占位（PSP 盖层走 -M 真 Vp）；面下才是 Vs0+g。
KAPPA_LID = 1.73
H = fwd.H
Z_CONV = fwd.Z_CONV
XMIN, XMAX, DX = fwd.XMIN, fwd.XMAX, fwd.DX
ZMAX, DZ = fwd.ZMAX, fwd.DZ
OBS_X = fwd.OBS_X
OBS_XS = (30.0, 40.0, 50.0, 60.0, 70.0)
OBS_Z = H
SHOT_Z = fwd.SHOT_Z
SHOT_DX = 2.0
SHOT_XS = tuple(round(x, 1) for x in [20.0 + i * SHOT_DX for i in range(31)])
LH, LV = 8.0, 0.5
WSV = 200.0
WTV = 20.0
PICK_U = 0.05


def write_smesh(path: Path, vs0: float) -> None:
    fwd.write_smesh(path, vs0=vs0, vs_grad=VS_GRAD)


def _vp_at(z: float, vs0: float) -> float:
    if z >= Z_CONV - 1e-9:
        return KAPPA_LID * (vs0 + VS_GRAD * max(0.0, z - Z_CONV))
    if z > H + 1e-9:
        return vp_sed(z)
    return V_WATER


def _vs_at(z: float, vs0: float) -> float:
    if z >= Z_CONV - 1e-9:
        return vs0 + VS_GRAD * max(0.0, z - Z_CONV)
    if z > H + 1e-9:
        return vp_sed(z) / KAPPA_LID
    return V_WATER


def _write_field(path: Path, vel_at) -> None:
    xs, zs = fwd._xs(), fwd._zs()
    lines = [
        f"{len(xs)} {len(zs)} {V_WATER:.4f} {fwd.V_AIR}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    for _x in xs:
        lines.append(" ".join(f"{vel_at(z):.4f}" for z in zs))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_vp(path: Path, vs0: float = VS_TRUE) -> None:
    """盖层真 Vp；面下只是占位（PSP 面下读 Vs）。"""
    _write_field(path, lambda z: _vp_at(z, vs0))


def write_vs(path: Path, vs0: float) -> None:
    """盖层 Vs 真/初相同（冻）；面下只错 Vs0。"""
    _write_field(path, lambda z: _vs_at(z, vs0))


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


def write_geom(path: Path) -> None:
    nrcv = len(SHOT_XS)
    lines = [str(len(OBS_XS))]
    for ox in OBS_XS:
        lines.append(fwd._fmt_s_line(ox, OBS_Z, nrcv))
        for x in SHOT_XS:
            lines.append(fwd._fmt_r_line(x, SHOT_Z, 6, 0.0, PICK_U))
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


def expected_vs_mean(vs0: float, z_lo: float, z_hi: float) -> float:
    """均匀 z 采样下线性梯度层的均值。"""
    return vs0 + VS_GRAD * 0.5 * ((z_lo - Z_CONV) + (z_hi - Z_CONV))


def main() -> None:
    here = Path(__file__).resolve().parent
    write_smesh(here / "true.smesh", VS_TRUE)
    write_smesh(here / "start.smesh", VS_START)
    write_vp(here / "true_vp.smesh", VS_TRUE)
    write_vs(here / "true_vs.smesh", VS_TRUE)
    write_vs(here / "start_vs.smesh", VS_START)
    write_seafloor(here / "seafloor.refl")
    write_conv(here / "conv.refl")
    write_geom(here / "geom_inv.dat")
    write_vcorr(here / "vcorr.dat")
    print(f"wrote {here}")
    print(
        f"  true.smesh   topo=0  H={H} Zc={Z_CONV}  "
        f"Vw={V_WATER} Vp_sed={VP_SED0}→{V_P_CRUST:.2f} Vs0={VS_TRUE} g={VS_GRAD}"
    )
    print(f"  start.smesh  Vs0={VS_START}（水/盖层与真值相同）")
    print(
        f"  true_vp/true_vs/start_vs  双场：盖层真 Vp，面下 Vs0={VS_TRUE}/{VS_START} 同梯度"
    )
    print("  conv.refl      （反演 -B / 正演 -X）")
    print("  seafloor.refl  （作图）")
    print(
        f"  geom_inv.dat  OBS x={list(OBS_XS)} z={OBS_Z}（海底）  "
        f"shots {min(SHOT_XS):g}..{max(SHOT_XS):g} step {SHOT_DX:g}  "
        f"z={SHOT_Z}（海面）  code 6"
    )
    print(f"  vcorr.dat     Lh={LH:g} Lv={LV:g}  （-CV，配 -SV{WSV:g} -TV{WTV:g}）")


if __name__ == "__main__":
    main()
