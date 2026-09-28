#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PPP / PPS / PSS / PSP 正演工区。

几何与 converse_fwd 同一套。PSP 用 mixed.smesh。PPP 用真 Vp（界面是壳幔 Vp）。
PPS/PSS 用 vp_psx+vs 双场（界面结点划给 Vs，盖层底 P 插值与 converse 相同）。
PSS 与 PSP 同构两星三段。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "converse_fwd"))
from make_converse_fwd_case import (  # noqa: E402
    H,
    OBS_X,
    SHOT_Z,
    V_AIR,
    V_WATER,
    VP_SED0,
    VP_SED_GRAD,
    VS0,
    VS_GRAD,
    Z_CONV,
    v_at as mixed_at,
    vp_sed,
)

# 比 converse 更长的排列：最大偏移 60 km。网格加宽加深，避免远炮贴边、贴底。
XMIN, XMAX, DX = 0.0, 125.0, 2.0
ZMAX, DZ = 16.0, 0.2
OFFSETS = (
    0.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0,
    32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0,
)

KAPPA = 1.73
VP_CRUST0 = KAPPA * VS0
VP_CRUST_GRAD = KAPPA * VS_GRAD


def vp_at(
    z: float,
    *,
    sed0: float = VP_SED0,
    sed_grad: float = VP_SED_GRAD,
    crust0: float = VP_CRUST0,
    crust_grad: float = VP_CRUST_GRAD,
) -> float:
    """1D：水；盖层沉积 Vp（与 converse 相同）；面下 κ·Vs。"""
    if z >= Z_CONV - 1e-9:
        return crust0 + crust_grad * max(0.0, z - Z_CONV)
    if z > H + 1e-9:
        return sed0 + sed_grad * max(0.0, min(z, Z_CONV) - H)
    return V_WATER


def vs_fwd_at(z: float) -> float:
    """双场 Vs：盖层 Vp/κ；面下与 converse 相同。"""
    if z >= Z_CONV - 1e-9:
        return mixed_at(z)
    if z > H + 1e-9:
        return vp_sed(z) / KAPPA
    return V_WATER


def _xs() -> list[float]:
    n = int(round((XMAX - XMIN) / DX)) + 1
    return [XMIN + i * DX for i in range(n)]


def _zs() -> list[float]:
    n = int(round(ZMAX / DZ)) + 1
    return [i * DZ for i in range(n)]


def _write_grid(path: Path, vel_at) -> None:
    xs, zs = _xs(), _zs()
    lines = [
        f"{len(xs)} {len(zs)} {V_WATER:.4f} {V_AIR}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    for _x in xs:
        lines.append(" ".join(f"{vel_at(z):.4f}" for z in zs))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_vp_smesh(path: Path, *, converse_iface: bool = False, **vp_kw) -> None:
    """converse_iface：转换面结点写成 Vs0，盖层底 P 插值与 mixed/converse 相同。"""

    def vel(z: float) -> float:
        if converse_iface and abs(z - Z_CONV) <= 1e-9:
            return VS0
        return vp_at(z, **vp_kw)

    _write_grid(path, vel)


def write_smesh(path: Path, **vp_kw) -> None:
    """反演用真/初 Vp：界面也是壳幔 Vp。"""
    write_vp_smesh(path, converse_iface=False, **vp_kw)


def write_vs_smesh(path: Path) -> None:
    _write_grid(path, vs_fwd_at)


def write_mixed_smesh(path: Path) -> None:
    _write_grid(path, mixed_at)


def write_seafloor(path: Path) -> None:
    path.write_text("".join(f"{x:.4f} {H:.4f}\n" for x in _xs()), encoding="utf-8")


def write_conv(path: Path) -> None:
    path.write_text("".join(f"{x:.4f} {Z_CONV:.4f}\n" for x in _xs()), encoding="utf-8")


def _fmt_s_line(x: float, z: float, npick: int) -> str:
    return f"s{x:10.3f}{z:10.3f}{npick:5d}"


def _fmt_r_line(x: float, z: float, kind: int, t: float, u: float) -> str:
    return f"r{x:10.3f}{z:10.3f}{kind:5d}{t:10.3f}{u:10.3f}"


def write_geom(path: Path, *, codes: tuple[int, ...] = (0, 6, 7, 8)) -> None:
    shots = [OBS_X + dx for dx in OFFSETS]
    nrcv = len(shots) * len(codes)
    lines = ["1", _fmt_s_line(OBS_X, H, nrcv)]
    for kind in codes:
        for x in shots:
            lines.append(_fmt_r_line(x, SHOT_Z, kind, 0.0, 0.0))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _r_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip().startswith("r")]


def _ray_blocks(text: str) -> list[str]:
    blocks: list[list[str]] = []
    cur: list[str] = []
    for ln in text.splitlines():
        if ln.startswith(">"):
            if cur:
                blocks.append(cur)
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        blocks.append(cur)
    return ["\n".join(b) + "\n" for b in blocks]


def merge_fwd(here: Path | None = None) -> None:
    """初至0 + 转换面反射1 + PSP6 + PPS/PSS 合成 syn/rays。"""
    here = here or HERE
    n = len(OFFSETS)
    r0 = _r_lines(here / "syn_ppp.dat")
    r1 = _r_lines(here / "syn_ppr.dat")
    r6 = _r_lines(here / "syn_psp.dat")
    rx = _r_lines(here / "syn_psx.dat")
    if len(r0) != n or len(r1) != n or len(r6) != n or len(rx) != 2 * n:
        raise RuntimeError(
            f"merge expected {n} 0, {n} 1, {n} 6, {2*n} 7/8 picks, "
            f"got {len(r0)}, {len(r1)}, {len(r6)}, {len(rx)}"
        )
    r7, r8 = rx[:n], rx[n:]
    body = "\n".join(r0 + r1 + r6 + r7 + r8)
    (here / "syn_ps.dat").write_text(
        f"1\n{_fmt_s_line(OBS_X, H, 5 * n)}\n{body}\n", encoding="utf-8"
    )
    b0 = _ray_blocks((here / "rays_ppp.dat").read_text(encoding="utf-8"))
    b1 = _ray_blocks((here / "rays_ppr.dat").read_text(encoding="utf-8"))
    b6 = _ray_blocks((here / "rays_psp.dat").read_text(encoding="utf-8"))
    bx = _ray_blocks((here / "rays_psx.dat").read_text(encoding="utf-8"))
    if len(b0) != n or len(b1) != n or len(b6) != n or len(bx) != 2 * n:
        raise RuntimeError(
            f"merge expected {n} 0, {n} 1, {n} 6, {2*n} 7/8 rays, "
            f"got {len(b0)}, {len(b1)}, {len(b6)}, {len(bx)}"
        )
    (here / "rays_ps.dat").write_text("".join(b0 + b1 + b6 + bx[:n] + bx[n:]))
    print(f"merged {here / 'syn_ps.dat'}  and  {here / 'rays_ps.dat'}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--merge", action="store_true")
    args = p.parse_args(argv)
    here = Path(__file__).resolve().parent
    if args.merge:
        merge_fwd(here)
        return
    write_mixed_smesh(here / "mixed.smesh")
    write_vp_smesh(here / "vp.smesh", converse_iface=False)
    write_vp_smesh(here / "vp_psx.smesh", converse_iface=True)
    write_vs_smesh(here / "vs.smesh")
    write_seafloor(here / "seafloor.refl")
    write_conv(here / "conv.refl")
    write_geom(here / "geom_ps.dat", codes=(0, 1, 6, 7, 8))
    write_geom(here / "geom_ppp.dat", codes=(0,))
    write_geom(here / "geom_ppr.dat", codes=(1,))
    write_geom(here / "geom_psp.dat", codes=(6,))
    write_geom(here / "geom_psx.dat", codes=(7, 8))
    print(f"wrote {here}")
    print(
        f"  mixed.smesh  PSP 与 converse 同一套  盖层 Vp {VP_SED0}+{VP_SED_GRAD}(z-{H:g})  "
        f"面下 Vs {VS0}+{VS_GRAD}(z-{Z_CONV:g})"
    )
    print("  vp.smesh  PPP 真 Vp，界面是壳幔 Vp")
    print(
        f"  vp_psx.smesh  7/8 界面结点 Vs0={VS0:g}  面下 Vp=κ·Vs  κ={KAPPA:g}"
    )
    print(
        f"  vs.smesh  盖层 Vp/κ  面下 {VS0:g}+{VS_GRAD:g}(z-{Z_CONV:g})  （-U）"
    )
    print(
        f"  geom_ppp.dat  0    geom_ppr.dat  1（转换面反射）  "
        f"geom_psp.dat  6    geom_psx.dat  7/8  "
        f"offsets {list(OFFSETS)}  mesh x={XMIN:g}–{XMAX:g} z=0–{ZMAX:g}"
    )


if __name__ == "__main__":
    main()
