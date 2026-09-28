#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""网格 I/O（不依赖 matplotlib，可供 WSL python3 调用）。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
import make_ppp_psp_inv_case as m2  # noqa: E402

H = 2.0
KAPPA = 1.73
ZC0, SLOPE, XREF = 4.0, 0.03, 50.0
DZ_IFACE = 0.2
ZMAX = 16.0
AMP1, LEN1 = 0.22, 12.0
AMP2, LEN2 = 0.10, 7.0
DAMP_LID, DAMP_BELOW = m2.DAMP_LID, m2.DAMP_BELOW
V_WATER, V_AIR = 1.5, 0.33


def z_conv(x: float) -> float:
    z = (
        ZC0
        + SLOPE * (x - XREF)
        + AMP1 * math.sin(2.0 * math.pi * (x - XREF) / LEN1)
        + AMP2 * math.sin(2.0 * math.pi * (x - 20.0) / LEN2)
    )
    k = int(round(z / DZ_IFACE))
    z = k * DZ_IFACE
    return max(H + DZ_IFACE, min(ZMAX - 2.0, z))


def write_mixed(cover: Path, below: Path, out: Path, kappa: float) -> None:
    xs, zs, cov = m2.parse_smesh(cover)
    xs2, zs2, bel = m2.parse_smesh(below)
    if xs2 != xs or zs2 != zs:
        raise SystemExit("mixed grid mismatch")
    lines = [
        f"{len(xs)} {len(zs)} {V_WATER:.4f} {V_AIR}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    for i, x in enumerate(xs):
        zi = z_conv(x)
        row = []
        for k, z in enumerate(zs):
            if z < zi - 1e-9:
                row.append(f"{cov[i][k]:.4f}")
            else:
                row.append(f"{(bel[i][k] / kappa):.4f}")
        lines.append(" ".join(row))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


HOT_DV = 0.50


def bump_below_depth(src: Path, dst: Path, dv: float, zmin_at) -> int:
    """z >= zmin_at(x) 的结点 +dv。"""
    xs, zs, vel = m2.parse_smesh(src)
    lines = [
        f"{len(xs)} {len(zs)} {V_WATER:.4f} {V_AIR}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    n = 0
    for i, x in enumerate(xs):
        z0 = zmin_at(x)
        row = []
        for k, z in enumerate(zs):
            v = vel[i][k]
            if z >= z0 - 1e-9:
                v = v + dv
                n += 1
            row.append(f"{v:.4f}")
        lines.append(" ".join(row))
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n


def bump_below_conv(src: Path, dst: Path, dv: float) -> int:
    """转换面及以下速度 +dv（水/盖层不动）。"""
    return bump_below_depth(src, dst, dv, z_conv)


def bump_below_seafloor(src: Path, dst: Path, dv: float) -> int:
    """海底以下（盖层+面下）+dv。PSS 台侧盖层是 S，盖层也要拉开。"""
    return bump_below_depth(src, dst, dv, lambda _x: H)


def bump_lid_only(src: Path, dst: Path, dv: float) -> int:
    """只加盖层（H < z < conv）。PPS 核只在台侧盖层。"""
    xs, zs, vel = m2.parse_smesh(src)
    lines = [
        f"{len(xs)} {len(zs)} {V_WATER:.4f} {V_AIR}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    n = 0
    for i, x in enumerate(xs):
        zi = z_conv(x)
        row = []
        for k, z in enumerate(zs):
            v = vel[i][k]
            if z > H + 1e-9 and z < zi - 1e-9:
                v = v + dv
                n += 1
            row.append(f"{v:.4f}")
        lines.append(" ".join(row))
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n


def graft_lid_vp(src_vp: Path, dst_vp: Path, out_path: Path | None = None) -> int:
    """把 src 转换面以上（盖层，不含转换面）拷到 dst，面下不动。"""
    xs, zs, src = m2.parse_smesh(src_vp)
    xs2, zs2, dst = m2.parse_smesh(dst_vp)
    if xs2 != xs or zs2 != zs:
        raise SystemExit("graft_lid_vp grid mismatch")
    out = out_path or dst_vp
    lines = [
        f"{len(xs)} {len(zs)} {V_WATER:.4f} {V_AIR}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    n = 0
    for i, x in enumerate(xs):
        zi = z_conv(x)
        row = []
        for k, z in enumerate(zs):
            if z < zi - 1e-9:
                v = src[i][k]
                if z > H + 1e-9:
                    n += 1
            else:
                v = dst[i][k]
            row.append(f"{v:.4f}")
        lines.append(" ".join(row))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n


def write_vs_from_vp(vp_path: Path, out_path: Path, kappa: float) -> None:
    """整网 Vs=Vp/κ（水保持水速）。-k 初值与 -U 用。"""
    xs, zs, vp = m2.parse_smesh(vp_path)
    lines = [
        f"{len(xs)} {len(zs)} {V_WATER:.4f} {V_AIR}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    for col in vp:
        row = [
            f"{(V_WATER if z <= H + 1e-9 else v / kappa):.4f}"
            for z, v in zip(zs, col)
        ]
        lines.append(" ".join(row))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_psp_speed(vp_path: Path, vs_path: Path, out_path: Path) -> None:
    """作图用：转换面以上 Vp，面上及面下 Vs（Vs 网格不再除 κ）。"""
    xs, zs, vp = m2.parse_smesh(vp_path)
    xs2, zs2, vs = m2.parse_smesh(vs_path)
    if xs2 != xs or zs2 != zs:
        raise SystemExit("psp-speed grid mismatch")
    lines = [
        f"{len(xs)} {len(zs)} {V_WATER:.4f} {V_AIR}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    for i, x in enumerate(xs):
        zi = z_conv(x)
        row = []
        for k, z in enumerate(zs):
            if z < zi - 1e-9:
                row.append(f"{vp[i][k]:.4f}")
            else:
                row.append(f"{vs[i][k]:.4f}")
        lines.append(" ".join(row))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latest(glob_pat: str, folder: Path | None = None) -> Path:
    root = folder or Path(".")
    cands = sorted(root.glob(glob_pat))
    if not cands:
        raise SystemExit(f"no {glob_pat}")
    return max(cands, key=lambda p: (int(p.name.split(".")[-2]), int(p.name.split(".")[-1])))
