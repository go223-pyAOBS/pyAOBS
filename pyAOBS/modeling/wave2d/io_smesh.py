# -*- coding: utf-8 -*-
"""只读 smesh / 界面 / 走时文件。不写回 tomo2d 工区。"""

from __future__ import annotations

from pathlib import Path


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


def load_xz(path: Path) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        a = ln.split()
        if len(a) >= 2:
            out.append((float(a[0]), float(a[1])))
    return out


def interp_z(xz: list[tuple[float, float]], x: float) -> float:
    if not xz:
        raise ValueError("empty interface")
    if x <= xz[0][0]:
        return xz[0][1]
    if x >= xz[-1][0]:
        return xz[-1][1]
    for i in range(len(xz) - 1):
        x0, z0 = xz[i]
        x1, z1 = xz[i + 1]
        if x0 <= x <= x1:
            if abs(x1 - x0) < 1e-12:
                return z0
            return z0 + (z1 - z0) * (x - x0) / (x1 - x0)
    return xz[-1][1]


def parse_pickfile(path: Path) -> list[tuple[int, float, float, float, float]]:
    """tomo2d syn：每行 (code, rec_x, rec_z, t, src_x)。"""
    recs: list[tuple[int, float, float, float, float]] = []
    src_x = float("nan")
    for ln in path.read_text(encoding="utf-8").splitlines():
        a = ln.split()
        if not a:
            continue
        if a[0] == "s" and len(a) >= 3:
            src_x = float(a[1])
            continue
        if a[0] == "r" and len(a) >= 5:
            recs.append((int(float(a[3])), float(a[1]), float(a[2]), float(a[4]), src_x))
    return recs


def parse_rays(path: Path) -> list[tuple[list[float], list[float]]]:
    segs: list[tuple[list[float], list[float]]] = []
    xs: list[float] = []
    zs: list[float] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith(">"):
            if len(xs) >= 2:
                segs.append((xs, zs))
            xs, zs = [], []
            continue
        a = s.split()
        if len(a) >= 2:
            xs.append(float(a[0]))
            zs.append(float(a[1]))
    if len(xs) >= 2:
        segs.append((xs, zs))
    return segs
