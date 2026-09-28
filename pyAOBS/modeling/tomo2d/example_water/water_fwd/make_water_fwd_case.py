#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 tt_forward 水波正演工区：topo=0、均匀水 1.5 + 水柱随机扰动、平底海底。

平坦常速 v=1.5 解析解只作对照:
  直达 2:  t = hypot(dx, H) / v
  多次 3:  t = hypot(dx, 3H) / v
水柱扰动后正演不应贴这条线；用来看射线是否仍贴海面/海底。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

V_WATER = 1.5
V_SEDIMENT = 1.8  # 海底以下浅沉积
V_AIR = 0.33
H = 2.0
XMIN, XMAX, DX = 0.0, 100.0, 2.0
ZMAX, DZ = 4.0, 0.1
OBS_X = 50.0
SHOT_Z = 0.01
# 水柱相关随机扰动：相对 1.5 的 RMS 百分数；沉积不扰动。seed 固定以便复现。
NOISE_AMP = 0.05
NOISE_SEED = 42
NOISE_LX = 6.0  # 水平相关长度 km（高斯 σ）
NOISE_LZ = 0.40  # 垂向相关长度 km
V_WATER_MIN = 1.35
V_WATER_MAX = 1.65
# 相对台的偏移（km）。0–24 每 1 km。
OFFSETS = tuple(float(i) for i in range(0, 25))


def v_at(x: float, z: float) -> float:
    """背景结点速度：水柱均匀 1.5；海底以下为沉积。扰动在 write_smesh 里叠加。"""
    _ = x
    if z > H + 1e-9:
        return V_SEDIMENT
    return V_WATER


def _gaussian_smooth(arr: np.ndarray, sigma: tuple[float, float]) -> np.ndarray:
    """最近邻延拓的可分高斯平滑。``sigma`` 为各轴结点单位。"""
    out = np.asarray(arr, dtype=float)
    for axis, sig in enumerate(sigma):
        if sig <= 0:
            continue
        radius = max(1, int(math.ceil(3.0 * float(sig))))
        t = np.arange(-radius, radius + 1, dtype=float)
        ker = np.exp(-0.5 * (t / float(sig)) ** 2)
        ker /= ker.sum()
        pad = [(0, 0)] * out.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="edge")
        out = np.apply_along_axis(
            lambda v, k=ker: np.convolve(v, k, mode="valid"), axis, padded
        )
    return out


def water_rel_noise(
    xs: list[float],
    zs: list[float],
    *,
    amp: float = NOISE_AMP,
    seed: int = NOISE_SEED,
    lx: float = NOISE_LX,
    lz: float = NOISE_LZ,
) -> np.ndarray:
    """水柱相对扰动场 ``(nx, nz)``；海底以下为 0。RMS(|p|_water)=amp。"""
    nx, nz = len(xs), len(zs)
    water = np.array([[z <= H + 1e-9 for z in zs] for _ in xs], dtype=bool)
    if amp <= 0:
        return np.zeros((nx, nz), dtype=float)
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(0.0, 1.0, size=(nx, nz))
    noise = np.where(water, noise, 0.0)
    noise = _gaussian_smooth(noise, (lx / DX, lz / DZ))
    noise = np.where(water, noise, 0.0)
    rms = float(np.sqrt(np.mean(noise[water] ** 2))) or 1.0
    noise = noise / rms * float(amp)
    noise = np.where(water, noise - float(noise[water].mean()), 0.0)
    rms = float(np.sqrt(np.mean(noise[water] ** 2))) or 1.0
    return noise / rms * float(amp)


def t_direct(dx: float, h: float = H, v: float = V_WATER, z_shot: float = SHOT_Z) -> float:
    """炮在 z_shot≈0，台在 z=h。"""
    return math.hypot(dx, h - z_shot) / v


def t_mult(dx: float, h: float = H, v: float = V_WATER, z_shot: float = SHOT_Z) -> float:
    """炮↓海底↑海面↓台；均匀介质展开高度 3h - z_shot。"""
    return math.hypot(dx, 3.0 * h - z_shot) / v


def _xs() -> list[float]:
    n = int(round((XMAX - XMIN) / DX)) + 1
    return [XMIN + i * DX for i in range(n)]


def _zs() -> list[float]:
    n = int(round(ZMAX / DZ)) + 1
    return [i * DZ for i in range(n)]


def write_smesh(
    path: Path,
    *,
    noise_amp: float = NOISE_AMP,
    seed: int = NOISE_SEED,
) -> None:
    xs, zs = _xs(), _zs()
    nx, nz = len(xs), len(zs)
    pert = water_rel_noise(xs, zs, amp=noise_amp, seed=seed)
    lines = [
        f"{nx} {nz} {V_WATER} {V_AIR}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    water_vals: list[float] = []
    for i, x in enumerate(xs):
        col: list[str] = []
        for k, z in enumerate(zs):
            v = v_at(x, z)
            if z <= H + 1e-9:
                v = float(
                    np.clip(v * (1.0 + pert[i, k]), V_WATER_MIN, V_WATER_MAX)
                )
                water_vals.append(v)
            col.append(f"{v:.4f}")
        lines.append(" ".join(col))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if water_vals:
        arr = np.asarray(water_vals, dtype=float)
        write_smesh.last_water_stats = (  # type: ignore[attr-defined]
            float(arr.min()),
            float(arr.max()),
            float(arr.mean()),
            float(arr.std()),
        )


def write_seafloor(path: Path) -> None:
    xs = _xs()
    path.write_text("".join(f"{x:.4f} {H:.4f}\n" for x in xs), encoding="utf-8")


def _fmt_s_line(x: float, z: float, npick: int) -> str:
    return f"s{x:10.3f}{z:10.3f}{npick:5d}"


def _fmt_r_line(x: float, z: float, kind: int, t: float, u: float) -> str:
    return f"r{x:10.3f}{z:10.3f}{kind:5d}{t:10.3f}{u:10.3f}"


def write_geom(path: Path) -> None:
    shots = [OBS_X + dx for dx in OFFSETS]
    nrcv = len(shots) * 2
    lines = ["1", _fmt_s_line(OBS_X, H, nrcv)]
    for x in shots:
        lines.append(_fmt_r_line(x, SHOT_Z, 2, 0.0, 0.0))
    for x in shots:
        lines.append(_fmt_r_line(x, SHOT_Z, 3, 0.0, 0.0))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analytic(path: Path) -> None:
    rows = [
        "# dx_km  t2_s  t3_s  t3_minus_t2_s",
    ]
    for dx in OFFSETS:
        t2, t3 = t_direct(dx), t_mult(dx)
        rows.append(f"{dx:.3f}  {t2:.6f}  {t3:.6f}  {t3 - t2:.6f}")
    rows.append(f"# H={H} v={V_WATER}  zero-offset 2H/v={2 * H / V_WATER:.6f}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def compare_syn_to_analytic(
    syn_text: str, *, atol: float = 0.05
) -> list[tuple[int, float, float, float, float]]:
    """解析 tt_forward stdout（与 -G 同构），返回 (code, dx, t_syn, t_ana, err)。"""
    lines = [ln.strip() for ln in syn_text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("空的正演走时")
    recs: list[tuple[int, float, float, float, float]] = []
    i = 1
    src_x = OBS_X
    while i < len(lines):
        parts = lines[i].split()
        i += 1
        if not parts:
            continue
        if parts[0] == "s":
            src_x = float(parts[1])
            nrcv = int(float(parts[-1]))
            for _ in range(nrcv):
                rp = lines[i].split()
                i += 1
                x = float(rp[1])
                code = int(float(rp[3]))
                t = float(rp[4])
                dx = abs(x - src_x)
                ana = t_direct(dx) if code == 2 else t_mult(dx) if code == 3 else float("nan")
                recs.append((code, dx, t, ana, t - ana))
        else:
            continue
    bad = [r for r in recs if abs(r[4]) > atol]
    if bad:
        msg = "; ".join(
            f"code={c} dx={dx:g} syn={ts:.4f} ana={ta:.4f} d={e:.4f}"
            for c, dx, ts, ta, e in bad
        )
        raise AssertionError(f"正演与解析解偏差 > {atol} s: {msg}")
    return recs


def main() -> None:
    here = Path(__file__).resolve().parent
    write_smesh(here / "water.smesh")
    write_seafloor(here / "seafloor.refl")
    write_geom(here / "geom_water.dat")
    write_analytic(here / "analytic.txt")
    print(f"wrote {here}")
    stats = getattr(write_smesh, "last_water_stats", None)
    if stats:
        vmin, vmax, vmean, vstd = stats
        print(
            f"  water.smesh  topo=0  H={H} km  v0={V_WATER}  "
            f"noise RMS={NOISE_AMP:.0%} seed={NOISE_SEED}  "
            f"water v=[{vmin:.3f},{vmax:.3f}] mean={vmean:.3f} std={vstd:.3f}"
        )
    else:
        print(f"  water.smesh  topo=0  H={H} km  v_water={V_WATER}")
    print(f"  seafloor.refl  (-F)  v_sed={V_SEDIMENT}")
    print(f"  geom_water.dat  OBS x={OBS_X} z={H}  codes 2+3")
    print(f"  analytic.txt  zero-offset t2={t_direct(0):.4f} t3={t_mult(0):.4f}")


if __name__ == "__main__":
    main()
