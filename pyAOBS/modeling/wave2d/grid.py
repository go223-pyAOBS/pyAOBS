# -*- coding: utf-8 -*-
"""smesh → 规则网格。水柱强制 Vs=0（smesh 水里常误写成 1.5）。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .io_smesh import interp_z, load_xz, parse_smesh


@dataclass
class RegularModel:
    x: np.ndarray
    z: np.ndarray
    vp: np.ndarray
    vs: np.ndarray
    rho: np.ndarray
    water: np.ndarray
    dx: float
    dz: float


def _as_col_major(xs: list[float], zs: list[float], vel: list[list[float]]) -> np.ndarray:
    """smesh 是 [ix][iz] → (nz, nx)。"""
    arr = np.asarray(vel, dtype=np.float64)
    if arr.shape == (len(xs), len(zs)):
        return arr.T.copy()
    if arr.shape == (len(zs), len(xs)):
        return arr.copy()
    raise ValueError(f"smesh shape {arr.shape} != ({len(xs)},{len(zs)})")


def _gardner_rho(vp_kms: np.ndarray) -> np.ndarray:
    vp_ms = np.clip(vp_kms, 0.2, 20.0) * 1000.0
    return 310.0 * np.power(vp_ms, 0.25)


def resample_dual(
    vp_path: Path,
    vs_path: Path,
    seafloor_path: Path,
    *,
    dx: float,
    dz: float,
    xmax: float | None = None,
    zmax: float | None = None,
) -> RegularModel:
    xs, zs, vp_raw = parse_smesh(vp_path)
    xs2, zs2, vs_raw = parse_smesh(vs_path)
    if xs != xs2 or zs != zs2:
        raise ValueError("true_vp / true_vs grid mismatch")
    sea = load_xz(seafloor_path)
    vp_n = _as_col_major(xs, zs, vp_raw)
    vs_n = _as_col_major(xs, zs, vs_raw)
    x0, x1 = float(xs[0]), float(xs[-1] if xmax is None else min(xmax, xs[-1]))
    z0, z1 = float(zs[0]), float(zs[-1] if zmax is None else min(zmax, zs[-1]))
    x = np.arange(x0, x1 + 0.5 * dx, dx, dtype=np.float64)
    z = np.arange(z0, z1 + 0.5 * dz, dz, dtype=np.float64)
    pts = np.stack(np.meshgrid(z, x, indexing="ij"), axis=-1)
    # RegularGridInterpolator 要 (z, x) 与 values (nz,nx)
    kw = dict(bounds_error=False, fill_value=None)
    fvp = RegularGridInterpolator((np.asarray(zs), np.asarray(xs)), vp_n, **kw)
    fvs = RegularGridInterpolator((np.asarray(zs), np.asarray(xs)), vs_n, **kw)
    vp = fvp(pts)
    vs = fvs(pts)
    water = np.zeros(vp.shape, dtype=bool)
    for i, xv in enumerate(x):
        zb = interp_z(sea, float(xv))
        water[:, i] = z < zb - 0.25 * dz
    vp = np.where(water, 1.5, vp)
    vs = np.where(water, 0.0, vs)
    vs = np.minimum(vs, vp / np.sqrt(2.0))
    vs = np.maximum(vs, 0.0)
    rho = np.where(water, 1030.0, _gardner_rho(vp))
    return RegularModel(x=x, z=z, vp=vp, vs=vs, rho=rho, water=water, dx=dx, dz=dz)


def boost_conv_impedance(
    model: RegularModel,
    conv: list[tuple[float, float]],
    *,
    lid_vs_scale: float = 0.55,
    below_vs_scale: float = 1.25,
) -> RegularModel:
    """加强转换面剪切阻抗差：盖层 Vs 降低、面下 Vs 升高。不改水柱、不写 smesh。"""
    vs = np.array(model.vs, copy=True)
    for i, xv in enumerate(model.x):
        zc = interp_z(conv, float(xv))
        solid = ~model.water[:, i]
        above = solid & (model.z < zc - 0.5 * model.dz)
        below = solid & (model.z > zc + 0.5 * model.dz)
        vs[above, i] *= lid_vs_scale
        vs[below, i] *= below_vs_scale
    vs = np.minimum(vs, model.vp / np.sqrt(2.0))
    vs = np.maximum(vs, 0.0)
    vs = np.where(model.water, 0.0, vs)
    return replace(model, vs=vs)


def lame(vp: np.ndarray, vs: np.ndarray, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """λ, μ；单位：km、s、kg/m³ → 与 v(km/s) 搭配时 μ=ρ_g/cm³ vs²。

    内部用 SI：vp/vs 先变成 m/s，ρ 已是 kg/m³。
    """
    vp_m = vp * 1000.0
    vs_m = vs * 1000.0
    mu = rho * vs_m * vs_m
    lam = rho * vp_m * vp_m - 2.0 * mu
    lam = np.maximum(lam, 0.0)
    return lam, mu
