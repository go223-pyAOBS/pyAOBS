# -*- coding: utf-8 -*-
"""2D P-SV 交错网格（Virieux 二阶）。水柱 μ=0；顶自由表面；吸收 Cerjan 或 C-PML。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import RegularModel, lame
from .pml import build_cpml, step_cpml
from .ricker import ricker, ricker_delay

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def suggest_dt(vp: np.ndarray, dx: float, dz: float, cfl: float = 0.40) -> float:
    vmax = float(np.max(vp)) * 1000.0
    h = min(dx, dz) * 1000.0
    return cfl * h / (vmax * np.sqrt(2.0))


def cerjan_sponge(nz: int, nx: int, nb: int, alpha: float = 0.018) -> np.ndarray:
    w = np.ones((nz, nx), dtype=np.float64)
    if nb <= 0:
        return w
    t = np.arange(nb, dtype=np.float64)
    ramp = np.exp(-((alpha * (nb - t)) ** 2))
    w[:, :nb] *= ramp[None, :]
    w[:, nx - nb :] *= ramp[None, ::-1]
    w[nz - nb :, :] *= ramp[::-1, None]
    return w


def _mu_face(mu: np.ndarray) -> np.ndarray:
    """txz 点调和平均；任一邻点 μ=0 则 0。"""
    a = mu[:-1, :-1]
    b = mu[:-1, 1:]
    c = mu[1:, :-1]
    d = mu[1:, 1:]
    zero = (a <= 0.0) | (b <= 0.0) | (c <= 0.0) | (d <= 0.0)
    inv = 0.25 * (1.0 / np.maximum(a, 1e-30) + 1.0 / np.maximum(b, 1e-30)
                  + 1.0 / np.maximum(c, 1e-30) + 1.0 / np.maximum(d, 1e-30))
    out = np.where(zero, 0.0, 1.0 / inv)
    return out


def _buoy_avg(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bx = 2.0 / (rho[:, :-1] + rho[:, 1:])
    bz = 2.0 / (rho[:-1, :] + rho[1:, :])
    return bx, bz


@dataclass
class Gather:
    t: np.ndarray
    rec_x: np.ndarray
    data: np.ndarray
    src_x: float
    src_z: float
    delay: float
    dt: float
    f0: float


def _nearest(coord: np.ndarray, val: float) -> int:
    return int(np.argmin(np.abs(coord - val)))


def _step_numpy(
    vx, vz, txx, tzz, txz,
    lam, mu, lam2mu, muxz, bx, bz, sponge,
    dx, dz, dt,
):
    nz, nx = txx.shape
    dvx_dx = (vx[:, 1:] - vx[:, :-1]) / dx
    dvz_dz = (vz[1:, :] - vz[:-1, :]) / dz
    txx[1:-1, 1:-1] += dt * (
        lam2mu[1:-1, 1:-1] * dvx_dx[1:-1, :]
        + lam[1:-1, 1:-1] * dvz_dz[:, 1:-1]
    )
    tzz[1:-1, 1:-1] += dt * (
        lam2mu[1:-1, 1:-1] * dvz_dz[:, 1:-1]
        + lam[1:-1, 1:-1] * dvx_dx[1:-1, :]
    )
    dvx_dz = (vx[1:, :] - vx[:-1, :]) / dz
    dvz_dx = (vz[:, 1:] - vz[:, :-1]) / dx
    txz += dt * muxz * (dvx_dz + dvz_dx)
    tzz[0, :] = 0.0
    txz[0, :] = 0.0

    dtxx_dx = (txx[:, 1:] - txx[:, :-1]) / dx
    dtxz_dz = np.zeros_like(vx)
    dtxz_dz[1:-1, :] = (txz[1:, :] - txz[:-1, :]) / dz
    dtxz_dz[0, :] = txz[0, :] / dz
    vx += dt * bx * (dtxx_dx + dtxz_dz)

    dtxz_dx = np.zeros_like(vz)
    dtxz_dx[:, 1:-1] = (txz[:, 1:] - txz[:, :-1]) / dx
    dtxz_dx[:, 0] = txz[:, 0] / dx
    dtzz_dz = (tzz[1:, :] - tzz[:-1, :]) / dz
    vz += dt * bz * (dtxz_dx + dtzz_dz)

    vx *= sponge[:, :-1]
    vz *= sponge[:-1, :]
    txx *= sponge
    tzz *= sponge
    txz *= sponge[:-1, :-1]
    tzz[0, :] = 0.0


if _HAS_NUMBA:

    @njit(cache=True)
    def _step_numba(
        vx, vz, txx, tzz, txz,
        lam, mu, lam2mu, muxz, bx, bz, sponge,
        dx, dz, dt,
    ):
        nz, nx = txx.shape
        for j in range(1, nz - 1):
            for i in range(1, nx - 1):
                dvx_dx = (vx[j, i] - vx[j, i - 1]) / dx
                dvz_dz = (vz[j, i] - vz[j - 1, i]) / dz
                txx[j, i] += dt * (lam2mu[j, i] * dvx_dx + lam[j, i] * dvz_dz)
                tzz[j, i] += dt * (lam2mu[j, i] * dvz_dz + lam[j, i] * dvx_dx)
        for j in range(nz - 1):
            for i in range(nx - 1):
                dvx_dz = (vx[j + 1, i] - vx[j, i]) / dz
                dvz_dx = (vz[j, i + 1] - vz[j, i]) / dx
                txz[j, i] += dt * muxz[j, i] * (dvx_dz + dvz_dx)
        for i in range(nx):
            tzz[0, i] = 0.0
        for i in range(nx - 1):
            txz[0, i] = 0.0
        for j in range(nz):
            for i in range(nx - 1):
                dtxx_dx = (txx[j, i + 1] - txx[j, i]) / dx
                if j == 0:
                    dtxz_dz = txz[0, i] / dz
                elif j >= nz - 1:
                    dtxz_dz = 0.0
                else:
                    dtxz_dz = (txz[j, i] - txz[j - 1, i]) / dz
                vx[j, i] += dt * bx[j, i] * (dtxx_dx + dtxz_dz)
                vx[j, i] *= sponge[j, i]
        for j in range(nz - 1):
            for i in range(nx):
                if i == 0:
                    dtxz_dx = txz[j, 0] / dx
                elif i >= nx - 1:
                    dtxz_dx = 0.0
                else:
                    dtxz_dx = (txz[j, i] - txz[j, i - 1]) / dx
                dtzz_dz = (tzz[j + 1, i] - tzz[j, i]) / dz
                vz[j, i] += dt * bz[j, i] * (dtxz_dx + dtzz_dz)
                vz[j, i] *= sponge[j, i]
        for j in range(nz):
            for i in range(nx):
                txx[j, i] *= sponge[j, i]
                tzz[j, i] *= sponge[j, i]
        for j in range(nz - 1):
            for i in range(nx - 1):
                txz[j, i] *= sponge[j, i]
        for i in range(nx):
            tzz[0, i] = 0.0


def propagate(
    model: RegularModel,
    *,
    src_x: float,
    src_z: float,
    rec_x: np.ndarray,
    rec_z: float,
    tmax: float,
    f0: float,
    dt: float | None = None,
    src_kind: str = "vz",
    nb: int = 28,
    absorb: str = "pml",
) -> Gather:
    """互易：源在 OBS（竖力），水中炮点记压力。长度单位 km，时间 s。

    absorb: ``\"pml\"``（默认，左右+底 C-PML）或 ``\"cerjan\"``（海绵）。
    """
    dx_m = model.dx * 1000.0
    dz_m = model.dz * 1000.0
    if dt is None:
        dt = suggest_dt(model.vp, model.dx, model.dz)
    nt = int(np.ceil(tmax / dt)) + 1
    lam, mu = lame(model.vp, model.vs, model.rho)
    lam2mu = lam + 2.0 * mu
    muxz = _mu_face(mu)
    bx, bz = _buoy_avg(model.rho)
    abs_mode = str(absorb).lower().strip()
    if abs_mode not in ("pml", "cerjan"):
        raise ValueError(f"absorb must be 'pml' or 'cerjan', got {absorb!r}")
    sponge = (
        cerjan_sponge(model.vp.shape[0], model.vp.shape[1], nb)
        if abs_mode == "cerjan"
        else None
    )
    nz, nx = model.vp.shape
    pml = None
    if abs_mode == "pml":
        vmax = float(np.max(model.vp)) * 1000.0
        pml = build_cpml(
            nz, nx, nb=nb, dt=dt, dx=dx_m, dz=dz_m, vmax=vmax, f0=f0
        )
    vx = np.zeros((nz, nx - 1), dtype=np.float64)
    vz = np.zeros((nz - 1, nx), dtype=np.float64)
    txx = np.zeros((nz, nx), dtype=np.float64)
    tzz = np.zeros((nz, nx), dtype=np.float64)
    txz = np.zeros((nz - 1, nx - 1), dtype=np.float64)

    isrc = min(max(_nearest(model.x, src_x), 1), nx - 2)
    # 顶面为自由面（tzz[0]=0），源至少放在 j=1，仍可在浅水内
    jsrc = min(max(_nearest(model.z, src_z), 1), nz - 2)
    # 源/检避开 PML 条带（左右+底）
    if abs_mode == "pml" and nb > 0:
        isrc = min(max(isrc, nb + 1), nx - nb - 2)
        jsrc = min(max(jsrc, 1), nz - nb - 2)
    rec_i = [min(max(_nearest(model.x, float(x)), 1), nx - 2) for x in rec_x]
    if abs_mode == "pml" and nb > 0:
        rec_i = [min(max(i, nb + 1), nx - nb - 2) for i in rec_i]
    jrec = min(max(_nearest(model.z, rec_z), 1), nz - 2)
    if abs_mode == "pml" and nb > 0:
        jrec = min(jrec, nz - nb - 2)
    delay = ricker_delay(f0)
    traces = np.zeros((len(rec_x), nt), dtype=np.float64)
    step_cerjan = _step_numba if _HAS_NUMBA else _step_numpy
    amp = 1.0e12
    for it in range(nt):
        t = it * dt
        st = amp * ricker(t, f0, delay)
        if src_kind == "vz":
            vz[jsrc, isrc] += dt * bz[jsrc, isrc] * st / (dx_m * dz_m)
        elif src_kind == "vx":
            # 水平力。互易：水中压力 ≡ 水中炮、固体 OBS 的水平质点速度。
            iv = min(max(isrc, 1), vx.shape[1] - 2)
            jv = min(max(jsrc, 1), vx.shape[0] - 2)
            vx[jv, iv] += dt * bx[jv, iv] * st / (dx_m * dz_m)
        elif src_kind == "expl":
            txx[jsrc, isrc] += st
            tzz[jsrc, isrc] += st
        else:
            raise ValueError(src_kind)
        if abs_mode == "pml":
            assert pml is not None
            step_cpml(
                vx, vz, txx, tzz, txz, lam, lam2mu, muxz, bx, bz, pml, dx_m, dz_m, dt
            )
        else:
            assert sponge is not None
            step_cerjan(
                vx, vz, txx, tzz, txz, lam, mu, lam2mu, muxz, bx, bz, sponge, dx_m, dz_m, dt
            )
        for k, i in enumerate(rec_i):
            traces[k, it] = -0.5 * (txx[jrec, i] + tzz[jrec, i])
    t = np.arange(nt, dtype=np.float64) * dt
    return Gather(
        t=t,
        rec_x=np.asarray(rec_x, dtype=np.float64),
        data=traces,
        src_x=float(src_x),
        src_z=float(src_z),
        delay=delay,
        dt=dt,
        f0=f0,
    )
