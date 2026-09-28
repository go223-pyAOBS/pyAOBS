# -*- coding: utf-8 -*-
"""2D 弹性速度-应力 C-PML（左右+底；顶面仍自由表面）。

CFS-PML：ψ ← b ψ + a ∂φ ， ∂̃φ = ∂φ/κ + ψ（Komatitsch–Martin 形式）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _cpml_abk(
    n: int,
    nb: int,
    *,
    dt: float,
    vmax: float,
    dh: float,
    f0: float,
    left: bool,
    right: bool,
    npower: float = 2.0,
    kappa_max: float = 7.0,
    rcoef: float = 1.0e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """一维剖面长度 n：返回 (kappa, a, b)。非 PML 区 κ=1,a=0,b=0。"""
    kappa = np.ones(n, dtype=np.float64)
    a = np.zeros(n, dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    if nb <= 0 or (not left and not right):
        return kappa, a, b
    thickness = float(nb) * dh
    d0 = -(npower + 1.0) * vmax * np.log(rcoef) / (2.0 * thickness)
    alpha_max = np.pi * max(f0, 1e-6)

    def fill(indices: range, from_interior: bool) -> None:
        for k, i in enumerate(indices):
            abscissa = (k + 1) / float(nb) if from_interior else (nb - k) / float(nb)
            abscissa = min(max(abscissa, 0.0), 1.0)
            d = d0 * abscissa**npower
            kap = 1.0 + (kappa_max - 1.0) * abscissa**npower
            alpha = alpha_max * (1.0 - abscissa)
            c1 = d / kap + alpha
            bb = float(np.exp(-c1 * dt))
            denom = kap * (d + kap * alpha)
            aa = 0.0 if abs(denom) < 1e-30 else d * (bb - 1.0) / denom
            kappa[i] = kap
            a[i] = aa
            b[i] = bb

    if left:
        # i=0 为外缘（abscissa→1），i=nb-1 贴内域
        fill(range(0, min(nb, n)), from_interior=False)
    if right:
        # i=n-nb 贴内域，i=n-1 为外缘
        fill(range(max(0, n - nb), n), from_interior=True)
    return kappa, a, b


@dataclass
class CpmlState:
    k_x: np.ndarray
    a_x: np.ndarray
    b_x: np.ndarray
    k_x_h: np.ndarray
    a_x_h: np.ndarray
    b_x_h: np.ndarray
    k_z: np.ndarray
    a_z: np.ndarray
    b_z: np.ndarray
    k_z_h: np.ndarray
    a_z_h: np.ndarray
    b_z_h: np.ndarray
    mem_dvx_dx: np.ndarray
    mem_dvz_dz: np.ndarray
    mem_dvx_dz: np.ndarray
    mem_dvz_dx: np.ndarray
    mem_dtxx_dx: np.ndarray
    mem_dtxz_dz: np.ndarray
    mem_dtxz_dx: np.ndarray
    mem_dtzz_dz: np.ndarray


def build_cpml(
    nz: int,
    nx: int,
    *,
    nb: int,
    dt: float,
    dx: float,
    dz: float,
    vmax: float,
    f0: float,
) -> CpmlState:
    k_x, a_x, b_x = _cpml_abk(
        nx, nb, dt=dt, vmax=vmax, dh=dx, f0=f0, left=True, right=True
    )
    k_x_h, a_x_h, b_x_h = _cpml_abk(
        nx - 1, nb, dt=dt, vmax=vmax, dh=dx, f0=f0, left=True, right=True
    )
    # 顶自由表面：z 向只在底部（right=True 对应高下标）
    k_z, a_z, b_z = _cpml_abk(
        nz, nb, dt=dt, vmax=vmax, dh=dz, f0=f0, left=False, right=True
    )
    k_z_h, a_z_h, b_z_h = _cpml_abk(
        nz - 1, nb, dt=dt, vmax=vmax, dh=dz, f0=f0, left=False, right=True
    )
    return CpmlState(
        k_x=k_x,
        a_x=a_x,
        b_x=b_x,
        k_x_h=k_x_h,
        a_x_h=a_x_h,
        b_x_h=b_x_h,
        k_z=k_z,
        a_z=a_z,
        b_z=b_z,
        k_z_h=k_z_h,
        a_z_h=a_z_h,
        b_z_h=b_z_h,
        mem_dvx_dx=np.zeros((nz, nx), dtype=np.float64),
        mem_dvz_dz=np.zeros((nz, nx), dtype=np.float64),
        mem_dvx_dz=np.zeros((nz - 1, nx - 1), dtype=np.float64),
        mem_dvz_dx=np.zeros((nz - 1, nx - 1), dtype=np.float64),
        mem_dtxx_dx=np.zeros((nz, nx - 1), dtype=np.float64),
        mem_dtxz_dz=np.zeros((nz, nx - 1), dtype=np.float64),
        mem_dtxz_dx=np.zeros((nz - 1, nx), dtype=np.float64),
        mem_dtzz_dz=np.zeros((nz - 1, nx), dtype=np.float64),
    )


def _step_cpml_py(
    vx, vz, txx, tzz, txz, lam, lam2mu, muxz, bx, bz, pml: CpmlState, dx, dz, dt
) -> None:
    nz, nx = txx.shape
    for j in range(1, nz - 1):
        for i in range(1, nx - 1):
            dvx_dx = (vx[j, i] - vx[j, i - 1]) / dx
            dvz_dz = (vz[j, i] - vz[j - 1, i]) / dz
            pml.mem_dvx_dx[j, i] = pml.b_x[i] * pml.mem_dvx_dx[j, i] + pml.a_x[i] * dvx_dx
            dvx_dx = dvx_dx / pml.k_x[i] + pml.mem_dvx_dx[j, i]
            pml.mem_dvz_dz[j, i] = pml.b_z[j] * pml.mem_dvz_dz[j, i] + pml.a_z[j] * dvz_dz
            dvz_dz = dvz_dz / pml.k_z[j] + pml.mem_dvz_dz[j, i]
            txx[j, i] += dt * (lam2mu[j, i] * dvx_dx + lam[j, i] * dvz_dz)
            tzz[j, i] += dt * (lam2mu[j, i] * dvz_dz + lam[j, i] * dvx_dx)
    for j in range(nz - 1):
        for i in range(nx - 1):
            dvx_dz = (vx[j + 1, i] - vx[j, i]) / dz
            dvz_dx = (vz[j, i + 1] - vz[j, i]) / dx
            pml.mem_dvx_dz[j, i] = (
                pml.b_z_h[j] * pml.mem_dvx_dz[j, i] + pml.a_z_h[j] * dvx_dz
            )
            dvx_dz = dvx_dz / pml.k_z_h[j] + pml.mem_dvx_dz[j, i]
            pml.mem_dvz_dx[j, i] = (
                pml.b_x_h[i] * pml.mem_dvz_dx[j, i] + pml.a_x_h[i] * dvz_dx
            )
            dvz_dx = dvz_dx / pml.k_x_h[i] + pml.mem_dvz_dx[j, i]
            txz[j, i] += dt * muxz[j, i] * (dvx_dz + dvz_dx)
    tzz[0, :] = 0.0
    txz[0, :] = 0.0
    for j in range(nz):
        for i in range(nx - 1):
            dtxx_dx = (txx[j, i + 1] - txx[j, i]) / dx
            if j == 0:
                dtxz_dz = txz[0, i] / dz
            elif j >= nz - 1:
                dtxz_dz = 0.0
            else:
                dtxz_dz = (txz[j, i] - txz[j - 1, i]) / dz
            pml.mem_dtxx_dx[j, i] = (
                pml.b_x_h[i] * pml.mem_dtxx_dx[j, i] + pml.a_x_h[i] * dtxx_dx
            )
            dtxx_dx = dtxx_dx / pml.k_x_h[i] + pml.mem_dtxx_dx[j, i]
            pml.mem_dtxz_dz[j, i] = (
                pml.b_z[j] * pml.mem_dtxz_dz[j, i] + pml.a_z[j] * dtxz_dz
            )
            dtxz_dz = dtxz_dz / pml.k_z[j] + pml.mem_dtxz_dz[j, i]
            vx[j, i] += dt * bx[j, i] * (dtxx_dx + dtxz_dz)
    for j in range(nz - 1):
        for i in range(nx):
            if i == 0:
                dtxz_dx = txz[j, 0] / dx
            elif i >= nx - 1:
                dtxz_dx = 0.0
            else:
                dtxz_dx = (txz[j, i] - txz[j, i - 1]) / dx
            dtzz_dz = (tzz[j + 1, i] - tzz[j, i]) / dz
            pml.mem_dtxz_dx[j, i] = (
                pml.b_x[i] * pml.mem_dtxz_dx[j, i] + pml.a_x[i] * dtxz_dx
            )
            dtxz_dx = dtxz_dx / pml.k_x[i] + pml.mem_dtxz_dx[j, i]
            pml.mem_dtzz_dz[j, i] = (
                pml.b_z_h[j] * pml.mem_dtzz_dz[j, i] + pml.a_z_h[j] * dtzz_dz
            )
            dtzz_dz = dtzz_dz / pml.k_z_h[j] + pml.mem_dtzz_dz[j, i]
            vz[j, i] += dt * bz[j, i] * (dtxz_dx + dtzz_dz)
    tzz[0, :] = 0.0


if _HAS_NUMBA:

    @njit(cache=True)
    def _step_cpml_numba(
        vx, vz, txx, tzz, txz, lam, lam2mu, muxz, bx, bz,
        k_x, a_x, b_x, k_x_h, a_x_h, b_x_h,
        k_z, a_z, b_z, k_z_h, a_z_h, b_z_h,
        mem_dvx_dx, mem_dvz_dz, mem_dvx_dz, mem_dvz_dx,
        mem_dtxx_dx, mem_dtxz_dz, mem_dtxz_dx, mem_dtzz_dz,
        dx, dz, dt,
    ):
        nz, nx = txx.shape
        for j in range(1, nz - 1):
            for i in range(1, nx - 1):
                dvx_dx = (vx[j, i] - vx[j, i - 1]) / dx
                dvz_dz = (vz[j, i] - vz[j - 1, i]) / dz
                mem_dvx_dx[j, i] = b_x[i] * mem_dvx_dx[j, i] + a_x[i] * dvx_dx
                dvx_dx = dvx_dx / k_x[i] + mem_dvx_dx[j, i]
                mem_dvz_dz[j, i] = b_z[j] * mem_dvz_dz[j, i] + a_z[j] * dvz_dz
                dvz_dz = dvz_dz / k_z[j] + mem_dvz_dz[j, i]
                txx[j, i] += dt * (lam2mu[j, i] * dvx_dx + lam[j, i] * dvz_dz)
                tzz[j, i] += dt * (lam2mu[j, i] * dvz_dz + lam[j, i] * dvx_dx)
        for j in range(nz - 1):
            for i in range(nx - 1):
                dvx_dz = (vx[j + 1, i] - vx[j, i]) / dz
                dvz_dx = (vz[j, i + 1] - vz[j, i]) / dx
                mem_dvx_dz[j, i] = b_z_h[j] * mem_dvx_dz[j, i] + a_z_h[j] * dvx_dz
                dvx_dz = dvx_dz / k_z_h[j] + mem_dvx_dz[j, i]
                mem_dvz_dx[j, i] = b_x_h[i] * mem_dvz_dx[j, i] + a_x_h[i] * dvz_dx
                dvz_dx = dvz_dx / k_x_h[i] + mem_dvz_dx[j, i]
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
                mem_dtxx_dx[j, i] = b_x_h[i] * mem_dtxx_dx[j, i] + a_x_h[i] * dtxx_dx
                dtxx_dx = dtxx_dx / k_x_h[i] + mem_dtxx_dx[j, i]
                mem_dtxz_dz[j, i] = b_z[j] * mem_dtxz_dz[j, i] + a_z[j] * dtxz_dz
                dtxz_dz = dtxz_dz / k_z[j] + mem_dtxz_dz[j, i]
                vx[j, i] += dt * bx[j, i] * (dtxx_dx + dtxz_dz)
        for j in range(nz - 1):
            for i in range(nx):
                if i == 0:
                    dtxz_dx = txz[j, 0] / dx
                elif i >= nx - 1:
                    dtxz_dx = 0.0
                else:
                    dtxz_dx = (txz[j, i] - txz[j, i - 1]) / dx
                dtzz_dz = (tzz[j + 1, i] - tzz[j, i]) / dz
                mem_dtxz_dx[j, i] = b_x[i] * mem_dtxz_dx[j, i] + a_x[i] * dtxz_dx
                dtxz_dx = dtxz_dx / k_x[i] + mem_dtxz_dx[j, i]
                mem_dtzz_dz[j, i] = b_z_h[j] * mem_dtzz_dz[j, i] + a_z_h[j] * dtzz_dz
                dtzz_dz = dtzz_dz / k_z_h[j] + mem_dtzz_dz[j, i]
                vz[j, i] += dt * bz[j, i] * (dtxz_dx + dtzz_dz)
        for i in range(nx):
            tzz[0, i] = 0.0


def step_cpml(
    vx, vz, txx, tzz, txz, lam, lam2mu, muxz, bx, bz, pml: CpmlState, dx, dz, dt
) -> None:
    if _HAS_NUMBA:
        _step_cpml_numba(
            vx, vz, txx, tzz, txz, lam, lam2mu, muxz, bx, bz,
            pml.k_x, pml.a_x, pml.b_x, pml.k_x_h, pml.a_x_h, pml.b_x_h,
            pml.k_z, pml.a_z, pml.b_z, pml.k_z_h, pml.a_z_h, pml.b_z_h,
            pml.mem_dvx_dx, pml.mem_dvz_dz, pml.mem_dvx_dz, pml.mem_dvz_dx,
            pml.mem_dtxx_dx, pml.mem_dtxz_dz, pml.mem_dtxz_dx, pml.mem_dtzz_dz,
            dx, dz, dt,
        )
    else:
        _step_cpml_py(
            vx, vz, txx, tzz, txz, lam, lam2mu, muxz, bx, bz, pml, dx, dz, dt
        )
