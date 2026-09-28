#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""倾斜层二维运动学射击：v=v(x,z) 时 p_x 不守恒。

射线方程（走时 T 为参数，z 向下为正）：
  dx/dT = v² p_x,  dz/dT = v² p_z,  dp/dT = −(∇v)/v,  |p|=1/v。
转换面用切向慢度连续（广义 Snell），不假设整条射线共 p。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

H = 2.0
ZMAX = 16.0
V_WATER = 1.5
KAPPA = 1.73
SED0 = 1.80
SED_GRAD = (4.00 - 1.80) / 3.0
SHOT_Z = 0.01
XMIN, XMAX = 0.0, 125.0

WAT, LID, BASE = 0, 1, 2


def _region(z: float, zc: float) -> int:
    if z < H - 1e-9:
        return WAT
    if z < zc - 1e-9:
        return LID
    return BASE


def _norm2(x: float, z: float) -> tuple[float, float]:
    n = math.hypot(x, z)
    return (x / n, z / n) if n > 1e-15 else (1.0, 0.0)


@dataclass
class Ray2dSpec:
    """转换面为折线（与 conv.refl 同一套结点），不假设共 p、也不假设单斜面。"""

    crust_vp: float
    vs0: float
    x_ifc: np.ndarray
    z_ifc: np.ndarray
    _dx: float = 0.0
    _x0: float = 0.0
    _nseg: int = 0
    _slopes: np.ndarray | None = None

    def __post_init__(self) -> None:
        xs = np.asarray(self.x_ifc, float)
        zs = np.asarray(self.z_ifc, float)
        self.x_ifc = xs
        self.z_ifc = zs
        nseg = max(len(xs) - 1, 1)
        self._nseg = nseg
        self._x0 = float(xs[0])
        self._dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
        sl = np.empty(nseg)
        for i in range(nseg):
            dxi = xs[i + 1] - xs[i]
            sl[i] = (zs[i + 1] - zs[i]) / dxi if abs(dxi) > 1e-12 else 0.0
        self._slopes = sl

    @classmethod
    def from_xy(cls, crust_vp: float, vs0: float, xs, zs) -> "Ray2dSpec":
        return cls(crust_vp, vs0, np.asarray(xs, float), np.asarray(zs, float))

    @classmethod
    def from_plane(cls, crust_vp: float, vs0: float, zc0: float, slope: float,
                   xref: float = 50.0) -> "Ray2dSpec":
        xs = np.linspace(XMIN, XMAX, 126)
        return cls.from_xy(crust_vp, vs0, xs, zc0 + slope * (xs - xref))

    def _seg(self, x: float) -> int:
        i = int((x - self._x0) / self._dx)
        if i < 0:
            return 0
        if i >= self._nseg:
            return self._nseg - 1
        return i

    def zc(self, x: float) -> float:
        i = self._seg(x)
        x0 = self.x_ifc[i]
        return float(self.z_ifc[i] + self._slopes[i] * (x - x0))

    def slope_at(self, x: float) -> float:
        return float(self._slopes[self._seg(x)])

    def vel(self, x: float, z: float, is_s: bool) -> tuple[float, float, float]:
        """返回 v, ∂v/∂x, ∂v/∂z。∇v 用当地折线斜率。"""
        zc = self.zc(x)
        sl = self.slope_at(x)
        if z < H:
            return V_WATER, 0.0, 0.0
        if z < zc:
            g = SED_GRAD / KAPPA if is_s else SED_GRAD
            v0 = SED0 / KAPPA if is_s else SED0
            return v0 + g * (z - H), 0.0, g
        if is_s:
            return self.vs0 + 0.12 * (z - zc), -0.12 * sl, 0.12
        g = KAPPA * 0.12
        return self.crust_vp + g * (z - zc), -g * sl, g


def _is_s_leg(code: int, reg: int, going_up: bool) -> bool:
    if code == 0 or reg == WAT:
        return False
    if reg == BASE:
        return code in (6, 8)
    return code in (7, 8) and going_up


def _snell(px: float, pz: float, tau: tuple[float, float], n_into: tuple[float, float], v2: float):
    pt = px * tau[0] + pz * tau[1]
    if abs(pt) * v2 >= 0.999:
        return None
    pn = math.sqrt(max(1.0 / (v2 * v2) - pt * pt, 0.0))
    qx = pt * tau[0] + pn * n_into[0]
    qz = pt * tau[1] + pn * n_into[1]
    return qx, qz


def _iface_frame(slope: float):
    tau = _norm2(1.0, slope)
    n_down = _norm2(-slope, 1.0)
    n_up = (-n_down[0], -n_down[1])
    return tau, n_up, n_down


def alpha_from_p(spec: Ray2dSpec, sx: float, p: float, sign: float) -> float:
    v, _, _ = spec.vel(sx, SHOT_Z, False)
    px = sign * min(abs(p), 0.999 / v)
    pz = math.sqrt(max(1.0 / (v * v) - px * px, 1e-12))
    return math.atan2(px, pz)


def trace_takeoff(
    spec: Ray2dSpec,
    sx: float,
    alpha: float,
    code: int,
    dt: float = 0.005,
    record: bool = False,
    why: list | None = None,
):
    """从炮点以出射角 alpha（相对向下 +z，朝 +x 为正）积分，直到上行过海底。"""
    vsrc, _, _ = spec.vel(sx, SHOT_Z, False)
    px = math.sin(alpha) / vsrc
    pz = math.cos(alpha) / vsrc
    px0 = px
    x, z, t = sx, SHOT_Z, 0.0
    xs = [x] if record else None
    zs = [z] if record else None
    zmax = z
    nkeep = 0
    tau_sf, n_up_sf, n_dn_sf = _iface_frame(0.0)
    zc_fun = spec.zc

    for _ in range(4000):
        zc_x = zc_fun(x)
        r0 = _region(z, zc_x)
        going_up = pz < 0
        is_s = _is_s_leg(code, r0, going_up)
        v, vx, vz = spec.vel(x, z, is_s)
        if v < 0.2:
            if why is not None:
                why.append(f"v={v:.3f}@{x:.2f},{z:.2f}")
            return None
        pnrm = math.hypot(px, pz)
        want = 1.0 / v
        if pnrm > 1e-12:
            s = want / pnrm
            px *= s
            pz *= s
        dx = v * v * px
        dz = v * v * pz
        dpx = -vx / v
        dpz = -vz / v
        # Heun RK2
        xh = x + dt * dx
        zh = z + dt * dz
        pxh = px + dt * dpx
        pzh = pz + dt * dpz
        vh, vxh, vzh = spec.vel(xh, zh, is_s)
        if vh < 0.2:
            if why is not None:
                why.append("vh")
            return None
        pn = math.hypot(pxh, pzh)
        if pn > 1e-12:
            s = (1.0 / vh) / pn
            pxh *= s
            pzh *= s
        dx2 = vh * vh * pxh
        dz2 = vh * vh * pzh
        x1 = x + 0.5 * dt * (dx + dx2)
        z1 = z + 0.5 * dt * (dz + dz2)
        px1 = px + 0.5 * dt * (dpx + (-vxh / vh))
        pz1 = pz + 0.5 * dt * (dpz + (-vzh / vh))
        if x1 < XMIN + 0.2 or x1 > XMAX - 0.2 or z1 > ZMAX - 0.05 or z1 < -0.05:
            if why is not None:
                why.append(f"bound x={x1:.2f} z={z1:.2f}")
            return None
        if pz > 0 and z > ZMAX - 1.5:
            if why is not None:
                why.append(f"no_turn z={z:.2f}")
            return None

        zc1 = zc_fun(x1)
        r1 = _region(z1, zc1)

        if r0 != r1:
            frac = 0.5
            if r0 == WAT or r1 == WAT:
                if abs(z1 - z) > 1e-12:
                    frac = (H - z) / (z1 - z)
            else:
                f0, f1 = z - zc_x, z1 - zc1
                if abs(f1 - f0) > 1e-12:
                    frac = -f0 / (f1 - f0)
            frac = min(1.0, max(0.0, frac))
            xc = x + frac * (x1 - x)
            zc = z + frac * (z1 - z)
            tc = t + frac * dt
            pxc = px + frac * (px1 - px)
            pzc = pz + frac * (pz1 - pz)
            zmax = max(zmax, zc)
            if record:
                xs.append(xc)
                zs.append(zc)

            pair = {r0, r1}
            if pair == {WAT, LID}:
                if going_up and r0 == LID:
                    return {
                        "xs": xs, "zs": zs, "t": tc, "zmax": zmax,
                        "xhit": xc, "alpha": alpha, "px0": px0,
                    }
                v2 = spec.vel(xc, H + 1e-3, False)[0]
                n_into = n_dn_sf if r1 == LID else n_up_sf
                got = _snell(pxc, pzc, tau_sf, n_into, v2)
                if got is None:
                    if why is not None:
                        why.append(f"snell_sf r{r0}->{r1} x={xc:.2f}")
                    return None
                px, pz = got
                x, z, t = xc, zc + (1e-4 if r1 == LID else -1e-4), tc
                continue

            if pair == {LID, BASE}:
                down = r1 == BASE
                is_s2 = _is_s_leg(code, BASE if down else LID, not down)
                v2 = spec.vel(xc, zc_fun(xc) + (1e-3 if down else -1e-3), is_s2)[0]
                tau_c, n_up_c, n_dn_c = _iface_frame(spec.slope_at(xc))
                n_into = n_dn_c if down else n_up_c
                got = _snell(pxc, pzc, tau_c, n_into, v2)
                if got is None:
                    if why is not None:
                        why.append(
                            f"snell_c {'down' if down else 'up'} x={xc:.2f} v2={v2:.3f}"
                        )
                    return None
                px, pz = got
                x, z, t = xc, zc + (1e-4 if down else -1e-4), tc
                continue
            if why is not None:
                why.append(f"bad_pair {r0}->{r1} x={xc:.2f} z={zc:.2f}")
            return None

        x, z, t = x1, z1, t + dt
        px, pz = px1, pz1
        zmax = max(zmax, z)
        nkeep += 1
        if record and nkeep % 5 == 0:
            xs.append(x)
            zs.append(z)
        if going_up and z <= H + 1e-3 and r0 == LID:
            if record:
                xs.append(x)
                zs.append(H)
            return {
                "xs": xs, "zs": zs, "t": t, "zmax": zmax,
                "xhit": x, "alpha": alpha, "px0": px0,
            }
    if why is not None:
        why.append(f"maxstep zmax={zmax:.2f} x={x:.2f} z={z:.2f}")
    return None


def _turning(got, zc_need: float) -> bool:
    return got is not None and got["zmax"] >= zc_need


def _alpha_turn_min(spec: Ray2dSpec, sx: float, code: int) -> float:
    """能在 ZMAX 前转折的最小 |出射角|（水层）。"""
    is_s = code in (6, 8)
    vbot = spec.vel(sx, ZMAX - 0.05, is_s)[0]
    pmin = 1.0 / vbot + 2e-4
    vsrc = spec.vel(sx, SHOT_Z, False)[0]
    return math.asin(min(0.999, pmin * vsrc))


def shoot_pair(
    spec: Ray2dSpec,
    sx: float,
    ox: float,
    code: int,
    x_tol: float = 0.30,
    alpha0: float | None = None,
):
    """两点射击：在可转折出射角窗口内扫描，二分打到 OBS。"""
    if abs(ox - sx) < 1.0:
        return None
    zc_need = spec.zc(0.5 * (sx + ox)) + 0.25
    sign = 1.0 if ox >= sx else -1.0
    amin = _alpha_turn_min(spec, sx, code)
    a_lo = amin * 0.98
    a_hi = min(1.15, amin + 0.22)
    if sign > 0:
        alphas = np.linspace(a_lo, a_hi, 21)
    else:
        alphas = np.linspace(-a_hi, -a_lo, 21)
    if alpha0 is not None and a_lo <= abs(alpha0) <= a_hi:
        a_seed = abs(alpha0) * sign
        extra = np.linspace(a_seed - 0.04, a_seed + 0.04, 7)
        alphas = np.unique(np.concatenate([alphas, extra]))

    hits = []
    for a in alphas:
        got = trace_takeoff(spec, sx, float(a), code, record=False)
        if _turning(got, zc_need):
            hits.append(got)
    if not hits:
        return None

    hits.sort(key=lambda h: h["alpha"])
    best = None
    for h0, h1 in zip(hits, hits[1:]):
        x0, x1 = h0["xhit"], h1["xhit"]
        if (x0 - ox) * (x1 - ox) > 0:
            continue
        lo, hi = h0["alpha"], h1["alpha"]
        x_lo, x_hi = x0, x1
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            hm = trace_takeoff(spec, sx, mid, code, record=False)
            if not _turning(hm, zc_need):
                lo = mid
                continue
            if abs(hm["xhit"] - ox) < x_tol:
                best = hm
                break
            if (hm["xhit"] - ox) * (x_lo - ox) <= 0:
                hi, x_hi = mid, hm["xhit"]
            else:
                lo, x_lo = mid, hm["xhit"]
        else:
            hm = trace_takeoff(spec, sx, 0.5 * (lo + hi), code, record=False)
            if _turning(hm, zc_need):
                if best is None or abs(hm["xhit"] - ox) < abs(best["xhit"] - ox):
                    best = hm
        if best is not None and abs(best["xhit"] - ox) < x_tol:
            break

    if best is None:
        h = min(hits, key=lambda q: abs(q["xhit"] - ox))
        if abs(h["xhit"] - ox) < 1.2:
            best = h
    if best is None or abs(best["xhit"] - ox) > 1.2:
        return None

    traced = trace_takeoff(spec, sx, best["alpha"], code, record=True)
    if not _turning(traced, zc_need):
        return None
    traced["xs"].append(ox)
    traced["zs"].append(H)
    return traced
