#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""台侧一阶 peg-leg 正演工区：topo=0、均匀水、壳内速度递增、莫霍速度跳跃。

几何与 water_fwd 相同：s 行是 OBS（海底），r 行是海面炮。
  raytype 0：折射
  raytype 4：折射的台侧一阶多次（OBS 侧先弹海面再进壳）
  raytype 1：壳内莫霍反射（PmP，-F=莫霍）
  raytype 5：莫霍反射的台侧一阶（水弹 + 莫霍反射，-B=海底、-F=莫霍）
对照：同一偏移上 t4 − t0 与 t5 − t1 大约 2H/v_water（镜像台）。
"""

from __future__ import annotations

import math
from pathlib import Path

V_WATER = 1.5
V_CRUST0 = 4.0  # 海底紧下方
V_GRAD = 0.35  # km/s per km
V_MANTLE = 8.0
V_AIR = 0.33
H = 2.0
H_MOHO = 8.0
XMIN, XMAX, DX = 0.0, 100.0, 2.0
ZMAX, DZ = 12.0, 0.2
OBS_X = 50.0
SHOT_Z = 0.01
OFFSETS = (10.0, 15.0, 20.0, 25.0, 30.0)


def v_at(x: float, z: float) -> float:
    _ = x
    if z > H_MOHO + 1e-9:
        return V_MANTLE
    if z > H + 1e-9:
        return V_CRUST0 + V_GRAD * (z - H)
    return V_WATER


def t_water_twt(h: float = H, v: float = V_WATER) -> float:
    """台侧一阶相对一次波的水柱双程（垂直近似）。"""
    return 2.0 * h / v


def v_moho_crust() -> float:
    return V_CRUST0 + V_GRAD * (H_MOHO - H)


def _eta(p: float, vel: float) -> float | None:
    a = 1.0 - (p * vel) ** 2
    if a < -1e-14:
        return None
    if a < 0.0:
        return 0.0
    return math.sqrt(a)


def water_dx_dt(p: float, thick: float, vw: float = V_WATER) -> tuple[float, float] | None:
    """常速水层：水平距离与走时。"""
    c = _eta(p, vw)
    if c is None or c < 1e-12:
        return None
    return thick * p * vw / c, thick / (vw * c)


def crust_dx_dt(p: float, z1: float, z2: float) -> tuple[float, float] | None:
    """线性壳速 v=v0+gζ，下行 ζ=z1→z2（海底起算）。"""
    if z2 < z1 - 1e-12:
        raise ValueError("crust_dx_dt expects z2>=z1")
    v1 = V_CRUST0 + V_GRAD * z1
    v2 = V_CRUST0 + V_GRAD * z2
    e1 = _eta(p, v1)
    e2 = _eta(p, v2)
    if e1 is None or e2 is None:
        return None
    dx = (e1 - e2) / (p * V_GRAD)
    dt = math.log((v2 / v1) * (1.0 + e1) / (1.0 + e2)) / V_GRAD
    return dx, dt


def _sample_p(pmin: float, pmax: float, n: int) -> list[float]:
    if n < 2 or pmax <= pmin:
        return []
    step = (pmax - pmin) / (n - 1)
    return [pmin + i * step for i in range(n)]


def _xt_primary_pmp(p: float) -> tuple[float, float] | None:
    """一次莫霍反射：炮侧水柱 + 壳内往返。OBS 在海底。"""
    w = water_dx_dt(p, H)
    c = crust_dx_dt(p, 0.0, H_MOHO - H)
    if w is None or c is None:
        return None
    return w[0] + 2.0 * c[0], w[1] + 2.0 * c[1]


def _xt_peg_pmp(p: float) -> tuple[float, float] | None:
    """莫霍反射台侧一阶：OBS 侧 2H 水弹 + 壳内往返 + 炮侧水柱。"""
    w1 = water_dx_dt(p, H)
    w2 = water_dx_dt(p, 2.0 * H)
    c = crust_dx_dt(p, 0.0, H_MOHO - H)
    if w1 is None or w2 is None or c is None:
        return None
    return w2[0] + 2.0 * c[0] + w1[0], w2[1] + 2.0 * c[1] + w1[1]


def _xt_primary_pg(p: float) -> tuple[float, float] | None:
    """一次折射：壳内回折（须在莫霍之上转折）+ 炮侧水柱。"""
    vt = 1.0 / p
    zt = (vt - V_CRUST0) / V_GRAD
    dmax = H_MOHO - H
    if zt <= 1e-6 or zt > dmax + 1e-9:
        return None
    c = crust_dx_dt(p, 0.0, zt)
    w = water_dx_dt(p, H)
    if c is None or w is None:
        return None
    return 2.0 * c[0] + w[0], 2.0 * c[1] + w[1]


def _xt_peg_pg(p: float) -> tuple[float, float] | None:
    """折射台侧一阶：OBS 侧 2H 水弹 + 壳内回折 + 炮侧水柱。"""
    vt = 1.0 / p
    zt = (vt - V_CRUST0) / V_GRAD
    dmax = H_MOHO - H
    if zt <= 1e-6 or zt > dmax + 1e-9:
        return None
    c = crust_dx_dt(p, 0.0, zt)
    w1 = water_dx_dt(p, H)
    w2 = water_dx_dt(p, 2.0 * H)
    if c is None or w1 is None or w2 is None:
        return None
    return 2.0 * c[0] + w2[0] + w1[0], 2.0 * c[1] + w2[1] + w1[1]


def _pn_x0_t0(peg: bool) -> tuple[float, float] | None:
    """地幔头波（Pn）临界点：p=1/v_mantle，L=0。"""
    p = 1.0 / V_MANTLE
    c = crust_dx_dt(p, 0.0, H_MOHO - H)
    w1 = water_dx_dt(p, H)
    if c is None or w1 is None:
        return None
    if peg:
        w2 = water_dx_dt(p, 2.0 * H)
        if w2 is None:
            return None
        return w2[0] + 2.0 * c[0] + w1[0], w2[1] + 2.0 * c[1] + w1[1]
    return w1[0] + 2.0 * c[0], w1[1] + 2.0 * c[1]


def _interp_sorted(xs: list[float], ts: list[float], dx: float) -> float | None:
    if len(xs) < 2 or dx < xs[0] - 1e-6 or dx > xs[-1] + 1e-6:
        return None
    lo, hi = 0, len(xs) - 1
    if dx <= xs[0]:
        return ts[0]
    if dx >= xs[-1]:
        return ts[-1]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= dx:
            lo = mid
        else:
            hi = mid
    x0, x1 = xs[lo], xs[hi]
    if abs(x1 - x0) < 1e-15:
        return ts[lo]
    r = (dx - x0) / (x1 - x0)
    return ts[lo] + r * (ts[hi] - ts[lo])


def _first_arrival_refr(peg: bool, n: int = 400, xmax: float = 40.0) -> tuple[list[float], list[float]]:
    """折射一次波/台侧多次：壳内回折与 Pn 头波的先到包络。"""
    p_moho = 1.0 / v_moho_crust()
    p_bot = 1.0 / V_CRUST0
    fn = _xt_peg_pg if peg else _xt_primary_pg
    pg_x: list[float] = []
    pg_t: list[float] = []
    for p in _sample_p(p_moho * (1.0 + 1e-6), p_bot * 0.999, n):
        xt = fn(p)
        if xt is None:
            continue
        pg_x.append(xt[0])
        pg_t.append(xt[1])
    order = sorted(range(len(pg_x)), key=lambda i: pg_x[i])
    pg_x = [pg_x[i] for i in order]
    pg_t = [pg_t[i] for i in order]
    pn0 = _pn_x0_t0(peg)
    xs: list[float] = []
    ts: list[float] = []
    xlo = pg_x[0] if pg_x else (pn0[0] if pn0 else 0.0)
    xhi = max(xmax, pg_x[-1] if pg_x else 0.0, pn0[0] if pn0 else 0.0)
    if xhi <= xlo:
        return pg_x, pg_t
    step = (xhi - xlo) / (n - 1)
    for i in range(n):
        x = xlo + i * step
        cands: list[float] = []
        tp = _interp_sorted(pg_x, pg_t, x)
        if tp is not None:
            cands.append(tp)
        if pn0 is not None and x + 1e-9 >= pn0[0]:
            cands.append(pn0[1] + (x - pn0[0]) / V_MANTLE)
        if cands:
            xs.append(x)
            ts.append(min(cands))
    return xs, ts


def analytic_curve(code: int, n: int = 400) -> tuple[list[float], list[float]]:
    """层状解析 T–X。0/1 一次波，4/5 台侧一阶（同一 p 的水弹）。"""
    p_moho = 1.0 / v_moho_crust()
    if code in (1, 5):
        fn = _xt_peg_pmp if code == 5 else _xt_primary_pmp
        ps = _sample_p(1e-4, p_moho * (1.0 - 1e-8), n)
        xs: list[float] = []
        ts: list[float] = []
        for p in ps:
            xt = fn(p)
            if xt is None:
                continue
            xs.append(xt[0])
            ts.append(xt[1])
        xt_c = fn(p_moho * (1.0 - 1e-12))
        if xt_c is not None:
            xs.append(xt_c[0])
            ts.append(xt_c[1])
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        return [xs[i] for i in order], [ts[i] for i in order]
    if code in (0, 4):
        return _first_arrival_refr(peg=(code == 4), n=n)
    raise ValueError(f"no analytic curve for raytype {code}")


def t_analytic(code: int, dx: float) -> float | None:
    """在解析曲线上按偏移插值；超出曲线存在区间则 None。"""
    xs, ts = analytic_curve(code)
    return _interp_sorted(xs, ts, dx)


def t_pmp_vertical() -> float:
    """零偏移 PmP：垂直水柱 + 壳内双程 ∫dζ/v。"""
    t_w = H / V_WATER
    t_c = math.log(v_moho_crust() / V_CRUST0) / V_GRAD
    return t_w + 2.0 * t_c


def _xs() -> list[float]:
    n = int(round((XMAX - XMIN) / DX)) + 1
    return [XMIN + i * DX for i in range(n)]


def _zs() -> list[float]:
    n = int(round(ZMAX / DZ)) + 1
    return [i * DZ for i in range(n)]


def write_smesh(path: Path) -> None:
    xs, zs = _xs(), _zs()
    nx, nz = len(xs), len(zs)
    lines = [
        f"{nx} {nz} {V_WATER} {V_AIR}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    for x in xs:
        lines.append(" ".join(f"{v_at(x, z):.4f}" for z in zs))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_seafloor(path: Path) -> None:
    xs = _xs()
    path.write_text("".join(f"{x:.4f} {H:.4f}\n" for x in xs), encoding="utf-8")


def write_moho(path: Path) -> None:
    xs = _xs()
    path.write_text("".join(f"{x:.4f} {H_MOHO:.4f}\n" for x in xs), encoding="utf-8")


def _fmt_s_line(x: float, z: float, npick: int) -> str:
    return f"s{x:10.3f}{z:10.3f}{npick:5d}"


def _fmt_r_line(x: float, z: float, kind: int, t: float, u: float) -> str:
    return f"r{x:10.3f}{z:10.3f}{kind:5d}{t:10.3f}{u:10.3f}"


def write_geom(path: Path) -> None:
    shots = [OBS_X + dx for dx in OFFSETS]
    codes = (0, 4, 1, 5)
    nrcv = len(shots) * len(codes)
    lines = ["1", _fmt_s_line(OBS_X, H, nrcv)]
    for kind in codes:
        for x in shots:
            lines.append(_fmt_r_line(x, SHOT_Z, kind, 0.0, 0.0))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    here = Path(__file__).resolve().parent
    write_smesh(here / "crust.smesh")
    write_seafloor(here / "seafloor.refl")
    write_moho(here / "moho.refl")
    write_geom(here / "geom_peg.dat")
    print(f"wrote {here}")
    print(
        f"  crust.smesh  topo=0  H={H}  Moho={H_MOHO}  "
        f"v_water={V_WATER}  crust={V_CRUST0}+{V_GRAD}*(z-H)  mantle={V_MANTLE}"
    )
    print("  seafloor.refl  (-B)")
    print("  moho.refl      (-F)")
    print(f"  geom_peg.dat  OBS x={OBS_X} z={H}  codes 0,4,1,5")
    print(f"  垂直近似 2H/v={t_water_twt():.4f} s")


if __name__ == "__main__":
    main()
