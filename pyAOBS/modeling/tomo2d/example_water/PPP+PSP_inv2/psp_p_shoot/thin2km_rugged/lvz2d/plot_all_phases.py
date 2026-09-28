#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全部震相：折合走时 + 沿射线左普利兹反射/透射能量积。"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inv_2d"))
sys.path.insert(0, str(ROOT.parents[1]))
sys.path.insert(0, str(ROOT.parents[1].parent / "ps_fwd"))
sys.path.insert(0, str(ROOT.parents[1].parent / "water_inv"))
import inv_grid as g  # noqa: E402
import make_ppp_psp_inv_case as m2  # noqa: E402
import plot_rugged_inv as pr  # noqa: E402
from check_joint import recs_for_draw  # noqa: E402
from check_ps_fwd import PHASE_TT_STYLE, split_psx_phase_segments  # noqa: E402
from check_water_inv import parse_picks  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

WORK = HERE / "fwd_all"
OBS_PLOT = 50.0
DX_MIN = 10.0  # 只画 / 统计 |Δx| > 10 km
VRED = 8.0
RHO_W = 1.03
VS_W = 0.02  # 水近似流体；过小会让 Zoeppritz 数值炸掉
H = g.H


def gardner(vp: float) -> float:
    return float(np.clip(1.74 * max(vp, 0.4) ** 0.25, 1.5, 3.4))


def aki_scattering(vp1, vs1, rho1, vp2, vs2, rho2, inc_deg, wave="P"):
    i1 = np.deg2rad(inc_deg)
    p = np.sin(i1) / (vp1 if wave == "P" else vs1)

    def cang(v, p):
        return np.arcsin(np.complex128(p * v))

    i1p, j1 = cang(vp1, p), cang(vs1, p)
    i2, j2 = cang(vp2, p), cang(vs2, p)
    a = rho2 * (1 - 2 * vs2**2 * p**2) - rho1 * (1 - 2 * vs1**2 * p**2)
    b = rho2 * (1 - 2 * vs2**2 * p**2) + 2 * rho1 * vs1**2 * p**2
    c = rho1 * (1 - 2 * vs1**2 * p**2) + 2 * rho2 * vs2**2 * p**2
    d = 2 * (rho2 * vs2**2 - rho1 * vs1**2)
    E = b * np.cos(i1p) / vp1 + c * np.cos(i2) / vp2
    F = b * np.cos(j1) / vs1 + c * np.cos(j2) / vs2
    G = a - d * np.cos(i1p) / vp1 * np.cos(j2) / vs2
    H = a - d * np.cos(i2) / vp2 * np.cos(j1) / vs1
    D = E * F + G * H * p**2
    Rpp = ((b * np.cos(i1p) / vp1 - c * np.cos(i2) / vp2) * F
           - (a + d * np.cos(i1p) / vp1 * np.cos(j2) / vs2) * H * p**2) / D
    Rps = -2 * np.cos(i1p) / vp1 * (a * b + c * d * np.cos(i2) / vp2 * np.cos(j2) / vs2) * p * vp1 / (vs1 * D)
    Tpp = 2 * rho1 * np.cos(i1p) / vp1 * F * vp1 / (vp2 * D)
    Tps = 2 * rho1 * np.cos(i1p) / vp1 * H * p * vp1 / (vs2 * D)
    Rss = -((b * np.cos(j1) / vs1 - c * np.cos(j2) / vs2) * E
            - (a + d * np.cos(i2) / vp2 * np.cos(j1) / vs1) * G * p**2) / D
    Rsp = -2 * np.cos(j1) / vs1 * (a * b + c * d * np.cos(i2) / vp2 * np.cos(j2) / vs2) * p * vs1 / (vp1 * D)
    Tsp = -2 * rho1 * np.cos(j1) / vs1 * G * p * vs1 / (vp2 * D)
    Tss = 2 * rho1 * np.cos(j1) / vs1 * E * vs1 / (vs2 * D)
    return dict(Rpp=Rpp, Rps=Rps, Tpp=Tpp, Tps=Tps, Rss=Rss, Rsp=Rsp, Tsp=Tsp, Tss=Tss,
                i1p=i1p, j1=j1, i2=i2, j2=j2)


def eflux(c, rho_o, v_o, ang, rho_i, v_i, ci):
    co = np.real(np.cos(ang))
    if ci <= 1e-12 or not np.isfinite(ci):
        return 0.0
    val = (rho_o * v_o * co) / (rho_i * v_i * ci) * (np.abs(c) ** 2)
    if not np.isfinite(val):
        return 0.0
    return float(np.clip(val, 0.0, 1.2))


def energy_coeff(kind: str, inc_s: bool, vp1, vs1, rho1, vp2, vs2, rho2, inc_deg: float) -> float:
    inc_deg = float(np.clip(inc_deg, 0.2, 75.0))
    wave = "S" if inc_s else "P"
    try:
        c = aki_scattering(vp1, vs1, rho1, vp2, vs2, rho2, inc_deg, wave)
    except Exception:
        return 0.0
    if inc_s:
        ci = np.real(np.cos(c["j1"]))
        key = {"Rss": ("Rss", rho1, vs1, c["j1"]), "Rsp": ("Rsp", rho1, vp1, c["i1p"]),
               "Tsp": ("Tsp", rho2, vp2, c["i2"]), "Tss": ("Tss", rho2, vs2, c["j2"])}[kind]
    else:
        ci = np.real(np.cos(c["i1p"]))
        key = {"Rpp": ("Rpp", rho1, vp1, c["i1p"]), "Rps": ("Rps", rho1, vs1, c["j1"]),
               "Tpp": ("Tpp", rho2, vp2, c["i2"]), "Tps": ("Tps", rho2, vs2, c["j2"])}[kind]
    name, rho_o, v_o, ang = key
    flux = eflux(c[name], rho_o, v_o, ang, rho1, vs1 if inc_s else vp1, ci)
    mag2 = float(np.clip(np.abs(c[name]) ** 2, 0.0, 1.0))
    if not np.isfinite(flux):
        flux = 0.0
    if flux < 1e-4:
        return mag2 if mag2 > 1e-6 else 0.0
    return flux


class Grids:
    def __init__(self, vp_p: Path, vs_p: Path, moho_p: Path):
        self.xs, zs, vp = m2.parse_smesh(vp_p)
        xs2, self.zs, vs = m2.parse_smesh(vs_p)
        self.xs = np.asarray(self.xs, dtype=float)
        self.zs = np.asarray(self.zs, dtype=float)
        self.vp = np.asarray(vp, dtype=float)
        self.vs = np.asarray(vs, dtype=float)
        mx, mz = np.loadtxt(moho_p).T
        self.mx, self.mz = np.asarray(mx), np.asarray(mz)

    def sample(self, x: float, z: float) -> tuple[float, float]:
        ix = int(np.argmin(np.abs(self.xs - x)))
        iz = int(np.argmin(np.abs(self.zs - z)))
        return float(self.vp[ix, iz]), float(self.vs[ix, iz])

    def z_moho(self, x: float) -> float:
        return float(np.interp(x, self.mx, self.mz))

    def pair(self, x: float, z_iface: float, dz: float = 0.2) -> tuple[float, float, float, float]:
        vp1, vs1 = self.sample(x, z_iface - dz)
        vp2, vs2 = self.sample(x, z_iface + dz)
        return vp1, vs1, vp2, vs2


def _fmt_jump(name, vp1, vs1, vp2, vs2) -> str:
    r1, r2 = gardner(vp1), gardner(vp2)
    return (f"{name}\n"
            f"Vp {vp1:.2f}→{vp2:.2f}  Δ{vp2-vp1:+.2f}\n"
            f"Vs {vs1:.2f}→{vs2:.2f}  Δ{vs2-vs1:+.2f}\n"
            f"ρ  {r1:.2f}→{r2:.2f}")


def plot_true_model(grids: Grids, out: Path) -> dict:
    """画出真 Vp 剖面 + 两台下方 1D，并标出三个界面的速度差。"""
    xs, zs, vp, vs = grids.xs, grids.zs, grids.vp, grids.vs
    XLO, XHI = 0.0, 114.0
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.6),
                             gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]})
    ax, ax30, ax50 = axes
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    im = ax.imshow(vp.T, extent=extent, cmap="RdYlBu_r", vmin=1.4, vmax=8.6,
                   aspect="auto", interpolation="nearest", zorder=0)
    xs_l = np.linspace(XLO, XHI, 160)
    ax.plot(xs_l, [g.H] * len(xs_l), "k--", lw=1.0, label="海底")
    ax.plot(xs_l, [g.z_conv(x) for x in xs_l], "k-.", lw=1.1, label="转换面")
    ax.plot(grids.mx, grids.mz, "k:", lw=1.4, label="莫霍")
    ax.plot([30, 50], [g.H, g.H], "k^", ms=8, zorder=7)
    # 图上直接标三个界面的 ΔVp（取 x=30）
    sf = grids.pair(30.0, g.H, dz=0.2)
    conv = grids.pair(30.0, g.z_conv(30.0), dz=0.2)
    moho = grids.pair(30.0, grids.z_moho(30.0), dz=0.2)
    ax.text(98.0, 1.15, f"海底  ΔVp={sf[2]-1.50:+.2f}", fontsize=8, color="k")
    ax.text(98.0, g.z_conv(30.0) - 0.35, f"转换面  ΔVp={conv[2]-conv[0]:+.2f}", fontsize=8, color="k")
    ax.text(98.0, grids.z_moho(30.0) - 0.35, f"莫霍  ΔVp={moho[2]-moho[0]:+.2f}", fontsize=8, color="k")
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(16.0, 0.0)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("深度 (km)")
    ax.set_title("真 Vp（盖层/面下低速 · 地壳底 7.0 · 地幔顶 8.0）")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.92)
    fig.colorbar(im, ax=ax, location="right", shrink=0.82, pad=0.02, label="Vp (km/s)")

    jumps = {}
    for x0, axp, title in ((30.0, ax30, "x = 30 km（OBS）"), (50.0, ax50, "x = 50 km（OBS）")):
        ix = int(np.argmin(np.abs(xs - x0)))
        zc = g.z_conv(x0)
        zm = grids.z_moho(x0)
        axp.plot(vp[ix], zs, color="#b2182b", lw=1.6, label="Vp")
        axp.plot(vs[ix], zs, color="#2166ac", lw=1.4, label="Vs")
        axp.axhline(g.H, color="k", ls="--", lw=0.8)
        axp.axhline(zc, color="k", ls="-.", lw=0.8)
        axp.axhline(zm, color="k", ls=":", lw=1.0)
        # 水→固：水 1.5，固 H+0.2
        sf = grids.pair(x0, g.H, dz=0.2)
        sf = (1.5, 0.0, sf[2], sf[3])
        conv = grids.pair(x0, zc, dz=0.2)
        moho = grids.pair(x0, zm, dz=0.2)
        jumps[int(x0)] = {
            "seafloor": dict(vp1=sf[0], vs1=sf[1], vp2=sf[2], vs2=sf[3]),
            "conv": dict(vp1=conv[0], vs1=conv[1], vp2=conv[2], vs2=conv[3]),
            "moho": dict(vp1=moho[0], vs1=moho[1], vp2=moho[2], vs2=moho[3]),
            "zc": zc, "zm": zm,
        }
        axp.text(0.98, g.H, _fmt_jump("海底", *sf),
                 transform=axp.get_yaxis_transform(), ha="right", va="bottom",
                 fontsize=7, color="0.15",
                 bbox=dict(boxstyle="round,pad=0.25", fc="w", ec="0.7", alpha=0.92))
        axp.text(0.98, zc, _fmt_jump("转换面", *conv),
                 transform=axp.get_yaxis_transform(), ha="right", va="bottom",
                 fontsize=7, color="0.15",
                 bbox=dict(boxstyle="round,pad=0.25", fc="w", ec="0.7", alpha=0.92))
        axp.text(0.98, zm, _fmt_jump("莫霍", *moho),
                 transform=axp.get_yaxis_transform(), ha="right", va="bottom",
                 fontsize=7, color="0.15",
                 bbox=dict(boxstyle="round,pad=0.25", fc="w", ec="0.7", alpha=0.92))
        axp.set_ylim(16.0, 0.0)
        axp.set_xlim(0.0, 9.2)
        axp.set_xlabel("速度 (km/s)")
        axp.set_title(title)
        axp.grid(True, alpha=0.3)
        axp.legend(loc="lower left", fontsize=8)
    ax30.set_ylabel("深度 (km)")
    fig.suptitle("lvz2d / fwd_all 真模型   转换面 7.20 → 地壳底 7.00 → 地幔顶 8.00", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    return jumps


def seg_is_s(xs, zs, code: int) -> list[bool]:
    flags = []
    for sx, sz, is_s in split_psx_phase_segments(list(xs), list(zs), code, z_conv=g.z_conv):
        flags.extend([is_s] * (len(sx) - 1))
    n = max(0, len(xs) - 1)
    if len(flags) < n:
        flags.extend([False] * (n - len(flags)))
    return flags[:n]


def incidence_deg(x0, z0, x1, z1) -> float:
    ds = math.hypot(x1 - x0, z1 - z0)
    if ds < 1e-9:
        return 12.0
    i = math.degrees(math.acos(min(1.0, abs(z1 - z0) / ds)))
    return float(np.clip(i, 0.2, 70.0))


def incoming_incidence(xs, zs, i, min_ds: float = 0.35, max_back: int = 50) -> float:
    """避开贴界面的碎点，用回望一段真实入射方向。"""
    i = int(np.clip(i, 1, len(xs) - 1))
    for j in range(i - 1, max(-1, i - max_back) - 1, -1):
        if math.hypot(xs[i] - xs[j], zs[i] - zs[j]) >= min_ds:
            return incidence_deg(xs[j], zs[j], xs[i], zs[i])
    return incidence_deg(xs[max(0, i - 1)], zs[max(0, i - 1)], xs[i], zs[i])


def interface_incidence(xs, zs, i, zfun, min_clear: float = 0.45, cap: float | None = 36.0) -> float:
    """用离开界面一段距离的来向估计入射角，避免贴面碎点把角度抬到超临界。"""
    n = len(xs)
    i = int(np.clip(i, 1, n - 1))
    j = i
    for k in range(i, -1, -1):
        if abs(zs[k] - zfun(xs[k])) >= min_clear:
            j = k
            break
    inc = incidence_deg(xs[j], zs[j], xs[i], zs[i])
    if cap is not None:
        inc = min(inc, cap)
    return inc


def collapse_path(xs, zs, tol: float = 0.03) -> tuple[list[float], list[float]]:
    nx, nz = [float(xs[0])], [float(zs[0])]
    for x, z in zip(xs[1:], zs[1:]):
        if math.hypot(x - nx[-1], z - nz[-1]) > tol:
            nx.append(float(x))
            nz.append(float(z))
    return nx, nz


# 系数种类按 raytype 物理清单，不从几何“同侧/异侧”猜。
# 转换面：下行、上行；若射线贴面转折则退化为一次反射。
CONV_PLAN = {
    0: [("cross_dn", "Tpp", False), ("cross_up", "Tpp", False)],
    1: [("cross_dn", "Tpp", False), ("cross_up", "Tpp", False)],
    6: [("cross_dn", "Tps", False), ("cross_up", "Tsp", True)],
    7: [("cross_dn", "Tpp", False), ("cross_up", "Tps", False)],
    8: [("cross_dn", "Tps", False), ("cross_up", "Tss", True)],
    9: [("cross_dn", "Tps", False), ("cross_up", "Tsp", True)],
    10: [("cross_dn", "Tpp", False), ("cross_up", "Tps", False), ("bounce", "Rss", True)],
    11: [("cross_dn", "Tps", False), ("cross_up", "Tss", True), ("bounce", "Rss", True)],
    12: [("cross_dn", "Tps", False), ("cross_up", "Tsp", True)],
    13: [("cross_dn", "Tps", False), ("cross_up", "Tss", True)],
    14: [("cross_dn", "Tpp", False), ("cross_up", "Tps", False)],
    15: [("cross_dn", "Tps", False), ("cross_up", "Tss", True)],
}
CONV_BOUNCE_FALLBACK = {
    0: ("Rpp", False),
    1: ("Rpp", False),
    6: ("Rps", False),
    7: ("Rps", False),
    8: ("Rps", False),
    9: ("Rps", False),
    10: ("Rps", False),
    11: ("Rps", False),
    12: ("Rps", False),
    13: ("Rps", False),
    14: ("Rps", False),
    15: ("Rps", False),
}


def iface_clusters(xs, zs, zfun, eps: float = 0.22) -> list[dict]:
    n = len(xs)
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i in range(n):
        on = abs(zs[i] - zfun(xs[i])) <= eps
        if on and not in_run:
            in_run = True
            start = i
        elif (not on) and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, n - 1))

    merged: list[tuple[int, int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= 8:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))

    out: list[dict] = []
    for a, b in merged:
        ia = max(0, a - 1)
        ib = min(n - 1, b + 1)
        zc = zfun(xs[(a + b) // 2])
        above_in = zs[ia] < zc - 0.06
        above_out = zs[ib] < zc - 0.06
        zmax_run = max(zs[a : b + 1])
        went_below = zmax_run > zc + 0.12
        if above_in and above_out and went_below:
            kind = "through"
        elif above_in and above_out:
            kind = "bounce"
        elif (not above_in) and (not above_out):
            kind = "bounce_below"
        elif above_in:
            kind = "cross_dn"
        else:
            kind = "cross_up"
        out.append({
            "i": a, "j": b, "kind": kind, "x": xs[a], "z": zs[a], "zc": zc,
            "above_in": above_in, "above_out": above_out, "went_below": went_below,
        })
    return out


def seafloor_events(xs, zs) -> dict[str, list[int]]:
    n = len(xs)
    entry, exit_, bounce_s = [], [], []
    for i in range(n - 1):
        z0, z1 = zs[i], zs[i + 1]
        if z0 < H - 0.05 and z1 >= H - 0.05:
            entry.append(i)
        elif z0 >= H - 0.05 and z1 < H - 0.05:
            exit_.append(i)
    for i in range(1, n - 1):
        if abs(zs[i] - H) > 0.18:
            continue
        if zs[i - 1] < H - 0.12 or zs[i + 1] < H - 0.12:
            continue
        if zs[i] <= zs[i - 1] and zs[i] <= zs[i + 1] and (zs[i - 1] + zs[i + 1] - 2 * zs[i]) > 0.06:
            if i > n // 4:
                bounce_s.append(i)
    return {"entry": entry, "exit": exit_, "bounce": bounce_s}


def surface_bounce_index(xs, zs) -> int | None:
    n = len(zs)
    best = None
    best_z = 1e9
    for i in range(max(6, n // 5), n - 1):
        if zs[i] < 0.40 and zs[i] <= zs[i - 1] and zs[i] <= zs[i + 1]:
            if zs[i] < best_z:
                best_z = zs[i]
                best = i
    return best


def ray_energy(xs, zs, code: int, grids: Grids) -> tuple[float, list[dict]]:
    xs, zs = collapse_path(xs, zs)
    n = len(xs)
    if n < 3:
        return 1.0, []
    sflag = seg_is_s(xs, zs, code)
    factors: list[dict] = []

    def add(name, val):
        v = float(val)
        if not np.isfinite(v) or v < 1e-8:
            v = 1e-8
        factors.append({"name": name, "E": min(v, 1.2)})

    def props_at(x, z):
        vp, vs = grids.sample(x, z)
        if z <= H + 0.05:
            return 1.5, VS_W, RHO_W
        vs = vs if vs > 0.15 else vp / 1.73
        return vp, vs, gardner(vp)

    def solid_props(x):
        return props_at(x, H + 0.25)

    def inc_s_at(i):
        i0 = max(0, min(i, len(sflag) - 1))
        return bool(sflag[i0])

    sf = seafloor_events(xs, zs)
    mid = n // 3

    entered_solid = max(zs) > H + 0.18
    # 炮侧：水→固 Tpp（水直达不到固体则跳过）
    if entered_solid:
        i_ent = sf["entry"][0] if sf["entry"] else min(range(n), key=lambda j: abs(zs[j] - H))
        vp2, vs2, r2 = solid_props(xs[i_ent])
        inc = incoming_incidence(xs, zs, i_ent)
        add("海底 Tpp(水→固)", energy_coeff("Tpp", False, 1.5, VS_W, RHO_W, vp2, vs2, r2, inc))

    # 转换面：through = 下行+上行一次走完；10/11 的 Rss 用最后一次盖层回弹
    convs = iface_clusters(xs, zs, g.z_conv, eps=0.24)
    expanded: list[dict] = []
    for c in convs:
        if c["kind"] == "through":
            expanded.append({**c, "kind": "cross_dn"})
            expanded.append({**c, "kind": "cross_up"})
        else:
            expanded.append(c)
    plan = list(CONV_PLAN.get(int(code), []))
    used = [False] * len(expanded)

    def conv_energy(clu, kind_c, inc_s):
        i = clu["i"]
        zc = clu["zc"]
        inc = interface_incidence(xs, zs, i, g.z_conv, min_clear=0.45, cap=36.0)
        vpa, vsa, ra = props_at(xs[i], zc - 0.18)
        vpb, vsb, rb = props_at(xs[i], zc + 0.18)
        if clu["kind"] == "cross_up":
            vp1, vs1, r1, vp2, vs2, r2 = vpb, vsb, rb, vpa, vsa, ra
        else:
            vp1, vs1, r1, vp2, vs2, r2 = vpa, vsa, ra, vpb, vsb, rb
        return energy_coeff(kind_c, inc_s, vp1, vs1, r1, vp2, vs2, r2, inc)

    bounce_idxs = [k for k, c in enumerate(expanded) if c["kind"] == "bounce"]
    rss_idx = bounce_idxs[-1] if bounce_idxs and int(code) in (10, 11) else None

    for want_kind, coeff, inc_s in plan:
        idx = None
        if want_kind == "bounce" and rss_idx is not None:
            idx = rss_idx
        else:
            idx = next((k for k, c in enumerate(expanded)
                        if (not used[k]) and k != rss_idx and c["kind"] == want_kind), None)
            if idx is None and want_kind.startswith("cross"):
                idx = next((k for k, c in enumerate(expanded)
                            if (not used[k]) and k != rss_idx and c["kind"].startswith("cross")), None)
        if idx is None:
            continue
        used[idx] = True
        val = conv_energy(expanded[idx], coeff, inc_s)
        # 超临界均匀 Tpp≈0：图论仍有 P 路径，但不计入能量积
        if coeff == "Tpp" and val < 0.03:
            continue
        add(f"转换面 {coeff}", val)

    leftover = [c for k, c in enumerate(expanded) if not used[k] and k != rss_idx]
    if leftover and not any(f["name"].startswith("转换面") and f["name"] != "转换面 Rss" for f in factors):
        fb, inc_s = CONV_BOUNCE_FALLBACK.get(int(code), ("Rpp", False))
        add(f"转换面 {fb}", conv_energy(leftover[0], fb, inc_s))

    # 莫霍反射：仅 1 / 12 / 13
    if int(code) in (1, 12, 13):
        i_deep = max(range(n), key=lambda j: zs[j])
        zm = grids.z_moho(xs[i_deep])
        if abs(zs[i_deep] - zm) < 1.8:
            inc_s = inc_s_at(max(0, i_deep - 1))
            inc = incoming_incidence(xs, zs, i_deep, min_ds=0.6)
            vp1, vs1, r1 = props_at(xs[i_deep], zm - 0.25)
            vp2, vs2, r2 = props_at(xs[i_deep], zm + 0.25)
            k = "Rss" if inc_s or int(code) in (12, 13) else "Rpp"
            add(f"莫霍 {k}", energy_coeff(k, k == "Rss", vp1, vs1, r1, vp2, vs2, r2, inc))

    # 台侧盖层 SS（10/11）：固体一侧海底回弹
    if int(code) in (10, 11):
        ib = next((i for i in sf["bounce"] if i > mid), None)
        if ib is None:
            ib = next((i for i in range(mid, n - 1)
                       if abs(zs[i] - H) < 0.18 and zs[i] <= zs[i - 1] and zs[i] <= zs[i + 1]
                       and min(zs[i - 1], zs[i + 1]) >= H - 0.05), None)
        if ib is not None:
            vp1, vs1, r1 = solid_props(xs[ib])
            inc = incoming_incidence(xs, zs, ib)
            add("海底 Rss", energy_coeff("Rss", True, vp1, vs1, r1, 1.5, VS_W, RHO_W, inc))

    # 台侧水柱 peg（9/14/15）
    if int(code) in (9, 14, 15):
        ix = next((i for i in sf["exit"] if i > mid), None)
        if ix is not None:
            vp1, vs1, r1 = solid_props(xs[ix])
            inc = incoming_incidence(xs, zs, ix)
            if int(code) == 9:
                add("海底 Tpp(固→水)", energy_coeff("Tpp", False, vp1, vs1, r1, 1.5, VS_W, RHO_W, inc))
            else:
                add("海底 Tsp(固→水)", energy_coeff("Tsp", True, vp1, vs1, r1, 1.5, VS_W, RHO_W, inc))
        isurf = surface_bounce_index(xs, zs)
        if isurf is not None:
            add("海面 Rpp", 0.95)

    if not factors:
        return 1.0, [{"name": "水直达", "E": 1.0}]
    prod = 1.0
    for f in factors:
        prod *= max(f["E"], 1e-6)
    return float(prod), factors


def main() -> int:
    syn = WORK / "syn_true.dat"
    rays_p = WORK / "rays_true.dat"
    if not syn.is_file() or not rays_p.is_file():
        raise SystemExit(f"missing {syn} or {rays_p}")
    picks = parse_picks(syn.read_text(encoding="utf-8"))
    rays = pr.parse_rays(rays_p)
    recs = recs_for_draw(syn)
    grids = Grids(WORK / "true_vp.smesh", WORK / "true_vs.smesh", WORK / "moho_true.refl")
    jumps = plot_true_model(grids, WORK / "check_all_model.png")

    rows = []
    for (code, dx, t), (xs, zs), p in zip(recs, rays, picks):
        e_intf, facs = ray_energy(xs, zs, code, grids)
        L = 0.0
        for i in range(len(xs) - 1):
            L += math.hypot(xs[i + 1] - xs[i], zs[i + 1] - zs[i])
        amp = math.sqrt(max(e_intf, 1e-12) * 10.0 / max(L, 1.0))
        bottleneck = min(facs, key=lambda f: f["E"])["name"] if facs else "—"
        shot, obs = float(p[1]), float(p[4])
        dx_s = shot - obs
        rows.append({
            "code": int(code),
            "name": PHASE_TT_STYLE.get(int(code), ("", "", f"{code}"))[2],
            "obs": round(obs, 3),
            "shot": round(shot, 3),
            "dx": float(dx_s),
            "t": float(t),
            "tred": float(t) - abs(dx_s) / VRED,
            "E": e_intf,
            "amp": amp,
            "L": L,
            "nfac": len(facs),
            "bottleneck": bottleneck,
            "factors": facs,
        })

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2))
    ax_t, ax_e = axes[0, 0], axes[0, 1]
    ax_a, ax_b = axes[1, 0], axes[1, 1]

    one = [r for r in rows if abs(r["obs"] - OBS_PLOT) < 0.2 and abs(r["dx"]) > DX_MIN]
    codes = [c for c in PHASE_TT_STYLE if any(r["code"] == c for r in one)]
    for code in codes:
        color, fmt, name = PHASE_TT_STYLE[code]
        sub = [r for r in one if r["code"] == code]
        if not sub:
            continue
        sub.sort(key=lambda r: r["dx"])
        ax_t.plot([r["dx"] for r in sub], [r["tred"] for r in sub], fmt, color=color,
                  ms=5, lw=1.2, label=name, alpha=0.95)
        epos = np.clip([r["E"] for r in sub], 1e-8, 1.0)
        ax_e.semilogy([r["dx"] for r in sub], epos, fmt, color=color, ms=5, lw=1.2,
                      label=name, alpha=0.95)
        ax_a.semilogy([r["dx"] for r in sub], np.clip([r["amp"] for r in sub], 1e-6, 2),
                      fmt, color=color, ms=5, lw=1.2, label=name, alpha=0.95)

    for ax in (ax_t, ax_e, ax_a):
        ax.axvline(0.0, color="0.55", ls=":", lw=0.9, zorder=1)
        ax.axvline(-DX_MIN, color="0.75", ls="--", lw=0.7, zorder=1)
        ax.axvline(DX_MIN, color="0.75", ls="--", lw=0.7, zorder=1)
        ax.set_xlim(-55.0, 65.0)
    ax_t.set_xlabel("偏移 Δx = x炮 − x台 (km)")
    ax_t.set_ylabel(rf"$t - |\Delta x|/{VRED:g}$ (s)")
    ax_t.set_title(f"理论折合走时  OBS {OBS_PLOT:.0f} km  |Δx| > {DX_MIN:.0f} km")
    ax_t.invert_yaxis()
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.92)

    ax_e.set_xlabel("偏移 Δx = x炮 − x台 (km)")
    ax_e.set_ylabel("界面能量积  Π E_i")
    ax_e.set_title(f"左普利兹反射/透射能量积  |Δx| > {DX_MIN:.0f} km")
    ax_e.set_ylim(1e-8, 2)
    ax_e.grid(True, which="both", alpha=0.3)

    ax_a.set_xlabel("偏移 Δx = x炮 − x台 (km)")
    ax_a.set_ylabel(r"相对振幅  $\sqrt{E_{\mathrm{intf}}\,L_0/L}$")
    ax_a.set_title(f"相对振幅（能量积 × 几何扩散）  |Δx| > {DX_MIN:.0f} km")
    ax_a.set_ylim(1e-5, 1)
    ax_a.grid(True, which="both", alpha=0.3)

    mean_e = []
    labels = []
    colors = []
    for code in codes:
        sub = [r["E"] for r in one if r["code"] == code]
        mean_e.append(float(np.exp(np.mean(np.log(np.clip(sub, 1e-8, 1.0))))))
        labels.append(PHASE_TT_STYLE[code][2])
        colors.append(PHASE_TT_STYLE[code][0])
    order = np.argsort(mean_e)[::-1]
    ax_b.barh([labels[i] for i in order], [max(mean_e[i], 1e-8) for i in order],
              color=[colors[i] for i in order], log=True)
    ax_b.set_xlabel("几何平均界面能量积")
    ax_b.set_title(f"各震相平均能量  |Δx| > {DX_MIN:.0f} km")
    ax_b.grid(True, axis="x", which="both", alpha=0.3)

    fig.suptitle(
        f"OBS {OBS_PLOT:.0f} km   |Δx| > {DX_MIN:.0f} km   炮检距至 60 km   地壳底 7.0 / 地幔顶 8.0",
        fontsize=11,
    )
    fig.tight_layout()
    out = WORK / "check_all_phases.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)

    summary = []
    for code in codes:
        sub = [r for r in one if r["code"] == code]
        bottlenecks: dict[str, int] = {}
        for r in sub:
            bottlenecks[r["bottleneck"]] = bottlenecks.get(r["bottleneck"], 0) + 1
        top_b = max(bottlenecks, key=bottlenecks.get) if bottlenecks else "—"
        e = np.clip([r["E"] for r in sub], 1e-8, 1.0)
        e_crust = []
        for r in sub:
            prod = 1.0
            n_f = 0
            for f in r["factors"]:
                if str(f["name"]).startswith("莫霍"):
                    continue
                prod *= max(f["E"], 1e-6)
                n_f += 1
            e_crust.append(prod if n_f else r["E"])
        # representative factors near 22 km on OBS 30
        ex = min(sub, key=lambda r: abs(abs(r["dx"]) - 22.0))
        summary.append({
            "code": code,
            "name": PHASE_TT_STYLE[code][2],
            "n": len(sub),
            "tmin": min(r["t"] for r in sub),
            "tmax": max(r["t"] for r in sub),
            "E_geo": float(np.exp(np.mean(np.log(e)))),
            "E_med": float(np.median(e)),
            "E_crust": float(np.exp(np.mean(np.log(np.clip(e_crust, 1e-8, 1.2))))),
            "amp_geo": float(np.exp(np.mean(np.log(np.clip([r["amp"] for r in sub], 1e-8, 2))))),
            "bottleneck": top_b,
            "example_dx": ex["dx"],
            "example_obs": ex["obs"],
            "example_factors": ex["factors"],
        })

    plotted = [r for r in rows if abs(r["dx"]) > DX_MIN]
    payload = {
        "vred": VRED,
        "dx_min": DX_MIN,
        "offsets": sorted({round(r["dx"], 1) for r in plotted}),
        "obs": [30.0, 50.0],
        "model": jumps,
        "summary": summary,
        "series": {},
    }
    for obs in (30.0, 50.0):
        payload["series"][str(int(obs))] = {}
        for code in codes:
            sub = [r for r in plotted if r["code"] == code and abs(r["obs"] - obs) < 0.2]
            sub.sort(key=lambda r: r["dx"])
            payload["series"][str(int(obs))][str(code)] = [
                {"dx": r["dx"], "t": r["t"], "tred": r["tred"], "E": r["E"], "amp": r["amp"]}
                for r in sub
            ]

    js = WORK / "all_phases.json"
    js.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"n={len(rows)}  wrote {out}")
    print(f"wrote {js}")
    print(f"{'code':6s} {'name':16s} {'tmin':7s} {'tmax':7s} {'E_geo':9s} {'amp':8s} bottleneck")
    for s in sorted(summary, key=lambda x: -x["E_geo"]):
        print(f"{s['code']:4d}   {s['name']:16s} {s['tmin']:7.3f} {s['tmax']:7.3f} "
              f"{s['E_geo']:9.2e} {s['amp_geo']:8.2e}  {s['bottleneck']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
