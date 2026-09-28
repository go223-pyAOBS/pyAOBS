#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""沉积 2 km（H=2, Zc=4）正演：PPP/PPS/PSP/PSS。图论用现成 tt_forward，共 p 射击不改源码。"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent / "ps_inv"))
sys.path.insert(0, str(HERE.parent.parent.parent / "ps_fwd"))
import plot_ps_inv_models as psp  # noqa: E402
from check_ps_fwd import split_psx_phase_segments  # noqa: E402

PHASE_TT_STYLE = {
    0: ("#1f77b4", "o-", "PPP  0"),
    7: ("#ff7f0e", "^-", "PPS  7"),
    6: ("#2ca02c", "s-", "PSP  6"),
    8: ("#c44e8a", "D-", "PSS  8"),
}

H, ZC, ZMAX = 2.0, 4.0, 16.0
SED_H = ZC - H
V_WATER, V_AIR, KAPPA = 1.5, 0.33, 1.73
SED0 = 1.80
# 与 3 km 盖层同一梯度，薄了以后盖层底更慢。
SED_GRAD = (4.00 - SED0) / 3.0
LID_VP = SED0 + SED_GRAD * SED_H
SHOT_Z = 0.01
OBS_XS = (30.0, 40.0, 50.0, 60.0, 70.0)
SHOT_XS = tuple(round(20.0 + i * 2.0, 1) for i in range(31))
XMIN, XMAX, DX = 0.0, 125.0, 2.0
DZ = 0.2
VRED = 8.0
P_COLOR, S_COLOR = "#1f77b4", "#c44e8a"
PHASES = (
    (0, "PPP", False, False),  # basement P, lid-up P
    (7, "PPS", False, True),
    (6, "PSP", True, False),
    (8, "PSS", True, True),
)


def _xs() -> list[float]:
    return [XMIN + i * DX for i in range(int(round((XMAX - XMIN) / DX)) + 1)]


def _zs() -> list[float]:
    return [i * DZ for i in range(int(round(ZMAX / DZ)) + 1)]


@dataclass(frozen=True)
class Model:
    name: str
    crust_vp: float

    @property
    def vs0(self) -> float:
        return self.crust_vp / KAPPA

    def vp(self, z: float) -> float:
        if z <= H + 1e-12:
            return V_WATER
        if z < ZC:
            return SED0 + SED_GRAD * (z - H)
        return self.crust_vp + KAPPA * 0.12 * (z - ZC)

    def vs(self, z: float) -> float:
        if z <= H + 1e-12:
            return V_WATER
        if z < ZC:
            return self.vp(z) / KAPPA
        return self.vs0 + 0.12 * (z - ZC)


def models(only: str | None = None) -> tuple[Model, ...]:
    # equal：面下顶 Vs = 盖层底 Vp，κ 仍 1.73。
    allm = (
        Model("slow", 5.00),
        Model("equal", LID_VP * KAPPA),
        Model("fast", 7.20),
    )
    if only:
        picked = tuple(m for m in allm if m.name == only)
        if not picked:
            raise SystemExit(f"unknown model {only!r}; choose slow|equal|fast")
        return picked
    return allm


def write_smesh(path: Path, vfun) -> None:
    xs, zs = _xs(), _zs()
    lines = [
        f"{len(xs)} {len(zs)} {V_WATER:.4f} {V_AIR}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    for _x in xs:
        lines.append(" ".join(f" {vfun(z):.4f}" for z in zs))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_iface(path: Path, z: float) -> None:
    path.write_text("".join(f"{x:.4f} {z:.4f}\n" for x in _xs()), encoding="utf-8")


def write_geom(path: Path, codes: tuple[int, ...]) -> None:
    nrcv = len(SHOT_XS) * len(codes)
    lines = [str(len(OBS_XS))]
    for ox in OBS_XS:
        lines.append(f"s{ox:10.3f}{H:10.3f}{nrcv:5d}")
        for kind in codes:
            for x in SHOT_XS:
                lines.append(f"r{x:10.3f}{SHOT_Z:10.3f}{kind:5d}{0.0:10.3f}{0.05:10.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_case(md: Model) -> Path:
    d = HERE / md.name
    d.mkdir(parents=True, exist_ok=True)
    write_smesh(d / "vp.smesh", md.vp)
    write_smesh(d / "vs.smesh", md.vs)

    def mixed(z: float) -> float:
        return md.vs(z) if z >= ZC - 1e-9 else md.vp(z)

    write_smesh(d / "mixed.smesh", mixed)

    def vp_psx(z: float) -> float:
        # 7/8 双场：转换面结点划给 Vs，盖层底 P 插值与 mixed 一致。
        if abs(z - ZC) <= 1e-9:
            return md.vs0
        return md.vp(z)

    write_smesh(d / "vp_psx.smesh", vp_psx)
    write_iface(d / "seafloor.refl", H)
    write_iface(d / "conv.refl", ZC)
    write_geom(d / "geom0.dat", (0,))
    write_geom(d / "geom6.dat", (6,))
    write_geom(d / "geom78.dat", (7, 8))
    return d


def wsl_path(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        return f"/mnt/{s[0].lower()}{s[2:]}"
    return s


def tt_forward(md: Model) -> None:
    d = write_case(md)
    bin_p = HERE.parents[3] / "src" / "build-tomo2d" / "tt_forward"
    n = "-N8/8/0.8/8/1e-4/1e-5"
    wd = wsl_path(d)
    b = wsl_path(bin_p)
    jobs = [
        f"{b} -Mvp.smesh -Ggeom0.dat -Bseafloor.refl {n} -Rrays0.dat > syn0.dat",
        f"{b} -Mmixed.smesh -Ggeom6.dat -Xconv.refl -Bseafloor.refl {n} -Rrays6.dat > syn6.dat",
        f"{b} -Mvp_psx.smesh -Uvs.smesh -Ggeom78.dat -Xconv.refl -Bseafloor.refl "
        f"{n} -Rrays78.dat > syn78.dat",
    ]
    env = "export TOMO2D_FWD_OMP=1 OMP_NUM_THREADS=8"
    for cmd in jobs:
        print(f"== [{md.name}] {cmd[:90]}…")
        r = subprocess.run(
            ["wsl", "bash", "-lc", f"{env}; cd {wd} && {cmd}"],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            raise SystemExit(f"tt_forward failed in {md.name}")


def _step(dz: float, p: float, v: float) -> tuple[float, float] | None:
    pv = p * v
    if pv >= 0.999:
        return None
    den = math.sqrt(max(1.0 - pv * pv, 1e-16))
    return abs(dz) * pv / den, abs(dz) / (v * den)


def integrate_no_turn(z0: float, z1: float, p: float, vfun, dz: float = 0.02):
    if abs(z1 - z0) < 1e-9:
        return 0.0, 0.0, [z0], [0.0]
    n = max(8, int(abs(z1 - z0) / dz))
    zs = np.linspace(z0, z1, n + 1)
    x = t = 0.0
    xs = [0.0]
    for a, b in zip(zs[:-1], zs[1:]):
        got = _step(b - a, p, vfun(0.5 * (a + b)))
        if got is None:
            return None
        dx, dt = got
        x += dx
        t += dt
        xs.append(x)
    return x, t, zs.tolist(), xs


def circle_turn(v0: float, g: float, p: float, z_top: float):
    if g <= 1e-9 or p * v0 >= 0.999:
        return None
    c0 = math.sqrt(max(1.0 - (p * v0) ** 2, 0.0))
    if c0 < 1e-5:
        return None
    r = 1.0 / (p * g)
    x_half = c0 / (p * g)
    z_turn = z_top + (1.0 / p - v0) / g
    if z_turn > ZMAX - 0.05 or z_turn <= z_top + 0.05:
        return None
    t = (2.0 / g) * math.log((1.0 + c0) / (p * v0))
    alpha = math.acos(max(-1.0, min(1.0, p * v0)))
    zc = z_top - v0 / g
    thetas = np.linspace(-alpha, alpha, 72)
    xs = [x_half + r * math.sin(th) for th in thetas]
    zs = [zc + r * math.cos(th) for th in thetas]
    xs[0], zs[0] = 0.0, z_top
    xs[-1], zs[-1] = 2.0 * x_half, z_top
    return 2.0 * x_half, t, z_turn, zs, xs


@dataclass
class ShotRay:
    code: int
    xs: list[float]
    zs: list[float]
    p: float
    t: float
    z_turn: float
    offset: float  # 有符号：x_shot − x_OBS
    lid_up_s: bool
    base_s: bool
    obs_x: float = 0.0


def p_window(md: Model, base_s: bool, lid_up_s: bool) -> tuple[float, float] | None:
    p_hi = 0.999 / LID_VP
    if base_s:
        p_hi = min(p_hi, 0.999 / md.vs0)
        vbot = md.vs(ZMAX)
        p_lo = 1.0 / vbot + 1e-4
    else:
        p_hi = min(p_hi, 0.999 / md.crust_vp)
        vbot = md.vp(ZMAX)
        p_lo = 1.0 / vbot + 1e-4
    if lid_up_s:
        p_hi = min(p_hi, 0.999 / md.vs(ZC - 1e-3))
    if p_hi <= p_lo + 1e-5:
        return None
    return p_lo, p_hi


def trace_phase(md: Model, p: float, base_s: bool, lid_up_s: bool):
    down = integrate_no_turn(SHOT_Z, ZC, p, md.vp)
    if down is None:
        return None
    if base_s:
        base = circle_turn(md.vs0, 0.12, p, ZC)
    else:
        base = circle_turn(md.crust_vp, KAPPA * 0.12, p, ZC)
    if base is None:
        return None
    vup = md.vs if lid_up_s else md.vp
    up = integrate_no_turn(ZC, H, p, vup)
    if up is None:
        return None
    xd, td, zd, xr_d = down
    xb, tb, z_turn, zb, xr_b = base
    xu, tu, zu, xr_u = up
    return xd + xb + xu, td + tb + tu, z_turn, {
        "d": (zd, xr_d), "b": (zb, xr_b), "u": (zu, xr_u),
    }


def assemble(shot_x: float, obs_x: float, pack: dict) -> tuple[list[float], list[float]]:
    sgn = 1.0 if obs_x >= shot_x else -1.0
    xs: list[float] = []
    zs: list[float] = []
    x0 = shot_x
    for z, xrel in zip(*pack["d"]):
        xs.append(x0 + sgn * xrel)
        zs.append(z)
    x1 = xs[-1]
    for z, xrel in zip(pack["b"][0][1:], pack["b"][1][1:]):
        xs.append(x1 + sgn * xrel)
        zs.append(z)
    x2 = xs[-1]
    for z, xrel in zip(pack["u"][0][1:], pack["u"][1][1:]):
        xs.append(x2 + sgn * xrel)
        zs.append(z)
    return xs, zs


def shoot_all(md: Model, shot_stride: int = 1) -> list[ShotRay]:
    out: list[ShotRay] = []
    shots = SHOT_XS[::shot_stride]
    for code, _name, base_s, lid_up_s in PHASES:
        win = p_window(md, base_s, lid_up_s)
        xmin = xmax = None
        if win:
            for p in (win[0], win[1]):
                got = trace_phase(md, p, base_s, lid_up_s)
                if got:
                    xmin = got[0] if xmin is None else min(xmin, got[0])
                    xmax = got[0] if xmax is None else max(xmax, got[0])
        print(f"  {md.name} {_name}: p {win}  X~[{xmin},{xmax}]" if win else f"  {md.name} {_name}: no p window")
        for ox in OBS_XS:
            for sx in shots:
                off = abs(ox - sx)
                hit = _match(md, off, base_s, lid_up_s)
                if hit is None:
                    continue
                xs, zs, p, t, zt, xest, pack = hit
                if abs(xest - off) > 2.5:
                    continue
                px, pz = assemble(sx, ox, pack)
                out.append(ShotRay(code, px, pz, p, t, zt, sx - ox, lid_up_s, base_s, ox))
    return out


def _match(md, offset, base_s, lid_up_s):
    win = p_window(md, base_s, lid_up_s)
    if win is None or offset < 1.0:
        return None
    p_lo, p_hi = win
    rows = []
    for p in np.linspace(p_lo, p_hi, 40):
        got = trace_phase(md, float(p), base_s, lid_up_s)
        if got:
            rows.append((float(p), got))
    if len(rows) < 2:
        return None
    best = None
    for (p0, g0), (p1, g1) in zip(rows, rows[1:]):
        x0, x1 = g0[0], g1[0]
        if (x0 - offset) * (x1 - offset) > 0 or abs(x1 - x0) < 1e-9:
            continue
        a = (offset - x0) / (x1 - x0)
        p = p0 + a * (p1 - p0)
        got = trace_phase(md, p, base_s, lid_up_s)
        if got is None:
            continue
        err = abs(got[0] - offset)
        if best is None or err < best[0]:
            best = (err, p, got)
    if best is None:
        p, got = min(rows, key=lambda r: abs(r[1][0] - offset))
        best = (abs(got[0] - offset), p, got)
    err, p, got = best
    x, t, zt, pack = got
    return None if err > 2.5 else (None, None, p, t, zt, x, pack)


def parse_syn_file(path: Path) -> list[tuple[int, float, float, float]]:
    """(code, x_shot−x_OBS, t, x_OBS)。"""
    recs: list[tuple[int, float, float, float]] = []
    if not path.is_file():
        return recs
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    i = 1
    src_x = OBS_XS[0]
    while i < len(lines):
        parts = lines[i].split()
        i += 1
        if not parts or parts[0] != "s":
            continue
        src_x = float(parts[1])
        nrcv = int(float(parts[-1]))
        for _ in range(nrcv):
            rp = lines[i].split()
            i += 1
            recs.append((int(float(rp[3])), float(rp[1]) - src_x, float(rp[4]), src_x))
    return recs


def shoot_recs(rays: list[ShotRay]) -> list[tuple[int, float, float, float]]:
    return [(r.code, r.offset, r.t, r.obs_x) for r in rays]


def _zmax_segs(segs) -> float:
    return max((max(zs) for _x, zs in segs if zs), default=0.0)


def _zmax_rays(rays: list[ShotRay], code: int) -> float:
    zm = 0.0
    for r in rays:
        if r.code == code and r.zs:
            zm = max(zm, max(r.zs))
    return zm


def draw_phase_rays(ax, rays: list[ShotRay], code: int) -> None:
    n = 0
    for r in rays:
        if r.code != code:
            continue
        n += 1
        for j in range(len(r.xs) - 1):
            zm = 0.5 * (r.zs[j] + r.zs[j + 1])
            if zm > ZC + 1e-3:
                s = r.base_s
            elif zm > H + 1e-3:
                going_home = abs(r.xs[j] - r.xs[-1]) < abs(r.xs[j] - r.xs[0])
                s = bool(r.lid_up_s and going_home)
            else:
                s = False
            ax.plot(r.xs[j : j + 2], r.zs[j : j + 2],
                    color=S_COLOR if s else P_COLOR, lw=0.7, alpha=0.75, zorder=3)
    ax.scatter(list(OBS_XS), [H] * 5, marker="^", c="k", s=22, zorder=5)


def draw_graph_phase_rays(ax, segs, recs, code: int) -> None:
    """图论射线：P 蓝虚线，S 粉实线（后画，避免被 P 盖住）。路径按炮→台。"""
    n = min(len(segs), len(recs)) if recs else len(segs)
    pending_s: list[tuple[list[float], list[float]]] = []
    for i in range(n):
        rx, rz = segs[i]
        ci = recs[i][0] if recs and i < len(recs) else code
        for sx, sz, is_s in split_psx_phase_segments(rx, rz, ci, z_conv=ZC, z_sf=H):
            if is_s:
                pending_s.append((sx, sz))
            else:
                ax.plot(sx, sz, color=P_COLOR, lw=0.5, ls="--", alpha=0.65, zorder=3)
    for sx, sz in pending_s:
        ax.plot(sx, sz, color=S_COLOR, lw=0.85, ls="-", alpha=0.85, zorder=5)
    ax.scatter(list(OBS_XS), [H] * 5, marker="^", c="k", s=22, zorder=6)


def plot_rays(md: Model, shoot: list[ShotRay], d: Path, out_png: Path) -> None:
    sys.path.insert(0, str(HERE.parent.parent))
    from make_ppp_psp_inv_case import parse_smesh  # noqa: E402

    xs, zs, vel = parse_smesh(d / "mixed.smesh")
    grid = psp._grid(xs, zs, vel)
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    fig, axes = plt.subplots(4, 2, figsize=(13.2, 14.5), facecolor="w", layout="constrained")
    names = {0: "PPP 0", 7: "PPS 7", 6: "PSP 6", 8: "PSS 8"}
    rayfiles = {0: d / "rays0.dat", 6: d / "rays6.dat", 7: d / "rays78.dat", 8: d / "rays78.dat"}
    recs78 = parse_syn_file(d / "syn78.dat")
    recs6 = parse_syn_file(d / "syn6.dat")
    recs0 = parse_syn_file(d / "syn0.dat")
    recmap = {0: recs0, 6: recs6, 7: recs78, 8: recs78}

    for row, (code, name, _a, _b) in enumerate(PHASES):
        for col, title_pref in enumerate(("图论 tt_forward", "共 p 射击")):
            ax = axes[row, col]
            ax.imshow(grid, extent=extent, cmap="RdYlBu_r", vmin=1.2, vmax=6.2,
                      aspect="auto", interpolation="nearest", zorder=0)
            ax.axhline(H, color="0.15", ls="--", lw=0.8)
            ax.axhline(ZC, color="0.15", ls="-.", lw=1.0)
            ax.set_xlim(18, 82)
            ax.set_ylim(ZMAX, 0)
            ax.grid(True, alpha=0.25)
            if col == 0:
                rf = rayfiles[code]
                segs = psp.parse_rays(rf) if rf.is_file() else []
                recs = recmap[code]
                if code in (7, 8) and segs and recs:
                    n = min(len(segs), len(recs))
                    segs = [segs[i] for i in range(n) if recs[i][0] == code]
                    recs = [recs[i] for i in range(n) if recs[i][0] == code]
                zm = _zmax_segs(segs)
                draw_graph_phase_rays(ax, segs, recs, code)
                ax.set_title(f"{name}  {title_pref}  zmax={zm:.2f} km")
                ax.set_ylabel("深度 (km)")
            else:
                draw_phase_rays(ax, shoot, code)
                ax.set_title(f"{name}  {title_pref}  zmax={_zmax_rays(shoot, code):.2f} km  n={sum(1 for r in shoot if r.code==code)}")
            if row == 3:
                ax.set_xlabel("模型距离 (km)")
    fig.suptitle(
        f"沉积 {SED_H:.0f} km（Zc={ZC:g}）  {md.name}  盖层底Vp={LID_VP:.2f}  面下Vs0={md.vs0:.2f}",
        fontsize=12,
    )
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"wrote {out_png}")


def _tred(dx: float, t: float) -> float:
    return t - abs(dx) / VRED


def _draw_signed_tt(ax, recs, *, filter_ppp_abs: float | None, title: str) -> None:
    """每个台一条线。折合仍用 |Δx|。"""
    for code, (color, fmt, name) in PHASE_TT_STYLE.items():
        by_src: dict[float, list[tuple[float, float]]] = {}
        for row in recs:
            c, dx, tt, src = row[0], row[1], row[2], row[3]
            if c != code:
                continue
            if filter_ppp_abs is not None and code == 0 and abs(dx) < filter_ppp_abs - 1e-6:
                continue
            by_src.setdefault(round(src, 3), []).append((dx, _tred(dx, tt)))
        labeled = False
        for pts in by_src.values():
            pts.sort()
            kw = dict(color=color, ms=3.2, lw=1.05, zorder=3, clip_on=True)
            if not labeled:
                kw["label"] = name
                labeled = True
            ax.plot([p[0] for p in pts], [p[1] for p in pts], fmt, **kw)
    ax.axvline(0.0, color="0.55", lw=0.7, zorder=1)
    ax.set_xlabel(r"有符号偏移 $x_{\mathrm{shot}}-x_{\mathrm{OBS}}$ (km)")
    ax.set_ylabel(rf"$t - |x_{{\mathrm{{shot}}}}-x_{{\mathrm{{OBS}}}}| / {VRED:g}$ (s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", framealpha=0.9, fontsize=8)


def plot_ttimes_signed(
    g: list,
    s: list,
    out_png: Path,
    *,
    left_title: str,
    right_title: str,
    suptitle: str,
) -> None:
    shoot_abs = [abs(dx) for _c, dx, _t, _src in s]
    abs_lo = min(shoot_abs) if shoot_abs else 0.0
    cmp_tr = [_tred(dx, tt) for _c, dx, tt, _src in s]
    cmp_tr += [
        _tred(dx, tt) for c, dx, tt, _src in g
        if abs(dx) >= abs_lo - 1e-6 or c != 0
    ]
    pad = 0.35
    tmin = min(cmp_tr) - pad if cmp_tr else 0.0
    tmax = max(cmp_tr) + pad if cmp_tr else 1.0
    xs = [dx for _c, dx, _t, _src in g + s]
    x0, x1 = (min(xs), max(xs)) if xs else (-50.0, 50.0)
    padx = max(1.0, 0.04 * (x1 - x0))
    fig, axes = plt.subplots(
        1, 2, figsize=(12.4, 5.4), facecolor="w", sharex=True, sharey=True, layout="constrained",
    )
    _draw_signed_tt(axes[0], g, filter_ppp_abs=abs_lo, title=left_title)
    _draw_signed_tt(axes[1], s, filter_ppp_abs=None, title=right_title)
    axes[0].set_xlim(x0 - padx, x1 + padx)
    axes[0].set_ylim(tmax, tmin)
    fig.suptitle(suptitle, fontsize=12)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")


def plot_ttimes(md: Model, d: Path, shoot: list[ShotRay], out_png: Path) -> None:
    g = parse_syn_file(d / "syn0.dat") + parse_syn_file(d / "syn6.dat") + parse_syn_file(d / "syn78.dat")
    s = shoot_recs(shoot)
    plot_ttimes_signed(
        g, s, out_png,
        left_title="图论 tt_forward",
        right_title="共 p 射击（转折续至枝）",
        suptitle=(
            f"折合走时  沉积{SED_H:.0f} km  {md.name}  "
            f"盖层底Vp={LID_VP:.2f}  Vs0={md.vs0:.2f}"
        ),
    )


def match_dt(
    g: list[tuple[int, float, float, float]],
    shoot: list[ShotRay],
    dx_tol: float = 0.6,
) -> list[tuple[int, float, float, float, float, float]]:
    """Δt = t_图论 − t_共p。同一台、同一有符号偏移配对。"""
    curves: dict[tuple[int, float], list[tuple[float, float]]] = {}
    for c, dx, tt, src in g:
        curves.setdefault((c, round(src, 3)), []).append((dx, tt))
    for k in curves:
        curves[k].sort()
    rows: list[tuple[int, float, float, float, float, float]] = []
    for r in shoot:
        pts = curves.get((r.code, round(r.obs_x, 3)), [])
        if not pts:
            continue
        xs = np.array([p[0] for p in pts])
        ts = np.array([p[1] for p in pts])
        j = int(np.argmin(np.abs(xs - r.offset)))
        if abs(float(xs[j]) - r.offset) > dx_tol:
            continue
        tg = float(ts[j])
        rows.append((r.code, float(r.offset), tg - r.t, tg, r.t, r.obs_x))
    return rows


def dt_stats(rows: list[tuple]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for code, name, *_ in PHASES:
        d = np.array([dt for c, _dx, dt, *_rest in rows if c == code])
        if d.size == 0:
            out[name] = {"n": 0, "mean_s": None, "rms_s": None, "maxabs_s": None}
            continue
        out[name] = {
            "n": int(d.size),
            "mean_s": float(d.mean()),
            "rms_s": float(np.sqrt(np.mean(d * d))),
            "maxabs_s": float(np.max(np.abs(d))),
        }
        print(
            f"  Δt {name}: n={d.size}  mean={d.mean():+.4f} s  "
            f"rms={out[name]['rms_s']:.4f} s  max|Δt|={out[name]['maxabs_s']:.4f} s"
        )
    return out


def plot_dt(md: Model, rows: list[tuple], out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.0), facecolor="w", layout="constrained")
    ax.axhline(0.0, color="0.4", lw=0.9, zorder=1)
    ax.axvline(0.0, color="0.55", lw=0.7, zorder=1)
    for code, (color, fmt, name) in PHASE_TT_STYLE.items():
        by_src: dict[float, list[tuple[float, float]]] = {}
        for row in rows:
            c, dx, dt = row[0], row[1], row[2]
            src = row[5] if len(row) > 5 else 0.0
            if c != code:
                continue
            by_src.setdefault(round(src, 3), []).append((dx, dt))
        labeled = False
        for pts in by_src.values():
            pts.sort()
            kw = dict(color=color, ms=4, lw=1.1, zorder=3)
            if not labeled:
                kw["label"] = name
                labeled = True
            ax.plot([p[0] for p in pts], [p[1] for p in pts], fmt, **kw)
    ax.set_xlabel(r"有符号偏移 $x_{\mathrm{shot}}-x_{\mathrm{OBS}}$ (km)")
    ax.set_ylabel(r"$\Delta t = t_{\mathrm{graph}} - t_{\mathrm{p}}$ (s)")
    ax.set_title(f"走时差  沉积{SED_H:.0f} km  {md.name}")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", framealpha=0.9, fontsize=8)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")


def report_model(md: Model) -> None:
    if abs(md.vs0 - LID_VP) < 1e-6:
        rel = "Vs=Vp盖层底"
    elif md.vs0 < LID_VP:
        rel = "Vs<Vp盖层底"
    else:
        rel = "Vs>Vp盖层底"
    print(
        f"{md.name}: 沉积 {SED_H:.0f} km  盖层底 Vp={LID_VP:.3f}  "
        f"面下 Vp={md.crust_vp:.2f}  Vs0={md.vs0:.2f}  {rel}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fwd", action="store_true")
    ap.add_argument("--only", choices=("slow", "equal", "fast"), default=None)
    args = ap.parse_args()
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    print(f"Z_CONV={ZC}  SED_H={SED_H}  LID_VP={LID_VP:.3f} (同 3 km 盖层梯度)")
    for md in models(args.only):
        report_model(md)
        d = write_case(md)
        if not args.skip_fwd:
            tt_forward(md)
        shoot = shoot_all(md)
        print(f"  shoot n={len(shoot)}  by code " +
              ", ".join(f"{c}:{sum(1 for r in shoot if r.code==c)}" for c, *_ in PHASES))
        plot_rays(md, shoot, d, HERE / f"check_rays_{md.name}.png")
        plot_ttimes(md, d, shoot, HERE / f"check_ttimes_{md.name}.png")
        g = parse_syn_file(d / "syn0.dat") + parse_syn_file(d / "syn6.dat") + parse_syn_file(d / "syn78.dat")
        drows = match_dt(g, shoot)
        print(f"  matched n={len(drows)}")
        dt_stats(drows)
        plot_dt(md, drows, HERE / f"check_dt_{md.name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
