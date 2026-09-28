#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""倾斜转换面正演（thin2km_incline）。

海底仍平 H=2 km。转换面 z = ZC0 + SLOPE*(x-XREF)，沉积厚度随 x 变。
图论：现成 tt_forward + 二维网格。
二维积分：与 conv.refl 同一条折线，当地法向 Snell（不共 p）。
--shoot 1d 可退回中点一维共 p 近似。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "thin2km"))
sys.path.insert(0, str(HERE.parent.parent.parent / "ps_inv"))
sys.path.insert(0, str(HERE.parent.parent.parent / "ps_fwd"))
import plot_ps_inv_models as psp  # noqa: E402
import run_thin2km as t  # noqa: E402
import ray2d as r2  # noqa: E402

H, ZMAX = t.H, t.ZMAX
V_WATER, V_AIR, KAPPA = t.V_WATER, t.V_AIR, t.KAPPA
SED0, SED_GRAD = t.SED0, t.SED_GRAD
SHOT_Z, OBS_XS, SHOT_XS = t.SHOT_Z, t.OBS_XS, t.SHOT_XS
XMIN, XMAX, DX, DZ = t.XMIN, t.XMAX, t.DX, t.DZ
VRED = t.VRED
P_COLOR, S_COLOR = t.P_COLOR, t.S_COLOR
PHASES = t.PHASES
PHASE_TT_STYLE = t.PHASE_TT_STYLE

XREF = 50.0
ZC0 = 4.0
SLOPE = 0.03  # dz/dx；对齐到 z 网格，避免 -X 取路径时在面下成环
DZ_IFACE = DZ


def z_conv_smooth(x: float) -> float:
    return ZC0 + SLOPE * (x - XREF)


def z_conv(x: float) -> float:
    z = z_conv_smooth(x)
    k = int(round(z / DZ_IFACE))
    z = k * DZ_IFACE
    z = max(H + DZ_IFACE, min(ZMAX - 2.0, z))
    return z


def lid_vp_at(x: float) -> float:
    return SED0 + SED_GRAD * max(z_conv(x) - H, 0.2)


def _xs() -> list[float]:
    return t._xs()


def _zs() -> list[float]:
    return t._zs()


@dataclass(frozen=True)
class Model:
    name: str
    crust_vp: float

    @property
    def vs0(self) -> float:
        return self.crust_vp / KAPPA

    def vp(self, x: float, z: float) -> float:
        zc = z_conv(x)
        if z <= H + 1e-12:
            return V_WATER
        if z < zc:
            return SED0 + SED_GRAD * (z - H)
        return self.crust_vp + KAPPA * 0.12 * (z - zc)

    def vs(self, x: float, z: float) -> float:
        zc = z_conv(x)
        if z <= H + 1e-12:
            return V_WATER
        if z < zc:
            return self.vp(x, z) / KAPPA
        return self.vs0 + 0.12 * (z - zc)


def models(only: str | None = None) -> tuple[Model, ...]:
    allm = (
        Model("vslow", 4.00),  # Vs0≈2.31，比 slow 更慢
        Model("slow", 5.00),
        Model("equal", lid_vp_at(XREF) * KAPPA),
        Model("fast", 7.20),
    )
    if only:
        picked = tuple(m for m in allm if m.name == only)
        if not picked:
            raise SystemExit(f"unknown model {only!r}; choose vslow|slow|equal|fast")
        return picked
    return allm


@dataclass
class FlatView:
    """共 p 用：把中点处的倾斜界面当成水平层。"""

    crust_vp: float
    vs0: float
    zc: float

    def vp(self, z: float) -> float:
        if z <= H + 1e-12:
            return V_WATER
        if z < self.zc:
            return SED0 + SED_GRAD * (z - H)
        return self.crust_vp + KAPPA * 0.12 * (z - self.zc)

    def vs(self, z: float) -> float:
        if z <= H + 1e-12:
            return V_WATER
        if z < self.zc:
            return self.vp(z) / KAPPA
        return self.vs0 + 0.12 * (z - self.zc)


def write_smesh(path: Path, vfun) -> None:
    xs, zs = _xs(), _zs()
    lines = [
        f"{len(xs)} {len(zs)} {V_WATER:.4f} {V_AIR}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    for x in xs:
        lines.append(" ".join(f" {vfun(x, z):.4f}" for z in zs))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_conv(path: Path) -> None:
    path.write_text("".join(f"{x:.4f} {z_conv(x):.4f}\n" for x in _xs()), encoding="utf-8")


def write_case(md: Model) -> Path:
    d = HERE / md.name
    d.mkdir(parents=True, exist_ok=True)
    write_smesh(d / "vp.smesh", md.vp)
    write_smesh(d / "vs.smesh", md.vs)

    def mixed(x: float, z: float) -> float:
        return md.vs(x, z) if z >= z_conv(x) - 1e-9 else md.vp(x, z)

    write_smesh(d / "mixed.smesh", mixed)

    def vp_psx(x: float, z: float) -> float:
        if abs(z - z_conv(x)) <= 0.5 * DZ + 1e-12:
            return md.vs0
        return md.vp(x, z)

    write_smesh(d / "vp_psx.smesh", vp_psx)
    t.write_iface(d / "seafloor.refl", H)
    write_conv(d / "conv.refl")
    t.write_geom(d / "geom0.dat", (0,))
    t.write_geom(d / "geom6.dat", (6,))
    t.write_geom(d / "geom78.dat", (7, 8))
    return d


def tt_forward(md: Model) -> None:
    d = write_case(md)
    bin_p = HERE.parents[3] / "src" / "build-tomo2d" / "tt_forward"
    n = "-N8/8/0.8/8/1e-4/1e-5"
    wd = t.wsl_path(d)
    b = t.wsl_path(bin_p)
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
            print(f"WARN: tt_forward failed [{md.name}] {cmd[:80]}")
            continue


def p_window(fv: FlatView, base_s: bool, lid_up_s: bool) -> tuple[float, float] | None:
    lid = SED0 + SED_GRAD * max(fv.zc - H, 0.2)
    p_hi = 0.999 / lid
    if base_s:
        p_hi = min(p_hi, 0.999 / fv.vs0)
        vbot = fv.vs(ZMAX)
        p_lo = 1.0 / vbot + 1e-4
    else:
        p_hi = min(p_hi, 0.999 / fv.crust_vp)
        vbot = fv.vp(ZMAX)
        p_lo = 1.0 / vbot + 1e-4
    if lid_up_s:
        p_hi = min(p_hi, 0.999 / fv.vs(fv.zc - 1e-3))
    if p_hi <= p_lo + 1e-5:
        return None
    return p_lo, p_hi


def trace_phase(fv: FlatView, p: float, base_s: bool, lid_up_s: bool):
    down = t.integrate_no_turn(SHOT_Z, fv.zc, p, fv.vp)
    if down is None:
        return None
    if base_s:
        base = t.circle_turn(fv.vs0, 0.12, p, fv.zc)
    else:
        base = t.circle_turn(fv.crust_vp, KAPPA * 0.12, p, fv.zc)
    if base is None:
        return None
    vup = fv.vs if lid_up_s else fv.vp
    up = t.integrate_no_turn(fv.zc, H, p, vup)
    if up is None:
        return None
    xd, td, zd, xr_d = down
    xb, tb, z_turn, zb, xr_b = base
    xu, tu, zu, xr_u = up
    return xd + xb + xu, td + tb + tu, z_turn, {
        "d": (zd, xr_d), "b": (zb, xr_b), "u": (zu, xr_u),
    }


def _match(fv: FlatView, offset: float, base_s: bool, lid_up_s: bool):
    win = p_window(fv, base_s, lid_up_s)
    if win is None or offset < 1.0:
        return None
    p_lo, p_hi = win
    rows = []
    for p in np.linspace(p_lo, p_hi, 40):
        got = trace_phase(fv, float(p), base_s, lid_up_s)
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
        got = trace_phase(fv, p, base_s, lid_up_s)
        if got is None:
            continue
        err = abs(got[0] - offset)
        if best is None or err < best[0]:
            best = (err, p, got)
    if best is None:
        p, got = min(rows, key=lambda r: abs(r[1][0] - offset))
        best = (abs(got[0] - offset), p, got)
    err, p, got = best
    x, tt, zt, pack = got
    return None if err > 2.5 else (p, tt, zt, x, pack)


def shoot_all_1d(md: Model, shot_stride: int = 1) -> list[t.ShotRay]:
    out: list[t.ShotRay] = []
    shots = SHOT_XS[::shot_stride]
    for code, name, base_s, lid_up_s in PHASES:
        fv0 = FlatView(md.crust_vp, md.vs0, z_conv(XREF))
        win = p_window(fv0, base_s, lid_up_s)
        print(f"  {md.name} {name}: p {win}  (共 p 用各对中点 zc)")
        for ox in OBS_XS:
            for sx in shots:
                off = abs(ox - sx)
                fv = FlatView(md.crust_vp, md.vs0, z_conv(0.5 * (sx + ox)))
                hit = _match(fv, off, base_s, lid_up_s)
                if hit is None:
                    continue
                p, tt, zt, xest, pack = hit
                if abs(xest - off) > 2.5:
                    continue
                px, pz = t.assemble(sx, ox, pack)
                out.append(t.ShotRay(code, px, pz, p, tt, zt, sx - ox, lid_up_s, base_s, ox))
    return out


def conv_polyline() -> tuple[np.ndarray, np.ndarray]:
    xs = np.array(_xs(), dtype=float)
    zs = np.array([z_conv(x) for x in xs], dtype=float)
    return xs, zs


def shoot_all_2d(md: Model, shot_stride: int = 1) -> list[t.ShotRay]:
    """真二维：在与 conv.refl 相同的折线上 ODE + 当地 Snell。"""
    xs_ifc, zs_ifc = conv_polyline()
    spec = r2.Ray2dSpec.from_xy(md.crust_vp, md.vs0, xs_ifc, zs_ifc)
    out: list[t.ShotRay] = []
    shots = SHOT_XS[::shot_stride]
    ntot = len(OBS_XS) * len(shots)
    for code, name, base_s, lid_up_s in PHASES:
        n_ok = 0
        print(f"  {md.name} {name}: 二维射击 {ntot} 对 …", flush=True)
        for ox in OBS_XS:
            for sx in shots:
                got = r2.shoot_pair(spec, sx, ox, code)
                if got is None:
                    continue
                n_ok += 1
                out.append(
                    t.ShotRay(
                        code, got["xs"], got["zs"], abs(got["px0"]),
                        got["t"], got["zmax"], sx - ox, lid_up_s, base_s, ox,
                    )
                )
        print(f"    hit {n_ok}/{ntot}", flush=True)
    return out


def shoot_all(md: Model, shot_stride: int = 1, mode: str = "2d") -> list[t.ShotRay]:
    if mode == "1d":
        return shoot_all_1d(md, shot_stride)
    return shoot_all_2d(md, shot_stride)


def _is_s(x: float, z: float, code: int, xs: list[float], j: int, base_s: bool, lid_up_s: bool,
          zc_fun=None) -> bool:
    zc = (zc_fun or z_conv)(x)
    if z <= H + 1e-3:
        return False
    if z > zc + 1e-3:
        return base_s if code >= 0 else False
    going_home = abs(xs[j] - xs[-1]) < abs(xs[j] - xs[0])
    return bool(lid_up_s and going_home)


def draw_phase_rays(ax, rays: list[t.ShotRay], code: int, zc_fun=None) -> None:
    flags = {c: (bs, lu) for c, _n, bs, lu in PHASES}
    base_s, lid_up_s = flags[code]
    zf = zc_fun or z_conv
    for r in rays:
        if r.code != code:
            continue
        for j in range(len(r.xs) - 1):
            xm = 0.5 * (r.xs[j] + r.xs[j + 1])
            zm = 0.5 * (r.zs[j] + r.zs[j + 1])
            s = _is_s(xm, zm, code, r.xs, j, base_s, lid_up_s, zf)
            ax.plot(r.xs[j : j + 2], r.zs[j : j + 2],
                    color=S_COLOR if s else P_COLOR, lw=0.7, alpha=0.75, zorder=3)
    ax.scatter(list(OBS_XS), [H] * 5, marker="^", c="k", s=22, zorder=6)


def draw_graph_phase_rays(ax, segs, recs, code: int) -> None:
    flags = {c: (bs, lu) for c, _n, bs, lu in PHASES}
    base_s, lid_up_s = flags[code]
    n = min(len(segs), len(recs)) if recs else len(segs)
    pending_s: list[tuple[list[float], list[float]]] = []
    for i in range(n):
        rx, rz = segs[i]
        ci = recs[i][0] if recs and i < len(recs) else code
        bs, lu = flags.get(ci, (base_s, lid_up_s))
        cur_x = [rx[0]]
        cur_z = [rz[0]]
        cur_s = _is_s(rx[0], rz[0], ci, rx, 0, bs, lu) if len(rx) > 1 else False
        for j in range(len(rx) - 1):
            xm = 0.5 * (rx[j] + rx[j + 1])
            zm = 0.5 * (rz[j] + rz[j + 1])
            s1 = _is_s(xm, zm, ci, rx, j, bs, lu)
            if s1 == cur_s:
                cur_x.append(rx[j + 1])
                cur_z.append(rz[j + 1])
                continue
            if len(cur_x) >= 2:
                if cur_s:
                    pending_s.append((cur_x, cur_z))
                else:
                    ax.plot(cur_x, cur_z, color=P_COLOR, lw=0.5, ls="--", alpha=0.65, zorder=3)
            cur_x = [rx[j], rx[j + 1]]
            cur_z = [rz[j], rz[j + 1]]
            cur_s = s1
        if len(cur_x) >= 2:
            if cur_s:
                pending_s.append((cur_x, cur_z))
            else:
                ax.plot(cur_x, cur_z, color=P_COLOR, lw=0.5, ls="--", alpha=0.65, zorder=3)
    for sx, sz in pending_s:
        ax.plot(sx, sz, color=S_COLOR, lw=0.85, ls="-", alpha=0.85, zorder=5)
    ax.scatter(list(OBS_XS), [H] * 5, marker="^", c="k", s=22, zorder=6)


def _draw_iface(ax, smooth: bool = False) -> None:
    xs = [x for x in _xs() if 18 <= x <= 82]
    ax.plot(xs, [H] * len(xs), color="0.15", ls="--", lw=0.8, zorder=2)
    zfun = z_conv_smooth if smooth else z_conv
    ax.plot(xs, [zfun(x) for x in xs], color="0.15", ls="-.", lw=1.0, zorder=2)


def plot_rays(md: Model, shoot: list[t.ShotRay], d: Path, out_png: Path, shoot_title: str) -> None:
    sys.path.insert(0, str(HERE.parent.parent))
    from make_ppp_psp_inv_case import parse_smesh  # noqa: E402

    xs, zs, vel = parse_smesh(d / "vp.smesh")
    grid = psp._grid(xs, zs, vel)
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    fig, axes = plt.subplots(4, 2, figsize=(13.2, 14.5), facecolor="w", layout="constrained")
    names = {0: "PPP 0", 7: "PPS 7", 6: "PSP 6", 8: "PSS 8"}
    rayfiles = {0: d / "rays0.dat", 6: d / "rays6.dat", 7: d / "rays78.dat", 8: d / "rays78.dat"}
    recs78 = t.parse_syn_file(d / "syn78.dat")
    recs6 = t.parse_syn_file(d / "syn6.dat")
    recs0 = t.parse_syn_file(d / "syn0.dat")
    recmap = {0: recs0, 6: recs6, 7: recs78, 8: recs78}

    for row, (code, name, _a, _b) in enumerate(PHASES):
        for col, title_pref in enumerate(("图论 tt_forward", shoot_title)):
            ax = axes[row, col]
            ax.imshow(grid, extent=extent, cmap="RdYlBu_r", vmin=1.2, vmax=6.2,
                      aspect="auto", interpolation="nearest", zorder=0)
            _draw_iface(ax, smooth=False)
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
                zm = t._zmax_segs(segs)
                draw_graph_phase_rays(ax, segs, recs, code)
                ax.set_title(f"{name}  {title_pref}  zmax={zm:.2f} km")
                ax.set_ylabel("深度 (km)")
            else:
                draw_phase_rays(ax, shoot, code, z_conv)
                n = sum(1 for r in shoot if r.code == code)
                ax.set_title(f"{name}  {title_pref}  zmax={t._zmax_rays(shoot, code):.2f} km  n={n}")
            if row == 3:
                ax.set_xlabel("模型距离 (km)")
    fig.suptitle(
        f"倾斜转换面  dz/dx={SLOPE:g}  {md.name}  "
        f"zc(50)={ZC0:g}  Vs0={md.vs0:.2f}  盖层底Vp(50)={lid_vp_at(XREF):.2f}",
        fontsize=12,
    )
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"wrote {out_png}")


def plot_ttimes(md: Model, d: Path, shoot: list[t.ShotRay], out_png: Path, shoot_title: str) -> None:
    g = t.parse_syn_file(d / "syn0.dat") + t.parse_syn_file(d / "syn6.dat") + t.parse_syn_file(d / "syn78.dat")
    s = t.shoot_recs(shoot)
    t.plot_ttimes_signed(
        g, s, out_png,
        left_title="图论 tt_forward",
        right_title=shoot_title,
        suptitle=f"折合走时  倾斜面 dz/dx={SLOPE:g}  {md.name}  Vs0={md.vs0:.2f}",
    )


def plot_dt(md: Model, rows, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.0), facecolor="w", layout="constrained")
    ax.axhline(0.0, color="0.4", lw=0.9, zorder=1)
    ax.axvline(0.0, color="0.55", lw=0.7, zorder=1)
    for code, (color, fmt, name) in PHASE_TT_STYLE.items():
        by_src: dict[float, list[tuple[float, float]]] = {}
        for row in rows:
            c, dx, dtt = row[0], row[1], row[2]
            src = row[5] if len(row) > 5 else 0.0
            if c != code:
                continue
            by_src.setdefault(round(src, 3), []).append((dx, dtt))
        labeled = False
        for pts in by_src.values():
            pts.sort()
            kw = dict(color=color, ms=4, lw=1.1, zorder=3)
            if not labeled:
                kw["label"] = name
                labeled = True
            ax.plot([p[0] for p in pts], [p[1] for p in pts], fmt, **kw)
    ax.set_xlabel(r"有符号偏移 $x_{\mathrm{shot}}-x_{\mathrm{OBS}}$ (km)")
    ax.set_ylabel(r"$\Delta t = t_{\mathrm{graph}} - t_{\mathrm{2d}}$ (s)")
    ax.set_title(f"走时差  倾斜面  {md.name}")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", framealpha=0.9, fontsize=8)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fwd", action="store_true")
    ap.add_argument("--only", choices=("vslow", "slow", "equal", "fast"), default=None)
    ap.add_argument("--shoot", choices=("2d", "1d"), default="2d",
                    help="2d=运动学 ODE；1d=中点一维共 p")
    args = ap.parse_args()
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    print(
        f"incline ZC0={ZC0}  SLOPE={SLOPE}  zc(20)={z_conv(20):.2f}  "
        f"zc(50)={z_conv(50):.2f}  zc(80)={z_conv(80):.2f}  "
        f"lidVp(50)={lid_vp_at(XREF):.3f}"
    )
    for md in models(args.only):
        print(
            f"{md.name}: Vs0={md.vs0:.2f}  crustVp={md.crust_vp:.2f}  "
            f"lidVp(30)={lid_vp_at(30):.2f}  lidVp(70)={lid_vp_at(70):.2f}"
        )
        d = write_case(md)
        if not args.skip_fwd:
            tt_forward(md)
        shoot = shoot_all(md, mode=args.shoot)
        shoot_title = "二维射线积分" if args.shoot == "2d" else "共 p（中点一维近似）"
        tag = md.name if args.shoot == "2d" else f"{md.name}_1d"
        print(
            "  shoot n=" + str(len(shoot)) + "  by code " +
            ", ".join(f"{c}:{sum(1 for r in shoot if r.code==c)}" for c, *_ in PHASES)
        )
        plot_rays(md, shoot, d, HERE / f"check_rays_{tag}.png", shoot_title)
        plot_ttimes(md, d, shoot, HERE / f"check_ttimes_{tag}.png", shoot_title)
        g = t.parse_syn_file(d / "syn0.dat") + t.parse_syn_file(d / "syn6.dat") + t.parse_syn_file(d / "syn78.dat")
        drows = t.match_dt(g, shoot)
        print(f"  matched n={len(drows)}")
        t.dt_stats(drows)
        plot_dt(md, drows, HERE / f"check_dt_{tag}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
