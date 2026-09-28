#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""6+12 Vs 段：真/初/反 Vs，莫霍，走时残差。"""

from __future__ import annotations

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
sys.path.insert(0, str(ROOT.parents[1].parent / "water_inv"))
sys.path.insert(0, str(ROOT.parents[1].parent / "ps_fwd"))
import inv_grid as g  # noqa: E402
import make_inv_612 as case612  # noqa: E402
import make_ppp_psp_inv_case as m2  # noqa: E402
import plot_rugged_inv as pr  # noqa: E402
from check_joint import mask_crust, recs_for_draw, region_rms, ttrms  # noqa: E402
from check_ps_fwd import draw_ps_rays  # noqa: E402
from make_lvz import BEL_LVZ  # noqa: E402

g.z_conv = case612.z_conv

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

WORK = HERE / "inv_612"
XLO, XHI = 10.0, 140.0
VCLIM = (1.8, 4.8)
DCLIM = 0.80


def _load_xz(path: Path) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        a = ln.split()
        if len(a) >= 2:
            out.append((float(a[0]), float(a[1])))
    return out


def _z_at(xz: list[tuple[float, float]], x: float) -> float:
    return float(np.interp(x, [p[0] for p in xz], [p[1] for p in xz]))


def _moho_rms(rec: list[tuple[float, float]], true: list[tuple[float, float]]) -> float:
    s = n = 0
    for x, zt in true:
        if not (XLO <= x <= XHI):
            continue
        d = _z_at(rec, x) - zt
        s += d * d
        n += 1
    return math.sqrt(s / n) if n else float("nan")


def _style(ax, title, moho=None, rec_peak=None, rays=None, recs=None, legend=False, true_moho=None):
    if rays and recs:
        draw_ps_rays(
            ax,
            rays,
            recs,
            thin=True,
            mark_conv=False,
            z_conv=case612.z_conv,
            codes=(6, 12),
        )
        if legend:
            ax.legend(loc="lower right", framealpha=0.88, fontsize=6, ncol=2)
    xs_l = np.linspace(XLO, XHI, 200)
    ax.plot(xs_l, [case612.z_conv(x) for x in xs_l], "k-.", lw=0.8, zorder=2)
    ax.plot(xs_l, [g.H] * len(xs_l), "k--", lw=0.6, zorder=2)
    if true_moho:
        ax.plot([p[0] for p in true_moho], [p[1] for p in true_moho], "k:", lw=1.2, zorder=3)
    if moho:
        ax.plot([p[0] for p in moho], [p[1] for p in moho], color="#c44e8a", lw=1.4, zorder=4)
    ax.plot(list(case612.OBS_XS), [g.H] * len(case612.OBS_XS), "k^", ms=6, zorder=7)
    ax.plot(BEL_LVZ["x0"], BEL_LVZ["z0"], "kx", ms=8, mew=1.4, zorder=8)
    if rec_peak is not None:
        ax.plot(rec_peak[0], rec_peak[1], "w+", ms=9, mew=1.4, zorder=8)
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(16, 0)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)


def _below_peak(xs, zs, vel, ref=None, *, xlo=None, xhi=None, zlo=None, zhi=None):
    xlo = XLO if xlo is None else xlo
    xhi = XHI if xhi is None else xhi
    best = (1e9, float("nan"), float("nan"))
    for i, x in enumerate(xs):
        if not (xlo <= x <= xhi):
            continue
        zi = case612.z_conv(x)
        for k, z in enumerate(zs):
            if z < zi - 1e-9 or z > 11.0:
                continue
            if zlo is not None and z < zlo:
                continue
            if zhi is not None and z > zhi:
                continue
            v = vel[i][k] if ref is None else vel[i][k] - ref[i][k]
            if v < best[0]:
                best = (v, x, z)
    return best


def _max_z(rays, recs, code: int) -> tuple[int, float]:
    zm = 0.0
    n = 0
    if not rays or not recs:
        return 0, float("nan")
    for rec, (_xs, zs) in zip(recs, rays):
        if rec[0] != code or not zs:
            continue
        n += 1
        zm = max(zm, max(zs))
    return n, zm


def _load_rays(ray_p: Path, syn_p: Path):
    if not ray_p.is_file() or not syn_p.is_file():
        return None, None
    return pr.parse_rays(ray_p), recs_for_draw(syn_p)


def _vs_pair(folder: Path):
    xs, zs, t_raw = m2.parse_smesh(folder / "true_vs.smesh")
    _, _, s_raw = m2.parse_smesh(folder / "start_vs.smesh")
    _, _, r_raw = m2.parse_smesh(folder / "rec_vs.smesh")
    return (
        xs,
        zs,
        mask_crust(xs, zs, t_raw),
        mask_crust(xs, zs, s_raw),
        mask_crust(xs, zs, r_raw),
    )


def main() -> int:
    xs, zs, t_vs, s_vs, r_vs = _vs_pair(WORK)
    true_m = _load_xz(WORK / "moho_true.refl")
    start_m = _load_xz(WORK / "moho.refl")
    rec_m = _load_xz(WORK / "rec_moho.refl")
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))

    print("Vs RMS vs true")
    print(
        f"  start  lid {region_rms(xs, zs, s_vs, t_vs, lid=True):.4f}  "
        f"below {region_rms(xs, zs, s_vs, t_vs, lid=False):.4f}"
    )
    print(
        f"  rec    lid {region_rms(xs, zs, r_vs, t_vs, lid=True):.4f}  "
        f"below {region_rms(xs, zs, r_vs, t_vs, lid=False):.4f}"
    )
    _, _, t_raw = m2.parse_smesh(WORK / "true_vs.smesh")
    _, _, s_raw = m2.parse_smesh(WORK / "start_vs.smesh")
    _, _, r_raw = m2.parse_smesh(WORK / "rec_vs.smesh")
    box = dict(xlo=48.0, xhi=68.0, zlo=6.2, zhi=9.8)
    tv, tx, tz = _below_peak(xs, zs, t_raw, **box)
    rv, rx, rz = _below_peak(xs, zs, r_raw, **box)
    dv, dx, dz = _below_peak(xs, zs, r_raw, s_raw, **box)
    ev, ex, ez = _below_peak(xs, zs, r_raw, s_raw)
    print(
        f"  LVZ 窗  true ({BEL_LVZ['x0']:.0f},{BEL_LVZ['z0']:.1f})  "
        f"true-min ({tx:.1f},{tz:.1f})={tv:.3f}  "
        f"rec-min ({rx:.1f},{rz:.1f})={rv:.3f}  "
        f"rec-start ({dx:.1f},{dz:.1f})={dv:+.3f}  "
        f"Δ=({dx - BEL_LVZ['x0']:+.1f},{dz - BEL_LVZ['z0']:+.1f})"
    )
    print(f"  全局 rec-start 最低 ({ex:.1f},{ez:.1f})={ev:+.3f}  （边缘假异常）")
    print("Moho RMS vs true (km)  (可动，初值 moho.refl)")
    print(f"  start  {_moho_rms(start_m, true_m):.4f}")
    print(f"  rec    {_moho_rms(rec_m, true_m):.4f}")
    syn_t, syn_s, syn_r = WORK / "syn_inv.dat", WORK / "syn_start.dat", WORK / "syn_rec.dat"
    if syn_t.is_file() and syn_s.is_file() and syn_r.is_file():
        for code in (6, 12):
            print(
                f"  tt{code}  start {ttrms(syn_t, syn_s, code):.4f} s  "
                f"rec {ttrms(syn_t, syn_r, code):.4f} s"
            )

    other = HERE / "inv_1213"
    if (other / "rec_vs.smesh").is_file() and (other / "rec_moho.refl").is_file():
        oxs, ozs, _, _, o_vs = _vs_pair(other)
        o_m = _load_xz(other / "rec_moho.refl")
        print("对照 inv_1213（仅 12/13）")
        print(f"  below Vs RMS  {region_rms(oxs, ozs, o_vs, t_vs, lid=False):.4f}")
        print(f"  Moho RMS      {_moho_rms(o_m, true_m):.4f}")

    rays_t, recs_t = _load_rays(WORK / "rays_true.dat", syn_t)
    rays_s, recs_s = _load_rays(WORK / "rays_start.dat", syn_s)
    rays_r, recs_r = _load_rays(WORK / "rays_rec.dat", syn_r)
    print("射线最大深度")
    for lab, rays, recs in (
        ("true", rays_t, recs_t),
        ("start", rays_s, recs_s),
        ("rec", rays_r, recs_r),
    ):
        n6, z6 = _max_z(rays, recs, 6)
        n12, z12 = _max_z(rays, recs, 12)
        print(f"  {lab:5s}  n6={n6} zmax={z6:.2f}  n12={n12} zmax={z12:.2f}")

    fig, axes = plt.subplots(2, 3, figsize=(16.4, 7.6), facecolor="w", layout="constrained")
    panels = (
        (axes[0, 0], t_vs, "RdYlBu_r", *VCLIM, "真 Vs + 真射线", true_m, rays_t, recs_t),
        (axes[0, 1], s_vs, "RdYlBu_r", *VCLIM, "初 Vs + 初射线", start_m, rays_s, recs_s),
        (axes[0, 2], r_vs, "RdYlBu_r", *VCLIM, "反 Vs + 反射线", rec_m, rays_r, recs_r),
        (axes[1, 0], s_vs - t_vs, "RdBu_r", -DCLIM, DCLIM, "初 − 真", start_m, rays_s, recs_s),
        (axes[1, 1], r_vs - t_vs, "RdBu_r", -DCLIM, DCLIM, "反 − 真", rec_m, rays_r, recs_r),
        (axes[1, 2], r_vs - s_vs, "RdBu_r", -DCLIM, DCLIM, "反 − 初", rec_m, rays_r, recs_r),
    )
    last = None
    for ax, arr, cmap, vmin, vmax, title, moho, rays, recs in panels:
        ax.imshow(
            arr,
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
            zorder=0,
        )
        peak = (dx, dz) if ax in (axes[0, 2], axes[1, 1], axes[1, 2]) else None
        _style(
            ax,
            title,
            moho,
            rec_peak=peak,
            rays=rays,
            recs=recs,
            legend=ax is axes[0, 2],
            true_moho=true_m,
        )
        last = ax.images[-1]
        if ax in (axes[0, 0], axes[1, 0]):
            ax.set_ylabel("深度 (km)")
        if ax in axes[1]:
            ax.set_xlabel("x (km)")
    fig.colorbar(axes[0, 2].images[0], ax=axes[0, :].ravel().tolist(), shrink=0.86, label="Vs (km/s)")
    fig.colorbar(last, ax=axes[1, :].ravel().tolist(), shrink=0.86, label="ΔVs (km/s)")
    fig.suptitle(
        "lvz2d / inv_612   5 OBS · 可动莫霍   转换面≤4 km · 面上Vp=面下Vs=3.5 · 壳底7.4",
        fontsize=12,
    )
    out = WORK / "check_inv_612.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
