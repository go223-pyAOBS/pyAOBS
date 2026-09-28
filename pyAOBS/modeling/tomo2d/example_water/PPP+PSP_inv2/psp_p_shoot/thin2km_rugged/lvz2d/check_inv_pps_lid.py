#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PPS(7) vs PPS 盖层 SS 多次(10) vs 7+10：面上 Vs / 盖层 LVZ。"""

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
from check_joint import mask_crust, recs_for_draw, ttrms  # noqa: E402
from check_ps_fwd import draw_ps_rays  # noqa: E402
from make_lvz import LID_LVZ  # noqa: E402

g.z_conv = case612.z_conv

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

WORK = HERE / "inv_pps_lid"
XLO, XHI = 15.0, 95.0
VCLIM = (1.8, 3.6)
DCLIM = 0.50
JOBS = (
    ("7", (7,), "PPS 7"),
    ("10", (10,), "PPS-SS 10"),
    ("710", (7, 10), "7+10"),
)


def region_rms(xs, zs, a, b, *, lid: bool | None) -> float:
    s = n = 0
    for i, x in enumerate(xs):
        if not (XLO <= x <= XHI):
            continue
        zi = case612.z_conv(x)
        for k, z in enumerate(zs):
            if z < g.H - 1e-9:
                continue
            if lid is True and z >= zi - 1e-9:
                continue
            if lid is False and z < zi - 1e-9:
                continue
            d = a[k, i] - b[k, i]
            if np.isnan(d):
                continue
            s += float(d * d)
            n += 1
    return math.sqrt(s / n) if n else float("nan")



def _style(ax, title, rays=None, recs=None, codes=(), rec_peak=None, legend=False):
    if rays and recs:
        draw_ps_rays(
            ax,
            rays,
            recs,
            thin=True,
            mark_conv=False,
            z_conv=case612.z_conv,
            codes=codes,
        )
        if legend:
            ax.legend(loc="lower right", framealpha=0.88, fontsize=6, ncol=2)
    xs_l = np.linspace(XLO, XHI, 200)
    ax.plot(xs_l, [case612.z_conv(x) for x in xs_l], "k-.", lw=0.8, zorder=2)
    ax.plot(xs_l, [g.H] * len(xs_l), "k--", lw=0.6, zorder=2)
    ax.plot(list(case612.OBS_XS), [g.H] * len(case612.OBS_XS), "k^", ms=6, zorder=7)
    ax.plot(LID_LVZ["x0"], LID_LVZ["z0"], "kx", ms=8, mew=1.4, zorder=8)
    if rec_peak is not None:
        ax.plot(rec_peak[0], rec_peak[1], "w+", ms=9, mew=1.4, zorder=8)
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(6.2, 0)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)


LVZ_BOX = dict(xlo=32.0, xhi=48.0, zlo=2.55, zhi=3.55)


def _lid_peak(xs, zs, vel, ref=None, *, xlo=None, xhi=None, zlo=None, zhi=None):
    xlo = XLO if xlo is None else xlo
    xhi = XHI if xhi is None else xhi
    best = (1e9, float("nan"), float("nan"))
    for i, x in enumerate(xs):
        if not (xlo <= x <= xhi):
            continue
        zi = case612.z_conv(x)
        for k, z in enumerate(zs):
            if z <= g.H + 1e-9 or z >= zi - 1e-9:
                continue
            if zlo is not None and z < zlo:
                continue
            if zhi is not None and z > zhi:
                continue
            v = vel[i][k] if ref is None else vel[i][k] - ref[i][k]
            if v < best[0]:
                best = (v, x, z)
    return best


def _v_at(xs, zs, vel, x0: float, z0: float) -> tuple[float, float, float]:
    i = min(range(len(xs)), key=lambda j: abs(xs[j] - x0))
    k = min(range(len(zs)), key=lambda j: abs(zs[j] - z0))
    return float(vel[i][k]), float(xs[i]), float(zs[k])


def region_rms_box(xs, zs, a, b, box: dict) -> float:
    s = n = 0
    for i, x in enumerate(xs):
        if not (box["xlo"] <= x <= box["xhi"]):
            continue
        for k, z in enumerate(zs):
            if z < box["zlo"] or z > box["zhi"]:
                continue
            d = a[k, i] - b[k, i]
            if np.isnan(d):
                continue
            s += float(d * d)
            n += 1
    return math.sqrt(s / n) if n else float("nan")


def _load_rays(ray_p: Path, syn_p: Path):
    if not ray_p.is_file() or not syn_p.is_file():
        return None, None
    return pr.parse_rays(ray_p), recs_for_draw(syn_p)


def main() -> int:
    xs, zs, t_raw = m2.parse_smesh(WORK / "true_vs.smesh")
    _, _, s_raw = m2.parse_smesh(WORK / "start_vs.smesh")
    t_vs = mask_crust(xs, zs, t_raw)
    s_vs = mask_crust(xs, zs, s_raw)
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    tv, tx, tz = _lid_peak(xs, zs, t_raw, **LVZ_BOX)
    vt0, xt0, zt0 = _v_at(xs, zs, t_raw, LID_LVZ["x0"], LID_LVZ["z0"])
    vs0, _, _ = _v_at(xs, zs, s_raw, LID_LVZ["x0"], LID_LVZ["z0"])
    print("Vs RMS vs true (lid only; below frozen)")
    print(
        f"  start  lid {region_rms(xs, zs, s_vs, t_vs, lid=True):.4f}  "
        f"below {region_rms(xs, zs, s_vs, t_vs, lid=False):.4f}  "
        f"LVZ窗 {region_rms_box(xs, zs, s_vs, t_vs, LVZ_BOX):.4f}"
    )
    print(
        f"  LVZ 真 ({LID_LVZ['x0']:.0f},{LID_LVZ['z0']:.2f}) amp_Vp={LID_LVZ['amp']:+.2f}  "
        f"true@node ({xt0:.1f},{zt0:.2f})={vt0:.3f}  start={vs0:.3f}  "
        f"true-min窗 ({tx:.1f},{tz:.2f})={tv:.3f}"
    )

    recs = {}
    peaks = {}
    for tag, codes, lab in JOBS:
        rec_p = WORK / f"rec_vs_{tag}.smesh"
        if not rec_p.is_file():
            print(f"  skip {tag}: missing {rec_p.name}")
            continue
        _, _, r_raw = m2.parse_smesh(rec_p)
        r_vs = mask_crust(xs, zs, r_raw)
        recs[tag] = (r_vs, r_raw, codes, lab)
        dv, dx, dz = _lid_peak(xs, zs, r_raw, s_raw, **LVZ_BOX)
        vr0, _, _ = _v_at(xs, zs, r_raw, LID_LVZ["x0"], LID_LVZ["z0"])
        peaks[tag] = (dv, dx, dz)
        print(
            f"  {lab:10s}  lid {region_rms(xs, zs, r_vs, t_vs, lid=True):.4f}  "
            f"LVZ窗 {region_rms_box(xs, zs, r_vs, t_vs, LVZ_BOX):.4f}  "
            f"@LVZ {vr0:.3f} (真{vt0:.3f} 初{vs0:.3f} Δ初{vr0 - vs0:+.3f})  "
            f"rec-start窗 ({dx:.1f},{dz:.2f})={dv:+.3f}  "
            f"Δ=({dx - LID_LVZ['x0']:+.1f},{dz - LID_LVZ['z0']:+.2f})"
        )
        syn_t = WORK / f"syn_inv_{tag}.dat"
        syn_s = WORK / f"syn_start_{tag}.dat"
        syn_r = WORK / f"syn_rec_{tag}.dat"
        if syn_t.is_file() and syn_s.is_file() and syn_r.is_file():
            for code in codes:
                print(
                    f"    tt{code}  start {ttrms(syn_t, syn_s, code):.4f} s  "
                    f"rec {ttrms(syn_t, syn_r, code):.4f} s"
                )

    fig, axes = plt.subplots(2, 4, figsize=(18.4, 7.2), facecolor="w", layout="constrained")
    rays_t, recs_t = _load_rays(WORK / "rays_true_7.dat", WORK / "syn_inv_7.dat")
    panels_top = [
        (axes[0, 0], t_vs, "RdYlBu_r", *VCLIM, "真 Vs + PPS 射线", rays_t, recs_t, (7,), None),
    ]
    panels_bot = [
        (axes[1, 0], s_vs - t_vs, "RdBu_r", -DCLIM, DCLIM, "初 − 真", None, None, (), None),
    ]
    col = 1
    for tag, codes, lab in JOBS:
        if tag not in recs:
            axes[0, col].axis("off")
            axes[1, col].axis("off")
            col += 1
            continue
        r_vs, _r_raw, codes, lab = recs[tag]
        rays_r, recs_r = _load_rays(WORK / f"rays_rec_{tag}.dat", WORK / f"syn_rec_{tag}.dat")
        pk = peaks[tag]
        panels_top.append(
            (axes[0, col], r_vs, "RdYlBu_r", *VCLIM, f"{lab} 反 Vs", rays_r, recs_r, codes, (pk[1], pk[2]))
        )
        panels_bot.append(
            (
                axes[1, col],
                r_vs - t_vs,
                "RdBu_r",
                -DCLIM,
                DCLIM,
                f"{lab} − 真",
                rays_r,
                recs_r,
                codes,
                (pk[1], pk[2]),
            )
        )
        col += 1

    last = None
    for ax, arr, cmap, vmin, vmax, title, rays, recs_d, codes, peak in panels_top + panels_bot:
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
        _style(ax, title, rays, recs_d, codes, rec_peak=peak, legend=ax is axes[0, 3])
        last = ax.images[-1]
        if ax in (axes[0, 0], axes[1, 0]):
            ax.set_ylabel("深度 (km)")
        if ax in axes[1]:
            ax.set_xlabel("x (km)")
    fig.colorbar(axes[0, 1].images[0] if axes[0, 1].images else axes[0, 0].images[0],
                 ax=axes[0, :].ravel().tolist(), shrink=0.86, label="Vs (km/s)")
    fig.colorbar(last, ax=axes[1, :].ravel().tolist(), shrink=0.86, label="ΔVs (km/s)")
    fig.suptitle(
        "lvz2d / inv_pps_lid   冻真 Vp · 冻面下 · 反盖层 Vs   "
        "PPS(7) vs 盖层SS多次(10) vs 7+10",
        fontsize=12,
    )
    out = WORK / "check_inv_pps_lid.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
