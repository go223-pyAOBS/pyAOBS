#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""公平两步：各自 PPP，再用 rec_vp/κ 反 Vs；观测/初值走时各自正演。"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

HERE = Path(__file__).resolve().parent
DUAL = HERE / "inv_graph6k_hot"
SINGLE = HERE / "inv_graph6_hot"
sys.path.insert(0, str(HERE / "inv_2d"))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1].parent / "water_inv"))
sys.path.insert(0, str(HERE.parents[1].parent / "ps_fwd"))
import inv_grid as g  # noqa: E402
import make_ppp_psp_inv_case as m2  # noqa: E402
import plot_rugged_inv as pr  # noqa: E402
from check_ps_fwd import draw_ps_rays  # noqa: E402
from check_water_inv import parse_picks  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DCLIM = 0.60
XLO, XHI = 30.0, 70.0


def mask_region(xs, zs, vel, *, include_lid: bool):
    arr = np.asarray(vel, float).T
    for j, z in enumerate(zs):
        for i, x in enumerate(xs):
            zmin = g.H if include_lid else g.z_conv(x)
            if z < zmin - 1e-9:
                arr[j, i] = np.nan
    return arr


def load_vs(folder: Path, *, include_lid: bool):
    xs, zs, true = m2.parse_smesh(folder / "true_vs.smesh")
    _, _, start = m2.parse_smesh(folder / "start_vs.smesh")
    rec_p = folder / "rec_vs.smesh"
    _, _, rec = m2.parse_smesh(rec_p)
    return (
        xs,
        zs,
        mask_region(xs, zs, true, include_lid=include_lid),
        mask_region(xs, zs, start, include_lid=include_lid),
        mask_region(xs, zs, rec, include_lid=include_lid),
    )


def ttrms(obs: Path, pred: Path) -> float:
    o = parse_picks(obs.read_text(encoding="utf-8"))
    p = parse_picks(pred.read_text(encoding="utf-8"))
    key = lambda t: (round(t[4], 3), round(t[1], 3), int(t[0]))
    md = {key(x): x[3] for x in p}
    ds = [md[key(x)] - x[3] for x in o if key(x) in md]
    return math.sqrt(sum(v * v for v in ds) / len(ds))


def vs_rms(xs, zs, a, b, *, include_lid: bool, lid_only: bool = False) -> float:
    s = n = 0
    for i, x in enumerate(xs):
        if not (XLO <= x <= XHI):
            continue
        zi = g.z_conv(x)
        zmin = g.H if (include_lid or lid_only) else zi
        for k, z in enumerate(zs):
            if z < zmin - 1e-9:
                continue
            if lid_only and z >= zi - 1e-9:
                continue
            d = a[k, i] - b[k, i]
            if np.isnan(d):
                continue
            s += d * d
            n += 1
    return math.sqrt(s / n) if n else float("nan")


OBS_XS = (30.0, 40.0, 50.0, 60.0, 70.0)


def recs_for_draw(syn_path: Path) -> list[tuple[int, float, float]]:
    picks = parse_picks(syn_path.read_text(encoding="utf-8"))
    return [(int(p[0]), abs(p[1] - p[4]), p[3]) for p in picks]


def load_folder_rays(folder: Path, stem: str):
    rp, sp = folder / f"rays_{stem}.dat", folder / f"syn_{stem}.dat"
    if not rp.is_file() or not sp.is_file():
        return [], []
    return pr.parse_rays(rp), recs_for_draw(sp)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--title",
        default="PSP 公平两步：双场 -M/-U  vs  单场 -M/-k    红=偏快 蓝=偏慢",
    )
    p.add_argument("--dual", type=Path, default=DUAL)
    p.add_argument("--single", type=Path, default=SINGLE)
    p.add_argument("--out", type=Path, default=HERE / "check_single_vs_dual.png")
    p.add_argument("--include-lid", action="store_true")
    p.add_argument(
        "--pss",
        action="store_true",
        help="PSS 热初值对照（盖层+面下）",
    )
    p.add_argument(
        "--pps",
        action="store_true",
        help="PPS 热初值对照（只盖层）",
    )
    args = p.parse_args()
    if args.pss:
        args.dual = HERE / "inv_pss_dual"
        args.single = HERE / "inv_pss_single"
        args.out = HERE / "check_pss_single_vs_dual.png"
        args.include_lid = True
        if args.title.startswith("PSP"):
            args.title = "PSS 热初值：盖层+面下 Vs = rec_vp/κ + 0.50    红=偏快 蓝=偏慢"
    if args.pps:
        args.dual = HERE / "inv_pps_dual"
        args.single = HERE / "inv_pps_single"
        args.out = HERE / "check_pps_single_vs_dual.png"
        args.include_lid = True
        if args.title.startswith("PSP"):
            args.title = "PPS 热初值：盖层 Vs = rec_vp/κ + 0.50（面下是 P）    红=偏快 蓝=偏慢"
    dual, single = args.dual, args.single
    lid = args.include_lid
    lid_only = bool(args.pps)
    codes = (8,) if args.pss else ((7,) if args.pps else (6,))
    xs, zs, t_d, s_d, r_d = load_vs(dual, include_lid=lid)
    _, _, t_s, s_s, r_s = load_vs(single, include_lid=lid)
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    rows = [
        ("双场 -U", s_d - t_d, r_d - t_d, r_d - s_d, dual),
        ("单场 -k", s_s - t_s, r_s - t_s, r_s - s_s, single),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.4), facecolor="w", layout="constrained")
    last = None
    legend_done = False
    for i, (name, d0, d1, d2, folder) in enumerate(rows):
        tt0 = ttrms(folder / "syn_inv.dat", folder / "syn_start.dat")
        tt1 = ttrms(folder / "syn_inv.dat", folder / "syn_rec.dat")
        v_st = vs_rms(xs, zs, (s_d if i == 0 else s_s), (t_d if i == 0 else t_s), include_lid=lid, lid_only=lid_only)
        v_rt = vs_rms(xs, zs, (r_d if i == 0 else r_s), (t_d if i == 0 else t_s), include_lid=lid, lid_only=lid_only)
        titles = (
            f"{name}  start−true\nVs RMS {v_st:.3f}  t {tt0:.3f} s",
            f"{name}  rec−true\nVs RMS {v_rt:.3f}  t {tt1:.3f} s",
            f"{name}  rec−start",
        )
        ray_sets = (
            load_folder_rays(folder, "start"),
            load_folder_rays(folder, "rec"),
            load_folder_rays(folder, "rec"),
        )
        for ax, grid, title, (rays, recs) in zip(axes[i], (d0, d1, d2), titles, ray_sets):
            last = ax.imshow(
                grid, extent=extent, cmap="RdBu_r", vmin=-DCLIM, vmax=DCLIM,
                aspect="auto", interpolation="nearest", zorder=0,
            )
            if rays and recs:
                draw_ps_rays(
                    ax, rays, recs, thin=True, mark_conv=(ax is axes[i][1]),
                    z_conv=g.z_conv, codes=codes,
                )
                if not legend_done:
                    ax.legend(loc="upper right", framealpha=0.88, fontsize=6, ncol=2)
                    legend_done = True
            xs_l = np.linspace(18, 82, 80)
            ax.plot(xs_l, [g.z_conv(x) for x in xs_l], "k-.", lw=0.8, zorder=2)
            if lid:
                ax.plot(xs_l, [g.H] * len(xs_l), "k--", lw=0.6, zorder=2)
            ax.plot(list(OBS_XS), [g.H] * len(OBS_XS), "k^", ms=5, zorder=7)
            ax.set_xlim(18, 82)
            ax.set_ylim(16, 0)
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.25)
            if ax is axes[i, 0]:
                ax.set_ylabel("深度 (km)")
    fig.colorbar(last, ax=axes, shrink=0.55, label="ΔVs (km/s)")
    fig.suptitle(args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"wrote {args.out}")
    print(f"dual   start-true {vs_rms(xs,zs,s_d,t_d,include_lid=lid,lid_only=lid_only):.4f}  rec-true {vs_rms(xs,zs,r_d,t_d,include_lid=lid,lid_only=lid_only):.4f}  "
          f"t {ttrms(dual/'syn_inv.dat', dual/'syn_start.dat'):.4f} -> {ttrms(dual/'syn_inv.dat', dual/'syn_rec.dat'):.4f}")
    print(f"single start-true {vs_rms(xs,zs,s_s,t_s,include_lid=lid,lid_only=lid_only):.4f}  rec-true {vs_rms(xs,zs,r_s,t_s,include_lid=lid,lid_only=lid_only):.4f}  "
          f"t {ttrms(single/'syn_inv.dat', single/'syn_start.dat'):.4f} -> {ttrms(single/'syn_inv.dat', single/'syn_rec.dat'):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
