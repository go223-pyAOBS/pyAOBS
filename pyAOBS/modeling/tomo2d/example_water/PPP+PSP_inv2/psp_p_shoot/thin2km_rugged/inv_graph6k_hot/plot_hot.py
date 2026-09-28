#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""-k 热初值：面下 Vs 紧色标 + 差分（start/rec/true）。"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "inv_2d"))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[2].parent / "ps_inv"))
sys.path.insert(0, str(HERE.parents[2].parent / "water_inv"))
import inv_grid as g  # noqa: E402
import make_ppp_psp_inv_case as m2  # noqa: E402
import plot_rugged_inv as pr  # noqa: E402
from check_ps_inv import plot_ttimes_fit  # noqa: E402
from check_water_inv import parse_picks  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

VMIN, VMAX = 3.70, 5.20
DCLIM = 0.60
START_TITLE = "start Vs  rec_vp/1.73+0.50"
REC_TITLE = "rec Vs"
TRUE_TITLE = "true Vs"
SUPTITLE_MODELS = "崎岖面  双场 type 6  PPP→Vs  观测=真双场正演   色标 3.70–5.20"
SUPTITLE_RAYS = "射线  收回 vs 真值（双场 PSP=6，真模型观测）"


def _mask_lid(xs, zs, vel) -> np.ndarray:
    arr = np.asarray(vel, float).T
    for j, z in enumerate(zs):
        for i, x in enumerate(xs):
            if z < g.z_conv(x) - 1e-9:
                arr[j, i] = np.nan
    return arr


def _axes_style(ax, title, *, ylabel=False):
    xs = np.linspace(18, 82, 80)
    ax.plot(xs, [2.0] * len(xs), "k--", lw=0.7, zorder=2)
    ax.plot(xs, [g.z_conv(x) for x in xs], "k-.", lw=0.9, zorder=2)
    ax.plot(list(pr.OBS_XS), [2.0] * len(pr.OBS_XS), "^", color="k", ms=6, zorder=7)
    ax.set_xlim(18, 82)
    ax.set_ylim(16, 0)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if ylabel:
        ax.set_ylabel("深度 (km)")


def _cbar(ax, im, label: str) -> None:
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.05)
    cb = ax.figure.colorbar(im, cax=cax)
    cb.set_label(label)


def main() -> int:
    xs, zs, true_vs = m2.parse_smesh(HERE / "true_vs.smesh")
    _, _, start_vs = m2.parse_smesh(HERE / "start_vs.smesh")
    rec_p = HERE / "rec_vs.smesh"
    if not rec_p.is_file():
        rec_p = HERE / "start_vs.smesh"
    _, _, rec_vs = m2.parse_smesh(rec_p)
    gtrue = _mask_lid(xs, zs, true_vs)
    gstart = _mask_lid(xs, zs, start_vs)
    grec = _mask_lid(xs, zs, rec_vs)
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))

    rec_rays = []
    if (HERE / "rays_rec.dat").is_file() and (HERE / "syn_rec.dat").is_file():
        rec_rays = pr.rays_used_in_inv(
            HERE / "rays_rec.dat", HERE / "syn_rec.dat", HERE / "syn_inv.dat",
        )
    true_rays = []
    if (HERE / "rays_true.dat").is_file():
        true_rays = pr.parse_rays(HERE / "rays_true.dat")
    elif (HERE.parent / "inv_graph6k" / "rays_true.dat").is_file():
        true_rays = pr.parse_rays(HERE.parent / "inv_graph6k" / "rays_true.dat")

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.8), facecolor="w", layout="constrained")
    for ax, grid, title, rays in (
        (axes[0], gstart, START_TITLE, None),
        (axes[1], grec, f"{REC_TITLE}  n={len(rec_rays)}", rec_rays),
        (axes[2], gtrue, f"{TRUE_TITLE}  n={len(true_rays)}", true_rays),
    ):
        im = ax.imshow(
            grid, extent=extent, cmap="RdYlBu_r", vmin=VMIN, vmax=VMAX,
            aspect="auto", interpolation="nearest", zorder=0,
        )
        if rays:
            pr.draw_iface_rays(ax, rays, legend=(ax is axes[1]))
        _axes_style(ax, title, ylabel=(ax is axes[0]))
        _cbar(ax, im, "Vs (km/s)")
    fig.suptitle(SUPTITLE_MODELS)
    out = HERE / "check_inv_models.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")

    d_st = gstart - gtrue
    d_rt = grec - gtrue
    d_rs = grec - gstart
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.8), facecolor="w", layout="constrained")
    for ax, grid, title in (
        (axes[0], d_st, "start − true"),
        (axes[1], d_rt, "rec − true"),
        (axes[2], d_rs, "rec − start"),
    ):
        im = ax.imshow(
            grid, extent=extent, cmap="RdBu_r", vmin=-DCLIM, vmax=DCLIM,
            aspect="auto", interpolation="nearest", zorder=0,
        )
        _axes_style(ax, title, ylabel=(ax is axes[0]))
        _cbar(ax, im, "ΔVs (km/s)")
    fig.suptitle("差分  红=偏快  蓝=偏慢  ±0.60 km/s")
    out_d = HERE / "check_inv_diff.png"
    fig.savefig(out_d, dpi=140)
    plt.close(fig)
    print(f"wrote {out_d}")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), facecolor="w", layout="constrained")
    for ax, grid, title, rays, legend in (
        (axes[0], grec, f"收回  {REC_TITLE}  n={len(rec_rays)}", rec_rays, True),
        (axes[1], gtrue, f"真值  {TRUE_TITLE}  n={len(true_rays)}", true_rays, False),
    ):
        im = ax.imshow(
            grid, extent=extent, cmap="RdYlBu_r", vmin=VMIN, vmax=VMAX,
            aspect="auto", interpolation="nearest", zorder=0,
        )
        if rays:
            pr.draw_iface_rays(ax, rays, legend=legend)
        _axes_style(ax, title, ylabel=(ax is axes[0]))
        ax.set_xlabel("模型距离 (km)")
        _cbar(ax, im, "Vs (km/s)")
    fig.suptitle(SUPTITLE_RAYS)
    out_r = HERE / "check_inv_rays.png"
    fig.savefig(out_r, dpi=140)
    plt.close(fig)
    print(f"wrote {out_r}")

    if (HERE / "syn_inv.dat").is_file() and (HERE / "syn_start.dat").is_file():
        rec = None
        if (HERE / "syn_rec.dat").is_file():
            rec = parse_picks((HERE / "syn_rec.dat").read_text(encoding="utf-8"))
        plot_ttimes_fit(
            parse_picks((HERE / "syn_inv.dat").read_text(encoding="utf-8")),
            parse_picks((HERE / "syn_start.dat").read_text(encoding="utf-8")),
            rec,
            HERE / "check_inv_ttimes.png",
            show=False,
        )
        print(f"wrote {HERE / 'check_inv_ttimes.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
