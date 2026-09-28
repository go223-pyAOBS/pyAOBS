#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PPP+PPS：Vp 图用 ps_inv 的 plot_models，Vs 图叠 PPS 射线。"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ps_inv"))
sys.path.insert(0, str(HERE.parent / "pps_inv"))
sys.path.insert(0, str(HERE.parent / "ps_fwd"))
sys.path.insert(0, str(HERE.parent / "converse_fwd"))
from check_ps_fwd import draw_ps_rays  # noqa: E402
from make_pps_inv_case import (  # noqa: E402
    H,
    KAPPA_START,
    KAPPA_TRUE,
    OBS_XS,
    OBS_Z,
    V_WATER,
    ZMAX,
    Z_CONV,
    illum_x_range,
    node_stats,
    parse_smesh,
)
import plot_ps_inv_models as psp  # noqa: E402

S_Z1, S_Z2 = Z_CONV, Z_CONV + 2.5
VS_VMIN, VS_VMAX = 0.80, 4.20

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _zmax(rays) -> float:
    if not rays:
        return 0.0
    return max(max(zs) for _, zs in rays if zs)


def plot_models(rec_path, true_path, start_path, out_png, geom_path, rays, recs):
    xs, zs, vrec = parse_smesh(rec_path)
    _, _, vtrue = parse_smesh(true_path)
    _, _, vstart = parse_smesh(start_path)
    rec, tru, sta = psp._grid(xs, zs, vrec), psp._grid(xs, zs, vtrue), psp._grid(xs, zs, vstart)
    x_lo, x_hi = illum_x_range()
    lid_m, *_ = node_stats(xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, z_lo=H + 1e-6, z_hi=Z_CONV)
    s_m, *_ = node_stats(xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, z_lo=S_Z1, z_hi=S_Z2, z_hi_inclusive=True)
    obs, shots = psp.parse_geom(geom_path) if geom_path.is_file() else ([(x, OBS_Z) for x in OBS_XS], [])
    xlim = (min(x_lo, 18.0), max(x_hi, 82.0))
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    fig = plt.figure(figsize=(12.6, 8.4), facecolor="w", layout="constrained")
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 0.055], height_ratios=[1.05, 1.0])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    cax_v = fig.add_subplot(gs[0, 3])
    ax_ds, ax_dt, ax_pr = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])
    cax_d = fig.add_subplot(gs[1, 3])
    titles = [
        f"初值  Vs=初值Vp/{KAPPA_START:g}",
        f"反演  {rec_path.name}",
        f"真值  Vs=Vp/{KAPPA_TRUE:g}",
    ]
    vnorm = Normalize(vmin=VS_VMIN, vmax=VS_VMAX, clip=False)
    dnorm = TwoSlopeNorm(vmin=-0.50, vcenter=0.0, vmax=0.50)
    im0 = None
    for i, (ax, data, title) in enumerate(zip(axes, (sta, rec, tru), titles)):
        im0 = ax.imshow(data, extent=extent, cmap="RdYlBu_r", norm=vnorm,
                        aspect="auto", interpolation="nearest", zorder=0)
        if i == 1 and rays and recs:
            draw_ps_rays(ax, rays, recs, thin=True, codes=(7,))
        psp.draw_geom(ax, obs, shots, legend=(i == 1))
        ax.set_title(title)
        ax.set_xlim(*xlim)
        ax.set_ylim(float(zs[-1]), float(zs[0]))
        ax.grid(True, alpha=0.28)
        if i == 0:
            ax.set_ylabel("深度 (km)")
        else:
            ax.tick_params(labelleft=False)
        if i == 1:
            ax.legend(loc="upper right", framealpha=0.9, fontsize=7, ncol=2)
    fig.colorbar(im0, cax=cax_v).set_label("Vs (km/s)")
    im_d = ax_ds.imshow(rec - sta, extent=extent, cmap="RdBu_r", norm=dnorm,
                        aspect="auto", interpolation="nearest", zorder=0)
    ax_dt.imshow(rec - tru, extent=extent, cmap="RdBu_r", norm=dnorm,
                 aspect="auto", interpolation="nearest", zorder=0)
    for ax, title, ylab in ((ax_ds, "反演 − 初值", True), (ax_dt, "反演 − 真值", False)):
        psp.draw_geom(ax, obs, shots, legend=False)
        ax.set_title(title)
        ax.set_xlabel("模型距离 (km)")
        ax.set_xlim(*xlim)
        ax.set_ylim(float(zs[-1]), float(zs[0]))
        ax.grid(True, alpha=0.28)
        if ylab:
            ax.set_ylabel("深度 (km)")
        else:
            ax.tick_params(labelleft=False)
    fig.colorbar(im_d, cax=cax_d).set_label("ΔVs (km/s)")
    ix = min(range(len(xs)), key=lambda i: abs(xs[i] - 50.0))
    ax_pr.plot(sta[:, ix], zs, color="0.45", lw=1.6, label="初值")
    ax_pr.plot(tru[:, ix], zs, color="C0", lw=1.8, label="真值")
    ax_pr.plot(rec[:, ix], zs, color="C3", lw=1.8, label="反演")
    ax_pr.axhline(H, color="0.15", ls="--", lw=1.0)
    ax_pr.axhline(Z_CONV, color="0.25", ls="--", lw=1.1)
    ax_pr.axvline(V_WATER, color="0.4", ls=":", lw=0.8)
    ax_pr.set_ylim(float(zs[-1]), float(zs[0]))
    ax_pr.set_xlabel("Vs (km/s)")
    ax_pr.set_title(f"剖面 x={xs[ix]:.0f} km")
    ax_pr.grid(True, alpha=0.28)
    ax_pr.legend(loc="lower right", framealpha=0.9, fontsize=8)
    fig.suptitle(
        f"PPS 台侧盖层 Vs（联合 PPP+PPS，面下是 P，-k{KAPPA_START:g}）  "
        f"盖层均值 {lid_m:.3f}  面下 {s_m:.3f}",
        fontsize=11,
    )
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")


def plot_rec_vs_true_rays(rec_path, true_path, rec_rays, recs, true_rays, geom_path, out_png):
    xs, zs, vrec = parse_smesh(rec_path)
    _, _, vtrue = parse_smesh(true_path)
    rec_g, tru_g = psp._grid(xs, zs, vrec), psp._grid(xs, zs, vtrue)
    obs, shots = psp.parse_geom(geom_path)
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    vnorm = Normalize(vmin=VS_VMIN, vmax=VS_VMAX)
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2), facecolor="w", layout="constrained")
    panels = (
        (axes[0], rec_g, rec_rays, f"收回  zmax={_zmax(rec_rays):.2f} km"),
        (axes[1], tru_g, true_rays, f"真值  zmax={_zmax(true_rays):.2f} km"),
    )
    im = None
    for i, (ax, grid, rays, title) in enumerate(panels):
        im = ax.imshow(
            grid, extent=extent, cmap="RdYlBu_r", norm=vnorm,
            aspect="auto", interpolation="nearest", zorder=0,
        )
        if rays and recs:
            draw_ps_rays(ax, rays, recs, thin=len(rays) > 40, codes=(7,))
        psp.draw_geom(ax, obs, shots, legend=(i == 0))
        ax.axhline(Z_CONV, color="0.15", ls="--", lw=1.0, zorder=2)
        ax.set_title(title)
        ax.set_xlim(*illum_x_range())
        ax.set_ylim(ZMAX, 0.0)
        ax.set_xlabel("模型距离 (km)")
        if i == 0:
            ax.set_ylabel("深度 (km)")
            ax.legend(loc="upper right", framealpha=0.9, fontsize=7, ncol=2)
        else:
            ax.tick_params(labelleft=False)
    fig.colorbar(im, ax=axes, shrink=0.72).set_label("Vs (km/s)")
    fig.suptitle("PPS：炮侧盖层 P、面下 P、台侧盖层 S（粉）", fontsize=11)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")


def main() -> int:
    rec = max(
        HERE.glob("out.smesh.*.*"),
        key=lambda p: (int(p.name.split(".")[-2]), int(p.name.split(".")[-1])),
    )
    ppp_rays = psp.parse_rays(HERE / "rays_ppp_rec.dat")
    ppp_recs = psp.parse_syn(HERE / "syn_ppp.dat")
    psp.plot_models(
        HERE / "rec_vp.smesh",
        HERE / "true_vp.smesh",
        HERE / "start_vp.smesh",
        HERE / "check_inv_vp_models.png",
        HERE / "geom_ppp.dat",
        ppp_rays,
        ppp_recs,
        "vp",
    )
    vs_rays = psp.parse_rays(HERE / "rays_rec.dat")
    vs_recs = psp.parse_syn(HERE / "syn_inv.dat")
    plot_models(
        rec,
        HERE / "true_vs.smesh",
        HERE / "start_vs.smesh",
        HERE / "check_inv_models.png",
        HERE / "geom_inv.dat",
        vs_rays,
        vs_recs,
    )
    true_rays = (
        psp.parse_rays(HERE / "rays_true.dat")
        if (HERE / "rays_true.dat").is_file()
        else []
    )
    plot_rec_vs_true_rays(
        rec,
        HERE / "true_vs.smesh",
        vs_rays,
        vs_recs,
        true_rays,
        HERE / "geom_inv.dat",
        HERE / "check_inv_rays.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
