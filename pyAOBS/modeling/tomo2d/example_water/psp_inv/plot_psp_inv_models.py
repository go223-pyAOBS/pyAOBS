#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""写出 check_inv_models / check_inv_rays（PSP：只面下 S，冻盖层）。"""

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
sys.path.insert(0, str(HERE.parent / "ps_fwd"))
sys.path.insert(0, str(HERE.parent / "converse_fwd"))
from check_ps_fwd import draw_ps_rays  # noqa: E402
from make_psp_inv_case import (  # noqa: E402
    H,
    KAPPA_START,
    KAPPA_TRUE,
    OBS_XS,
    OBS_Z,
    V_WATER,
    Z_CONV,
    ZMAX,
    illum_x_range,
    node_stats,
    parse_smesh,
)
import plot_ps_inv_models as psp  # noqa: E402

S_Z1, S_Z2 = Z_CONV, Z_CONV + 2.5
VEL_VMIN, VEL_VMAX = 1.20, 4.40


def _mix_psp(xs, zs, vp, vs):
    mixed = [
        [sv if z >= Z_CONV - 1e-9 else pv for z, pv, sv in zip(zs, col_p, col_s)]
        for col_p, col_s in zip(vp, vs)
    ]
    return psp._grid(xs, zs, mixed)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_models(rec_path, true_path, start_path, out_png, geom_path, rays, recs,
                vp_rec=None, vp_true=None, vp_start=None):
    xs, zs, vs_rec = parse_smesh(rec_path)
    _, _, vs_true = parse_smesh(true_path)
    _, _, vs_start = parse_smesh(start_path)
    vp_rec = vp_rec if vp_rec and Path(vp_rec).is_file() else HERE / "true_vp.smesh"
    vp_true = vp_true if vp_true and Path(vp_true).is_file() else HERE / "true_vp.smesh"
    vp_start = vp_start if vp_start and Path(vp_start).is_file() else HERE / "true_vp.smesh"
    _, _, p_rec = parse_smesh(Path(vp_rec))
    _, _, p_true = parse_smesh(Path(vp_true))
    _, _, p_sta = parse_smesh(Path(vp_start))
    rec, tru, sta = (
        _mix_psp(xs, zs, p_rec, vs_rec),
        _mix_psp(xs, zs, p_true, vs_true),
        _mix_psp(xs, zs, p_sta, vs_start),
    )
    x_lo, x_hi = illum_x_range()
    lid_m, *_ = node_stats(xs, zs, p_rec, x_lo=x_lo, x_hi=x_hi, z_lo=H + 1e-6, z_hi=Z_CONV)
    s_m, *_ = node_stats(xs, zs, vs_rec, x_lo=x_lo, x_hi=x_hi, z_lo=S_Z1, z_hi=S_Z2, z_hi_inclusive=True)
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
        f"初值  盖层真Vp / 面下真Vp/{KAPPA_START:g}",
        f"反演  {rec_path.name}",
        f"真值  盖层Vp / 面下Vp/{KAPPA_TRUE:g}",
    ]
    vnorm = Normalize(vmin=VEL_VMIN, vmax=VEL_VMAX, clip=False)
    dnorm = TwoSlopeNorm(vmin=-0.50, vcenter=0.0, vmax=0.50)
    im0 = None
    for i, (ax, data, title) in enumerate(zip(axes, (sta, rec, tru), titles)):
        im0 = ax.imshow(data, extent=extent, cmap="RdYlBu_r", norm=vnorm,
                        aspect="auto", interpolation="nearest", zorder=0)
        if i == 1 and rays and recs:
            draw_ps_rays(ax, rays, recs, thin=True, codes=(6,))
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
        ax.axhline(Z_CONV, color="0.15", ls="--", lw=1.0, zorder=2)
    fig.colorbar(im0, cax=cax_v).set_label("PSP 用速 (盖层Vp / 面下Vs)")
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
    fig.colorbar(im_d, cax=cax_d).set_label("Δ用速 (km/s)")
    ix = min(range(len(xs)), key=lambda i: abs(xs[i] - 50.0))
    ax_pr.plot(sta[:, ix], zs, color="0.45", lw=1.6, label="初值")
    ax_pr.plot(tru[:, ix], zs, color="C0", lw=1.8, label="真值")
    ax_pr.plot(rec[:, ix], zs, color="C3", lw=1.8, label="反演")
    ax_pr.axhline(H, color="0.15", ls="--", lw=1.0)
    ax_pr.axhline(Z_CONV, color="0.25", ls="--", lw=1.1)
    ax_pr.axvline(V_WATER, color="0.4", ls=":", lw=0.8)
    ax_pr.set_ylim(float(zs[-1]), float(zs[0]))
    ax_pr.set_xlabel("用速 (km/s)")
    ax_pr.set_title(f"剖面 x={xs[ix]:.0f} km（盖层Vp / 面下Vs）")
    ax_pr.grid(True, alpha=0.28)
    ax_pr.legend(loc="lower right", framealpha=0.9, fontsize=8)
    fig.suptitle(
        f"PSP 用速（冻真 Vp，盖层=Vp，面下=Vs，-k{KAPPA_START:g}）  "
        f"盖层Vp {lid_m:.3f}  面下Vs {s_m:.3f}",
        fontsize=11,
    )
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")


def plot_inv_rays(smesh_path, rays, recs, geom_path, out_png, vp_path=None):
    xs, zs, vel = parse_smesh(smesh_path)
    vp_path = Path(vp_path) if vp_path else HERE / "true_vp.smesh"
    if vp_path.is_file():
        _, _, vp = parse_smesh(vp_path)
        grid = _mix_psp(xs, zs, vp, vel)
    else:
        grid = psp._grid(xs, zs, vel)
    obs, shots = psp.parse_geom(geom_path)
    fig, ax = plt.subplots(figsize=(11.2, 5.2), facecolor="w", layout="constrained")
    im = ax.imshow(
        grid,
        extent=(float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0])),
        cmap="RdYlBu_r",
        norm=Normalize(vmin=VEL_VMIN, vmax=VEL_VMAX),
        aspect="auto",
        interpolation="nearest",
        zorder=0,
    )
    draw_ps_rays(ax, rays, recs, thin=len(rays) > 40, codes=(6,))
    psp.draw_geom(ax, obs, shots, legend=True)
    ax.set_xlim(*illum_x_range())
    ax.set_ylim(ZMAX, 0.0)
    ax.set_title(f"PSP 收回射线（{len(obs)} 台）：盖层 P、面下 S（冻盖层）")
    ax.set_xlabel("模型距离 (km)")
    ax.set_ylabel("深度 (km)")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=7, ncol=2)
    fig.colorbar(im, ax=ax, shrink=0.72).set_label("PSP 用速 (盖层Vp / 面下Vs)")
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")


def main() -> int:
    rec = max(HERE.glob("out.smesh.*.*"), key=lambda p: (int(p.name.split(".")[-2]), int(p.name.split(".")[-1])))
    vs_rays = psp.parse_rays(HERE / "rays_rec.dat")
    vs_recs = psp.parse_syn(HERE / "syn_inv.dat")
    plot_models(
        rec, HERE / "true_vs.smesh", HERE / "start_vs.smesh",
        HERE / "check_inv_models.png", HERE / "geom_inv.dat",
        vs_rays, vs_recs,
    )
    plot_inv_rays(rec, vs_rays, vs_recs, HERE / "geom_inv.dat", HERE / "check_inv_rays.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
