#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PPP+PSP：Vp 图用 ps_inv 的 plot_models；PSP 图画射线实际用速（盖层Vp / 面下Vs）。"""

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
sys.path.insert(0, str(HERE.parent / "psp_inv"))
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
    ZMAX,
    Z_CONV,
    illum_x_range,
    node_stats,
    parse_smesh,
)
import plot_ps_inv_models as psp  # noqa: E402

S_Z1, S_Z2 = Z_CONV, Z_CONV + 2.5
# 盖层是 Vp（~2.5），面下是 Vs（~3.5），不能再用纯 Vs 色标的 0.8–4.2 去标「全是 Vs」
VEL_VMIN, VEL_VMAX = 1.20, 5.80


def _mix_psp(xs, zs, vp, vs):
    """PSP 用速：转换面以上（水+盖层）取 Vp，面上及面下取 Vs。"""
    mixed = [
        [sv if z >= Z_CONV - 1e-9 else pv for z, pv, sv in zip(zs, col_p, col_s)]
        for col_p, col_s in zip(vp, vs)
    ]
    return psp._grid(xs, zs, mixed)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _zmax(rays) -> float:
    if not rays:
        return 0.0
    return max(max(zs) for _, zs in rays if zs)


def plot_models(rec_vs, true_vs, start_vs, rec_vp, true_vp, start_vp, out_png, geom_path, rays, recs):
    xs, zs, vs_rec = parse_smesh(rec_vs)
    _, _, vs_true = parse_smesh(true_vs)
    _, _, vs_start = parse_smesh(start_vs)
    _, _, vp_rec = parse_smesh(rec_vp)
    _, _, vp_true = parse_smesh(true_vp)
    _, _, vp_start = parse_smesh(start_vp)
    rec = _mix_psp(xs, zs, vp_rec, vs_rec)
    tru = _mix_psp(xs, zs, vp_true, vs_true)
    sta = _mix_psp(xs, zs, vp_start, vs_start)
    x_lo, x_hi = illum_x_range()
    lid_m, *_ = node_stats(xs, zs, vp_rec, x_lo=x_lo, x_hi=x_hi, z_lo=H + 1e-6, z_hi=Z_CONV)
    t_lid, *_ = node_stats(xs, zs, vp_true, x_lo=x_lo, x_hi=x_hi, z_lo=H + 1e-6, z_hi=Z_CONV)
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
        "初值  折合（盖层Vp / 面下 Vp/1.73）",
        f"反演  {rec_vs.name}",
        "真值  折合（盖层Vp / 面下 Vp/1.73）",
    ]
    vnorm = Normalize(vmin=VEL_VMIN, vmax=VEL_VMAX, clip=False)
    dnorm = TwoSlopeNorm(vmin=-0.50, vcenter=0.0, vmax=0.50)
    im0 = None
    for i, (ax, data, title) in enumerate(zip(axes, (sta, rec, tru), titles)):
        im0 = ax.imshow(data, extent=extent, cmap="RdYlBu_r", norm=vnorm,
                        aspect="auto", interpolation="nearest", zorder=0)
        if i == 1 and rays and recs:
            draw_ps_rays(ax, rays, recs, thin=True, codes=(0, 6))
        psp.draw_geom(ax, obs, shots, legend=(i == 1))
        ax.axhline(Z_CONV, color="0.15", ls="--", lw=1.0, zorder=2)
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
    fig.colorbar(im0, cax=cax_v).set_label("PSP 用速 (盖层Vp / 面下Vs)")
    im_d = ax_ds.imshow(rec - sta, extent=extent, cmap="RdBu_r", norm=dnorm,
                        aspect="auto", interpolation="nearest", zorder=0)
    ax_dt.imshow(rec - tru, extent=extent, cmap="RdBu_r", norm=dnorm,
                 aspect="auto", interpolation="nearest", zorder=0)
    for ax, title, ylab in ((ax_ds, "反演 − 初值", True), (ax_dt, "反演 − 真值", False)):
        psp.draw_geom(ax, obs, shots, legend=False)
        ax.axhline(Z_CONV, color="0.15", ls="--", lw=1.0, zorder=2)
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
        f"PSP 用速（折合：盖层 Vp，面下 Vp/1.73）  "
        f"盖层Vp {lid_m:.3f}（真 {t_lid:.3f}）  面下Vs {s_m:.3f}",
        fontsize=11,
    )
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")


def plot_true_fwd_rays(true_mixed, rays6, recs6, rays0, recs0, geom6, geom0, out_png):
    """真模型正演：左 PSP(-X/6)，右同网格初至(0)。"""
    xs, zs, mix = parse_smesh(true_mixed)
    grid = psp._grid(xs, zs, mix)
    obs6, shots6 = psp.parse_geom(geom6) if geom6.is_file() else ([], [])
    obs0, shots0 = psp.parse_geom(geom0) if geom0.is_file() else (obs6, shots6)
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    vnorm = Normalize(vmin=VEL_VMIN, vmax=VEL_VMAX)
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2), facecolor="w", layout="constrained")
    panels = (
        (axes[0], rays6, recs6, obs6, shots6, (6,),
         f"真 PSP（-X / 6）  zmax={_zmax(rays6):.2f} km"),
        (axes[1], rays0, recs0, obs0, shots0, (0,),
         f"真折合初至（0，观测）  zmax={_zmax(rays0):.2f} km"),
    )
    im = None
    for i, (ax, rays, recs, obs, shots, codes, title) in enumerate(panels):
        im = ax.imshow(
            grid, extent=extent, cmap="RdYlBu_r", norm=vnorm,
            aspect="auto", interpolation="nearest", zorder=0,
        )
        if rays and recs:
            draw_ps_rays(ax, rays, recs, thin=len(rays) > 40, codes=codes)
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
    fig.colorbar(im, ax=axes, shrink=0.72).set_label("折合用速 (盖层Vp / 面下Vs)")
    fig.suptitle("真模型正演射线：PSP 必须过转换面；初至可只走盖层", fontsize=11)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")


def plot_rec_vs_true_rays(rec_vs, true_vs, rec_vp, true_vp, rec_rays, recs, true_rays, geom_path, out_png):
    xs, zs, vs_rec = parse_smesh(rec_vs)
    _, _, vs_true = parse_smesh(true_vs)
    _, _, vp_rec = parse_smesh(rec_vp)
    _, _, vp_true = parse_smesh(true_vp)
    rec_g = _mix_psp(xs, zs, vp_rec, vs_rec)
    tru_g = _mix_psp(xs, zs, vp_true, vs_true)
    obs, shots = psp.parse_geom(geom_path)
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    vnorm = Normalize(vmin=VEL_VMIN, vmax=VEL_VMAX)
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
            draw_ps_rays(ax, rays, recs, thin=len(rays) > 40, codes=(0, 6))
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
    fig.colorbar(im, ax=axes, shrink=0.72).set_label("PSP 用速 (盖层Vp / 面下Vs)")
    fig.suptitle("PSP：盖层 P（Vp）、面下 S（粉，Vs）", fontsize=11)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")


def _latest_vs():
    cands = list(HERE.glob("out_psp.smesh.*.*")) or list(HERE.glob("out.smesh.*.*"))
    if not cands:
        raise FileNotFoundError("缺 out_psp.smesh.* / out.smesh.*")
    return max(cands, key=lambda p: (int(p.name.split(".")[-2]), int(p.name.split(".")[-1])))


def main() -> int:
    rec = _latest_vs()
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
    if (HERE / "true_mixed.smesh").is_file():
        r6 = HERE / "rays_true6.dat" if (HERE / "rays_true6.dat").is_file() else HERE / "rays_true.dat"
        s6 = HERE / "syn_psp6.dat" if (HERE / "syn_psp6.dat").is_file() else HERE / "syn_inv.dat"
        r0 = HERE / "rays_true.dat"
        s0 = HERE / "syn_inv.dat"
        plot_true_fwd_rays(
            HERE / "true_mixed.smesh",
            psp.parse_rays(r6) if r6.is_file() else [],
            psp.parse_syn(s6) if s6.is_file() else [],
            psp.parse_rays(r0) if r0.is_file() else [],
            psp.parse_syn(s0) if s0.is_file() else [],
            HERE / "geom_inv.dat",
            HERE / "geom_psp0.dat",
            HERE / "check_fwd_psp_rays.png",
        )
    vs_rays = psp.parse_rays(HERE / "rays_rec.dat")
    vs_recs = psp.parse_syn(HERE / "syn_inv.dat")
    true_vs = HERE / "true_mixed.smesh" if (HERE / "true_mixed.smesh").is_file() else HERE / "true_vs.smesh"
    start_vs = HERE / "start_mixed.smesh" if (HERE / "start_mixed.smesh").is_file() else HERE / "start_vs.smesh"
    plot_models(
        rec,
        true_vs,
        start_vs,
        HERE / "rec_vp.smesh",
        HERE / "true_vp.smesh",
        HERE / "start_vp.smesh",
        HERE / "check_inv_models.png",
        HERE / "geom_psp0.dat" if (HERE / "geom_psp0.dat").is_file() else HERE / "geom_inv.dat",
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
        true_vs,
        HERE / "rec_vp.smesh",
        HERE / "true_vp.smesh",
        vs_rays,
        vs_recs,
        true_rays,
        HERE / "geom_psp0.dat" if (HERE / "geom_psp0.dat").is_file() else HERE / "geom_inv.dat",
        HERE / "check_inv_rays.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
