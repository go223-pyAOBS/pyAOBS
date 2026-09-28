#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""写出 check_inv_vp_models / check_inv_models / check_inv_rays（不经过 GUI 导入链）。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

HERE = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(HERE))
from make_ps_inv_case import (  # noqa: E402
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

S_Z1, S_Z2 = Z_CONV, Z_CONV + 2.5
VS_VMIN, VS_VMAX = 0.80, 4.20
VP_VMIN, VP_VMAX = 1.45, 8.00
P_RAY, S_RAY = "#1f77b4", "#e377c2"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _grid(xs, zs, vel):
    arr = np.asarray(vel, dtype=float).T
    if arr.shape != (len(zs), len(xs)):
        raise ValueError(f"grid shape {arr.shape} != ({len(zs)}, {len(xs)})")
    return arr


def parse_geom(path: Path):
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    stations, shots, seen = [], [], set()
    i = 1
    while i < len(lines):
        parts = lines[i].split()
        i += 1
        if not parts or parts[0] != "s":
            continue
        stations.append((float(parts[1]), float(parts[2])))
        nrcv = int(float(parts[-1]))
        for _ in range(nrcv):
            rp = lines[i].split()
            i += 1
            xy = (float(rp[1]), float(rp[2]))
            if xy not in seen:
                seen.add(xy)
                shots.append(xy)
    return stations, shots


def parse_rays(path: Path):
    segs, xs, zs = [], [], []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith(">"):
            if len(xs) >= 2:
                segs.append((xs, zs))
            xs, zs = [], []
            continue
        a = s.split()
        if len(a) >= 2:
            xs.append(float(a[0]))
            zs.append(float(a[1]))
    if len(xs) >= 2:
        segs.append((xs, zs))
    return segs


def parse_syn(path: Path):
    recs = []
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
            recs.append((int(float(rp[3])), abs(float(rp[1]) - src_x), float(rp[4])))
    return recs


def split_ps(xs, zs, code):
    if len(xs) < 2:
        return []
    out, cx, cz = [], [xs[0]], [zs[0]]
    def is_s(i):
        zm = 0.5 * (zs[i] + zs[i + 1])
        if zm < H - 1e-3 or code in (0, 1):
            return False
        return zm > Z_CONV + 1e-3
    cur = is_s(0)
    for i in range(len(xs) - 1):
        s1 = is_s(i)
        if s1 == cur:
            cx.append(xs[i + 1])
            cz.append(zs[i + 1])
            continue
        if len(cx) >= 2:
            out.append((cx, cz, cur))
        cx, cz, cur = [xs[i], xs[i + 1]], [zs[i], zs[i + 1]], s1
    if len(cx) >= 2:
        out.append((cx, cz, cur))
    return out


def draw_rays(ax, rays, recs, *, thin, codes=None, ppp=False):
    halo, core = (1.15, 0.55) if thin else (2.05, 1.15)
    labeled = set()
    n = min(len(rays), len(recs))
    pending = []
    for i in range(n):
        code = recs[i][0]
        if codes and code not in codes:
            continue
        xs, zs = rays[i]
        if ppp or code in (0, 1):
            ax.plot(xs, zs, color=P_RAY, lw=0.7, ls="--", alpha=0.55, zorder=3)
            continue
        for sx, sz, is_s in split_ps(xs, zs, code):
            kw = dict(lw=core, ls="-", alpha=0.95, solid_capstyle="round")
            key = "S（粉）" if is_s else "P（蓝）"
            if key not in labeled:
                kw["label"] = key
                labeled.add(key)
            if is_s:
                pending.append((sx, sz, kw))
            else:
                ax.plot(sx, sz, color="#FFFFFF", lw=halo, alpha=0.75, zorder=4)
                ax.plot(sx, sz, color=P_RAY, zorder=4.1, **kw)
    for sx, sz, kw in pending:
        ax.plot(sx, sz, color="#FFFFFF", lw=halo + 0.35, alpha=0.9, zorder=5)
        ax.plot(sx, sz, color=S_RAY, zorder=5.1, **kw)


def draw_geom(ax, obs, shots, *, legend):
    x_lo, x_hi = illum_x_range()
    ax.axhline(H, color="0.15", ls="--", lw=1.2, zorder=3, label="海底" if legend else None)
    ax.axhline(Z_CONV, color="0.25", ls="--", lw=1.3, zorder=3, label="转换面" if legend else None)
    ax.axvline(x_lo, color="0.25", ls=":", lw=0.9, zorder=3)
    ax.axvline(x_hi, color="0.25", ls=":", lw=0.9, zorder=3)
    if shots:
        sx, sz = zip(*shots)
        ax.plot(sx, sz, marker="o", color="#ff7f0e", ms=4.0, mew=0.4, mec="k",
                ls="none", zorder=6, label="炮" if legend else None)
    if obs:
        ox, oz = zip(*obs)
        ax.plot(ox, oz, marker="^", color="k", ms=9, mew=0.7, mec="w",
                ls="none", zorder=7, label="OBS 台" if legend else None)


def plot_models(rec_path, true_path, start_path, out_png, geom_path, rays, recs, kind):
    xs, zs, vrec = parse_smesh(rec_path)
    _, _, vtrue = parse_smesh(true_path)
    _, _, vstart = parse_smesh(start_path)
    rec, tru, sta = _grid(xs, zs, vrec), _grid(xs, zs, vtrue), _grid(xs, zs, vstart)
    x_lo, x_hi = illum_x_range()
    lid_m, *_ = node_stats(xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, z_lo=H + 1e-6, z_hi=Z_CONV)
    s_m, *_ = node_stats(xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, z_lo=S_Z1, z_hi=S_Z2, z_hi_inclusive=True)
    obs, shots = parse_geom(geom_path) if geom_path.is_file() else ([(x, OBS_Z) for x in OBS_XS], [])
    xlim = (min(x_lo, 18.0), max(x_hi, 82.0))
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    is_vp = kind == "vp"
    vlo, vhi = (VP_VMIN, VP_VMAX) if is_vp else (VS_VMIN, VS_VMAX)
    unit = "Vp" if is_vp else "Vs"
    fig = plt.figure(figsize=(12.6, 8.4), facecolor="w", layout="constrained")
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 0.055], height_ratios=[1.05, 1.0])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    cax_v = fig.add_subplot(gs[0, 3])
    ax_ds, ax_dt, ax_pr = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])
    cax_d = fig.add_subplot(gs[1, 3])
    titles = (
        ["初值 Vp", f"反演  {rec_path.name}", "真值 Vp"]
        if is_vp
        else [f"初值  Vs=收回Vp/{KAPPA_START:g}", f"反演  {rec_path.name}", f"真值  Vs=Vp/{KAPPA_TRUE:g}"]
    )
    vnorm = Normalize(vmin=vlo, vmax=vhi, clip=False)
    dnorm = TwoSlopeNorm(vmin=-0.50, vcenter=0.0, vmax=0.50)
    im0 = None
    for i, (ax, data, title) in enumerate(zip(axes, (sta, rec, tru), titles)):
        im0 = ax.imshow(data, extent=extent, cmap="RdYlBu_r", norm=vnorm,
                        aspect="auto", interpolation="nearest", zorder=0)
        if i == 1 and rays and recs:
            draw_rays(ax, rays, recs, thin=True, ppp=is_vp)
        draw_geom(ax, obs, shots, legend=(i == 1))
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
    fig.colorbar(im0, cax=cax_v).set_label(f"{unit} (km/s)")
    im_d = ax_ds.imshow(rec - sta, extent=extent, cmap="RdBu_r", norm=dnorm,
                        aspect="auto", interpolation="nearest", zorder=0)
    ax_dt.imshow(rec - tru, extent=extent, cmap="RdBu_r", norm=dnorm,
                 aspect="auto", interpolation="nearest", zorder=0)
    for ax, title, ylab in ((ax_ds, "反演 − 初值", True), (ax_dt, "反演 − 真值", False)):
        draw_geom(ax, obs, shots, legend=False)
        ax.set_title(title)
        ax.set_xlabel("模型距离 (km)")
        ax.set_xlim(*xlim)
        ax.set_ylim(float(zs[-1]), float(zs[0]))
        ax.grid(True, alpha=0.28)
        if ylab:
            ax.set_ylabel("深度 (km)")
        else:
            ax.tick_params(labelleft=False)
    fig.colorbar(im_d, cax=cax_d).set_label(f"Δ{unit} (km/s)")
    ix = min(range(len(xs)), key=lambda i: abs(xs[i] - 50.0))
    ax_pr.plot(sta[:, ix], zs, color="0.45", lw=1.6, label="初值")
    ax_pr.plot(tru[:, ix], zs, color="C0", lw=1.8, label="真值")
    ax_pr.plot(rec[:, ix], zs, color="C3", lw=1.8, label="反演")
    ax_pr.axhline(H, color="0.15", ls="--", lw=1.0)
    ax_pr.axhline(Z_CONV, color="0.25", ls="--", lw=1.1)
    ax_pr.axvline(V_WATER, color="0.4", ls=":", lw=0.8)
    ax_pr.set_ylim(float(zs[-1]), float(zs[0]))
    ax_pr.set_xlabel(f"{unit} (km/s)")
    ax_pr.set_title(f"剖面 x={xs[ix]:.0f} km")
    ax_pr.grid(True, alpha=0.28)
    ax_pr.legend(loc="lower right", framealpha=0.9, fontsize=8)
    fig.suptitle(
        f"PPP 反 Vp  盖层均值 {lid_m:.3f}  面下 {s_m:.3f}"
        if is_vp
        else f"PSP 面下 Vs（冻收回 Vp，-k{KAPPA_START:g}）  盖层均值 {lid_m:.3f}  面下 {s_m:.3f}",
        fontsize=11,
    )
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")


def plot_inv_rays(smesh_path, rays, recs, geom_path, out_png):
    xs, zs, vel = parse_smesh(smesh_path)
    grid = _grid(xs, zs, vel)
    obs, shots = parse_geom(geom_path)
    fig, ax = plt.subplots(figsize=(11.2, 5.2), facecolor="w", layout="constrained")
    im = ax.imshow(
        grid,
        extent=(float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0])),
        cmap="RdYlBu_r",
        norm=Normalize(vmin=VS_VMIN, vmax=VS_VMAX),
        aspect="auto",
        interpolation="nearest",
        zorder=0,
    )
    draw_rays(ax, rays, recs, thin=len(rays) > 40, codes=(6,))
    draw_geom(ax, obs, shots, legend=True)
    ax.set_xlim(*illum_x_range())
    ax.set_ylim(ZMAX, 0.0)
    ax.set_title(f"PSP（{len(obs)} 台）：盖层 P、面下 S（只反面下）")
    ax.set_xlabel("模型距离 (km)")
    ax.set_ylabel("深度 (km)")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=7, ncol=2)
    fig.colorbar(im, ax=ax, shrink=0.72).set_label("Vs (km/s)")
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")


def main() -> int:
    rec = max(HERE.glob("out.smesh.*.*"), key=lambda p: (int(p.name.split(".")[-2]), int(p.name.split(".")[-1])))
    ppp_rays = parse_rays(HERE / "rays_ppp_rec.dat")
    ppp_recs = parse_syn(HERE / "syn_ppp.dat")
    plot_models(
        HERE / "rec_vp.smesh", HERE / "true_vp.smesh", HERE / "start_vp.smesh",
        HERE / "check_inv_vp_models.png", HERE / "geom_ppp.dat",
        ppp_rays, ppp_recs, "vp",
    )
    vs_rays = parse_rays(HERE / "rays_rec.dat")
    vs_recs = parse_syn(HERE / "syn_inv.dat")
    plot_models(
        rec, HERE / "true_vs.smesh", HERE / "start_vs.smesh",
        HERE / "check_inv_models.png", HERE / "geom_inv.dat",
        vs_rays, vs_recs, "vs",
    )
    plot_inv_rays(rec, vs_rays, vs_recs, HERE / "geom_inv.dat", HERE / "check_inv_rays.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
