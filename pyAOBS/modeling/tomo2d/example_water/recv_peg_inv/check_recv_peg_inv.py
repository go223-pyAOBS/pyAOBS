#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对照壳内高速体：0/1 vs 0/1/4/5，以及 0/1 vs 4/5。

用法（在本目录）:
  python check_recv_peg_inv.py --no-show --smesh out01.smesh.5.1 --tag 01 ...
  python check_recv_peg_inv.py --no-show --compare out01.smesh.5.1 out0145.smesh.5.1
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "water_fwd"))
sys.path.insert(0, str(HERE.parent / "water_inv"))
from check_analytic import _setup_mpl, parse_ray_file  # noqa: E402
from check_water_inv import (  # noqa: E402
    Pick,
    _align_picks,
    _as_stations,
    _draw_rays_on,
    _grid,
    latest_smesh,
    parse_geom,
    parse_picks,
)
from make_recv_peg_inv_case import (  # noqa: E402
    AX0,
    AX1,
    AZ0,
    AZ1,
    DV_ANOM,
    H,
    H_MOHO,
    OBS_X,
    OBS_XS,
    V_WATER,
    box_stats,
    crust_field_compare,
    illum_x_range,
    node_stats,
    parse_smesh,
)


def _cmap():
    from matplotlib import colormaps

    return colormaps["RdYlBu_r"], 3.7, 6.8, "RdYlBu_r"


def _map_xlim(shots: list[tuple[float, float]]) -> tuple[float, float]:
    if shots:
        xs = [p[0] for p in shots]
        return min(xs) - 3.0, max(xs) + 3.0
    x_lo, x_hi = illum_x_range()
    return x_lo, x_hi


def _draw_geometry(ax, obs, shots, *, legend: bool) -> None:
    x_lo, x_hi = illum_x_range()
    ax.axhline(H, color="crimson", ls="--", lw=1.2, zorder=3, label="海底" if legend else None)
    ax.axhline(H_MOHO, color="0.2", ls=":", lw=1.0, zorder=3, label="莫霍" if legend else None)
    ax.axvline(x_lo, color="0.25", ls=":", lw=0.9, zorder=3)
    ax.axvline(x_hi, color="0.25", ls=":", lw=0.9, zorder=3)
    if shots:
        sx, sz = zip(*shots)
        ax.plot(
            sx,
            sz,
            marker="o",
            color="#ff7f0e",
            ms=3.2,
            mew=0.35,
            mec="k",
            ls="none",
            zorder=6,
            label="炮" if legend else None,
        )
    stations = _as_stations(obs)
    if stations:
        ox, oz = zip(*stations)
        ax.plot(
            ox,
            oz,
            marker="^",
            color="k",
            ms=9,
            mew=0.7,
            mec="w",
            ls="none",
            zorder=7,
            label="OBS 台" if legend else None,
        )


def _draw_anomaly(ax, *, legend: bool) -> None:
    from matplotlib.patches import Rectangle

    ax.add_patch(
        Rectangle(
            (AX0, AZ0),
            AX1 - AX0,
            AZ1 - AZ0,
            fill=False,
            ec="#00c853",
            lw=1.2,
            ls="--",
            zorder=5,
            label="高速体" if legend else None,
        )
    )


def plot_inversion(
    rec_path: Path,
    true_path: Path,
    start_path: Path,
    out_png: Path,
    *,
    show: bool,
    geom_path: Path | None = None,
    rec_label: str = "反演",
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize, TwoSlopeNorm

    _setup_mpl()
    xs, zs, vrec = parse_smesh(rec_path)
    _, _, vtrue = parse_smesh(true_path)
    _, _, vstart = parse_smesh(start_path)
    rec = _grid(xs, zs, vrec)
    tru = _grid(xs, zs, vtrue)
    sta = _grid(xs, zs, vstart)
    x_lo, x_hi = illum_x_range()
    w_m, w_lo, w_hi, n_w = node_stats(
        xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, layer="water"
    )
    a_m, a_lo, a_hi, n_a = box_stats(
        xs, zs, vrec, x_lo=AX0, x_hi=AX1, z_lo=AZ0, z_hi=AZ1
    )
    ta_m, *_ = box_stats(xs, zs, vtrue, x_lo=AX0, x_hi=AX1, z_lo=AZ0, z_hi=AZ1)
    sa_m, *_ = box_stats(xs, zs, vstart, x_lo=AX0, x_hi=AX1, z_lo=AZ0, z_hi=AZ1)
    cmap, vlo, vhi, _spec = _cmap()
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    vnorm = Normalize(vmin=vlo, vmax=vhi, clip=False)
    dnorm = TwoSlopeNorm(vmin=-0.55, vcenter=0.0, vmax=0.55)
    obs, shots = parse_geom(geom_path) if geom_path and geom_path.is_file() else (
        [(x, H) for x in OBS_XS],
        [],
    )
    xlim = _map_xlim(shots)

    fig = plt.figure(figsize=(12.6, 8.4), facecolor="w", layout="constrained")
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 0.055], height_ratios=[1.05, 1.0])
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
    ]
    cax_v = fig.add_subplot(gs[0, 3])
    ax_ds = fig.add_subplot(gs[1, 0])
    ax_dt = fig.add_subplot(gs[1, 1])
    ax_pr = fig.add_subplot(gs[1, 2])
    cax_d = fig.add_subplot(gs[1, 3])

    titles = ["初值  无异常背景", f"{rec_label}  {rec_path.name}", "真值  壳内高速体"]
    im0 = None
    for i, (ax, data, title) in enumerate(zip(axes, (sta, rec, tru), titles)):
        im0 = ax.imshow(
            data,
            extent=extent,
            cmap=cmap,
            norm=vnorm,
            aspect="auto",
            interpolation="nearest",
            zorder=0,
        )
        _draw_geometry(ax, obs, shots, legend=(i == 1))
        _draw_anomaly(ax, legend=(i == 1))
        ax.set_title(title)
        ax.set_xlim(*xlim)
        ax.set_ylim(float(zs[-1]), float(zs[0]))
        ax.grid(True, alpha=0.28)
        if i == 0:
            ax.set_ylabel("深度 (km)")
        else:
            ax.tick_params(labelleft=False)
        if i == 1:
            ax.legend(loc="upper right", framealpha=0.9, fontsize=8, ncol=2)

    fig.colorbar(im0, cax=cax_v).set_label("Vp (km/s)")
    im_d = ax_ds.imshow(
        rec - sta,
        extent=extent,
        cmap="RdBu_r",
        norm=dnorm,
        aspect="auto",
        interpolation="nearest",
        zorder=0,
    )
    ax_dt.imshow(
        rec - tru,
        extent=extent,
        cmap="RdBu_r",
        norm=dnorm,
        aspect="auto",
        interpolation="nearest",
        zorder=0,
    )
    for ax, title, ylab in (
        (ax_ds, "反演 − 初值", True),
        (ax_dt, "反演 − 真值", False),
    ):
        _draw_geometry(ax, obs, shots, legend=False)
        _draw_anomaly(ax, legend=False)
        ax.set_title(title)
        ax.set_xlabel("模型距离 (km)")
        ax.set_xlim(*xlim)
        ax.set_ylim(float(zs[-1]), float(zs[0]))
        ax.grid(True, alpha=0.28)
        if ylab:
            ax.set_ylabel("深度 (km)")
        else:
            ax.tick_params(labelleft=False)
    fig.colorbar(im_d, cax=cax_d).set_label("ΔV (km/s)")

    ix = min(range(len(xs)), key=lambda i: abs(xs[i] - OBS_X))
    ax_pr.plot(sta[:, ix], zs, color="0.45", lw=1.6, label="初值")
    ax_pr.plot(tru[:, ix], zs, color="C0", lw=1.8, label="真值")
    ax_pr.plot(rec[:, ix], zs, color="C3", lw=1.8, label=rec_label)
    ax_pr.axhline(H, color="crimson", ls="--", lw=1.1)
    ax_pr.axhline(H_MOHO, color="0.2", ls=":", lw=1.1)
    ax_pr.axhline(AZ0, color="#00c853", ls="--", lw=0.8)
    ax_pr.axhline(AZ1, color="#00c853", ls="--", lw=0.8)
    ax_pr.set_ylim(float(zs[-1]), float(zs[0]))
    ax_pr.set_xlabel("速度 (km/s)")
    ax_pr.set_ylabel("深度 (km)")
    ax_pr.set_title(f"剖面 x={xs[ix]:.0f} km")
    ax_pr.grid(True, alpha=0.28)
    ax_pr.legend(loc="lower right", framealpha=0.9)

    fig.suptitle(
        f"{rec_label}  水 {w_m:.3f}（应 {V_WATER:g}）[{w_lo:.3f},{w_hi:.3f}] n={n_w}    "
        f"框内 {a_m:.3f}（真 {ta_m:.3f}，初 {sa_m:.3f}）[{a_lo:.3f},{a_hi:.3f}] n={n_a}",
        fontsize=11,
    )
    fig.savefig(out_png, dpi=140)
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)


def plot_inv_rays(
    smesh_path: Path,
    seaf_path: Path,
    moho_path: Path,
    rays: list[tuple[list[float], list[float]]],
    picks: list[Pick],
    geom_path: Path,
    out_png: Path,
    *,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    _setup_mpl()
    xs, zs, vel = parse_smesh(smesh_path)
    data = _grid(xs, zs, vel)
    cmap, vlo, vhi, _spec = _cmap()
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    obs, shots = parse_geom(geom_path)
    fig = plt.figure(figsize=(10.4, 5.6), facecolor="w", layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[28, 1.85])
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(
        data,
        extent=extent,
        cmap=cmap,
        norm=Normalize(vmin=vlo, vmax=vhi, clip=False),
        aspect="auto",
        interpolation="nearest",
        zorder=0,
    )
    fig.colorbar(im, cax=cax).set_label("Vp (km/s)")
    for path, color, ls in ((seaf_path, "crimson", "--"), (moho_path, "0.15", ":")):
        if not path.is_file():
            continue
        rx, rz = [], []
        for ln in path.read_text(encoding="utf-8").splitlines():
            p = ln.split()
            if len(p) >= 2:
                rx.append(float(p[0]))
                rz.append(float(p[1]))
        if rx:
            ax.plot(rx, rz, color=color, ls=ls, lw=1.3, zorder=3)
    _draw_anomaly(ax, legend=True)
    _draw_rays_on(ax, rays, picks, legend=True, thin=False)
    _draw_geometry(ax, obs, shots, legend=True)
    ax.set_xlim(*_map_xlim(shots))
    ax.set_ylim(float(zs[-1]), 0.0)
    ax.set_title("反演模型上的射线")
    ax.set_xlabel("模型距离 (km)")
    ax.set_ylabel("深度 (km)")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8, ncol=2)
    fig.savefig(out_png, dpi=140)
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)


def plot_compare(
    rec_a: Path,
    rec_b: Path,
    true_path: Path,
    start_path: Path,
    out_png: Path,
    *,
    show: bool,
    geom_path: Path | None = None,
    labels: tuple[str, str] = ("A", "B"),
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm

    _setup_mpl()
    xs, zs, va = parse_smesh(rec_a)
    _, _, vb = parse_smesh(rec_b)
    _, _, vtrue = parse_smesh(true_path)
    _, _, vstart = parse_smesh(start_path)
    ga, gb = _grid(xs, zs, va), _grid(xs, zs, vb)
    tru = _grid(xs, zs, vtrue)
    x_lo, x_hi = illum_x_range()
    rms_a, c_a, _ = crust_field_compare(xs, zs, va, vtrue, x_lo=x_lo, x_hi=x_hi)
    rms_b, c_b, _ = crust_field_compare(xs, zs, vb, vtrue, x_lo=x_lo, x_hi=x_hi)
    rms_st, _, _ = crust_field_compare(xs, zs, vstart, vtrue, x_lo=x_lo, x_hi=x_hi)
    cmap, vlo, vhi, _spec = _cmap()
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    vnorm = Normalize(vmin=vlo, vmax=vhi, clip=False)
    dnorm = TwoSlopeNorm(vmin=-0.55, vcenter=0.0, vmax=0.55)
    obs, shots = parse_geom(geom_path) if geom_path and geom_path.is_file() else (
        [(x, H) for x in OBS_XS],
        [],
    )
    xlim = _map_xlim(shots)
    fig = plt.figure(figsize=(16.0, 8.6), facecolor="w", layout="constrained")
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 0.055], height_ratios=[1.05, 1.0])
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
    ]
    cax_v = fig.add_subplot(gs[0, 3])
    ax_da = fig.add_subplot(gs[1, 0])
    ax_db = fig.add_subplot(gs[1, 1])
    ax_pr = fig.add_subplot(gs[1, 2])
    cax_d = fig.add_subplot(gs[1, 3])
    titles = [
        f"真值  {AX1 - AX0:g}×{AZ1 - AZ0:g} km  +{DV_ANOM:g}",
        labels[0],
        labels[1],
    ]
    im0 = None
    for i, (ax, data, title) in enumerate(zip(axes, (tru, ga, gb), titles)):
        im0 = ax.imshow(
            data,
            extent=extent,
            cmap=cmap,
            norm=vnorm,
            aspect="auto",
            interpolation="nearest",
            zorder=0,
        )
        _draw_geometry(ax, obs, shots, legend=(i == 0))
        _draw_anomaly(ax, legend=(i == 0))
        ax.set_title(title)
        ax.set_xlim(*xlim)
        ax.set_ylim(float(zs[-1]), float(zs[0]))
        ax.grid(True, alpha=0.28)
        if i == 0:
            ax.set_ylabel("深度 (km)")
        else:
            ax.tick_params(labelleft=False)
        if i == 0:
            ax.legend(loc="upper right", framealpha=0.9, fontsize=8, ncol=2)
    fig.colorbar(im0, cax=cax_v).set_label("Vp (km/s)")
    im_d = None
    for ax, data, title, ylab in (
        (ax_da, ga - tru, f"{labels[0]} − 真值", True),
        (ax_db, gb - tru, f"{labels[1]} − 真值", False),
    ):
        im_d = ax.imshow(
            data,
            extent=extent,
            cmap="RdBu_r",
            norm=dnorm,
            aspect="auto",
            interpolation="nearest",
            zorder=0,
        )
        _draw_geometry(ax, obs, shots, legend=False)
        _draw_anomaly(ax, legend=False)
        ax.set_title(title)
        ax.set_xlabel("模型距离 (km)")
        ax.set_xlim(*xlim)
        ax.set_ylim(float(zs[-1]), float(zs[0]))
        ax.grid(True, alpha=0.28)
        if ylab:
            ax.set_ylabel("深度 (km)")
        else:
            ax.tick_params(labelleft=False)
    fig.colorbar(im_d, cax=cax_d).set_label("ΔV (km/s)")
    ix = min(range(len(xs)), key=lambda i: abs(xs[i] - OBS_X))
    ax_pr.plot(tru[:, ix], zs, color="C0", lw=1.8, label="真值")
    ax_pr.plot(ga[:, ix], zs, color="C3", lw=1.8, label=labels[0])
    ax_pr.plot(gb[:, ix], zs, color="C1", lw=1.6, label=labels[1])
    ax_pr.axhline(H, color="crimson", ls="--", lw=1.1)
    ax_pr.axhline(AZ0, color="#00c853", ls="--", lw=0.8)
    ax_pr.axhline(AZ1, color="#00c853", ls="--", lw=0.8)
    ax_pr.set_ylim(float(zs[-1]), float(zs[0]))
    ax_pr.set_xlabel("速度 (km/s)")
    ax_pr.set_ylabel("深度 (km)")
    ax_pr.set_title(f"剖面 x={xs[ix]:.0f} km")
    ax_pr.grid(True, alpha=0.28)
    ax_pr.legend(loc="lower right", framealpha=0.9)
    ba, *_ = box_stats(xs, zs, va, x_lo=AX0, x_hi=AX1, z_lo=AZ0, z_hi=AZ1)
    bb, *_ = box_stats(xs, zs, vb, x_lo=AX0, x_hi=AX1, z_lo=AZ0, z_hi=AZ1)
    ta, *_ = box_stats(xs, zs, vtrue, x_lo=AX0, x_hi=AX1, z_lo=AZ0, z_hi=AZ1)
    upa, *_ = box_stats(xs, zs, va, x_lo=AX0, x_hi=AX1, z_lo=H + 0.05, z_hi=AZ0 - 0.05)
    upb, *_ = box_stats(xs, zs, vb, x_lo=AX0, x_hi=AX1, z_lo=H + 0.05, z_hi=AZ0 - 0.05)
    dna, *_ = box_stats(xs, zs, va, x_lo=AX0, x_hi=AX1, z_lo=AZ1 + 0.05, z_hi=H_MOHO)
    dnb, *_ = box_stats(xs, zs, vb, x_lo=AX0, x_hi=AX1, z_lo=AZ1 + 0.05, z_hi=H_MOHO)
    fig.suptitle(
        f"壳 RMS  {labels[0]}={rms_a:.3f}  {labels[1]}={rms_b:.3f}（初值 {rms_st:.3f}）  "
        f"corr {c_a:.2f}/{c_b:.2f}    "
        f"框内 {ba:.3f}/{bb:.3f}（真 {ta:.3f}）  框上 {upa:.3f}/{upb:.3f}  框下 {dna:.3f}/{dnb:.3f}",
        fontsize=10,
    )
    fig.savefig(out_png, dpi=140)
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)
    print(
        f"compare  crust RMS vs true  {labels[0]}={rms_a:.4f}  {labels[1]}={rms_b:.4f}  "
        f"start={rms_st:.4f}  corr  {c_a:.3f}/{c_b:.3f}"
    )
    print(
        f"  localize  box  {labels[0]}={ba:.4f}  {labels[1]}={bb:.4f}  true={ta:.4f}  "
        f"above  {upa:.4f}/{upb:.4f}  below  {dna:.4f}/{dnb:.4f}"
    )


def _mark_obs_x(ax, ox: float) -> None:
    ax.axvline(ox, color="0.35", ls="--", lw=0.9, zorder=1)
    ax.plot(
        ox,
        1.02,
        marker="^",
        color="k",
        ms=8,
        mew=0.6,
        mec="w",
        transform=ax.get_xaxis_transform(),
        clip_on=False,
        zorder=5,
    )


def plot_ttimes_fit(
    obs: list[Pick],
    start: list[Pick],
    rec: list[Pick],
    out_png: Path,
    *,
    show: bool,
) -> tuple[float, float, int]:
    """按真实 OBS 分栏，横轴用炮点 x，不把多台折到同一 |dx|。"""
    import matplotlib.pyplot as plt

    _setup_mpl()
    rows = _align_picks(obs, start, rec)
    obs_xs = sorted({round(o[4], 6) for o, _s, _r in rows})
    n_obs = max(1, len(obs_xs))
    fig, axes = plt.subplots(
        3,
        n_obs,
        figsize=(max(8.2, 4.4 * n_obs), 8.6),
        sharex="col",
        squeeze=False,
        gridspec_kw={"height_ratios": [1.55, 0.9, 0.9]},
    )
    for j in range(1, n_obs):
        axes[0][j].sharey(axes[0][0])
        axes[1][j].sharey(axes[1][0])
    for j in range(n_obs):
        axes[2][j].sharey(axes[1][0])
    styles = {
        0: ("#1f77b4", "折射 0"),
        1: ("#d62728", "反射 1"),
        4: ("#2ca02c", "台侧折射 4"),
        5: ("#ff7f0e", "台侧反射 5"),
    }
    rms_s: list[float] = []
    rms_r: list[float] = []
    for j, ox in enumerate(obs_xs):
        ax, ax0, ax1 = axes[0][j], axes[1][j], axes[2][j]
        _mark_obs_x(ax, ox)
        _mark_obs_x(ax0, ox)
        _mark_obs_x(ax1, ox)
        ax0.axhline(0.0, color="0.4", lw=0.8)
        ax1.axhline(0.0, color="0.4", lw=0.8)
        for code, (color, name) in styles.items():
            sub = [
                r
                for r in rows
                if r[0][0] == code and abs(r[0][4] - ox) < 1e-4
            ]
            if not sub:
                continue
            sx = [o[1] for o, _s, _r in sub]
            to = [o[3] for o, _s, _r in sub]
            ts = [s[3] for _o, s, _r in sub]
            tr = [r[3] for _o, _s, r in sub]
            order = sorted(range(len(sx)), key=lambda i: sx[i])
            sx = [sx[i] for i in order]
            to = [to[i] for i in order]
            ts = [ts[i] for i in order]
            tr = [tr[i] for i in order]
            ds = [a - b for a, b in zip(ts, to)]
            dr = [a - b for a, b in zip(tr, to)]
            lab = j == 0
            ax.plot(
                sx,
                to,
                "o",
                color=color,
                ms=5,
                zorder=3,
                label=f"{name} 观测" if lab else None,
            )
            ax.plot(
                sx,
                ts,
                "--",
                color=color,
                lw=1.25,
                alpha=0.85,
                label=f"{name} 初值正演" if lab else None,
            )
            ax.plot(
                sx,
                tr,
                "-",
                color=color,
                lw=1.5,
                label=f"{name} 反演正演" if lab else None,
            )
            ax0.plot(sx, ds, "-", color=color, lw=0.8, alpha=0.45, zorder=2)
            ax0.plot(
                sx,
                ds,
                "x",
                color=color,
                ms=6.5,
                mew=1.5,
                zorder=3,
                label=name if lab else None,
            )
            ax1.plot(sx, dr, "-", color=color, lw=0.9, alpha=0.55, zorder=2)
            ax1.plot(
                sx,
                dr,
                "o",
                color=color,
                ms=5.5,
                zorder=3,
                label=name if lab else None,
            )
            rms_s.extend(ds)
            rms_r.extend(dr)
        ax.set_title(f"OBS x={ox:g} km")
        ax.grid(True, alpha=0.35)
        ax0.grid(True, alpha=0.35)
        ax1.grid(True, alpha=0.35)
        ax1.set_xlabel("炮 x (km)")
    rs = math.sqrt(sum(v * v for v in rms_s) / len(rms_s)) if rms_s else float("nan")
    rr = math.sqrt(sum(v * v for v in rms_r) / len(rms_r)) if rms_r else float("nan")
    axes[0][0].invert_yaxis()
    axes[0][0].set_ylabel("走时 t (s)")
    axes[1][0].set_ylabel("初值−观测 (s)")
    axes[2][0].set_ylabel("反演−观测 (s)")
    axes[0][0].legend(loc="lower left", fontsize=7.5, ncol=2, framealpha=0.9)
    axes[1][0].legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    axes[2][0].legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    fig.suptitle(
        f"走时拟合：观测=真模型正演    RMS 初值 {rs:.3f} s → 反演 {rr:.3f} s"
        "    横轴=炮点 x，三角/竖线=该栏 OBS；残差分行：叉=初值，圆=反演",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)
    print(f"  ttimes RMS  start={rs:.4f} s  rec={rr:.4f} s  n={len(rows)}")
    return rs, rr, len(rows)


def report(rec_path: Path, true_path: Path, start_path: Path) -> int:
    xs, zs, vrec = parse_smesh(rec_path)
    _, _, vtrue = parse_smesh(true_path)
    _, _, vstart = parse_smesh(start_path)
    x_lo, x_hi = illum_x_range()
    w_m, w_lo, w_hi, n_w = node_stats(
        xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, layer="water"
    )
    rms_rt, corr_rt, _ = crust_field_compare(xs, zs, vrec, vtrue, x_lo=x_lo, x_hi=x_hi)
    rms_st, _, _ = crust_field_compare(xs, zs, vstart, vtrue, x_lo=x_lo, x_hi=x_hi)
    a_m, a_lo, a_hi, n_a = box_stats(
        xs, zs, vrec, x_lo=AX0, x_hi=AX1, z_lo=AZ0, z_hi=AZ1
    )
    ta_m, *_ = box_stats(xs, zs, vtrue, x_lo=AX0, x_hi=AX1, z_lo=AZ0, z_hi=AZ1)
    sa_m, *_ = box_stats(xs, zs, vstart, x_lo=AX0, x_hi=AX1, z_lo=AZ0, z_hi=AZ1)
    print(f"recovered  {rec_path.name}")
    print(
        f"  water  z<H  n={n_w}  mean={w_m:.4f}  [{w_lo:.4f},{w_hi:.4f}]  expect={V_WATER}"
    )
    print(
        f"  crust  RMS vs true  rec={rms_rt:.4f}  start={rms_st:.4f}  "
        f"corr={corr_rt:.3f}"
    )
    print(
        f"  anom   n={n_a}  rec={a_m:.4f} [{a_lo:.4f},{a_hi:.4f}]  "
        f"true={ta_m:.4f}  start={sa_m:.4f}"
    )
    ok = True
    if abs(w_m - V_WATER) > 1e-3 or abs(w_hi - V_WATER) > 1e-3:
        print(f"FAIL water above seafloor not frozen at {V_WATER}")
        ok = False
    if a_m <= sa_m + 0.02:
        print(f"FAIL anomaly box mean {a_m:.4f} did not rise from start {sa_m:.4f}")
        ok = False
    if ok:
        print("OK")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=HERE / "out")
    p.add_argument("--true", dest="true_smesh", type=Path, default=HERE / "true.smesh")
    p.add_argument("--start", dest="start_smesh", type=Path, default=HERE / "start.smesh")
    p.add_argument("--smesh", type=Path, default=None)
    p.add_argument("--geom", type=Path, default=HERE / "geom_inv_0145.dat")
    p.add_argument("--obs", dest="syn_obs", type=Path, default=HERE / "syn_inv_0145.dat")
    p.add_argument("--syn-start", type=Path, default=HERE / "syn_start_0145.dat")
    p.add_argument("--syn-rec", type=Path, default=HERE / "syn_rec_0145.dat")
    p.add_argument("--rays", type=Path, default=HERE / "rays_rec_0145.dat")
    p.add_argument("--seafloor", type=Path, default=HERE / "seafloor.refl")
    p.add_argument("--moho", type=Path, default=HERE / "moho.refl")
    p.add_argument("--tag", default="")
    p.add_argument("--rec-label", default="反演")
    p.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"))
    p.add_argument("--compare-labels", nargs=2, default=None)
    p.add_argument("--compare-out", type=Path, default=None)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    show = not args.no_show
    rc = 0
    if args.compare:
        labels = tuple(args.compare_labels) if args.compare_labels else ("A", "B")
        cmp_png = args.compare_out or HERE / "check_inv_compare.png"
        plot_compare(
            args.compare[0],
            args.compare[1],
            args.true_smesh,
            args.start_smesh,
            cmp_png,
            show=show,
            geom_path=args.geom if args.geom.is_file() else None,
            labels=(labels[0], labels[1]),
        )
        print(f"wrote {cmp_png}")
        return 0
    rec = args.smesh if args.smesh else latest_smesh(args.out_root)
    rc = report(rec, args.true_smesh, args.start_smesh)
    geom = args.geom if args.geom.is_file() else None
    tag = f"_{args.tag}" if args.tag else ""
    models_png = HERE / f"check_inv_models{tag}.png"
    plot_inversion(
        rec,
        args.true_smesh,
        args.start_smesh,
        models_png,
        show=show,
        geom_path=geom,
        rec_label=args.rec_label,
    )
    print(f"wrote {models_png}")
    rays: list[tuple[list[float], list[float]]] = []
    picks_rec: list[Pick] = []
    if args.rays.is_file():
        rays = parse_ray_file(args.rays)
    if args.syn_rec.is_file():
        picks_rec = parse_picks(args.syn_rec.read_text(encoding="utf-8"))
    if rays and picks_rec and rec.is_file() and args.seafloor.is_file() and geom:
        rays_png = HERE / f"check_inv_rays{tag}.png"
        plot_inv_rays(
            rec, args.seafloor, args.moho, rays, picks_rec, geom, rays_png, show=show
        )
        print(f"wrote {rays_png}")
    if args.syn_obs.is_file() and args.syn_start.is_file() and args.syn_rec.is_file():
        t_png = HERE / f"check_inv_ttimes{tag}.png"
        plot_ttimes_fit(
            parse_picks(args.syn_obs.read_text(encoding="utf-8")),
            parse_picks(args.syn_start.read_text(encoding="utf-8")),
            parse_picks(args.syn_rec.read_text(encoding="utf-8")),
            t_png,
            show=show,
        )
        print(f"wrote {t_png}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
