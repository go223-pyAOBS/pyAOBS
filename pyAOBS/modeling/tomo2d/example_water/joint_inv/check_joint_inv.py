#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对照水+壳联合反演：模型、台/炮、射线、走时拟合。

用法（在本目录）:
  python check_joint_inv.py
  python check_joint_inv.py --no-show
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
    _draw_geometry,
    _draw_rays_on,
    _grid,
    _map_xlim,
    latest_smesh,
    parse_geom,
    parse_picks,
)
from make_joint_inv_case import (  # noqa: E402
    H,
    H_REFL,
    OBS_X,
    V_SED_START,
    V_SED_TRUE,
    V_WATER_START,
    V_WATER_TRUE,
    illum_x_range,
    node_stats,
    parse_smesh,
)


def _joint_cmap():
    from matplotlib import colormaps

    return colormaps["RdYlBu_r"], 1.40, 2.10


def _draw_ifaces(ax, *, legend: bool) -> None:
    ax.axhline(
        H_REFL, color="#8B4513", ls=":", lw=1.2, zorder=3, label="反射面" if legend else None
    )


def plot_inversion(
    rec_path: Path,
    true_path: Path,
    start_path: Path,
    out_png: Path,
    *,
    show: bool,
    geom_path: Path | None = None,
    rays: list[tuple[list[float], list[float]]] | None = None,
    picks: list[Pick] | None = None,
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
        xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, water=True, strict=True
    )
    s_m, s_lo, s_hi, n_s = node_stats(xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, water=False)
    cmap, vlo, vhi = _joint_cmap()
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    vnorm = Normalize(vmin=vlo, vmax=vhi, clip=False)
    dnorm = TwoSlopeNorm(vmin=-0.25, vcenter=0.0, vmax=0.25)
    obs, shots = parse_geom(geom_path) if geom_path and geom_path.is_file() else (
        [(OBS_X, H)],
        [],
    )
    xlim = _map_xlim(shots)

    fig = plt.figure(figsize=(12.6, 8.4), facecolor="w", layout="constrained")
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 0.055], height_ratios=[1.05, 1.0])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax_v = fig.add_subplot(gs[0, 3])
    ax_ds = fig.add_subplot(gs[1, 0])
    ax_dt = fig.add_subplot(gs[1, 1])
    ax_pr = fig.add_subplot(gs[1, 2])
    cax_d = fig.add_subplot(gs[1, 3])

    titles = [
        f"初值  水 {V_WATER_START:g}  沉积 {V_SED_START:g}",
        f"反演  {rec_path.name}",
        f"真值  水 {V_WATER_TRUE:g}  沉积 {V_SED_TRUE:g}",
    ]
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
        if i == 1 and rays and picks:
            _draw_rays_on(ax, rays, picks, legend=True, thin=True)
        _draw_geometry(ax, obs, shots, legend=(i == 1))
        _draw_ifaces(ax, legend=(i == 1))
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

    cb = fig.colorbar(im0, cax=cax_v)
    cb.set_label("Vp (km/s)")
    cb.set_ticks(np.linspace(vlo, vhi, 8))

    dstart = rec - sta
    dtrue = rec - tru
    im_d = ax_ds.imshow(
        dstart,
        extent=extent,
        cmap="RdBu_r",
        norm=dnorm,
        aspect="auto",
        interpolation="nearest",
        zorder=0,
    )
    ax_dt.imshow(
        dtrue,
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
        _draw_ifaces(ax, legend=False)
        ax.set_title(title)
        ax.set_xlabel("模型距离 (km)")
        ax.set_xlim(*xlim)
        ax.set_ylim(float(zs[-1]), float(zs[0]))
        ax.grid(True, alpha=0.28)
        if ylab:
            ax.set_ylabel("深度 (km)")
        else:
            ax.tick_params(labelleft=False)
    cb_d = fig.colorbar(im_d, cax=cax_d)
    cb_d.set_label("ΔV (km/s)")

    ix = min(range(len(xs)), key=lambda i: abs(xs[i] - OBS_X))
    ax_pr.plot(sta[:, ix], zs, color="0.45", lw=1.6, label="初值")
    ax_pr.plot(tru[:, ix], zs, color="C0", lw=1.8, label="真值")
    ax_pr.plot(rec[:, ix], zs, color="C3", lw=1.8, label="反演")
    ax_pr.axhline(H, color="crimson", ls="--", lw=1.1)
    ax_pr.axhline(H_REFL, color="#8B4513", ls=":", lw=1.1)
    ax_pr.axvline(V_WATER_TRUE, color="C0", ls=":", lw=0.8)
    ax_pr.axvline(V_SED_TRUE, color="0.3", ls=":", lw=0.8)
    ax_pr.set_ylim(float(zs[-1]), float(zs[0]))
    ax_pr.set_xlabel("速度 (km/s)")
    ax_pr.set_ylabel("深度 (km)")
    ax_pr.set_title(f"剖面 x={xs[ix]:.0f} km（台下）")
    ax_pr.grid(True, alpha=0.28)
    ax_pr.legend(loc="lower right", framealpha=0.9)

    fig.suptitle(
        f"联合反演（0/1+2/3，无 -y/-w）  "
        f"水 {w_m:.3f}（真 {V_WATER_TRUE:g}，初 {V_WATER_START:g}）"
        f"  [{w_lo:.3f},{w_hi:.3f}]    "
        f"沉积 {s_m:.3f}（真 {V_SED_TRUE:g}，初 {V_SED_START:g}）"
        f"  [{s_lo:.3f},{s_hi:.3f}]",
        fontsize=11,
    )
    fig.savefig(out_png, dpi=140)
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)


def _draw_joint_background(
    ax, cax, fig, smesh_path: Path, seafloor_path: Path, basement_path: Path
) -> None:
    import numpy as np
    from matplotlib.colors import Normalize

    xs, zs, vel = parse_smesh(smesh_path)
    data = _grid(xs, zs, vel)
    cmap, vlo, vhi = _joint_cmap()
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    im = ax.imshow(
        data,
        extent=extent,
        cmap=cmap,
        norm=Normalize(vmin=vlo, vmax=vhi, clip=False),
        aspect="auto",
        interpolation="nearest",
        zorder=0,
    )
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Vp (km/s)")
    cb.set_ticks(np.linspace(vlo, vhi, 8))
    for path, color, ls in (
        (seafloor_path, "crimson", "--"),
        (basement_path, "#8B4513", ":"),
    ):
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


def plot_inv_rays(
    smesh_path: Path,
    seafloor_path: Path,
    basement_path: Path,
    rays: list[tuple[list[float], list[float]]],
    picks: list[Pick],
    geom_path: Path,
    out_png: Path,
    *,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    _setup_mpl()
    obs, shots = parse_geom(geom_path)
    fig = plt.figure(figsize=(10.4, 5.8), facecolor="w", layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[28, 1.85])
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    _draw_joint_background(ax, cax, fig, smesh_path, seafloor_path, basement_path)
    _draw_rays_on(ax, rays, picks, legend=True, thin=False)
    _draw_geometry(ax, obs, shots, legend=True)
    _draw_ifaces(ax, legend=True)
    ax.set_xlim(*_map_xlim(shots))
    ax.set_ylim(4.0, 0.0)
    ax.set_title("反演模型上的射线（0/1 海底炮，2/3 海面炮）")
    ax.set_xlabel("模型距离 (km)")
    ax.set_ylabel("深度 (km)")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=7, ncol=2)
    fig.savefig(out_png, dpi=140)
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)


def plot_ttimes_fit(
    obs: list[Pick],
    start: list[Pick],
    rec: list[Pick],
    out_png: Path,
    *,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    _setup_mpl()
    rows = _align_picks(obs, start, rec)
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(8.4, 7.4), sharex=True, gridspec_kw={"height_ratios": [1.6, 1.0]}
    )
    styles = {
        0: ("#1f77b4", "折射 0"),
        1: ("#d62728", "反射 1"),
        2: ("#9B30FF", "直达 2"),
        3: ("#111111", "多次 3"),
    }
    rms_s: list[float] = []
    rms_r: list[float] = []
    for code, (color, name) in styles.items():
        sub = [r for r in rows if r[0][0] == code]
        if not sub:
            continue
        dx = [abs(o[1] - o[4]) for o, _s, _r in sub]
        to = [o[3] for o, _s, _r in sub]
        ts = [s[3] for _o, s, _r in sub]
        tr = [r[3] for _o, _s, r in sub]
        order = sorted(range(len(dx)), key=lambda i: dx[i])
        dx = [dx[i] for i in order]
        to = [to[i] for i in order]
        ts = [ts[i] for i in order]
        tr = [tr[i] for i in order]
        ax.plot(dx, to, "o", color=color, ms=6, zorder=3, label=f"{name} 观测")
        ax.plot(dx, ts, "--", color=color, lw=1.2, alpha=0.85, label=f"{name} 初值")
        ax.plot(dx, tr, "-", color=color, lw=1.5, label=f"{name} 反演")
        axr.plot(dx, [a - b for a, b in zip(ts, to)], "s", color=color, ms=4.5, alpha=0.7)
        axr.plot(dx, [a - b for a, b in zip(tr, to)], "o", color=color, ms=5)
        rms_s.extend(a - b for a, b in zip(ts, to))
        rms_r.extend(a - b for a, b in zip(tr, to))
    rs = math.sqrt(sum(v * v for v in rms_s) / len(rms_s)) if rms_s else float("nan")
    rr = math.sqrt(sum(v * v for v in rms_r) / len(rms_r)) if rms_r else float("nan")
    ax.set_ylabel("走时 t (s)")
    ax.set_title(f"走时拟合：观测=真模型正演    RMS 初值 {rs:.3f} s → 反演 {rr:.3f} s")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower left", fontsize=7, ncol=2, framealpha=0.9)
    axr.axhline(0.0, color="0.4", lw=0.8)
    axr.set_xlabel("偏移 dx (km)")
    axr.set_ylabel("残差 (s)")
    axr.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)
    print(f"  ttimes RMS  start={rs:.4f} s  rec={rr:.4f} s  n={len(rows)}")


def report(rec_path: Path, true_path: Path, start_path: Path) -> int:
    xs, zs, vrec = parse_smesh(rec_path)
    _, _, vtrue = parse_smesh(true_path)
    _, _, vstart = parse_smesh(start_path)
    x_lo, x_hi = illum_x_range()
    w_m, w_lo, w_hi, n_w = node_stats(
        xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, water=True, strict=True
    )
    s_m, s_lo, s_hi, n_s = node_stats(xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, water=False)
    tw, _, _, _ = node_stats(xs, zs, vtrue, x_lo=x_lo, x_hi=x_hi, water=True, strict=True)
    sw, _, _, _ = node_stats(xs, zs, vstart, x_lo=x_lo, x_hi=x_hi, water=True, strict=True)
    ts, _, _, _ = node_stats(xs, zs, vtrue, x_lo=x_lo, x_hi=x_hi, water=False)
    ss, _, _, _ = node_stats(xs, zs, vstart, x_lo=x_lo, x_hi=x_hi, water=False)
    print(f"recovered  {rec_path.name}")
    print(
        f"  water  z<H  n={n_w}  mean={w_m:.4f}  [{w_lo:.4f},{w_hi:.4f}]  "
        f"true={tw:.4f}  start={sw:.4f}"
    )
    print(
        f"  sed    n={n_s}  mean={s_m:.4f}  [{s_lo:.4f},{s_hi:.4f}]  "
        f"true={ts:.4f}  start={ss:.4f}"
    )
    ok = True
    if abs(w_m - V_WATER_TRUE) >= abs(sw - V_WATER_TRUE):
        print("FAIL water did not move toward true vs start")
        ok = False
    if abs(s_m - V_SED_TRUE) >= abs(ss - V_SED_TRUE):
        print("FAIL sediment did not move toward true vs start")
        ok = False
    if abs(w_m - V_WATER_TRUE) > 0.05:
        print(f"FAIL water mean not within 0.05 of {V_WATER_TRUE}")
        ok = False
    if abs(s_m - V_SED_TRUE) > 0.05:
        print(f"FAIL sediment mean not within 0.05 of {V_SED_TRUE}")
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
    p.add_argument("--geom", type=Path, default=HERE / "geom_inv.dat")
    p.add_argument("--obs", dest="syn_obs", type=Path, default=HERE / "syn_inv.dat")
    p.add_argument("--syn-start", type=Path, default=HERE / "syn_start.dat")
    p.add_argument("--syn-rec", type=Path, default=HERE / "syn_rec.dat")
    p.add_argument("--rays", type=Path, default=HERE / "rays_rec.dat")
    p.add_argument("--seafloor", type=Path, default=HERE / "seafloor.refl")
    p.add_argument("--basement", type=Path, default=HERE / "basement.refl")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()
    rec = args.smesh if args.smesh else latest_smesh(args.out_root)
    rc = report(rec, args.true_smesh, args.start_smesh)
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    show = not args.no_show
    geom = args.geom if args.geom.is_file() else None
    rays: list[tuple[list[float], list[float]]] = []
    picks_rec: list[Pick] = []
    if args.rays.is_file():
        rays = parse_ray_file(args.rays)
    if args.syn_rec.is_file():
        picks_rec = parse_picks(args.syn_rec.read_text(encoding="utf-8"))
    elif args.syn_obs.is_file():
        picks_rec = parse_picks(args.syn_obs.read_text(encoding="utf-8"))
    models_png = HERE / "check_inv_models.png"
    plot_inversion(
        rec,
        args.true_smesh,
        args.start_smesh,
        models_png,
        show=show,
        geom_path=geom,
        rays=rays or None,
        picks=picks_rec or None,
    )
    print(f"wrote {models_png}")
    if rays and picks_rec and rec.is_file() and geom:
        rays_png = HERE / "check_inv_rays.png"
        plot_inv_rays(
            rec, args.seafloor, args.basement, rays, picks_rec, geom, rays_png, show=show
        )
        print(f"wrote {rays_png}")
    else:
        print("缺射线/走时，跳过 check_inv_rays.png", file=sys.stderr)
    if args.syn_obs.is_file() and args.syn_start.is_file() and args.syn_rec.is_file():
        t_png = HERE / "check_inv_ttimes.png"
        plot_ttimes_fit(
            parse_picks(args.syn_obs.read_text(encoding="utf-8")),
            parse_picks(args.syn_start.read_text(encoding="utf-8")),
            parse_picks(args.syn_rec.read_text(encoding="utf-8")),
            t_png,
            show=show,
        )
        print(f"wrote {t_png}")
    else:
        print("缺 syn_inv / syn_start / syn_rec，跳过走时拟合图", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
