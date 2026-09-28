#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对照折合 PSP 反演：模型、台/炮几何、射线、走时拟合。

用法（在本目录）:
  python check_converse_inv.py
  python check_converse_inv.py --no-show
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "converse_fwd"))
sys.path.insert(0, str(HERE.parent / "water_fwd"))
sys.path.insert(0, str(HERE.parent / "water_inv"))
from check_analytic import _setup_mpl, parse_ray_file  # noqa: E402
from check_converse_fwd import _draw_hybrid_background, draw_psp_ps_rays  # noqa: E402
from check_water_inv import (  # noqa: E402
    Pick,
    _align_picks,
    _grid,
    latest_smesh,
    parse_geom,
    parse_picks,
)
from make_converse_inv_case import (  # noqa: E402
    H,
    OBS_X,
    OBS_XS,
    OBS_Z,
    V_P_CRUST,
    V_WATER,
    expected_lid_mean,
    VS_START,
    VS_TRUE,
    Z_CONV,
    ZMAX,
    illum_x_range,
    node_stats,
    parse_smesh,
)

S_ILLUM_Z1 = Z_CONV
S_ILLUM_Z2 = Z_CONV + 2.5


def _as_stations(
    obs: tuple[float, float] | list[tuple[float, float]] | None,
) -> list[tuple[float, float]]:
    if obs is None:
        return []
    if isinstance(obs, tuple) and len(obs) == 2 and isinstance(obs[0], (int, float)):
        return [(float(obs[0]), float(obs[1]))]
    return [(float(p[0]), float(p[1])) for p in obs]  # type: ignore[union-attr]


def _draw_geometry(
    ax,
    obs: tuple[float, float] | list[tuple[float, float]] | None,
    shots: list[tuple[float, float]],
    *,
    legend: bool,
) -> None:
    x_lo, x_hi = illum_x_range()
    ax.axhline(H, color="0.15", ls="--", lw=1.2, zorder=3, label="海底" if legend else None)
    ax.axhline(
        Z_CONV, color="0.25", ls="--", lw=1.3, zorder=3, label="转换面" if legend else None
    )
    ax.axvline(x_lo, color="0.25", ls=":", lw=0.9, zorder=3)
    ax.axvline(x_hi, color="0.25", ls=":", lw=0.9, zorder=3)
    if shots:
        sx, sz = zip(*shots)
        ax.plot(
            sx,
            sz,
            marker="o",
            color="#ff7f0e",
            ms=4.5,
            mew=0.5,
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
            ms=10,
            mew=0.8,
            mec="w",
            ls="none",
            zorder=7,
            label="OBS 台" if legend else None,
        )


def _draw_rays_on(
    ax,
    rays: list[tuple[list[float], list[float]]],
    picks: list[Pick],
    *,
    legend: bool,
    thin: bool = False,
) -> None:
    del picks
    draw_psp_ps_rays(ax, rays, z_conv=Z_CONV, legend=legend, thin=thin)


def _map_xlim(shots: list[tuple[float, float]]) -> tuple[float, float]:
    if shots:
        xs = [p[0] for p in shots]
        pad = 3.0
        return min(xs) - pad, max(xs) + pad
    x_lo, x_hi = illum_x_range()
    return x_lo, x_hi


def _vp_cmap():
    from pathlib import Path as P

    from pyAOBS.modeling.tomo2d.gui.services.smesh_plot_core import (
        builtin_smesh_cmap_path,
        cmap_blank_air,
    )
    from pyAOBS.visualization.gmt_cpt import parse_gmt_cpt_for_matplotlib

    spec = str(builtin_smesh_cmap_path("vp"))
    if spec.lower().endswith(".cpt") and P(spec).is_file():
        cmap, lo, hi = parse_gmt_cpt_for_matplotlib(spec)
        return cmap_blank_air(cmap), float(lo), float(hi), spec
    from matplotlib import colormaps

    return colormaps["RdYlBu_r"], 1.40, 6.50, spec


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
    lid_m, *_ = node_stats(
        xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, z_lo=H + 1e-6, z_hi=Z_CONV
    )
    s_m, s_lo, s_hi, n_s = node_stats(
        xs,
        zs,
        vrec,
        x_lo=x_lo,
        x_hi=x_hi,
        z_lo=S_ILLUM_Z1,
        z_hi=S_ILLUM_Z2,
        z_hi_inclusive=True,
    )
    cmap, vlo, vhi, cmap_spec = _vp_cmap()
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    vnorm = Normalize(vmin=vlo, vmax=vhi, clip=False)
    dlim = 0.40
    dnorm = TwoSlopeNorm(vmin=-dlim, vcenter=0.0, vmax=dlim)
    obs, shots = (
        parse_geom(geom_path)
        if geom_path and geom_path.is_file()
        else ([(x, OBS_Z) for x in OBS_XS], [])
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

    titles = [
        f"初值  Vs0 {VS_START:g}",
        f"反演  {rec_path.name}",
        f"真值  Vs0 {VS_TRUE:g}",
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
    from pyAOBS.modeling.tomo2d.gui.services.smesh_plot_core import (
        colorbar_label_for_cmap,
    )

    cb = fig.colorbar(im0, cax=cax_v)
    cb.set_label(colorbar_label_for_cmap(cmap_spec))
    cb.set_ticks(np.linspace(vlo, vhi, 7))

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
    ax_pr.plot(sta[:, ix], zs, color="0.45", lw=1.6, label=f"初值 Vs0={VS_START:g}")
    ax_pr.plot(tru[:, ix], zs, color="C0", lw=1.8, label=f"真值 Vs0={VS_TRUE:g}")
    ax_pr.plot(rec[:, ix], zs, color="C3", lw=1.8, label="反演")
    ax_pr.axhline(H, color="0.15", ls="--", lw=1.0)
    ax_pr.axhline(Z_CONV, color="0.25", ls="--", lw=1.1)
    ax_pr.axvline(V_WATER, color="0.4", ls=":", lw=0.8)
    ax_pr.axvline(V_P_CRUST, color="0.4", ls=":", lw=0.8)
    ax_pr.set_ylim(float(zs[-1]), float(zs[0]))
    ax_pr.set_xlabel("速度 (km/s)")
    ax_pr.set_ylabel("深度 (km)")
    ax_pr.set_title(f"剖面 x={xs[ix]:.0f} km（台下）")
    ax_pr.grid(True, alpha=0.28)
    ax_pr.legend(loc="lower right", framealpha=0.9, fontsize=8)

    fig.suptitle(
        f"折合 PSP 反演（海面炮 / 海底台，-B 核冻盖层）  浅 S 均值 {s_m:.3f}"
        f"（真 Vs0 {VS_TRUE:g}，初 {VS_START:g}）"
        f"  [{s_lo:.3f},{s_hi:.3f}]  n={n_s}    盖层 {lid_m:.3f}（应 {expected_lid_mean():.2f}）",
        fontsize=11,
    )
    fig.savefig(out_png, dpi=140)
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)


def plot_inv_rays(
    smesh_path: Path,
    seafloor: Path,
    conv: Path,
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
    fig = plt.figure(figsize=(10.6, 5.8), facecolor="w", layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[28, 1.85])
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    _draw_hybrid_background(ax, cax, fig, smesh_path, seafloor, conv)
    _draw_rays_on(ax, rays, picks, legend=True, thin=len(rays) > 40)
    _draw_geometry(ax, obs, shots, legend=True)
    ax.set_xlim(*_map_xlim(shots))
    ax.set_ylim(ZMAX, 0.0)
    ax.set_title(f"反演模型上的 PSP（P 蓝 / S 粉，{len(obs)} 台海底 OBS + 海面炮）")
    ax.set_xlabel("模型距离 (km)")
    ax.set_ylabel("深度 (km)")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8, ncol=2)
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
        2, 1, figsize=(8.4, 7.0), sharex=True, gridspec_kw={"height_ratios": [1.6, 1.0]}
    )
    styles = {6: ("#d62728", "折合 PSP 6")}
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
        ax.plot(dx, ts, "--", color=color, lw=1.35, alpha=0.85, label=f"{name} 初值正演")
        ax.plot(dx, tr, "-", color=color, lw=1.6, label=f"{name} 反演正演")
        axr.plot(
            dx,
            [a - b for a, b in zip(ts, to)],
            "s",
            color=color,
            ms=5,
            alpha=0.7,
            label=f"{name} 初值−观测",
        )
        axr.plot(
            dx,
            [a - b for a, b in zip(tr, to)],
            "o",
            color=color,
            ms=6,
            label=f"{name} 反演−观测",
        )
        rms_s.extend(a - b for a, b in zip(ts, to))
        rms_r.extend(a - b for a, b in zip(tr, to))
    rs = math.sqrt(sum(v * v for v in rms_s) / len(rms_s)) if rms_s else float("nan")
    rr = math.sqrt(sum(v * v for v in rms_r) / len(rms_r)) if rms_r else float("nan")
    ax.set_ylabel("走时 t (s)")
    ax.set_title(f"走时拟合：观测=真模型正演    RMS 初值 {rs:.3f} s → 反演 {rr:.3f} s")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower left", fontsize=8, ncol=2, framealpha=0.9)
    axr.axhline(0.0, color="0.4", lw=0.8)
    axr.set_xlabel("偏移 dx (km)")
    axr.set_ylabel("残差 (s)")
    axr.grid(True, alpha=0.35)
    axr.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)
    print(f"  ttimes RMS  start={rs:.4f} s  rec={rr:.4f} s  n={len(rows)}")


def report(rec_path: Path, true_path: Path, start_path: Path, *, dual: bool = False) -> int:
    xs, zs, vrec = parse_smesh(rec_path)
    _, _, vtrue = parse_smesh(true_path)
    _, _, vstart = parse_smesh(start_path)
    x_lo, x_hi = illum_x_range()
    w_m, w_lo, w_hi, n_w = node_stats(
        xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, z_lo=0.0, z_hi=H, z_hi_inclusive=True
    )
    lid_m, lid_lo, lid_hi, n_lid = node_stats(
        xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, z_lo=H + 1e-6, z_hi=Z_CONV
    )
    lid_up_m, *_ = node_stats(
        xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, z_lo=H + 1e-6, z_hi=H + 1.0
    )
    s_m, s_lo, s_hi, n_s = node_stats(
        xs,
        zs,
        vrec,
        x_lo=x_lo,
        x_hi=x_hi,
        z_lo=S_ILLUM_Z1,
        z_hi=S_ILLUM_Z2,
        z_hi_inclusive=True,
    )
    t_s, *_ = node_stats(
        xs,
        zs,
        vtrue,
        x_lo=x_lo,
        x_hi=x_hi,
        z_lo=S_ILLUM_Z1,
        z_hi=S_ILLUM_Z2,
        z_hi_inclusive=True,
    )
    st_s, *_ = node_stats(
        xs,
        zs,
        vstart,
        x_lo=x_lo,
        x_hi=x_hi,
        z_lo=S_ILLUM_Z1,
        z_hi=S_ILLUM_Z2,
        z_hi_inclusive=True,
    )
    print(f"recovered  {rec_path.name}")
    print(
        f"  water  n={n_w}  mean={w_m:.4f}  [{w_lo:.4f},{w_hi:.4f}]  expect={V_WATER}"
    )
    print(
        f"  lid    n={n_lid}  mean={lid_m:.4f}  [{lid_lo:.4f},{lid_hi:.4f}]  "
        f"expect≈{expected_lid_mean():.2f}  upper(z<{H+1:.0f})={lid_up_m:.4f}"
    )
    print(
        f"  S illum z=[{S_ILLUM_Z1:.1f},{S_ILLUM_Z2:.1f}]  n={n_s}  "
        f"mean={s_m:.4f}  [{s_lo:.4f},{s_hi:.4f}]"
    )
    print(
        f"           true={t_s:.4f}  start={st_s:.4f}  |mean-true|={abs(s_m - t_s):.4f}"
    )
    ok = True
    if abs(w_m - V_WATER) > 0.01 or abs(w_hi - V_WATER) > 0.05:
        print(f"FAIL water not frozen at {V_WATER}（反演应加 -Y -w）")
        ok = False
    if dual:
        st_lid, *_ = node_stats(
            xs, zs, vstart, x_lo=x_lo, x_hi=x_hi, z_lo=H + 1e-6, z_hi=Z_CONV
        )
        print(f"           lid start={st_lid:.4f}  (dual: freeze lid Vs)")
        if abs(lid_m - st_lid) > 0.03:
            print(
                f"FAIL lid Vs moved {abs(lid_m - st_lid):.3f} from start "
                f"(psx 只反面下应冻盖层)"
            )
            ok = False
    elif abs(lid_m - expected_lid_mean()) > 0.03:
        print(
            f"FAIL lid not frozen (mean {lid_m:.3f}, expect {expected_lid_mean():.3f}; "
            f"-B 应把转换面以上从核/平滑/dm 里去掉)"
        )
        ok = False
    if abs(s_m - t_s) > 0.15:
        print(f"FAIL shallow S mean not within 0.15 of true {t_s:.3f}")
        ok = False
    if abs(s_m - t_s) >= abs(st_s - t_s) - 1e-6:
        print("FAIL shallow S did not move toward true vs start")
        ok = False
    if ok:
        print("OK")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=HERE / "out")
    p.add_argument("--true", dest="true_smesh", type=Path, default=HERE / "true.smesh")
    p.add_argument("--start", dest="start_smesh", type=Path, default=HERE / "start.smesh")
    p.add_argument(
        "--smesh",
        type=Path,
        default=None,
        help="直接指定反演 smesh，否则取 --out-root 最新一步",
    )
    p.add_argument("--geom", type=Path, default=HERE / "geom_inv.dat")
    p.add_argument("--obs", dest="syn_obs", type=Path, default=HERE / "syn_inv.dat")
    p.add_argument("--syn-start", type=Path, default=HERE / "syn_start.dat")
    p.add_argument("--syn-rec", type=Path, default=HERE / "syn_rec.dat")
    p.add_argument("--rays", type=Path, default=HERE / "rays_rec.dat")
    p.add_argument("--seafloor", type=Path, default=HERE / "seafloor.refl")
    p.add_argument("--conv", type=Path, default=HERE / "conv.refl")
    p.add_argument("--no-show", action="store_true")
    p.add_argument(
        "--dual",
        action="store_true",
        help="双场 Vs：盖层对照初值（冻），面下对照 true_vs",
    )
    args = p.parse_args()
    rec = args.smesh if args.smesh else latest_smesh(args.out_root)
    rc = report(rec, args.true_smesh, args.start_smesh, dual=args.dual)
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
    tag = "psx" if args.dual else "inv"
    models_png = HERE / f"check_{tag}_models.png"
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
    if rays and picks_rec and rec.is_file() and args.conv.is_file() and geom:
        rays_png = HERE / f"check_{tag}_rays.png"
        plot_inv_rays(
            rec,
            args.seafloor,
            args.conv,
            rays,
            picks_rec,
            geom,
            rays_png,
            show=show,
        )
        print(f"wrote {rays_png}")
    else:
        print(
            f"缺射线/走时，跳过 check_inv_rays.png"
            f"（对收回模型跑 tt_forward -X -R{args.rays.name} > {args.syn_rec.name}）",
            file=sys.stderr,
        )
    if args.syn_obs.is_file() and args.syn_start.is_file() and args.syn_rec.is_file():
        t_png = HERE / f"check_{tag}_ttimes.png"
        plot_ttimes_fit(
            parse_picks(args.syn_obs.read_text(encoding="utf-8")),
            parse_picks(args.syn_start.read_text(encoding="utf-8")),
            parse_picks(args.syn_rec.read_text(encoding="utf-8")),
            t_png,
            show=show,
        )
        print(f"wrote {t_png}")
    else:
        print(
            "缺 syn_inv / syn_start / syn_rec，跳过走时拟合图"
            "（初值、收回模型各跑一次 tt_forward -X）",
            file=sys.stderr,
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
