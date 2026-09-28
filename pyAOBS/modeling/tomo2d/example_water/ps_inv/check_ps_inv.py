#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对照 PSP 面下 Vs 反演：模型图、射线（P 蓝 / S 粉）、走时拟合。"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ps_fwd"))
sys.path.insert(0, str(HERE.parent / "converse_fwd"))
sys.path.insert(0, str(HERE.parent / "water_fwd"))
sys.path.insert(0, str(HERE.parent / "water_inv"))
from check_analytic import _setup_mpl, parse_ray_file  # noqa: E402
from check_converse_fwd import P_RAY_COLOR, S_RAY_COLOR  # noqa: E402
from check_ps_fwd import draw_ps_rays  # noqa: E402
from check_water_inv import (  # noqa: E402
    Pick,
    _align_picks,
    _grid,
    latest_smesh,
    parse_geom,
    parse_picks,
)
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

S_Z1 = Z_CONV
S_Z2 = Z_CONV + 2.5
VS_VMIN, VS_VMAX = 0.80, 4.20
VP_VMIN, VP_VMAX = 1.45, 8.00


def _layer_ok(
    rec_m: float,
    true_m: float,
    start_m: float,
    *,
    name: str,
    abs_tol: float,
) -> bool:
    ok = True
    if abs(rec_m - true_m) >= abs(start_m - true_m) - 1e-6:
        print(f"FAIL {name} did not move toward true vs start")
        ok = False
    if abs(rec_m - true_m) > abs_tol:
        print(f"FAIL {name} mean not within {abs_tol:g} of true {true_m:.3f}")
        ok = False
    return ok


def report_vp(rec_path: Path, true_path: Path, start_path: Path) -> int:
    xs, zs, vrec = parse_smesh(rec_path)
    _, _, vtrue = parse_smesh(true_path)
    _, _, vstart = parse_smesh(start_path)
    x_lo, x_hi = illum_x_range()
    w_m, w_lo, w_hi, n_w = node_stats(
        xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, z_lo=0.0, z_hi=H, z_hi_inclusive=False
    )
    lid_m, lid_lo, lid_hi, n_lid = node_stats(
        xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, z_lo=H + 1e-6, z_hi=Z_CONV
    )
    t_lid, *_ = node_stats(
        xs, zs, vtrue, x_lo=x_lo, x_hi=x_hi, z_lo=H + 1e-6, z_hi=Z_CONV
    )
    st_lid, *_ = node_stats(
        xs, zs, vstart, x_lo=x_lo, x_hi=x_hi, z_lo=H + 1e-6, z_hi=Z_CONV
    )
    s_m, s_lo, s_hi, n_s = node_stats(
        xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, z_lo=S_Z1, z_hi=S_Z2, z_hi_inclusive=True
    )
    t_s, *_ = node_stats(
        xs, zs, vtrue, x_lo=x_lo, x_hi=x_hi, z_lo=S_Z1, z_hi=S_Z2, z_hi_inclusive=True
    )
    st_s, *_ = node_stats(
        xs, zs, vstart, x_lo=x_lo, x_hi=x_hi, z_lo=S_Z1, z_hi=S_Z2, z_hi_inclusive=True
    )
    print(f"recovered  {rec_path.name}  (Vp, PPP)")
    print(f"  water  n={n_w}  mean={w_m:.4f}  [{w_lo:.4f},{w_hi:.4f}]  expect={V_WATER}")
    print(
        f"  lid    n={n_lid}  mean={lid_m:.4f}  [{lid_lo:.4f},{lid_hi:.4f}]  "
        f"true={t_lid:.4f}  start={st_lid:.4f}"
    )
    print(
        f"  below  n={n_s}  mean={s_m:.4f}  [{s_lo:.4f},{s_hi:.4f}]  "
        f"true={t_s:.4f}  start={st_s:.4f}"
    )
    ok = True
    if abs(w_m - V_WATER) > 0.02 or abs(w_hi - V_WATER) > 0.08:
        print(f"FAIL water not frozen at {V_WATER}（PPP 应加 -Y -w）")
        ok = False
    ok = _layer_ok(lid_m, t_lid, st_lid, name="lid Vp", abs_tol=0.20) and ok
    ok = _layer_ok(s_m, t_s, st_s, name="below-conv Vp", abs_tol=0.30) and ok
    if ok:
        print("OK  PPP Vp")
    return 0 if ok else 1


def report(rec_path: Path, true_path: Path, start_path: Path) -> int:
    xs, zs, vrec = parse_smesh(rec_path)
    _, _, vtrue = parse_smesh(true_path)
    _, _, vstart = parse_smesh(start_path)
    x_lo, x_hi = illum_x_range()
    w_m, w_lo, w_hi, n_w = node_stats(
        xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, z_lo=0.0, z_hi=H, z_hi_inclusive=False
    )
    lid_m, lid_lo, lid_hi, n_lid = node_stats(
        xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, z_lo=H + 1e-6, z_hi=Z_CONV
    )
    t_lid, *_ = node_stats(
        xs, zs, vtrue, x_lo=x_lo, x_hi=x_hi, z_lo=H + 1e-6, z_hi=Z_CONV
    )
    st_lid, *_ = node_stats(
        xs, zs, vstart, x_lo=x_lo, x_hi=x_hi, z_lo=H + 1e-6, z_hi=Z_CONV
    )
    s_m, s_lo, s_hi, n_s = node_stats(
        xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, z_lo=S_Z1, z_hi=S_Z2, z_hi_inclusive=True
    )
    t_s, *_ = node_stats(
        xs, zs, vtrue, x_lo=x_lo, x_hi=x_hi, z_lo=S_Z1, z_hi=S_Z2, z_hi_inclusive=True
    )
    st_s, *_ = node_stats(
        xs, zs, vstart, x_lo=x_lo, x_hi=x_hi, z_lo=S_Z1, z_hi=S_Z2, z_hi_inclusive=True
    )
    print(f"recovered  {rec_path.name}  (Vs field)")
    print(f"  water  n={n_w}  mean={w_m:.4f}  [{w_lo:.4f},{w_hi:.4f}]  expect={V_WATER}")
    print(
        f"  lid    n={n_lid}  mean={lid_m:.4f}  [{lid_lo:.4f},{lid_hi:.4f}]  "
        f"true={t_lid:.4f}  start={st_lid:.4f}"
    )
    print(
        f"  below  n={n_s}  mean={s_m:.4f}  [{s_lo:.4f},{s_hi:.4f}]  "
        f"true={t_s:.4f}  start={st_s:.4f}"
    )
    ok = True
    if abs(w_m - V_WATER) > 0.02 or abs(w_hi - V_WATER) > 0.08:
        print(f"FAIL water not frozen at {V_WATER}（应加 -Y -w；水结点勿按 κ 缩放）")
        ok = False
    if abs(lid_m - st_lid) > 0.25:
        print(f"FAIL lid Vs moved {abs(lid_m - st_lid):.3f} from start (PSP 核不应写盖层)")
        ok = False
    else:
        print(f"  lid Vs stayed near start (PSP 无盖层核)")
    ok = _layer_ok(s_m, t_s, st_s, name="below-conv Vs", abs_tol=0.30) and ok
    if ok:
        print(f"OK  PSP 面下 Vs（冻收回 Vp，κ_start={KAPPA_START:g}）")
    return 0 if ok else 1


def _draw_geometry(ax, obs, shots, *, legend: bool) -> None:
    x_lo, x_hi = illum_x_range()
    ax.axhline(H, color="0.15", ls="--", lw=1.2, zorder=3, label="海底" if legend else None)
    ax.axhline(Z_CONV, color="0.25", ls="--", lw=1.3, zorder=3, label="转换面" if legend else None)
    ax.axvline(x_lo, color="0.25", ls=":", lw=0.9, zorder=3)
    ax.axvline(x_hi, color="0.25", ls=":", lw=0.9, zorder=3)
    if shots:
        sx, sz = zip(*shots)
        ax.plot(
            sx, sz, marker="o", color="#ff7f0e", ms=4.0, mew=0.4, mec="k",
            ls="none", zorder=6, label="炮" if legend else None,
        )
    if obs:
        ox, oz = zip(*obs)
        ax.plot(
            ox, oz, marker="^", color="k", ms=9, mew=0.7, mec="w",
            ls="none", zorder=7, label="OBS 台" if legend else None,
        )


def plot_models(
    rec_path: Path,
    true_path: Path,
    start_path: Path,
    out_png: Path,
    *,
    show: bool,
    geom_path: Path | None = None,
    rays: list[tuple[list[float], list[float]]] | None = None,
    recs: list[tuple[int, float, float]] | None = None,
    kind: str = "vs",
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
    lid_m, *_ = node_stats(xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, z_lo=H + 1e-6, z_hi=Z_CONV)
    s_m, *_ = node_stats(
        xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, z_lo=S_Z1, z_hi=S_Z2, z_hi_inclusive=True
    )
    obs, shots = (
        parse_geom(geom_path)
        if geom_path and geom_path.is_file()
        else ([(x, OBS_Z) for x in OBS_XS], [])
    )
    xlim = (min(x_lo, 18.0), max(x_hi, 82.0))
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    is_vp = kind == "vp"
    vlo, vhi = (VP_VMIN, VP_VMAX) if is_vp else (VS_VMIN, VS_VMAX)
    vnorm = Normalize(vmin=vlo, vmax=vhi, clip=False)
    dnorm = TwoSlopeNorm(vmin=-0.50, vcenter=0.0, vmax=0.50)
    unit = "Vp" if is_vp else "Vs"

    fig = plt.figure(figsize=(12.6, 8.4), facecolor="w", layout="constrained")
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 0.055], height_ratios=[1.05, 1.0])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    cax_v = fig.add_subplot(gs[0, 3])
    ax_ds = fig.add_subplot(gs[1, 0])
    ax_dt = fig.add_subplot(gs[1, 1])
    ax_pr = fig.add_subplot(gs[1, 2])
    cax_d = fig.add_subplot(gs[1, 3])

    titles = (
        ["初值 Vp", f"反演  {rec_path.name}", "真值 Vp"]
        if is_vp
        else [
            f"初值  Vs=收回Vp/{KAPPA_START:g}",
            f"反演  {rec_path.name}",
            f"真值  Vs=Vp/{KAPPA_TRUE:g}",
        ]
    )
    im0 = None
    for i, (ax, data, title) in enumerate(zip(axes, (sta, rec, tru), titles)):
        im0 = ax.imshow(
            data, extent=extent, cmap="RdYlBu_r", norm=vnorm,
            aspect="auto", interpolation="nearest", zorder=0,
        )
        if i == 1 and rays and recs:
            draw_ps_rays(ax, rays, recs, thin=True, mark_conv=False)
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
            ax.legend(loc="upper right", framealpha=0.9, fontsize=7, ncol=2)

    cb = fig.colorbar(im0, cax=cax_v)
    cb.set_label(f"{unit} (km/s)")
    im_d = ax_ds.imshow(rec - sta, extent=extent, cmap="RdBu_r", norm=dnorm,
                        aspect="auto", interpolation="nearest", zorder=0)
    ax_dt.imshow(rec - tru, extent=extent, cmap="RdBu_r", norm=dnorm,
                 aspect="auto", interpolation="nearest", zorder=0)
    for ax, title, ylab in ((ax_ds, "反演 − 初值", True), (ax_dt, "反演 − 真值", False)):
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
        (
            f"PPP 反 Vp  盖层均值 {lid_m:.3f}  面下 {s_m:.3f}"
            if is_vp
            else f"PSP 面下 Vs（冻收回 Vp，-k{KAPPA_START:g}）  盖层均值 {lid_m:.3f}  面下 {s_m:.3f}"
        ),
        fontsize=11,
    )
    fig.savefig(out_png, dpi=140)
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)
    del np


def plot_inv_rays(
    smesh_path: Path,
    rays: list[tuple[list[float], list[float]]],
    recs: list[tuple[int, float, float]],
    geom_path: Path,
    out_png: Path,
    *,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    _setup_mpl()
    xs, zs, vel = parse_smesh(smesh_path)
    grid = _grid(xs, zs, vel)
    obs, shots = parse_geom(geom_path)
    fig, ax6 = plt.subplots(figsize=(11.2, 5.2), facecolor="w", layout="constrained")
    im = None
    for ax, codes, title in (
        (ax6, (6,), f"PSP（{len(obs)} 台）：盖层 P、面下 S（只反面下）"),
    ):
        im = ax.imshow(
            grid,
            extent=(float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0])),
            cmap="RdYlBu_r",
            norm=Normalize(vmin=VS_VMIN, vmax=VS_VMAX),
            aspect="auto",
            interpolation="nearest",
            zorder=0,
        )
        draw_ps_rays(ax, rays, recs, thin=len(rays) > 40, codes=codes)
        _draw_geometry(ax, obs, shots, legend=True)
        ax.set_xlim(*illum_x_range())
        ax.set_ylim(ZMAX, 0.0)
        ax.set_title(title)
        ax.set_ylabel("深度 (km)")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=7, ncol=2)
    ax6.set_xlabel("模型距离 (km)")
    fig.colorbar(im, ax=ax6, shrink=0.72).set_label("Vs (km/s)")
    fig.savefig(out_png, dpi=140)
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)


def plot_ttimes_fit(
    obs: list[Pick],
    start: list[Pick],
    rec: list[Pick] | None,
    out_png: Path,
    *,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    _setup_mpl()
    groups = (obs, start, rec) if rec else (obs, start)
    rows = _align_picks(*groups)
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(8.6, 7.0), sharex=True, gridspec_kw={"height_ratios": [1.6, 1.0]}
    )
    styles = {
        0: ("#1f77b4", "PPP 0"),
        6: ("#2ca02c", "PSP 6"),
        7: (P_RAY_COLOR, "PPS 7"),
        8: (S_RAY_COLOR, "PSS 8"),
    }
    rms_s: list[float] = []
    rms_r: list[float] = []
    for code, (color, name) in styles.items():
        sub = [r for r in rows if r[0][0] == code]
        if not sub:
            continue
        dx = [abs(o[1] - o[4]) for o, *_rest in sub]
        to = [o[3] for o, *_rest in sub]
        ts = [s[3] for _o, s, *_rest in sub]
        order = sorted(range(len(dx)), key=lambda i: dx[i])
        dx = [dx[i] for i in order]
        to = [to[i] for i in order]
        ts = [ts[i] for i in order]
        ax.plot(dx, to, "o", color=color, ms=5, zorder=3, label=f"{name} 观测")
        ax.plot(dx, ts, "--", color=color, lw=1.3, alpha=0.85, label=f"{name} 初值正演")
        axr.plot(dx, [a - b for a, b in zip(ts, to)], "s", color=color, ms=4, alpha=0.7,
                 label=f"{name} 初值−观测")
        rms_s.extend(a - b for a, b in zip(ts, to))
        if rec:
            tr = [r[3] for _o, _s, r in sub]
            tr = [tr[i] for i in order]
            ax.plot(dx, tr, "-", color=color, lw=1.5, label=f"{name} 反演正演")
            axr.plot(dx, [a - b for a, b in zip(tr, to)], "o", color=color, ms=5,
                     label=f"{name} 反演−观测")
            rms_r.extend(a - b for a, b in zip(tr, to))
    rs = math.sqrt(sum(v * v for v in rms_s) / len(rms_s)) if rms_s else float("nan")
    rr = math.sqrt(sum(v * v for v in rms_r) / len(rms_r)) if rms_r else float("nan")
    title = f"走时拟合    RMS 初值 {rs:.3f} s"
    if rec:
        title += f" → 反演 {rr:.3f} s"
    ax.set_ylabel("走时 t (s)")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower left", fontsize=7, ncol=2, framealpha=0.9)
    axr.axhline(0.0, color="0.4", lw=0.8)
    axr.set_xlabel("偏移 dx (km)")
    axr.set_ylabel("残差 (s)")
    axr.grid(True, alpha=0.35)
    axr.legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)
    extra = f"  rec={rr:.4f} s" if rec else ""
    print(f"  ttimes RMS  start={rs:.4f} s{extra}  n={len(rows)}")


def _syn_recs(path: Path) -> list[tuple[int, float, float]]:
    from check_ps_fwd import parse_syn

    return parse_syn(path.read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=HERE / "out")
    p.add_argument("--true", dest="true_smesh", type=Path, default=HERE / "true_vs.smesh")
    p.add_argument("--start", dest="start_smesh", type=Path, default=HERE / "start_vs.smesh")
    p.add_argument("--smesh", type=Path, default=None)
    p.add_argument("--vp-rec", type=Path, default=HERE / "rec_vp.smesh")
    p.add_argument("--vp-true", type=Path, default=HERE / "true_vp.smesh")
    p.add_argument("--vp-start", type=Path, default=HERE / "start_vp.smesh")
    p.add_argument("--geom", type=Path, default=HERE / "geom_inv.dat")
    p.add_argument("--geom-ppp", type=Path, default=HERE / "geom_ppp.dat")
    p.add_argument("--obs", dest="syn_obs", type=Path, default=HERE / "syn_inv.dat")
    p.add_argument("--syn-start", type=Path, default=HERE / "syn_start.dat")
    p.add_argument("--syn-rec", type=Path, default=HERE / "syn_rec.dat")
    p.add_argument("--syn-ppp", type=Path, default=HERE / "syn_ppp.dat")
    p.add_argument("--syn-ppp-start", type=Path, default=HERE / "syn_ppp_start.dat")
    p.add_argument("--syn-ppp-rec", type=Path, default=HERE / "syn_ppp_rec.dat")
    p.add_argument("--rays", type=Path, default=HERE / "rays_true.dat")
    p.add_argument("--rays-ppp", type=Path, default=HERE / "rays_ppp_rec.dat")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()
    rec = args.smesh if args.smesh else latest_smesh(args.out_root)
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    show = not args.no_show
    rc = 0
    if args.vp_rec.is_file() and args.vp_true.is_file() and args.vp_start.is_file():
        rc |= report_vp(args.vp_rec, args.vp_true, args.vp_start)
        ppp_rays = parse_ray_file(args.rays_ppp) if args.rays_ppp.is_file() else []
        ppp_recs = _syn_recs(args.syn_ppp) if args.syn_ppp.is_file() else []
        vp_png = HERE / "check_inv_vp_models.png"
        plot_models(
            args.vp_rec,
            args.vp_true,
            args.vp_start,
            vp_png,
            show=show,
            geom_path=args.geom_ppp if args.geom_ppp.is_file() else None,
            rays=ppp_rays or None,
            recs=ppp_recs or None,
            kind="vp",
        )
        print(f"wrote {vp_png}")
        if args.syn_ppp.is_file() and args.syn_ppp_start.is_file():
            ppp_rec_picks = (
                parse_picks(args.syn_ppp_rec.read_text(encoding="utf-8"))
                if args.syn_ppp_rec.is_file()
                else None
            )
            vp_t_png = HERE / "check_inv_vp_ttimes.png"
            plot_ttimes_fit(
                parse_picks(args.syn_ppp.read_text(encoding="utf-8")),
                parse_picks(args.syn_ppp_start.read_text(encoding="utf-8")),
                ppp_rec_picks,
                vp_t_png,
                show=show,
            )
            print(f"wrote {vp_t_png}")
    else:
        print("缺 rec_vp / true_vp / start_vp，跳过 PPP 对照", file=sys.stderr)

    rc |= report(rec, args.true_smesh, args.start_smesh)
    rays: list[tuple[list[float], list[float]]] = []
    recs: list[tuple[int, float, float]] = []
    if args.rays.is_file():
        rays = parse_ray_file(args.rays)
    if args.syn_obs.is_file():
        recs = _syn_recs(args.syn_obs)
    models_png = HERE / "check_inv_models.png"
    plot_models(
        rec,
        args.true_smesh,
        args.start_smesh,
        models_png,
        show=show,
        geom_path=args.geom if args.geom.is_file() else None,
        rays=rays or None,
        recs=recs or None,
    )
    print(f"wrote {models_png}")
    if rays and recs and rec.is_file() and args.geom.is_file():
        rays_png = HERE / "check_inv_rays.png"
        plot_inv_rays(rec, rays, recs, args.geom, rays_png, show=show)
        print(f"wrote {rays_png}")
    else:
        print("缺射线，跳过 check_inv_rays.png（tt_forward -R rays_true.dat）", file=sys.stderr)
    if args.syn_obs.is_file() and args.syn_start.is_file():
        rec_picks = (
            parse_picks(args.syn_rec.read_text(encoding="utf-8"))
            if args.syn_rec.is_file()
            else None
        )
        t_png = HERE / "check_inv_ttimes.png"
        plot_ttimes_fit(
            parse_picks(args.syn_obs.read_text(encoding="utf-8")),
            parse_picks(args.syn_start.read_text(encoding="utf-8")),
            rec_picks,
            t_png,
            show=show,
        )
        print(f"wrote {t_png}")
    else:
        print("缺 syn_inv / syn_start，跳过走时拟合图", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
