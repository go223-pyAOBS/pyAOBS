#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对照反演结果：模型、台/炮几何、射线、走时拟合。

用法（在本目录）:
  python check_water_inv.py
  python check_water_inv.py --no-show   # 只写 PNG，不弹窗
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "water_fwd"))
from check_analytic import (  # noqa: E402
    _PHASE_RAY,
    _draw_smesh_background,
    _setup_mpl,
    parse_ray_file,
)
from make_water_inv_case import (  # noqa: E402
    H,
    OBS_X,
    OBS_XS,
    V_SEDIMENT,
    V_START,
    V_TRUE,
    illum_x_range,
    node_stats,
    parse_smesh,
)

# code, x, z, t, src_x
Pick = tuple[int, float, float, float, float]


def latest_smesh(out_root: Path) -> Path:
    parent = out_root.parent if out_root.suffix else out_root.parent
    stem = out_root.name
    cands = sorted(parent.glob(f"{stem}.smesh.*.*"))
    if not cands:
        raise FileNotFoundError(f"找不到 {stem}.smesh.<iter>.<iset> 于 {parent}")

    def key(p: Path) -> tuple[int, int]:
        parts = p.name.split(".")
        return int(parts[-2]), int(parts[-1])

    return max(cands, key=key)


def parse_geom(
    path: Path,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """返回 OBS 台列表与不重复炮点 (x,z)。"""
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    stations: list[tuple[float, float]] = []
    shots: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    i = 1
    while i < len(lines):
        parts = lines[i].split()
        i += 1
        if not parts:
            continue
        if parts[0] == "s":
            stations.append((float(parts[1]), float(parts[2])))
            nrcv = int(float(parts[-1]))
            for _ in range(nrcv):
                rp = lines[i].split()
                i += 1
                xy = (float(rp[1]), float(rp[2]))
                if xy not in seen:
                    seen.add(xy)
                    shots.append(xy)
    if not stations:
        raise ValueError(f"{path}: 没有 s 行")
    return stations, shots


def parse_picks(text: str) -> list[Pick]:
    recs: list[Pick] = []
    src_x = OBS_X
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    i = 1
    while i < len(lines):
        parts = lines[i].split()
        i += 1
        if not parts:
            continue
        if parts[0] == "s":
            src_x = float(parts[1])
            nrcv = int(float(parts[-1]))
            for _ in range(nrcv):
                rp = lines[i].split()
                i += 1
                x = float(rp[1])
                z = float(rp[2])
                code = int(float(rp[3]))
                t = float(rp[4])
                recs.append((code, x, z, t, src_x))
    return recs


def _pick_key(p: Pick) -> tuple[int, float, float]:
    return p[0], round(p[1], 3), round(p[4], 3)


def _align_picks(*groups: list[Pick]) -> list[tuple[Pick, ...]]:
    maps = [{_pick_key(p): p for p in g} for g in groups]
    keys = [k for k in maps[0] if all(k in m for m in maps[1:])]
    keys.sort(key=lambda k: (k[0], k[1]))
    return [tuple(m[k] for m in maps) for k in keys]


def _grid(xs: list[float], zs: list[float], vel: list[list[float]]):
    import numpy as np

    arr = np.asarray(vel, dtype=float).T
    if arr.shape != (len(zs), len(xs)):
        raise ValueError(f"grid shape {arr.shape} != ({len(zs)}, {len(xs)})")
    return arr


def _water_cmap():
    from pathlib import Path as P

    from pyAOBS.modeling.tomo2d.gui.services.smesh_plot_core import (
        builtin_smesh_cmap_path,
        cmap_blank_air,
    )
    from pyAOBS.visualization.gmt_cpt import parse_gmt_cpt_for_matplotlib

    spec = str(builtin_smesh_cmap_path("water"))
    if spec.lower().endswith(".cpt") and P(spec).is_file():
        cmap, lo, hi = parse_gmt_cpt_for_matplotlib(spec)
        return cmap_blank_air(cmap), float(lo), float(hi), spec
    from matplotlib import colormaps

    return colormaps["RdYlBu_r"], 1.35, 1.65, spec


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
    ax.axhline(H, color="crimson", ls="--", lw=1.3, zorder=3, label="海底" if legend else None)
    ax.axvline(x_lo, color="0.25", ls=":", lw=0.9, zorder=3)
    ax.axvline(x_hi, color="0.25", ls=":", lw=0.9, zorder=3)
    if shots:
        sx, sz = zip(*shots)
        ax.plot(
            sx,
            sz,
            marker="o",
            color="#ff7f0e",
            ms=5.5,
            mew=0.6,
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
            ms=11,
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
    n = min(len(rays), len(picks))
    seen: set[int] = set()
    for i in range(n):
        xs, zs = rays[i]
        code = picks[i][0]
        color, ls, name = _PHASE_RAY.get(code, ("#111111", "-", f"code {code}"))
        halo_w, core_w = (1.15, 0.48) if thin else (1.55, 0.72)
        if code == 3:
            halo_w += 0.2
            core_w += 0.05
        ax.plot(
            xs,
            zs,
            color="#FFFFFF",
            lw=halo_w,
            ls=ls,
            alpha=0.9,
            zorder=4,
            solid_capstyle="round",
        )
        kw: dict = dict(
            color=color,
            lw=core_w,
            ls=ls,
            alpha=0.95,
            zorder=4.1,
            solid_capstyle="round",
        )
        if legend and code not in seen:
            kw["label"] = name
            seen.add(code)
        ax.plot(xs, zs, **kw)


def _map_xlim(shots: list[tuple[float, float]]) -> tuple[float, float]:
    if shots:
        xs = [p[0] for p in shots]
        pad = 3.0
        return min(xs) - pad, max(xs) + pad
    x_lo, x_hi = illum_x_range()
    return x_lo, x_hi


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
    w_m, w_lo, w_hi, n_w = node_stats(xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, water=True)
    s_m, _, _, _ = node_stats(xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, water=False)
    cmap, vlo, vhi, cmap_spec = _water_cmap()
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    vnorm = Normalize(vmin=vlo, vmax=vhi, clip=False)
    dlim = 0.20
    dnorm = TwoSlopeNorm(vmin=-dlim, vcenter=0.0, vmax=dlim)
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

    titles = [
        f"初值  水 {V_START:g}",
        f"反演  {rec_path.name}",
        f"真值  水 {V_TRUE:g}",
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
    ax_pr.plot(sta[:, ix], zs, color="0.45", lw=1.6, label=f"初值 {V_START:g}")
    ax_pr.plot(tru[:, ix], zs, color="C0", lw=1.8, label=f"真值 {V_TRUE:g}")
    ax_pr.plot(rec[:, ix], zs, color="C3", lw=1.8, label="反演")
    ax_pr.axhline(H, color="crimson", ls="--", lw=1.1)
    ax_pr.axvline(V_TRUE, color="C0", ls=":", lw=0.8)
    ax_pr.axvline(V_SEDIMENT, color="0.3", ls=":", lw=0.8)
    ax_pr.set_ylim(float(zs[-1]), float(zs[0]))
    ax_pr.set_xlabel("速度 (km/s)")
    ax_pr.set_ylabel("深度 (km)")
    ax_pr.set_title(f"剖面 x={xs[ix]:.0f} km（台下）")
    ax_pr.grid(True, alpha=0.28)
    ax_pr.legend(loc="lower right", framealpha=0.9)

    fig.suptitle(
        f"水核反演（-y 冻壳）  照明区水均值 {w_m:.3f}（真 {V_TRUE:g}，初 {V_START:g}）"
        f"  [{w_lo:.3f},{w_hi:.3f}]  n={n_w}    沉积 {s_m:.3f}（应 {V_SEDIMENT:g}）",
        fontsize=11,
    )
    fig.savefig(out_png, dpi=140)
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)


def plot_inv_rays(
    smesh_path: Path,
    refl_path: Path,
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
    fig = plt.figure(figsize=(10.4, 5.6), facecolor="w", layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[28, 1.85])
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    _draw_smesh_background(ax, cax, fig, smesh_path, refl_path)
    _draw_rays_on(ax, rays, picks, legend=True, thin=len(rays) > 60)
    _draw_geometry(ax, obs, shots, legend=True)
    ax.set_xlim(*_map_xlim(shots))
    ax.set_title(f"反演模型上的水波射线（直达 2 / 多次 3，{len(obs)} 台）")
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
        2, 1, figsize=(8.2, 7.0), sharex=True, gridspec_kw={"height_ratios": [1.6, 1.0]}
    )
    styles = {
        2: ("#9B30FF", "直达 2"),
        3: ("#333333", "多次 3"),
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
        ax.plot(dx, to, "o", color=color, ms=7, zorder=3, label=f"{name} 观测")
        ax.plot(dx, ts, "--", color=color, lw=1.35, alpha=0.85, label=f"{name} 初值正演")
        ax.plot(dx, tr, "-", color=color, lw=1.6, label=f"{name} 反演正演")
        axr.plot(dx, [a - b for a, b in zip(ts, to)], "s", color=color, ms=5, alpha=0.7, label=f"{name} 初值−观测")
        axr.plot(dx, [a - b for a, b in zip(tr, to)], "o", color=color, ms=6, label=f"{name} 反演−观测")
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


def report(rec_path: Path, true_path: Path, start_path: Path) -> int:
    xs, zs, vrec = parse_smesh(rec_path)
    _, _, vtrue = parse_smesh(true_path)
    _, _, vstart = parse_smesh(start_path)
    x_lo, x_hi = illum_x_range()
    w_m, w_lo, w_hi, n_w = node_stats(xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, water=True)
    s_m, s_lo, s_hi, n_s = node_stats(xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, water=False)
    t_m, _, _, _ = node_stats(xs, zs, vtrue, x_lo=x_lo, x_hi=x_hi, water=True)
    st_m, _, _, _ = node_stats(xs, zs, vstart, x_lo=x_lo, x_hi=x_hi, water=True)
    print(f"recovered  {rec_path.name}")
    print(
        f"  water  illum x=[{x_lo:.1f},{x_hi:.1f}]  n={n_w}  "
        f"mean={w_m:.4f}  [{w_lo:.4f},{w_hi:.4f}]"
    )
    print(f"           true={t_m:.4f}  start={st_m:.4f}  |mean-true|={abs(w_m - V_TRUE):.4f}")
    print(f"  sed    n={n_s}  mean={s_m:.4f}  [{s_lo:.4f},{s_hi:.4f}]  expect={V_SEDIMENT}")
    ok = True
    if abs(w_m - V_TRUE) > 0.03:
        print(f"FAIL water mean not within 0.03 of {V_TRUE}")
        ok = False
    if abs(w_m - V_TRUE) >= abs(st_m - V_TRUE):
        print("FAIL water did not move toward true vs start")
        ok = False
    if abs(s_m - V_SEDIMENT) > 1e-3 or abs(s_hi - V_SEDIMENT) > 1e-3:
        print("FAIL sediment not frozen")
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
    p.add_argument("--refl", type=Path, default=HERE / "seafloor.refl")
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
    if rays and picks_rec and rec.is_file() and args.refl.is_file() and geom:
        rays_png = HERE / "check_inv_rays.png"
        plot_inv_rays(rec, args.refl, rays, picks_rec, geom, rays_png, show=show)
        print(f"wrote {rays_png}")
    else:
        print(
            f"缺射线/走时，跳过 check_inv_rays.png"
            f"（对收回模型跑 tt_forward -R{args.rays.name} > {args.syn_rec.name}）",
            file=sys.stderr,
        )
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
        print(
            "缺 syn_inv / syn_start / syn_rec，跳过走时拟合图"
            "（初值、收回模型各跑一次 tt_forward）",
            file=sys.stderr,
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
