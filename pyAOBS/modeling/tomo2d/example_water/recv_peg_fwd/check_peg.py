#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对照折射(0)/台侧(4) 与 莫霍反射(1)/台侧(5)，并画射线。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "water_fwd"))
from check_analytic import _setup_mpl, parse_ray_file  # noqa: E402
from make_recv_peg_case import (  # noqa: E402
    H,
    H_MOHO,
    OBS_X,
    analytic_curve,
    t_analytic,
    t_water_twt,
)


def parse_syn(text: str) -> list[tuple[int, float, float]]:
    recs: list[tuple[int, float, float]] = []
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
                code = int(float(rp[3]))
                t = float(rp[4])
                recs.append((code, abs(x - src_x), t))
    return recs


def _draw_crust_background(ax, cax, fig, smesh_path: Path) -> None:
    import numpy as np
    from matplotlib.colors import Normalize

    from pyAOBS.modeling.tomo2d.gui.services.smesh_plot_core import (
        air_axis_zlim,
        builtin_smesh_cmap_path,
        cmap_blank_air,
        colorbar_label_for_cmap,
        load_smesh_plot_data,
        mask_air_layer_for_plot,
        normalize_velocity_plot_dataset,
    )
    from pyAOBS.visualization.gmt_cpt import parse_gmt_cpt_for_matplotlib

    mesh, ds, _extra = load_smesh_plot_data(smesh_path, None, with_xarray=True)
    if ds is None:
        raise RuntimeError(f"无法把 {smesh_path.name} 转成绘图网格")
    ds = normalize_velocity_plot_dataset(ds)
    data = np.asarray(ds["velocity"].values, dtype=float)
    x = np.asarray(ds["x"].values, dtype=float)
    z = np.asarray(ds["z"].values, dtype=float)
    data = mask_air_layer_for_plot(data, x, z, mesh)
    cmap_spec = str(builtin_smesh_cmap_path("vp"))
    if cmap_spec.lower().endswith(".cpt") and Path(cmap_spec).is_file():
        cmap, lo, hi = parse_gmt_cpt_for_matplotlib(cmap_spec)
    else:
        from matplotlib import colormaps

        cmap = colormaps["jet"]
        finite = data[np.isfinite(data)]
        lo = float(np.nanmin(finite)) if finite.size else 1.5
        hi = float(np.nanmax(finite)) if finite.size else 8.0
    cmap = cmap_blank_air(cmap)
    data = np.ma.masked_invalid(data)
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    ax.set_facecolor("white")
    im = ax.imshow(
        data,
        extent=(xmin, xmax, zmax, zmin),
        cmap=cmap,
        aspect="auto",
        norm=Normalize(vmin=float(lo), vmax=float(hi), clip=False),
        interpolation="nearest",
        zorder=0,
    )
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(colorbar_label_for_cmap(cmap_spec))
    ticks = np.linspace(float(lo), float(hi), 6)
    cb.set_ticks(ticks)
    if mesh is not None and getattr(mesh, "xpos", None) is not None:
        ax.plot(
            np.asarray(mesh.xpos, dtype=float),
            np.asarray(mesh.topo, dtype=float),
            color="k",
            lw=0.9,
            zorder=2,
            label="海面",
        )
    ax.set_ylim(*air_axis_zlim(zmin, zmax))


def _plot_iface(ax, path: Path, *, color: str, ls: str, label: str) -> None:
    xs: list[float] = []
    zs: list[float] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        a, b = s.split()[:2]
        xs.append(float(a))
        zs.append(float(b))
    ax.plot(xs, zs, color=color, lw=1.6, ls=ls, zorder=3, label=label)


def _print_pair(by: dict[tuple[int, float], float], a: int, b: int, twt: float) -> None:
    print(f"{'dx':>7} {f't{a}':>9} {f't{b}':>9} {f't{b}-t{a}':>8} {'2H/v':>8} {'d-2H/v_ms':>10}")
    keys = sorted({d for c, d in by if c == a})
    for dx in keys:
        ta = by.get((a, dx))
        tb = by.get((b, dx))
        if ta is None or tb is None:
            print(f"{dx:7.2f}  missing")
            continue
        dt = tb - ta
        print(
            f"{dx:7.2f} {ta:9.4f} {tb:9.4f} {dt:8.4f} {twt:8.4f} {(dt - twt) * 1e3:10.1f}"
        )


def _print_syn_vs_theory(recs: list[tuple[int, float, float]]) -> None:
    print(f"{'code':>4} {'dx':>7} {'syn':>9} {'ana':>9} {'err_ms':>8}")
    for code, dx, ts in recs:
        ta = t_analytic(code, dx)
        if ta is None:
            print(f"{code:4d} {dx:7.2f} {ts:9.4f} {'n/a':>9} {'n/a':>8}")
            continue
        print(f"{code:4d} {dx:7.2f} {ts:9.4f} {ta:9.4f} {(ts - ta) * 1e3:8.1f}")


def plot_ttimes(
    recs: list[tuple[int, float, float]],
    out_png: Path,
    *,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    styles = {
        0: ("#1f77b4", "折射 0"),
        1: ("#2ca02c", "莫霍反射 1"),
        4: ("#d62728", "折射台侧 4"),
        5: ("#9467bd", "反射台侧 5"),
    }
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for code, (color, name) in styles.items():
        xs, ts = analytic_curve(code)
        if xs:
            ax.plot(xs, ts, "--", color=color, lw=1.5, label=f"{name} 理论")
        dxs = [r[1] for r in recs if r[0] == code]
        syn = [r[2] for r in recs if r[0] == code]
        if dxs:
            ax.plot(dxs, syn, "o", color=color, ms=7, zorder=3, label=f"{name} 正演")
    ax.set_xlabel("偏移 dx (km)")
    ax.set_ylabel("走时 t (s)")
    ax.set_title("台侧一阶：一次波与多次波（正演 vs 层状解析）")
    ax.set_xlim(0.0, 35.0)
    ax.grid(True, alpha=0.35)
    ax.invert_yaxis()
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args(argv)
    syn = HERE / "syn_peg.dat"
    rays_path = HERE / "rays_peg.dat"
    if not syn.is_file():
        print(f"缺少 {syn}：请先跑 tt_forward", file=sys.stderr)
        return 1
    recs = parse_syn(syn.read_text(encoding="utf-8"))
    by: dict[tuple[int, float], float] = {}
    for code, dx, t in recs:
        by[(code, round(dx, 3))] = t
    twt = t_water_twt()
    print("正演 vs 层状解析")
    _print_syn_vs_theory(recs)
    print("折射 0 vs 台侧 4")
    _print_pair(by, 0, 4, twt)
    print("莫霍反射 1 vs 台侧 5")
    _print_pair(by, 1, 5, twt)
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_mpl()
    show = not args.no_show
    t_png = HERE / "check_peg_ttimes.png"
    plot_ttimes(recs, t_png, show=show)
    print(f"写出 {t_png}")
    fig = plt.figure(figsize=(9.6, 5.2), facecolor="w", layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[28, 1.85])
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    _draw_crust_background(ax, cax, fig, HERE / "crust.smesh")
    _plot_iface(ax, HERE / "seafloor.refl", color="#ffffff", ls="-", label="海底")
    _plot_iface(ax, HERE / "moho.refl", color="crimson", ls="--", label="莫霍")
    ax.plot([OBS_X], [H], marker="^", color="k", ms=9, ls="none", label="OBS", zorder=5)
    styles = {
        0: ("#1f77b4", "折射 0"),
        4: ("#d62728", "折射台侧 4"),
        1: ("#2ca02c", "莫霍反射 1"),
        5: ("#9467bd", "反射台侧 5"),
    }
    if rays_path.is_file():
        rays = parse_ray_file(rays_path)
        n = min(len(rays), len(recs))
        seen: set[int] = set()
        for i in range(n):
            xs, zs = rays[i]
            code = recs[i][0]
            color, name = styles.get(code, ("#333", f"code {code}"))
            kw: dict = dict(color=color, lw=0.9, alpha=0.85, zorder=4)
            if code not in seen:
                kw["label"] = name
                seen.add(code)
            ax.plot(xs, zs, **kw)
    ax.axhline(H_MOHO, color="crimson", lw=0.4, ls=":", alpha=0.4, zorder=1)
    ax.set_title("台侧一阶：折射 0/4，莫霍反射 1/5（5 应弹海面并碰到莫霍）")
    ax.set_xlabel("模型距离 (km)")
    ax.set_ylabel("深度 (km)")
    ax.legend(loc="upper right", framealpha=0.88)
    out = HERE / "check_peg_rays.png"
    fig.savefig(out, dpi=140)
    print(f"写出 {out}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
