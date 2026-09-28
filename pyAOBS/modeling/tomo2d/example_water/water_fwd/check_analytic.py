#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对照 tt_forward 走时与平坦均匀水解析解，并画出 T–X 与射线。

用法（在本目录）:
  python check_analytic.py
  python check_analytic.py --no-show   # 只写 PNG，不弹窗
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import math

from make_water_fwd_case import H, OBS_X, compare_syn_to_analytic

HERE = Path(__file__).resolve().parent
Rec = tuple[int, float, float, float, float]
# 直达：细紫实线+白边；多次：黑实线+白边
_PHASE_RAY = {
    0: ("#1f77b4", "-", "折射 0"),
    1: ("#d62728", "-", "反射 1"),
    2: ("#9B30FF", "-", "直达 2"),
    3: ("#111111", "-", "多次 3"),
    4: ("#2ca02c", "-", "台侧折射 4"),
    5: ("#ff7f0e", "-", "台侧反射 5"),
    6: ("#c51b8a", "-", "折合 PSP 6"),
}
_PHASE_RAY_PAPER = {
    2: ("#9B30FF", "直达 2"),
    3: ("#333333", "多次 3"),
}


def parse_ray_file(path: Path) -> list[tuple[list[float], list[float]]]:
    """``>`` 分隔的 tomo2d -R 文件（printCurve：每行 x z）。"""
    segs: list[tuple[list[float], list[float]]] = []
    xs: list[float] = []
    zs: list[float] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith(">"):
            if len(xs) >= 2:
                segs.append((xs, zs))
            xs, zs = [], []
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            xs.append(float(parts[0]))
            zs.append(float(parts[1]))
        except ValueError:
            continue
    if len(xs) >= 2:
        segs.append((xs, zs))
    return segs


def _point_seg_dist(
    px: float, pz: float, ax: float, az: float, bx: float, bz: float
) -> float:
    dx, dz = bx - ax, bz - az
    l2 = dx * dx + dz * dz
    if l2 < 1e-18:
        return math.hypot(px - ax, pz - az)
    t = ((px - ax) * dx + (pz - az) * dz) / l2
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), pz - (az + t * dz))


def max_chord_sagitta(xs: list[float], zs: list[float]) -> float:
    """控制点折线相对端点弦的最大垂距（km）。直线段应接近 0。"""
    if len(xs) < 3:
        return 0.0
    ax, az, bx, bz = xs[0], zs[0], xs[-1], zs[-1]
    return max(
        _point_seg_dist(xs[i], zs[i], ax, az, bx, bz) for i in range(len(xs))
    )


def max_mult_sagitta(xs: list[float], zs: list[float]) -> float:
    """多次：按海底峰、海面谷拆成三段，各段相对弦的最大垂距。"""
    n = len(zs)
    if n < 6:
        return max_chord_sagitta(xs, zs)
    i_b = max(range(n), key=lambda i: zs[i])
    i_s = min(range(i_b, n), key=lambda i: zs[i])
    if i_b < 2:
        i_b = 2
    if i_s <= i_b:
        i_s = min(n - 2, i_b + 1)
    s1 = max_chord_sagitta(xs[: i_b + 1], zs[: i_b + 1])
    s2 = max_chord_sagitta(xs[i_b : i_s + 1], zs[i_b : i_s + 1])
    s3 = max_chord_sagitta(xs[i_s:], zs[i_s:])
    return max(s1, s2, s3)


def ray_sagitta(code: int, xs: list[float], zs: list[float]) -> float:
    if code == 3:
        return max_mult_sagitta(xs, zs)
    return max_chord_sagitta(xs, zs)


def _repo_root() -> Path:
    """仓库根：其下有可 import 的 ``pyAOBS`` 包。"""
    p = HERE
    for _ in range(8):
        if (p / "modeling" / "tomo2d").is_dir() and (p / "model_building").is_dir():
            return p.parent
        if (p / "pyAOBS" / "modeling" / "tomo2d").is_dir():
            return p
        p = p.parent
    return HERE.parents[4]


def _ensure_pkg() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _setup_mpl() -> None:
    _ensure_pkg()
    try:
        from pyAOBS.modeling.tomo2d.tt_inverse_log_analysis import (
            ensure_matplotlib_cjk_font,
        )

        ensure_matplotlib_cjk_font()
        return
    except Exception:
        pass
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _draw_smesh_background(ax, cax, fig, smesh_path: Path, refl_path: Path | None):
    """与 GUI ``draw_smesh_velocity_figure`` 同一套：smesh imshow + Vp CPT 色标 + 界面。"""
    import numpy as np
    from pyAOBS.modeling.tomo2d.gui.services.smesh_plot_core import (
        builtin_smesh_cmap_path,
        cmap_blank_air,
        colorbar_label_for_cmap,
        air_axis_zlim,
        load_smesh_plot_data,
        mask_air_layer_for_plot,
        normalize_velocity_plot_dataset,
    )
    from pyAOBS.visualization.gmt_cpt import parse_gmt_cpt_for_matplotlib

    refl = str(refl_path) if refl_path is not None and refl_path.is_file() else None
    mesh, ds, extra = load_smesh_plot_data(smesh_path, refl, with_xarray=True)
    if ds is None:
        raise RuntimeError(f"无法把 {smesh_path.name} 转成绘图网格")
    ds = normalize_velocity_plot_dataset(ds)
    data = np.asarray(ds["velocity"].values, dtype=float)
    x = np.asarray(ds["x"].values, dtype=float)
    z = np.asarray(ds["z"].values, dtype=float)
    data = mask_air_layer_for_plot(data, x, z, mesh)
    cmap_spec = str(builtin_smesh_cmap_path("water"))
    if not Path(cmap_spec).is_file():
        from pyAOBS.modeling.tomo2d.gui.services.smesh_plot_core import (
            default_plot_smesh_cmap,
        )

        cmap_spec = default_plot_smesh_cmap()
    if str(cmap_spec).lower().endswith(".cpt") and Path(cmap_spec).is_file():
        cmap, lo, hi = parse_gmt_cpt_for_matplotlib(cmap_spec)
    else:
        from matplotlib import colormaps

        try:
            cmap = colormaps[str(cmap_spec)]
        except Exception:
            cmap = colormaps["jet"]
        finite = data[np.isfinite(data)]
        lo = float(np.nanpercentile(finite, 1)) if finite.size else 1.5
        hi = float(np.nanpercentile(finite, 99)) if finite.size else 8.0
        if hi <= lo:
            hi = lo + 1e-6
    cmap = cmap_blank_air(cmap)
    from pyAOBS.modeling.tomo2d.gui.plots.velocity_contours import (
        DEFAULT_WATER_CONTOURS,
        overlay_velocity_contours,
    )

    data = np.ma.masked_invalid(data)
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    ax.set_facecolor("white")
    from matplotlib.colors import Normalize

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
    overlay_velocity_contours(
        ax, data, x, z, DEFAULT_WATER_CONTOURS, zorder=1
    )
    if mesh is not None and getattr(mesh, "xpos", None) is not None:
        ax.plot(
            np.asarray(mesh.xpos, dtype=float),
            np.asarray(mesh.topo, dtype=float),
            color="k",
            lw=0.9,
            zorder=2,
            label="topo (海面)",
        )
    if extra:
        for iface in extra:
            ax.plot(
                np.asarray(iface["x"], dtype=float),
                np.asarray(iface["z"], dtype=float),
                color=str(iface.get("color") or "crimson"),
                lw=float(iface.get("linewidth", 1.8)),
                ls=str(iface.get("linestyle", "--")),
                zorder=3,
                label=str(iface.get("label") or "海底"),
            )
    ax.set_ylim(*air_axis_zlim(zmin, zmax))
    return xmin, xmax, zmin, zmax


def plot_ttimes(recs: list[Rec], out_png: Path, *, show: bool) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    styles = {
        2: ("C0", "直达 2"),
        3: ("C1", "多次 3"),
    }
    for code, (color, name) in styles.items():
        dx = [r[1] for r in recs if r[0] == code]
        syn = [r[2] for r in recs if r[0] == code]
        ana = [r[3] for r in recs if r[0] == code]
        if not dx:
            continue
        ax.plot(dx, syn, "o", color=color, ms=7, zorder=3, label=f"{name} 正演")
        ax.plot(dx, ana, "--", color=color, lw=1.4, label=f"{name} 均匀1.5参考")
    ax.set_xlabel("偏移 dx (km)")
    ax.set_ylabel("走时 t (s)")
    ax.set_title("水波走时：正演（随机水速）vs 均匀 1.5 解析解")
    ax.grid(True, alpha=0.35)
    ax.invert_yaxis()
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)


def plot_rays(
    rays: list[tuple[list[float], list[float]]],
    recs: list[Rec],
    out_png: Path,
    *,
    show: bool,
    smesh_path: Path,
    refl_path: Path | None = None,
    h: float = H,
    obs_x: float = OBS_X,
) -> None:
    import matplotlib.pyplot as plt

    _setup_mpl()
    n = min(len(rays), len(recs))
    fig = plt.figure(figsize=(9.6, 5.2), facecolor="w", layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[28, 1.85])
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    _xmin, _xmax, _zmin, _zmax = _draw_smesh_background(ax, cax, fig, smesh_path, refl_path)
    ax.plot(
        [obs_x],
        [h],
        marker="^",
        color="k",
        ms=9,
        ls="none",
        label="OBS",
        zorder=5,
    )
    seen: set[int] = set()
    ray_xs: list[float] = []
    for i in range(n):
        xs, zs = rays[i]
        ray_xs.extend(xs)
        code = recs[i][0] if i < len(recs) else 2
        color, ls, name = _PHASE_RAY.get(code, ("#111111", "-", f"code {code}"))
        halo_c, halo_w, core_w = (
            ("#FFFFFF", 1.65, 0.75) if code == 3 else ("#FFFFFF", 1.35, 0.68)
        )
        ax.plot(
            xs,
            zs,
            color=halo_c,
            lw=halo_w,
            ls=ls,
            alpha=0.95,
            zorder=4,
            solid_capstyle="round",
        )
        kw: dict = dict(
            color=color,
            lw=core_w,
            ls=ls,
            alpha=0.96,
            zorder=4.1,
            solid_capstyle="round",
        )
        if code not in seen:
            kw["label"] = name
            seen.add(code)
        ax.plot(xs, zs, **kw)
    ax.set_title("射线路径（1.5 + 随机扰动；应贴海面/海底，勿进入沉积）")
    ax.set_xlabel("模型距离 (km)")
    ax.set_ylabel("深度 (km)")
    ax.tick_params(colors="black")
    ax.grid(True, alpha=0.3)
    if ray_xs:
        pad = 2.0
        ax.set_xlim(min(ray_xs) - pad, max(ray_xs) + pad)
    ax.legend(loc="upper right", framealpha=0.88)
    fig.savefig(out_png, dpi=140)
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)


def plot_sagitta(
    recs: list[Rec],
    sags: list[float],
    out_png: Path,
    *,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    _setup_mpl()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    by: dict[int, tuple[list[float], list[float]]] = {}
    n = min(len(recs), len(sags))
    for i in range(n):
        code, dx, *_rest = recs[i]
        by.setdefault(code, ([], []))
        by[code][0].append(dx)
        by[code][1].append(sags[i] * 1000.0)
    for code, (dx, sag_m) in by.items():
        color, name = _PHASE_RAY_PAPER.get(code, ("#333", f"code {code}"))
        ax.plot(dx, sag_m, "o-", color=color, ms=4.5, lw=1.1, label=name)
    ax.axhline(50.0, color="0.5", ls=":", lw=0.9, label="50 m")
    ax.set_xlabel("偏移 dx (km)")
    ax.set_ylabel("相对弦的最大垂距 (m)")
    ax.set_title("射线弯曲量 vs 偏移（随机水速；直达可弯）")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="水波正演走时对照，并画 T–X / 射线")
    p.add_argument("--no-show", action="store_true", help="只保存 PNG，不弹窗")
    p.add_argument("--syn", type=Path, default=HERE / "syn_water.dat")
    p.add_argument("--rays", type=Path, default=HERE / "rays_water.dat")
    p.add_argument("--smesh", type=Path, default=HERE / "water.smesh")
    p.add_argument("--refl", type=Path, default=HERE / "seafloor.refl")
    args = p.parse_args(argv)

    if not args.syn.is_file():
        print(f"缺少 {args.syn}：请先跑 tt_forward（见 README）", file=sys.stderr)
        return 1
    recs = compare_syn_to_analytic(args.syn.read_text(encoding="utf-8"), atol=99.0)
    rays: list[tuple[list[float], list[float]]] = []
    if args.rays.is_file():
        rays = parse_ray_file(args.rays)
        if len(rays) != len(recs):
            print(
                f"注意：射线条数 {len(rays)} 与走时条数 {len(recs)} 不一致",
                file=sys.stderr,
            )
    sags = [
        ray_sagitta(recs[i][0], rays[i][0], rays[i][1])
        if i < len(rays)
        else float("nan")
        for i in range(len(recs))
    ]
    print(f"{'code':>4} {'dx':>7} {'syn':>9} {'ana':>9} {'err_ms':>8} {'sag_m':>8}")
    for i, (code, dx, ts, ta, err) in enumerate(recs):
        sag_s = f" {sags[i] * 1000:8.1f}" if sags[i] == sags[i] else "     n/a"
        print(f"{code:4d} {dx:7.2f} {ts:9.4f} {ta:9.4f} {err * 1e3:8.1f}{sag_s}")

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    _setup_mpl()
    show = not args.no_show
    t_png = HERE / "check_ttimes.png"
    plot_ttimes(recs, t_png, show=show)
    print(f"写出 {t_png}")

    if not args.rays.is_file():
        print(f"没有 {args.rays}，跳过射线图（tt_forward 加 -R）", file=sys.stderr)
    else:
        if not args.smesh.is_file():
            print(f"缺少 {args.smesh}，无法画速度底图", file=sys.stderr)
            return 1
        r_png = HERE / "check_rays.png"
        plot_rays(
            rays,
            recs,
            r_png,
            show=show,
            smesh_path=args.smesh,
            refl_path=args.refl if args.refl.is_file() else None,
        )
        print(f"写出 {r_png}")
        s_png = HERE / "check_sagitta.png"
        plot_sagitta(recs, sags, s_png, show=show)
        print(f"写出 {s_png}")

    if show:
        import matplotlib.pyplot as plt

        print("关闭图窗后结束。")
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
