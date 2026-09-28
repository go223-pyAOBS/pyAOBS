#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对照 tt_forward 折合 PSP 走时与参考曲线，并画出 T–X 与射线。

用法（在本目录）:
  python check_converse_fwd.py
  python check_converse_fwd.py --no-show
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "water_fwd"))
from check_analytic import (  # noqa: E402
    _setup_mpl,
    parse_ray_file,
)
from make_converse_fwd_case import (  # noqa: E402
    H,
    OBS_X,
    VS0,
    Z_CONV,
    compare_syn_to_analytic,
    vp_sed,
)

Rec = tuple[int, float, float, float, float]

P_RAY_COLOR = "#1f77b4"
S_RAY_COLOR = "#e377c2"  # 粉：S 腿（盖层台侧 / PSS 面下 / PSP 面下）


def split_psp_phase_segments(
    xs: list[float],
    zs: list[float],
    z_conv: float = Z_CONV,
    *,
    eps: float = 1e-3,
) -> list[tuple[list[float], list[float], bool]]:
    """按转换面切开射线。``z > z_conv+eps`` 为 S，其余为 P；跨面段在界面切开。"""
    if len(xs) < 2 or len(xs) != len(zs):
        return []

    def is_s(z: float) -> bool:
        return z > z_conv + eps

    def cross(x0: float, z0: float, x1: float, z1: float) -> tuple[float, float]:
        if abs(z1 - z0) < 1e-15:
            return x0, z_conv
        t = (z_conv - z0) / (z1 - z0)
        return x0 + t * (x1 - x0), z_conv

    out: list[tuple[list[float], list[float], bool]] = []
    cur_x = [xs[0]]
    cur_z = [zs[0]]
    cur_s = is_s(zs[0])
    for x1, z1 in zip(xs[1:], zs[1:]):
        s1 = is_s(z1)
        if s1 == cur_s:
            cur_x.append(x1)
            cur_z.append(z1)
            continue
        xc, zc = cross(cur_x[-1], cur_z[-1], x1, z1)
        cur_x.append(xc)
        cur_z.append(zc)
        if len(cur_x) >= 2:
            out.append((cur_x, cur_z, cur_s))
        cur_x = [xc, x1]
        cur_z = [zc, z1]
        cur_s = s1
    if len(cur_x) >= 2:
        out.append((cur_x, cur_z, cur_s))
    return out


def draw_psp_ps_rays(
    ax,
    rays: list[tuple[list[float], list[float]]],
    *,
    z_conv: float = Z_CONV,
    legend: bool = False,
    thin: bool = False,
) -> None:
    """P 段蓝色、S 段粉色。"""
    halo_w, core_w = (1.05, 0.42) if thin else (1.45, 0.72)
    seen_p = False
    seen_s = False
    for xs, zs in rays:
        for sx, sz, is_s in split_psp_phase_segments(xs, zs, z_conv):
            color = S_RAY_COLOR if is_s else P_RAY_COLOR
            ax.plot(
                sx,
                sz,
                color="#FFFFFF",
                lw=halo_w,
                ls="-",
                alpha=0.88,
                zorder=4,
                solid_capstyle="round",
            )
            kw: dict = dict(
                color=color,
                lw=core_w,
                ls="-",
                alpha=0.94,
                zorder=4.1,
                solid_capstyle="round",
            )
            if legend and is_s and not seen_s:
                kw["label"] = "S（转换面以下）"
                seen_s = True
            elif legend and (not is_s) and not seen_p:
                kw["label"] = "P（水柱+盖层）"
                seen_p = True
            ax.plot(sx, sz, **kw)


def _incident_angle_deg(x0: float, z0: float, x1: float, z1: float) -> float:
    """相对铅垂的入射角（度）。界面水平，法向即铅垂。"""
    return math.degrees(math.atan2(abs(x1 - x0), abs(z1 - z0)))


def report_psp_snell(
    rays: list[tuple[list[float], list[float]]],
    recs: list[Rec],
    *,
    z_conv: float = Z_CONV,
) -> None:
    """事后对照：从最短路径上估计 p=sin i / v。连续介质里常接近 pP≈pS，不是选点约束。"""
    print(f"{'dx':>6} {'xP':>7} {'iP':>6} {'iS':>6} {'pP':>8} {'pS':>8} {'dp':>8}")
    n = min(len(rays), len(recs))
    vp = vp_sed(z_conv)
    vs = VS0
    for i in range(n):
        if recs[i][0] != 6:
            continue
        xs, zs = rays[i]
        if len(xs) < 3:
            continue
        def _step(k0: int, toward: int) -> tuple[int, int] | None:
            """跳过贴面钉点，找到一段真正离开界面的边。"""
            k = k0
            while 0 <= k < len(xs) - 1:
                kn = k + toward
                if kn < 0 or kn >= len(xs):
                    return None
                if abs(xs[kn] - xs[k]) + abs(zs[kn] - zs[k]) < 1e-6:
                    k = kn if toward > 0 else k - 1
                    continue
                if abs(zs[kn] - z_conv) < 1e-3 and abs(zs[k] - z_conv) < 1e-3:
                    k = kn if toward > 0 else k - 1
                    continue
                return (k, kn) if toward > 0 else (kn, k)
            return None

        hits: list[int] = []
        for k in range(len(zs) - 1):
            if (zs[k] - z_conv) * (zs[k + 1] - z_conv) <= 0.0:
                hits.append(k)
        if not hits:
            continue
        k_shot = hits[0]
        pair_p = _step(k_shot, -1) or _step(k_shot, 1)
        pair_s = None
        for k in hits:
            cand = _step(k, 1)
            if cand and max(zs[cand[0]], zs[cand[1]]) > z_conv + 1e-3:
                pair_s = cand
                break
            cand = _step(k, -1)
            if cand and max(zs[cand[0]], zs[cand[1]]) > z_conv + 1e-3:
                pair_s = cand
                break
        if pair_p is None or pair_s is None:
            continue
        i_p = _incident_angle_deg(xs[pair_p[0]], zs[pair_p[0]], xs[pair_p[1]], zs[pair_p[1]])
        i_s = _incident_angle_deg(xs[pair_s[0]], zs[pair_s[0]], xs[pair_s[1]], zs[pair_s[1]])
        p_p = math.sin(math.radians(i_p)) / vp
        p_s = math.sin(math.radians(i_s)) / vs
        print(
            f"{recs[i][1]:6.1f} {xs[k_shot]:7.2f} {i_p:6.1f} {i_s:6.1f} "
            f"{p_p:8.4f} {p_s:8.4f} {p_p - p_s:8.4f}"
        )


def _read_iface(path: Path) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    zs: list[float] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        p = raw.split()
        if len(p) >= 2:
            xs.append(float(p[0]))
            zs.append(float(p[1]))
    return xs, zs


def _draw_hybrid_background(
    ax,
    cax,
    fig,
    smesh_path: Path,
    seafloor: Path | None,
    conv: Path | None,
):
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

    refl = str(seafloor) if seafloor is not None and seafloor.is_file() else None
    mesh, ds, extra = load_smesh_plot_data(smesh_path, refl, with_xarray=True)
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
    if cax is not None:
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(colorbar_label_for_cmap(cmap_spec) + "（盖层 Vp / 面下 Vs）")
        cb.set_ticks(np.linspace(float(lo), float(hi), 7))
    if extra:
        for iface in extra:
            ax.plot(
                np.asarray(iface["x"], dtype=float),
                np.asarray(iface["z"], dtype=float),
                color=str(iface.get("color") or "0.15"),
                lw=float(iface.get("linewidth", 1.4)),
                ls=str(iface.get("linestyle", "--")),
                zorder=3,
                label=str(iface.get("label") or "海底"),
            )
    elif seafloor is not None and seafloor.is_file():
        sx, sz = _read_iface(seafloor)
        if sx:
            ax.plot(sx, sz, color="0.15", ls="--", lw=1.4, zorder=3, label="海底")
    if conv is not None and conv.is_file():
        cx, cz = _read_iface(conv)
        if cx:
            ax.plot(cx, cz, color="0.25", ls="--", lw=1.8, zorder=3, label="转换面")
    ax.set_ylim(*air_axis_zlim(zmin, zmax))
    return xmin, xmax, zmin, zmax, im


def plot_ttimes(recs: list[Rec], out_png: Path, *, show: bool) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    styles = {
        0: (P_RAY_COLOR, "折射 0"),
        6: (S_RAY_COLOR, "折合 PSP 6"),
    }
    for code, (color, name) in styles.items():
        dx = [r[1] for r in recs if r[0] == code]
        syn = [r[2] for r in recs if r[0] == code]
        ref = [r[3] for r in recs if r[0] == code]
        if not dx:
            continue
        ax.plot(dx, syn, "o", color=color, ms=7, zorder=3, label=f"{name} 正演")
        ref_name = "海底头波" if code == 0 else "界面 S 参考（垂直 P+界面 S）"
        ax.plot(dx, ref, "--", color=color, lw=1.4, label=f"{name} {ref_name}")
    ax.set_xlabel("偏移 dx (km)")
    ax.set_ylabel("走时 t (s)")
    ax.set_title("折合 PSP（弯曲 + 前向星 8）：0 初至参考；6 相对垂直 P+界面 S")
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
    seafloor: Path | None,
    conv: Path | None,
    h: float = H,
    obs_x: float = OBS_X,
) -> None:
    import matplotlib.pyplot as plt

    _setup_mpl()
    n = min(len(rays), len(recs))
    fig = plt.figure(figsize=(9.8, 5.6), facecolor="w", layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[28, 1.85])
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    _draw_hybrid_background(ax, cax, fig, smesh_path, seafloor, conv)
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
    ray_xs: list[float] = []
    z6_max = 0.0
    rays0: list[tuple[list[float], list[float]]] = []
    rays6: list[tuple[list[float], list[float]]] = []
    for i in range(n):
        xs, zs = rays[i]
        ray_xs.extend(xs)
        code = recs[i][0] if i < len(recs) else 6
        if code == 6:
            rays6.append((xs, zs))
            if zs:
                z6_max = max(z6_max, max(zs))
        else:
            rays0.append((xs, zs))
    for i, (xs, zs) in enumerate(rays0):
        ax.plot(
            xs,
            zs,
            color="#FFFFFF",
            lw=1.45,
            ls="--",
            alpha=0.9,
            zorder=3.8,
            solid_capstyle="round",
        )
        kw: dict = dict(
            color=P_RAY_COLOR,
            lw=0.85,
            ls="--",
            alpha=0.9,
            zorder=3.9,
            solid_capstyle="round",
        )
        if i == 0:
            kw["label"] = "折射 0（P）"
        ax.plot(xs, zs, **kw)
    draw_psp_ps_rays(ax, rays6, z_conv=Z_CONV, legend=True)
    title = "射线（弯曲 + 前向星 8）：P 蓝 / S 粉；6 斜过转换面"
    if z6_max > Z_CONV + 0.15:
        title += f"（6 最深 {z6_max:.2f} km）"
    ax.set_title(title)
    ax.set_xlabel("模型距离 (km)")
    ax.set_ylabel("深度 (km)")
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="折合 PSP 正演走时对照，并画 T–X / 射线")
    p.add_argument("--no-show", action="store_true", help="只保存 PNG，不弹窗")
    p.add_argument("--syn", type=Path, default=HERE / "syn_conv.dat")
    p.add_argument("--rays", type=Path, default=HERE / "rays_conv.dat")
    p.add_argument("--smesh", type=Path, default=HERE / "converse.smesh")
    p.add_argument("--seafloor", type=Path, default=HERE / "seafloor.refl")
    p.add_argument("--conv", type=Path, default=HERE / "conv.refl")
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
    print(f"{'code':>4} {'dx':>7} {'syn':>9} {'ref':>9} {'err_ms':>8}")
    for code, dx, ts, ta, err in recs:
        print(f"{code:4d} {dx:7.2f} {ts:9.4f} {ta:9.4f} {err * 1e3:8.1f}")
    if rays:
        print("PSP 转换点事后慢度（p=sin i / v；最短路径的结果，不是强制 Snell）")
        report_psp_snell(rays, recs)

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
            seafloor=args.seafloor if args.seafloor.is_file() else None,
            conv=args.conv if args.conv.is_file() else None,
        )
        print(f"写出 {r_png}")

    if show:
        import matplotlib.pyplot as plt

        print("关闭图窗后结束。")
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
