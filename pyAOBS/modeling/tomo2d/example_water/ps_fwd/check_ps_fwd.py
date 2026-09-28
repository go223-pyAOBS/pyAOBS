#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""打印 PPP/PPS/PSS/PSP 走时，检验 Δ(PPS−PPP)≈Δ(PSS−PSP)，并画射线。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from collections.abc import Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "converse_fwd"))
from check_converse_fwd import (  # noqa: E402
    P_RAY_COLOR,
    S_RAY_COLOR,
    _draw_hybrid_background,
    _setup_mpl,
    parse_ray_file,
)
from make_ps_fwd_case import H, KAPPA, OBS_X, VS0, VS_GRAD, Z_CONV  # noqa: E402

ConvZ = float | Callable[[float], float]


def conv_depth(x: float, z_conv: ConvZ) -> float:
    return float(z_conv(x)) if callable(z_conv) else float(z_conv)


def parse_syn(text: str) -> list[tuple[int, float, float]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    recs: list[tuple[int, float, float]] = []
    i = 1
    src_x = OBS_X
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
            x = float(rp[1])
            recs.append((int(float(rp[3])), abs(x - src_x), float(rp[4])))
    return recs


def find_psx_conversion_points(
    xs: list[float],
    zs: list[float],
    *,
    z_conv: ConvZ = Z_CONV,
    eps: float = 5e-2,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """路径 shot→OBS 上的炮侧 / 台侧转换点（落在转换面上）。

    取第一次和最后一次到达或穿过转换面的位置。选点只跟整条路径最短，
    不按 Snell 或铅垂投影。
    """
    if len(xs) < 2 or len(xs) != len(zs):
        return None
    hits: list[tuple[float, float]] = []
    for i, z in enumerate(zs):
        zc = conv_depth(xs[i], z_conv)
        if abs(z - zc) <= eps:
            hits.append((xs[i], zc))
        if i + 1 >= len(zs):
            continue
        z1 = zs[i + 1]
        zc1 = conv_depth(xs[i + 1], z_conv)
        d0, d1 = z - zc, z1 - zc1
        if d0 * d1 < 0.0:
            tfr = d0 / (d0 - d1)
            hits.append((
                xs[i] + tfr * (xs[i + 1] - xs[i]),
                zc + tfr * (zc1 - zc),
            ))
    if not hits:
        return None
    uniq: list[tuple[float, float]] = [hits[0]]
    for x, z in hits[1:]:
        if abs(x - uniq[-1][0]) > 1e-3:
            uniq.append((x, z))
    return uniq[0], uniq[-1]


def _iter_coded_rays(
    rays: list[tuple[list[float], list[float]]],
    recs: list[tuple[int, float, float]],
    codes: tuple[int, ...] | None = None,
):
    for ir, rec in enumerate(recs):
        if ir >= len(rays):
            break
        if codes is not None and rec[0] not in codes:
            continue
        yield rec, rays[ir]


def report_psx_conversion_points(
    rays: list[tuple[list[float], list[float]]],
    recs: list[tuple[int, float, float]],
    *,
    z_conv: ConvZ = Z_CONV,
) -> None:
    """打印转换点。1 反射钉在面上；6/8 两个星；7 取最后一次过面。"""
    print("转换点（整条路径最短；7 一个星，6/8 两个星）")
    print(f"{'code':>4} {'dx':>7} {'shot_x':>8} {'star1':>8} {'star2':>8} {'obs_x':>8}")
    for (code, dx, _t), (xs, zs) in _iter_coded_rays(rays, recs, (1, 6, 7, 8)):
        pts = find_psx_conversion_points(xs, zs, z_conv=z_conv)
        if pts is None:
            print(f"{code:4d} {dx:7.2f}  (no conversion hit)")
            continue
        c_shot, c_obs = pts
        if code in (1, 7):
            c_shot = c_obs
        print(
            f"{code:4d} {dx:7.2f} {xs[0]:8.2f} {c_shot[0]:8.2f} "
            f"{c_obs[0]:8.2f} {xs[-1]:8.2f}"
        )


def draw_ps_conversion_points(
    ax,
    rays: list[tuple[list[float], list[float]]],
    recs: list[tuple[int, float, float]],
    *,
    z_conv: ConvZ = Z_CONV,
    thin: bool = False,
    codes: tuple[int, ...] | None = None,
) -> None:
    shot_x: list[float] = []
    shot_z: list[float] = []
    obs_x: list[float] = []
    obs_z: list[float] = []
    for (code, _dx, _t), (xs, zs) in _iter_coded_rays(rays, recs, codes):
        if code not in (1, 6, 7, 8):
            continue
        pts = find_psx_conversion_points(xs, zs, z_conv=z_conv)
        if pts is None:
            continue
        c_shot, c_obs = pts
        if code in (1, 7):
            c_shot = c_obs
        shot_x.append(c_shot[0])
        shot_z.append(c_shot[1])
        if abs(c_obs[0] - c_shot[0]) > 1e-3:
            obs_x.append(c_obs[0])
            obs_z.append(c_obs[1])
    ms = 4.0 if thin else 6.4
    if shot_x:
        ax.plot(
            shot_x,
            shot_z,
            "o",
            ms=ms,
            mfc="#FFFFFF",
            mec=P_RAY_COLOR,
            mew=1.2,
            ls="none",
            zorder=6.4,
            label="炮侧转换点（联合最短）",
        )
    if obs_x:
        ax.plot(
            obs_x,
            obs_z,
            "D",
            ms=ms * 0.88,
            mfc=S_RAY_COLOR,
            mec="0.12",
            mew=0.7,
            ls="none",
            zorder=6.5,
            label="台侧转换点（联合最短）",
        )


def split_psx_phase_segments(
    xs: list[float],
    zs: list[float],
    code: int,
    *,
    z_conv: ConvZ = Z_CONV,
    z_sf: float = H,
    eps: float = 1e-3,
) -> list[tuple[list[float], list[float], bool]]:
    """Shot→OBS 着色。6：仅面下为 S。7：台侧盖层 S。8：过星之后全是 S。"""
    if len(xs) < 2 or len(xs) != len(zs):
        return []
    first_at_conv = -1
    last_at_conv = -1
    for i, z in enumerate(zs):
        if z >= conv_depth(xs[i], z_conv) - eps:
            if first_at_conv < 0:
                first_at_conv = i
            last_at_conv = i
    if last_at_conv < 0:
        last_at_conv = max(range(len(zs)), key=lambda i: zs[i])
        first_at_conv = last_at_conv

    def is_s_seg(i: int) -> bool:
        xm = 0.5 * (xs[i] + xs[i + 1])
        zm = 0.5 * (zs[i] + zs[i + 1])
        zc = conv_depth(xm, z_conv)
        if zm < z_sf - eps:
            return False
        if code in (0, 1):
            return False
        if code in (6, 9, 12):
            return zm > zc + eps
        if zm > zc + eps:
            return code in (8, 11, 13, 15)
        if code in (8, 11, 13, 15):
            return i >= first_at_conv
        if code == 10:
            return i >= first_at_conv
        return i >= last_at_conv

    out: list[tuple[list[float], list[float], bool]] = []
    cur_x = [xs[0]]
    cur_z = [zs[0]]
    cur_s = is_s_seg(0)
    for i in range(len(xs) - 1):
        s1 = is_s_seg(i)
        if s1 == cur_s:
            cur_x.append(xs[i + 1])
            cur_z.append(zs[i + 1])
            continue
        if len(cur_x) >= 2:
            out.append((cur_x, cur_z, cur_s))
        cur_x = [xs[i], xs[i + 1]]
        cur_z = [zs[i], zs[i + 1]]
        cur_s = s1
    if len(cur_x) >= 2:
        out.append((cur_x, cur_z, cur_s))
    return out


VRED = 8.0  # km/s，折合速度

PHASE_TT_STYLE = {
    0: ("#9ecae1", "o--", "初至  0"),
    1: ("#1f77b4", "o-", "PPP反射 1"),
    6: ("#2ca02c", "s-", "PSP  6"),
    7: ("#ff7f0e", "^-", "PPS  7"),
    8: (S_RAY_COLOR, "D-", "PSS  8"),
    9: ("#74c476", "s:", "PSP-peg 9"),
    10: ("#fd8d3c", "^:", "PPS-SS 10"),
    11: ("#dd3497", "D:", "PSS-SS 11"),
    12: ("#006d2c", "s--", "PSP-Moho 12"),
    13: ("#ae017e", "D--", "PSS-Moho 13"),
    14: ("#fdae6b", "^--", "PPS-water 14"),
    15: ("#fa9fb5", "D--", "PSS-water 15"),
}


def plot_reduced_ttimes(
    recs: list[tuple[int, float, float]],
    out_png: Path,
    *,
    vred: float = VRED,
    show: bool = False,
) -> None:
    """四种震相画在一张折合走时图上：t_red = t − Δx / vred。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 5.2), facecolor="w")
    for code, (color, fmt, name) in PHASE_TT_STYLE.items():
        rows = [(dx, t) for c, dx, t in recs if c == code]
        if not rows:
            continue
        rows.sort()
        dx = [r[0] for r in rows]
        tred = [r[1] - r[0] / vred for r in rows]
        ax.plot(dx, tred, fmt, color=color, ms=7, lw=1.6, label=name, zorder=3)
    ax.set_xlabel("偏移 Δx (km)")
    ax.set_ylabel(rf"$t - \Delta x / {vred:g}$ (s)")
    ax.set_title(f"折合走时（折合速度 {vred:g} km/s）")
    ax.grid(True, alpha=0.35)
    ax.invert_yaxis()
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)


def draw_ps_rays(
    ax,
    rays,
    recs,
    *,
    thin: bool = False,
    mark_conv: bool = True,
    z_conv: ConvZ = Z_CONV,
    codes: tuple[int, ...] | None = None,
) -> None:
    halo_w, core_w = (1.15, 0.55) if thin else (2.05, 1.15)
    labeled = set()
    pending_s: list[tuple[list[float], list[float], dict]] = []
    for (code, _dx, _t), (xs, zs) in _iter_coded_rays(rays, recs, codes):
        if code in (0, 1):
            ax.plot(
                xs,
                zs,
                color=P_RAY_COLOR,
                lw=0.7 if code == 0 else 1.15,
                ls="--",
                alpha=0.55 if code == 0 else 0.85,
                zorder=3.2 if code == 1 else 3,
            )
            continue
        for sx, sz, is_s in split_psx_phase_segments(xs, zs, code, z_conv=z_conv):
            kw: dict = dict(
                lw=core_w,
                ls="-",
                alpha=0.95,
                solid_capstyle="round",
            )
            key = "S（粉）" if is_s else "P（蓝）"
            if key not in labeled:
                kw["label"] = key
                labeled.add(key)
            if is_s:
                pending_s.append((sx, sz, kw))
            else:
                ax.plot(sx, sz, color="#FFFFFF", lw=halo_w, alpha=0.75, zorder=4)
                ax.plot(sx, sz, color=P_RAY_COLOR, zorder=4.1, **kw)
    for sx, sz, kw in pending_s:
        ax.plot(sx, sz, color="#FFFFFF", lw=halo_w + 0.35, alpha=0.9, zorder=5)
        ax.plot(sx, sz, color=S_RAY_COLOR, zorder=5.1, **kw)
    if mark_conv:
        draw_ps_conversion_points(ax, rays, recs, z_conv=z_conv, thin=thin, codes=codes)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--syn", type=Path, default=HERE / "syn_ps.dat")
    p.add_argument("--rays", type=Path, default=HERE / "rays_ps.dat")
    p.add_argument("--smesh", type=Path, default=HERE / "mixed.smesh")
    p.add_argument("--seafloor", type=Path, default=HERE / "seafloor.refl")
    p.add_argument("--conv", type=Path, default=HERE / "conv.refl")
    p.add_argument("--vred", type=float, default=VRED, help="折合速度 km/s")
    args = p.parse_args(argv)
    if not args.syn.is_file():
        print(f"缺少 {args.syn}", file=sys.stderr)
        return 1
    recs = parse_syn(args.syn.read_text(encoding="utf-8"))
    print(f"{'code':>4} {'dx':>7} {'t':>9} {'t-dx/8':>9}")
    for code, dx, t in recs:
        print(f"{code:4d} {dx:7.2f} {t:9.4f} {t - dx / args.vred:9.4f}")
    by: dict[tuple[int, float], float] = {}
    for code, dx, t in recs:
        by.setdefault((code, round(dx, 2)), t)
    print(f"{'dx':>7} {'t1':>8} {'t6':>8} {'t7':>8} {'t8':>8} "
          f"{'7-1':>8} {'8-6':>8} {'|d|':>8}")
    for dx in sorted({round(d, 2) for _c, d, _t in recs}):
        t1, t6 = by.get((1, dx)), by.get((6, dx))
        t7, t8 = by.get((7, dx)), by.get((8, dx))
        if None in (t1, t6, t7, t8):
            continue
        d71, d86 = t7 - t1, t8 - t6
        print(f"{dx:7.2f} {t1:8.4f} {t6:8.4f} {t7:8.4f} {t8:8.4f} "
              f"{d71:8.4f} {d86:8.4f} {abs(d71 - d86):8.4f}")
        if t8 + 1e-3 < t7:
            print(f"NOTE dx={dx}: t8 < t7 ({t8:.4f} < {t7:.4f})", file=sys.stderr)
    print("NOTE: 7-1 ~ 8-6 at zero offset (same vertical lid; 1 is conv reflection).")
    print("      Far offset: 1 stays in the lid; 7/6/8 dive, so the identity loosens.")

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    _setup_mpl()
    import matplotlib.pyplot as plt

    rays = parse_ray_file(args.rays) if args.rays.is_file() else []
    if rays:
        report_psx_conversion_points(rays, recs)
    fig = plt.figure(figsize=(13.0, 8.6), facecolor="w", layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.055])
    axs = (
        (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])),
        (fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])),
    )
    cax = fig.add_subplot(gs[:, 2])
    panels = (
        (axs[0][0], (0, 1), "PPP反射（虚线=初至）"),
        (axs[0][1], (1, 7), "PPS（细线=PPP反射）"),
        (axs[1][0], (6,), "PSP：面下 S，两端盖层 P"),
        (axs[1][1], (8,), "PSS：两星，面下 S 同 PSP"),
    )
    ray_xs: list[float] = []
    for xs, _zs in rays:
        ray_xs.extend(xs)
    for i, (ax, codes, title) in enumerate(panels):
        _draw_hybrid_background(
            ax,
            cax if i == 0 else None,
            fig,
            args.smesh,
            args.seafloor,
            args.conv,
        )
        ax.plot(OBS_X, H, "k^", ms=8, zorder=5)
        if rays:
            draw_ps_rays(ax, rays, recs, codes=codes)
        if ray_xs:
            pad = 2.0
            ax.set_xlim(min(ray_xs) - pad, max(ray_xs) + pad)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.88, fontsize=7)
    axs[0][0].set_ylabel("深度 (km)")
    axs[1][0].set_ylabel("深度 (km)")
    axs[1][0].set_xlabel("模型距离 (km)")
    axs[1][1].set_xlabel("模型距离 (km)")
    fig.suptitle(
        f"与 converse 同一套：盖层 Vp，面下 Vs={VS0:g}+{VS_GRAD:g}(z-{Z_CONV:g})  "
        f"（κ={KAPPA:g}）"
    )
    out = HERE / "check_rays.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"写出 {out}")

    out_tt = HERE / "check_tt_red.png"
    plot_reduced_ttimes(recs, out_tt, vred=args.vred, show=not args.no_show)
    print(f"写出 {out_tt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
