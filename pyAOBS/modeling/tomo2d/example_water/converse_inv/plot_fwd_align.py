#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对照 converse / 双场 PSP 正演：射线 + T–X + 残差。不依赖 GUI。"""

from __future__ import annotations

import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
H = 2.0
Z_CONV = 4.5
ZMAX = 12.0


def parse_picks(path: Path):
    recs = []
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    i = 1
    src_x = 0.0
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
                recs.append((int(float(rp[3])), float(rp[1]), float(rp[2]), float(rp[4]), src_x))
    return recs


def parse_geom(path: Path):
    stations, shots, seen = [], [], set()
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
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


def key(p):
    return (round(p[4], 3), round(p[1], 3), p[0])


def align(a, b):
    mb = {key(p): p for p in b}
    return [(p, mb[key(p)]) for p in a if key(p) in mb]


def rms(vals):
    return math.sqrt(sum(v * v for v in vals) / len(vals)) if vals else float("nan")


def split_ps(xs, zs, z_conv=Z_CONV, eps=1e-3):
    if len(xs) < 2:
        return []

    def is_s(z):
        return z > z_conv + eps

    def cross(x0, z0, x1, z1):
        if abs(z1 - z0) < 1e-15:
            return x0, z_conv
        t = (z_conv - z0) / (z1 - z0)
        return x0 + t * (x1 - x0), z_conv

    out = []
    cx, cz, cs = [xs[0]], [zs[0]], is_s(zs[0])
    for x1, z1 in zip(xs[1:], zs[1:]):
        s1 = is_s(z1)
        if s1 == cs:
            cx.append(x1)
            cz.append(z1)
            continue
        xc, zc = cross(cx[-1], cz[-1], x1, z1)
        cx.append(xc)
        cz.append(zc)
        if len(cx) >= 2:
            out.append((cx, cz, cs))
        cx, cz, cs = [xc, x1], [zc, z1], s1
    if len(cx) >= 2:
        out.append((cx, cz, cs))
    return out


def draw_rays(ax, rays, legend=False):
    seen_p = seen_s = False
    for xs, zs in rays:
        for sx, sz, is_s in split_ps(xs, zs):
            color = "#e377c2" if is_s else "#1f77b4"
            kw = dict(color=color, lw=0.45, ls="-", alpha=0.88, zorder=3)
            if legend and is_s and not seen_s:
                kw["label"] = "S（面下）"
                seen_s = True
            elif legend and (not is_s) and not seen_p:
                kw["label"] = "P（水+盖层）"
                seen_p = True
            ax.plot(sx, sz, **kw)


def draw_geom(ax, stations, shots, legend=False):
    ax.axhline(H, color="0.2", ls="--", lw=1.0, zorder=2, label="海底" if legend else None)
    ax.axhline(Z_CONV, color="0.35", ls="--", lw=1.1, zorder=2, label="转换面" if legend else None)
    if shots:
        sx, sz = zip(*shots)
        ax.plot(sx, sz, "v", color="0.15", ms=3.5, zorder=5, label="炮" if legend else None)
    if stations:
        ox, oz = zip(*stations)
        ax.plot(ox, oz, "^", color="k", ms=7, zorder=6, label="OBS" if legend else None)


def _roughness(xs, zs):
    if len(xs) < 3:
        return 0.0
    acc = 0.0
    n = 0
    for i in range(1, len(xs) - 1):
        ax, az = xs[i] - xs[i - 1], zs[i] - zs[i - 1]
        bx, bz = xs[i + 1] - xs[i], zs[i + 1] - zs[i]
        la = math.hypot(ax, az)
        lb = math.hypot(bx, bz)
        if la < 1e-8 or lb < 1e-8:
            continue
        acc += abs(ax * bz - az * bx) / (la * lb)
        n += 1
    return acc / n if n else 0.0


def lid_stats(rays, z1=H, z2=Z_CONV):
    rs = []
    for xs, zs in rays:
        xx, zz = [], []
        for x, z in zip(xs, zs):
            if z1 - 0.05 <= z <= z2 + 0.05:
                xx.append(x)
                zz.append(z)
        if len(xx) >= 4:
            rs.append(_roughness(xx, zz))
    return (sum(rs) / len(rs) if rs else 0.0), (max(rs) if rs else 0.0)


def plot_rays_fig(out_png: Path):
    import matplotlib.pyplot as plt

    stations, shots = parse_geom(HERE / "geom_inv.dat")
    xs = [p[0] for p in shots] + [p[0] for p in stations]
    xlim = (min(xs) - 3.0, max(xs) + 3.0)
    pairs = (
        ("converse  混合单场", HERE / "rays_cv_true.dat"),
        ("双场  -M true_vp -U true_vs", HERE / "rays_psx_true.dat"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharey=True, facecolor="w")
    for i, (title, rayp) in enumerate(pairs):
        ax = axes[i]
        rays = parse_rays(rayp)
        draw_rays(ax, rays, legend=(i == 0))
        draw_geom(ax, stations, shots, legend=(i == 0))
        zmax = max((max(zs) for _x, zs in rays), default=0.0)
        rmean, rmax = lid_stats(rays)
        print(f"lid roughness {title}: mean {rmean:.4f} max {rmax:.4f}")
        ax.set_title(f"{title}\n{len(rays)} 条  最深 {zmax:.2f} km")
        ax.set_xlim(*xlim)
        ax.set_ylim(ZMAX, 0.0)
        ax.set_xlabel("模型距离 (km)")
        ax.grid(True, alpha=0.28)
        if i == 0:
            ax.set_ylabel("深度 (km)")
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
    fig.suptitle("真模型 PSP 正演射线（蓝 P / 粉 S）", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")


def plot_tt_fig(out_png: Path):
    import matplotlib.pyplot as plt

    cv = parse_picks(HERE / "syn_inv.dat")
    du = parse_picks(HERE / "syn_psx.dat")
    cvs = parse_picks(HERE / "syn_start.dat")
    dus = parse_picks(HERE / "syn_psx_start.dat")
    true_pairs = align(cv, du)
    start_cv = align(cv, cvs)
    start_du = align(du, dus)
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(8.6, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.55, 1.0]}
    )
    dx = [abs(p[1] - p[4]) for p, _ in true_pairs]
    tcv = [p[3] for p, _ in true_pairs]
    tdu = [q[3] for _, q in true_pairs]
    order = sorted(range(len(dx)), key=lambda i: (dx[i], tcv[i]))
    dx_s = [dx[i] for i in order]
    ax.plot(dx_s, [tcv[i] for i in order], "o", color="#1f77b4", ms=5.5, zorder=3, label="converse 真模型")
    ax.plot(dx_s, [tdu[i] for i in order], "x", color="#d62728", ms=5.0, zorder=4, label="双场 真模型")
    d_td = [q[3] - p[3] for p, q in true_pairs]
    ax.set_ylabel("走时 t (s)")
    ax.set_title(f"PSP 正演 T–X    双场 − converse  RMS {rms(d_td):.3f} s  n={len(true_pairs)}")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    d_cv0 = [s[3] - o[3] for o, s in start_cv]
    d_du0 = [s[3] - o[3] for o, s in start_du]
    axr.plot([abs(p[1] - p[4]) for p, _ in true_pairs], d_td, "o", color="0.25", ms=5,
             label=f"双场−converse  RMS {rms(d_td):.3f}")
    axr.plot([abs(o[1] - o[4]) for o, _ in start_cv], d_cv0, "s", color="#1f77b4", ms=4, alpha=0.7,
             label=f"converse 初值−真  RMS {rms(d_cv0):.3f}")
    axr.plot([abs(o[1] - o[4]) for o, _ in start_du], d_du0, "^", color="#d62728", ms=4, alpha=0.7,
             label=f"双场 初值−真  RMS {rms(d_du0):.3f}")
    axr.axhline(0.0, color="0.4", lw=0.8)
    axr.set_xlabel("偏移 dx (km)")
    axr.set_ylabel("残差 (s)")
    axr.grid(True, alpha=0.35)
    axr.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"true dual-cv RMS {rms(d_td):.4f}  max {max(abs(v) for v in d_td):.4f}")
    print(f"start-true  cv {rms(d_cv0):.4f}  du {rms(d_du0):.4f}")
    print("offset-bin dual-cv RMS")
    dx_td = [abs(p[1] - p[4]) for p, _ in true_pairs]
    for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 50)):
        sub = [d for d, x in zip(d_td, dx_td) if lo <= x < hi]
        if sub:
            print(f"  {lo:2d}-{hi:<2d} km  n={len(sub):3d}  rms={rms(sub):.4f}  "
                  f"mean={sum(sub)/len(sub):+.4f}")


def plot_lid_zoom(out_png: Path):
    import matplotlib.pyplot as plt

    stations, shots = parse_geom(HERE / "geom_inv.dat")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 3.8), sharey=True, facecolor="w")
    pairs = (
        ("converse", HERE / "rays_cv_true.dat"),
        ("双场", HERE / "rays_psx_true.dat"),
    )
    for i, (title, rayp) in enumerate(pairs):
        ax = axes[i]
        rays = parse_rays(rayp)
        draw_rays(ax, rays, legend=(i == 0))
        draw_geom(ax, stations, shots, legend=False)
        ax.set_xlim(28, 52)
        ax.set_ylim(Z_CONV + 0.15, 0.0)
        ax.set_title(f"{title}  浅部放大")
        ax.set_xlabel("模型距离 (km)")
        ax.grid(True, alpha=0.28)
        if i == 0:
            ax.set_ylabel("深度 (km)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    print("plot tt", flush=True)
    plot_tt_fig(HERE / "check_fwd_align_tt.png")
    print("plot rays", flush=True)
    plot_rays_fig(HERE / "check_fwd_align_rays.png")
    print("plot lid zoom", flush=True)
    plot_lid_zoom(HERE / "check_fwd_align_lid.png")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
