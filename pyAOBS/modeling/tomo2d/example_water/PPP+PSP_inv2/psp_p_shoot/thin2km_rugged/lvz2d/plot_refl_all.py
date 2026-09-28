# -*- coding: utf-8 -*-
"""OBS=50：当前代码所有反射相关震相。不改 inv_*。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE
while REPO != REPO.parent:
    if (REPO / "pyAOBS" / "modeling" / "wave2d").is_dir():
        break
    REPO = REPO.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pyAOBS.modeling.wave2d.io_smesh import interp_z, load_xz, parse_pickfile, parse_rays
from pyAOBS.modeling.wave2d.phases import write_geom
from pyAOBS.modeling.wave2d.plot_model_rays import (
    VS_CMAP,
    VS_TICKS,
    VS_XP,
    VS_YP,
    _PiecewiseNorm,
    _style_ifaces,
    _vs_masked,
)

WORK = HERE / "inv_612"
OUT = HERE / "wave_fwd"

# 1 PmP, 3 水柱多次, 5 台侧 Moho peg, 9/14/15 水柱 peg,
# 10/11 盖层 SS, 12/13 转换波打莫霍
REFL_CODES = (1, 3, 5, 9, 10, 11, 12, 13, 14, 15)

STYLE = {
    1: ("#d95f02", "1 PmP"),
    3: ("#6baed6", "3 水多次"),
    5: ("#e6550d", "5 PmP-peg"),
    9: ("#3182bd", "9 PSP-peg"),
    10: ("#756bb1", "10 PPS-SS"),
    11: ("#54278f", "11 PSS-SS"),
    12: ("#c44e8a", "12 PSP-Moho"),
    13: ("#8c564b", "13 PSS-Moho"),
    14: ("#17becf", "14 PPS-peg"),
    15: ("#2ca02c", "15 PSS-peg"),
}


def _draw(ax, picks, rays, codes, moho, *, mark: bool) -> dict[int, tuple[int, int]]:
    drawn: set[int] = set()
    stats = {c: [0, 0] for c in codes}
    bounce_xy: dict[int, list[tuple[float, float]]] = {c: [] for c in codes}
    for (code, rx, _rz, tt, src), (xs, zs) in zip(picks, rays):
        if code not in codes:
            continue
        if not np.isfinite(tt) or tt <= 0:
            stats[code][1] += 1
            continue
        stats[code][0] += 1
        col, lab = STYLE[code]
        kw = dict(color=col, lw=0.8, alpha=0.85, zorder=5)
        if code not in drawn:
            kw["label"] = lab
            drawn.add(code)
        ax.plot(xs, zs, **kw)
        if mark and xs:
            i = int(np.argmax(zs))
            bounce_xy[code].append((float(xs[i]), float(zs[i])))
    if mark:
        for code, pts in bounce_xy.items():
            if not pts:
                continue
            col, _lab = STYLE[code]
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                "o",
                ms=3.2,
                mfc="none",
                mew=1.0,
                color=col,
                zorder=8,
            )
    return {c: (a[0], a[1]) for c, a in stats.items()}


def _report(picks, rays, moho, sea, conv) -> None:
    print("code  n  fail   tmin    tmax   zmax   penMoho  slideMoho  tag")
    for code in REFL_CODES:
        rows = []
        n_fail = 0
        for (c, rx, rz, tt, sx), (xx, zz) in zip(picks, rays):
            if c != code:
                continue
            if not np.isfinite(tt) or tt <= 0 or len(xx) < 2:
                n_fail += 1
                continue
            i = int(np.argmax(zz))
            zm = interp_z(moho, xx[i])
            pen = float(zz[i] - zm)
            xs_n = [x for x, z in zip(xx, zz) if abs(z - interp_z(moho, x)) < 0.25]
            slide = (max(xs_n) - min(xs_n)) if xs_n else 0.0
            rows.append((rx, tt, zz[i], pen, slide))
        if not rows and n_fail == 0:
            continue
        ts = [r[1] for r in rows]
        zms = [r[2] for r in rows]
        pens = [r[3] for r in rows]
        slides = [r[4] for r in rows]
        n_pen = sum(1 for p in pens if p > 0.15)
        n_slide = sum(1 for p, s in zip(pens, slides) if p <= 0.15 and s > 3.0)
        n_ok = len(rows) - n_pen - n_slide
        print(
            f"{code:4d} {len(rows):3d} {n_fail:4d}  "
            f"{(min(ts) if ts else float('nan')):6.2f} {(max(ts) if ts else float('nan')):6.2f}  "
            f"{(max(zms) if zms else float('nan')):5.2f}  "
            f"PEN={n_pen:2d} SLIDE={n_slide:2d} BOUNCE={n_ok:2d}"
        )
        if code in (1, 5, 12, 13):
            for rx, tt, zmax, pen, slide in rows:
                if pen > 0.15 or slide > 20.0 or tt > 20.0:
                    print(f"       rx={rx:6.1f} t={tt:7.3f} zmax={zmax:5.2f} pen={pen:+5.2f} slide={slide:5.1f}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-a", action="store_true", help="读无 -A 的 syn/rays，另存新图")
    p.add_argument("--write-geom", action="store_true")
    args = p.parse_args()
    geom = OUT / "geom_obs50_refl.dat"
    if args.write_geom or not geom.is_file():
        n = write_geom(geom, codes=REFL_CODES, obs_xs=(50.0,))
        print(f"wrote {geom} nrec={n}")
    syn = OUT / ("syn_obs50_refl_noA.dat" if args.no_a else "syn_obs50_refl.dat")
    ray_p = OUT / ("rays_obs50_refl_noA.dat" if args.no_a else "rays_obs50_refl.dat")
    if not syn.is_file() or not ray_p.is_file():
        print(f"missing {syn} or {ray_p}")
        return 0
    picks = parse_pickfile(syn)
    rays = parse_rays(ray_p)
    moho = load_xz(WORK / "moho_true.refl")
    sea = load_xz(WORK / "seafloor.refl")
    conv = load_xz(WORK / "conv.refl")
    print(f"npick={len(picks)} nray={len(rays)}")
    _report(picks, rays, moho, sea, conv)

    xs, zs, vs = _vs_masked(WORK)
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    zlim = float(zs[-1])
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.8), facecolor="w", layout="constrained", sharex=True, sharey=True)
    panels = (
        (axes[0, 0], (1, 5), "莫霍 P：1 PmP  5 PmP-peg"),
        (axes[0, 1], (12, 13), "莫霍 S：12 PSP-Moho  13 PSS-Moho"),
        (axes[1, 0], (3, 9, 14, 15), "水柱反射：3 / 9 PSP-peg / 14 PPS-peg / 15 PSS-peg"),
        (axes[1, 1], (10, 11), "盖层 SS：10 PPS-SS  11 PSS-SS"),
    )
    last = None
    for ax, codes, title in panels:
        last = ax.imshow(
            vs,
            extent=extent,
            cmap=VS_CMAP,
            norm=_PiecewiseNorm(VS_XP, VS_YP),
            aspect="auto",
            interpolation="nearest",
            zorder=0,
        )
        st = _draw(ax, picks, rays, codes, moho, mark=True)
        _style_ifaces(ax, WORK, 50.0)
        ax.set_xlim(0.0, 150.0)
        ax.set_ylim(zlim, 0.0)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.22)
        ax.legend(loc="lower right", fontsize=6.5, framealpha=0.9, ncol=2)
        for c, (keep, fail) in st.items():
            if keep or fail:
                print(f"  plot {c}: keep={keep} fail={fail}")
    axes[0, 0].set_ylabel("深度 (km)")
    axes[1, 0].set_ylabel("深度 (km)")
    axes[1, 0].set_xlabel("x (km)")
    axes[1, 1].set_xlabel("x (km)")
    cbar = fig.colorbar(last, ax=axes.ravel().tolist(), shrink=0.72)
    cbar.set_ticks(VS_TICKS)
    cbar.set_label("Vs (km/s)")
    fig.suptitle(
        f"当前 tt_forward  双场 {'无 -A' if args.no_a else '-A'}  全部反射震相  OBS=50  圆点=最深点",
        fontsize=12,
    )
    png = OUT / ("cmp_refl_all_noA.png" if args.no_a else "cmp_refl_all.png")
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
