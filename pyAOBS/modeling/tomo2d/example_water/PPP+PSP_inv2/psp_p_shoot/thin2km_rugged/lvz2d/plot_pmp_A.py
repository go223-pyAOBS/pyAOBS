# -*- coding: utf-8 -*-
"""OBS=50 震相 1：有/无 -A 对照图。不改 inv_*。"""

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
from pyAOBS.modeling.wave2d.plot_model_rays import (
    VS_CMAP,
    VS_TICKS,
    VS_XP,
    VS_YP,
    _PiecewiseNorm,
    _draw_rays,
    _style_ifaces,
    _vs_masked,
)

WORK = HERE / "inv_612"
OUT = HERE / "wave_fwd"


def _stats(tag: str, syn: Path, ray_p: Path, moho) -> tuple:
    picks = parse_pickfile(syn)
    rays = parse_rays(ray_p)
    n_bounce = n_slide = n_pen = 0
    print(f"=== {tag} ===")
    print("  rx       t    pen   slide")
    for (c, rx, rz, tt, sx), (xx, zz) in zip(picks, rays):
        i = int(np.argmax(zz))
        pen = float(zz[i] - interp_z(moho, xx[i]))
        xs_n = [x for x, z in zip(xx, zz) if abs(z - interp_z(moho, x)) < 0.25]
        slide = (max(xs_n) - min(xs_n)) if xs_n else 0.0
        if pen > 0.15:
            n_pen += 1
            lab = "PEN"
        elif slide > 3.0:
            n_slide += 1
            lab = "SLIDE"
        else:
            n_bounce += 1
            lab = "BOUNCE"
        print(f"  {rx:6.1f} {tt:7.3f} {pen:+5.2f} {slide:6.1f}  {lab}")
    print(f"  BOUNCE={n_bounce} SLIDE={n_slide} PEN={n_pen}")
    return picks, rays


def _tmap(p: Path) -> dict[float, float]:
    return {rx: t for c, rx, rz, t, sx in parse_pickfile(p)}


def plot_pair(
    syn_noa: Path,
    ray_noa: Path,
    syn_a: Path,
    ray_a: Path,
    png: Path,
    *,
    title: str,
    panel_noa: str,
    panel_a: str,
) -> None:
    moho = load_xz(WORK / "moho_true.refl")
    xs, zs, vs = _vs_masked(WORK)
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    zlim = float(zs[-1])
    picks_n, rays_n = _stats(panel_noa, syn_noa, ray_noa, moho)
    picks_a, rays_a = _stats(panel_a, syn_a, ray_a, moho)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 8.6), facecolor="w", layout="constrained", sharex=True)
    last = None
    for ax, picks, rays, lab in (
        (axes[0], picks_n, rays_n, panel_noa),
        (axes[1], picks_a, rays_a, panel_a),
    ):
        last = ax.imshow(
            vs,
            extent=extent,
            cmap=VS_CMAP,
            norm=_PiecewiseNorm(VS_XP, VS_YP),
            aspect="auto",
            interpolation="nearest",
            zorder=0,
        )
        _draw_rays(ax, picks, rays, (1,), moho, every=1, bounce_tol=None, mark_bounce=True)
        _style_ifaces(ax, WORK, 50.0)
        ax.set_xlim(0.0, 150.0)
        ax.set_ylim(zlim, 0.0)
        ax.set_ylabel("深度 (km)")
        ax.set_title(lab, fontsize=10)
        ax.grid(True, alpha=0.22)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    axes[1].set_xlabel("x (km)")
    cbar = fig.colorbar(last, ax=axes.ravel().tolist(), shrink=0.78)
    cbar.set_ticks(VS_TICKS)
    cbar.set_label("Vs (km/s)   蓝绿=盖层  黄橙=地壳  红=地幔")
    fig.suptitle(title, fontsize=12)
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"wrote {png}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("dual", "src"), default="dual")
    args = p.parse_args()
    if args.mode == "dual":
        plot_pair(
            OUT / "syn_pmp_dual_noA.dat",
            OUT / "rays_pmp_dual_noA.dat",
            OUT / "syn_pmp_dual_A.dat",
            OUT / "rays_pmp_dual_A.dat",
            OUT / "cmp_pmp_dual_A.png",
            title="双场（-U true_vs）震相 1   OBS=50   圆点=最深点   白黑粗线=莫霍",
            panel_noa="双场 PmP（1）无 -A：全部射线",
            panel_a="双场 PmP（1）有 -A：全部射线",
        )
        src_noa = _tmap(OUT / "syn_pmp_src_noA.dat")
        src_a = _tmap(OUT / "syn_pmp_src_A.dat")
        dual_noa = _tmap(OUT / "syn_pmp_dual_noA.dat")
        dual_a = _tmap(OUT / "syn_pmp_dual_A.dat")
        print("rx     d-noA   s-noA   d-A     s-A")
        for rx in sorted(dual_noa):
            print(
                f"{rx:6.1f} {dual_noa[rx]:7.3f} {src_noa.get(rx, np.nan):7.3f} "
                f"{dual_a[rx]:7.3f} {src_a.get(rx, np.nan):7.3f}"
            )
    else:
        plot_pair(
            OUT / "syn_pmp_src_noA.dat",
            OUT / "rays_pmp_src_noA.dat",
            OUT / "syn_pmp_src_A.dat",
            OUT / "rays_pmp_src_A.dat",
            OUT / "cmp_pmp_src_single_A.png",
            title="原版 tomo2d 单场（无 -U）震相 1   OBS=50   圆点=最深点   白黑粗线=莫霍",
            panel_noa="原版单场 PmP（1）无 -A：全部射线",
            panel_a="原版单场 PmP（1）有 -A：全部射线",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
