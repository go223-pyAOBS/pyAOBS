#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""面下 Vs：PSS vs 正演校正 PSP（全体 / >20 km），对照全真 PSP。"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inv_2d"))
sys.path.insert(0, str(ROOT.parents[1]))
sys.path.insert(0, str(ROOT.parents[1].parent / "water_inv"))
import inv_grid as g  # noqa: E402
from check_joint import DCLIM, OBS_XS, PHASES, region_rms, ttrms  # noqa: E402
from check_lvz import load_vp_vs  # noqa: E402
from check_sparse import _rms  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

XLO, XHI = 18.0, 82.0


def _print_vs(lab, xs, zs, rec, true):
    print(
        f"  {lab:22s}  lid {region_rms(xs, zs, rec, true, lid=True):.4f}  "
        f"below {region_rms(xs, zs, rec, true, lid=False):.4f}  "
        f"crust {region_rms(xs, zs, rec, true, lid=None):.4f}"
    )


def _style(ax, title):
    xs_l = np.linspace(XLO, XHI, 80)
    ax.plot(xs_l, [g.z_conv(x) for x in xs_l], "k-.", lw=0.8, zorder=2)
    ax.plot(xs_l, [g.H] * len(xs_l), "k--", lw=0.6, zorder=2)
    ax.plot(list(OBS_XS), [g.H] * len(OBS_XS), "k^", ms=5, zorder=7)
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(16, 0)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)


def main() -> int:
    folders = (
        ("A  PSS", HERE / "path_a"),
        ("C  corr PSP all", HERE / "path_c"),
        ("Cf corr PSP >20km", HERE / "path_cf"),
        ("B  true PSP (same lid)", HERE / "path_bf"),
    )
    xs, zs, _, _, _, t_vs, _, _ = load_vp_vs(HERE / "path_a")
    recs = []
    print("Vs rec vs true")
    for lab, folder in folders:
        if not (folder / "rec_vs.smesh").is_file():
            print(f"  {lab:22s}  no rec_vs")
            recs.append((lab, None))
            continue
        _, _, _, _, _, t_vs_f, _, r_vs = load_vp_vs(folder)
        _print_vs(lab, xs, zs, r_vs, t_vs_f)
        recs.append((lab, r_vs))
        hold_t, hold_r = folder / "syn_holdout_true.dat", folder / "syn_holdout_rec.dat"
        if hold_t.is_file() and hold_r.is_file():
            print(f"    holdout PSP {ttrms(hold_t, hold_r, 6):.4f}  PSS {ttrms(hold_t, hold_r, 8):.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4), facecolor="w", layout="constrained")
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    last = None
    for ax, (lab, rec) in zip(axes.ravel(), recs):
        if rec is None:
            ax.set_title(f"{lab} 无 rec_vs")
            ax.axis("off")
            continue
        grid = rec - t_vs
        im = ax.imshow(
            grid,
            extent=extent,
            cmap="RdBu_r",
            vmin=-DCLIM,
            vmax=DCLIM,
            aspect="auto",
            interpolation="nearest",
            zorder=0,
        )
        last = im
        _style(
            ax,
            f"{lab} rec-true Vs\n"
            f"lid {region_rms(xs, zs, rec, t_vs, lid=True):.3f}  "
            f"below {region_rms(xs, zs, rec, t_vs, lid=False):.3f}",
        )
    if last is not None:
        fig.colorbar(last, ax=axes.ravel().tolist(), shrink=0.72, label="dVs (km/s)")
    axes[0, 0].set_ylabel("深度 (km)")
    axes[1, 0].set_ylabel("深度 (km)")
    fig.suptitle("同一 PPP+PPS 盖层    A:PSS    C:校正PSP全体    Cf:>20km    B:真PSP")
    out = HERE / "check_corr_vs.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
