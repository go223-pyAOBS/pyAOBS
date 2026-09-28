#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""12/13 Vs 段：真/初/反 Vs，莫霍，走时残差。"""

from __future__ import annotations

import math
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
sys.path.insert(0, str(ROOT.parents[1].parent / "water_inv"))
import inv_grid as g  # noqa: E402
import make_ppp_psp_inv_case as m2  # noqa: E402
from check_joint import mask_crust, region_rms, ttrms  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

WORK = HERE / "inv_1213"
XLO, XHI = 18.0, 82.0
VCLIM = (1.8, 4.8)
DCLIM = 0.80


def _load_xz(path: Path) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        a = ln.split()
        if len(a) >= 2:
            out.append((float(a[0]), float(a[1])))
    return out


def _z_at(xz: list[tuple[float, float]], x: float) -> float:
    return float(np.interp(x, [p[0] for p in xz], [p[1] for p in xz]))


def _moho_rms(rec: list[tuple[float, float]], true: list[tuple[float, float]]) -> float:
    s = n = 0
    for x, zt in true:
        if not (XLO <= x <= XHI):
            continue
        d = _z_at(rec, x) - zt
        s += d * d
        n += 1
    return math.sqrt(s / n) if n else float("nan")


def _style(ax, title, moho=None):
    xs_l = np.linspace(XLO, XHI, 120)
    ax.plot(xs_l, [g.z_conv(x) for x in xs_l], "k-.", lw=0.8, zorder=2)
    ax.plot(xs_l, [g.H] * len(xs_l), "k--", lw=0.6, zorder=2)
    if moho:
        ax.plot([p[0] for p in moho], [p[1] for p in moho], "k:", lw=1.2, zorder=3)
    ax.plot([30.0, 50.0], [g.H, g.H], "k^", ms=6, zorder=7)
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(16, 0)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)


def main() -> int:
    xs, zs, t_raw = m2.parse_smesh(WORK / "true_vs.smesh")
    _, _, s_raw = m2.parse_smesh(WORK / "start_vs.smesh")
    _, _, r_raw = m2.parse_smesh(WORK / "rec_vs.smesh")
    t_vs = mask_crust(xs, zs, t_raw)
    s_vs = mask_crust(xs, zs, s_raw)
    r_vs = mask_crust(xs, zs, r_raw)
    true_m = _load_xz(WORK / "moho_true.refl")
    start_m = _load_xz(WORK / "moho.refl")
    rec_m = _load_xz(WORK / "rec_moho.refl")
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))

    print("Vs RMS vs true")
    print(
        f"  start  lid {region_rms(xs, zs, s_vs, t_vs, lid=True):.4f}  "
        f"below {region_rms(xs, zs, s_vs, t_vs, lid=False):.4f}"
    )
    print(
        f"  rec    lid {region_rms(xs, zs, r_vs, t_vs, lid=True):.4f}  "
        f"below {region_rms(xs, zs, r_vs, t_vs, lid=False):.4f}"
    )
    print("Moho RMS vs true (km)")
    print(f"  start  {_moho_rms(start_m, true_m):.4f}")
    print(f"  rec    {_moho_rms(rec_m, true_m):.4f}")
    syn_t, syn_s, syn_r = WORK / "syn_inv.dat", WORK / "syn_start.dat", WORK / "syn_rec.dat"
    if syn_t.is_file() and syn_s.is_file() and syn_r.is_file():
        for code in (12, 13):
            print(
                f"  tt{code}  start {ttrms(syn_t, syn_s, code):.4f} s  "
                f"rec {ttrms(syn_t, syn_r, code):.4f} s"
            )

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.6), facecolor="w", layout="constrained")
    panels = (
        (axes[0, 0], t_vs, "RdYlBu_r", *VCLIM, "真 Vs", true_m),
        (axes[0, 1], s_vs, "RdYlBu_r", *VCLIM, "初 Vs（真盖层）", start_m),
        (axes[0, 2], r_vs, "RdYlBu_r", *VCLIM, "反 Vs  12/13", rec_m),
        (axes[1, 0], s_vs - t_vs, "RdBu_r", -DCLIM, DCLIM, "初 − 真", start_m),
        (axes[1, 1], r_vs - t_vs, "RdBu_r", -DCLIM, DCLIM, "反 − 真", rec_m),
        (axes[1, 2], r_vs - s_vs, "RdBu_r", -DCLIM, DCLIM, "反 − 初", rec_m),
    )
    last = None
    for ax, arr, cmap, vmin, vmax, title, moho in panels:
        im = ax.imshow(
            arr,
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
            zorder=0,
        )
        _style(ax, title, moho)
        last = im
        if ax in (axes[0, 0], axes[1, 0]):
            ax.set_ylabel("深度 (km)")
        if ax in axes[1]:
            ax.set_xlabel("x (km)")
    fig.colorbar(axes[0, 2].images[0], ax=axes[0, :].ravel().tolist(), shrink=0.86, label="Vs (km/s)")
    fig.colorbar(last, ax=axes[1, :].ravel().tolist(), shrink=0.86, label="ΔVs (km/s)")
    fig.suptitle("lvz2d / inv_1213   冻真 Vp · 12/13 反面下 Vs + 莫霍", fontsize=12)
    out = WORK / "check_inv_1213.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
