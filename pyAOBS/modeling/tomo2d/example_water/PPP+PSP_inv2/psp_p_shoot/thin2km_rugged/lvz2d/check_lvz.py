#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比 A（PPS−PPP 盖层 + PSS 面下）与 B（PPP+PSP）。"""

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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inv_2d"))
sys.path.insert(0, str(ROOT.parents[1]))
sys.path.insert(0, str(ROOT.parents[1].parent / "water_inv"))
sys.path.insert(0, str(ROOT.parents[1].parent / "ps_fwd"))
import inv_grid as g  # noqa: E402
import make_ppp_psp_inv_case as m2  # noqa: E402
from check_joint import (  # noqa: E402
    DCLIM,
    OBS_XS,
    PHASES,
    mask_crust,
    plot_phase_ttimes,
    region_rms,
    ttrms,
)
from check_water_inv import parse_picks  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

XLO, XHI = 18.0, 82.0


def load_vp_vs(folder: Path):
    xs, zs, t_vp = m2.parse_smesh(folder / "true_vp.smesh")
    _, _, s_vp = m2.parse_smesh(folder / "start_vp.smesh")
    rec_vp_p = folder / "rec_vp.smesh"
    if not rec_vp_p.is_file():
        rec_vp_p = folder / "ppp_vp.smesh"
    _, _, r_vp = m2.parse_smesh(rec_vp_p)
    t_vs = s_vs = r_vs = None
    if (folder / "true_vs.smesh").is_file():
        _, _, t_vs = m2.parse_smesh(folder / "true_vs.smesh")
        t_vs = mask_crust(xs, zs, t_vs)
    if (folder / "start_vs.smesh").is_file():
        _, _, s_vs = m2.parse_smesh(folder / "start_vs.smesh")
        s_vs = mask_crust(xs, zs, s_vs)
    if (folder / "rec_vs.smesh").is_file():
        _, _, r_vs = m2.parse_smesh(folder / "rec_vs.smesh")
        r_vs = mask_crust(xs, zs, r_vs)
    return (
        xs,
        zs,
        mask_crust(xs, zs, t_vp),
        mask_crust(xs, zs, s_vp),
        mask_crust(xs, zs, r_vp),
        t_vs,
        s_vs,
        r_vs,
    )


def _rms(lab, xs, zs, a, b):
    print(
        f"  {lab:22s}  lid {region_rms(xs, zs, a, b, lid=True):.4f}  "
        f"below {region_rms(xs, zs, a, b, lid=False):.4f}  "
        f"crust {region_rms(xs, zs, a, b, lid=None):.4f}"
    )


def _tt(folder: Path, label: str, phases) -> None:
    hold_t, hold_r = folder / "syn_holdout_true.dat", folder / "syn_holdout_rec.dat"
    obs, start_p, rec_p = folder / "syn_inv.dat", folder / "syn_start.dat", folder / "syn_rec.dat"
    print(f"{label}  inverted geom  start→rec")
    if obs.is_file() and start_p.is_file() and rec_p.is_file():
        for code, name in phases:
            print(
                f"  {name:3s}({code})  {ttrms(obs, start_p, code):.4f} → {ttrms(obs, rec_p, code):.4f}"
            )
    if hold_t.is_file() and hold_r.is_file():
        print(f"{label}  holdout all vs true")
        for code, name in PHASES:
            print(f"  {name:3s}({code})  {ttrms(hold_t, hold_r, code):.4f}")


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
    ppp = HERE
    fa, fb = HERE / "path_a", HERE / "path_b"
    xs, zs, t_vp, s_vp, r_vp, _, _, _ = load_vp_vs(ppp)
    _, _, _, _, _, t_vs_a, s_vs_a, r_vs_a = load_vp_vs(fa)
    _, _, _, _, _, t_vs_b, s_vs_b, r_vs_b = load_vp_vs(fb)
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))

    print("PPP Vp")
    _rms("start-true", xs, zs, s_vp, t_vp)
    _rms("rec-true  ", xs, zs, r_vp, t_vp)
    ppp_obs, ppp_st, ppp_rec = HERE / "syn_ppp.dat", HERE / "syn_ppp_start.dat", HERE / "syn_ppp_rec.dat"
    if ppp_obs.is_file() and ppp_st.is_file() and ppp_rec.is_file():
        print(f"  PPP t  {ttrms(ppp_obs, ppp_st):.4f} → {ttrms(ppp_obs, ppp_rec):.4f}")
    print("A  PPS+PPS-PPP lid Vs, then PSS below Vs")
    if t_vs_a is not None and r_vs_a is not None:
        _rms("Vs start-true", xs, zs, s_vs_a, t_vs_a)
        _rms("Vs rec-true  ", xs, zs, r_vs_a, t_vs_a)
        if s_vs_a is not None:
            _rms("Vs rec-start ", xs, zs, r_vs_a, s_vs_a)
    _tt(fa, "A", ((0, "PPP"), (7, "PPS"), (8, "PSS")))
    print("B  PPP+PSP")
    if t_vs_b is not None and r_vs_b is not None:
        _rms("Vs start-true", xs, zs, s_vs_b, t_vs_b)
        _rms("Vs rec-true  ", xs, zs, r_vs_b, t_vs_b)
        if s_vs_b is not None:
            _rms("Vs rec-start ", xs, zs, r_vs_b, s_vs_b)
    _tt(fb, "B", ((6, "PSP"),))

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.6), facecolor="w", layout="constrained")
    panels = (
        (axes[0, 0], t_vp, "RdYlBu_r", 1.6, 8.4, "真 Vp（壳/幔低速）"),
        (
            axes[0, 1],
            r_vp - t_vp,
            "RdBu_r",
            -DCLIM,
            DCLIM,
            f"PPP rec-true Vp\nlid {region_rms(xs,zs,r_vp,t_vp,lid=True):.3f}  "
            f"below {region_rms(xs,zs,r_vp,t_vp,lid=False):.3f}",
        ),
        (
            axes[0, 2],
            s_vp - t_vp,
            "RdBu_r",
            -DCLIM,
            DCLIM,
            f"PPP start-true Vp\nlid {region_rms(xs,zs,s_vp,t_vp,lid=True):.3f}  "
            f"below {region_rms(xs,zs,s_vp,t_vp,lid=False):.3f}",
        ),
        (
            axes[1, 0],
            (r_vs_a - t_vs_a) if r_vs_a is not None else t_vp * np.nan,
            "RdBu_r",
            -DCLIM,
            DCLIM,
            (
                f"A rec−true Vs\nlid {region_rms(xs,zs,r_vs_a,t_vs_a,lid=True):.3f}  "
                f"below {region_rms(xs,zs,r_vs_a,t_vs_a,lid=False):.3f}"
                if r_vs_a is not None
                else "A 无 rec_vs"
            ),
        ),
        (
            axes[1, 1],
            (r_vs_b - t_vs_b) if r_vs_b is not None else t_vp * np.nan,
            "RdBu_r",
            -DCLIM,
            DCLIM,
            (
                f"B rec−true Vs\nlid {region_rms(xs,zs,r_vs_b,t_vs_b,lid=True):.3f}  "
                f"below {region_rms(xs,zs,r_vs_b,t_vs_b,lid=False):.3f}"
                if r_vs_b is not None
                else "B 无 rec_vs"
            ),
        ),
        (
            axes[1, 2],
            (r_vs_a - r_vs_b) if r_vs_a is not None and r_vs_b is not None else t_vp * np.nan,
            "RdBu_r",
            -DCLIM,
            DCLIM,
            "A-B  Vs（红=A 更快）",
        ),
    )
    last_d = None
    for ax, grid, cmap, vmin, vmax, title in panels:
        im = ax.imshow(
            grid,
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
            zorder=0,
        )
        if cmap == "RdBu_r":
            last_d = im
        else:
            fig.colorbar(im, ax=ax, shrink=0.72, label="Vp (km/s)")
        _style(ax, title)
        if ax is axes[0, 0] or ax is axes[1, 0]:
            ax.set_ylabel("深度 (km)")
    if last_d is not None:
        fig.colorbar(last_d, ax=axes[1, :].ravel().tolist(), shrink=0.72, label="ΔV (km/s)")
    fig.suptitle("二维非均匀  共享 PPP Vp    A: PPS+PPS-PPP -> PSS    B: PPP+PSP    红=偏快 蓝=偏慢")
    out = HERE / "check_lvz.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")

    for folder, phases, name in (
        (fa, ((0, "PPP"), (7, "PPS"), (8, "PSS")), "A"),
        (fb, ((6, "PSP"),), "B"),
    ):
        obs, start_p, rec_p = folder / "syn_inv.dat", folder / "syn_start.dat", folder / "syn_rec.dat"
        if obs.is_file() and start_p.is_file() and rec_p.is_file():
            plot_phase_ttimes(
                parse_picks(obs.read_text(encoding="utf-8")),
                parse_picks(start_p.read_text(encoding="utf-8")),
                parse_picks(rec_p.read_text(encoding="utf-8")),
                phases,
                HERE / f"check_lvz_{name.lower()}_ttimes.png",
                f"{name}  走时拟合",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
