#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""真 Vs、面下反演初值 rec_vs_lid、以及差。"""

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
import inv_grid as g  # noqa: E402
import make_ppp_psp_inv_case as m2  # noqa: E402
from check_joint import OBS_XS, mask_crust, region_rms  # noqa: E402
from check_lvz import load_vp_vs  # noqa: E402
from make_lvz import BEL_LVZ, LID_LVZ  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

XLO, XHI = 18.0, 82.0
VCLIM = (1.8, 4.8)
DCLIM = 1.00


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
    fa = HERE / "path_a"
    xs, zs, _, _, _, t_vs, s_vs, _ = load_vp_vs(fa)
    _, _, vel = m2.parse_smesh(fa / "rec_vs_lid.smesh")
    i_vs = mask_crust(xs, zs, vel)
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    dlt = i_vs - t_vs
    print(
        "true Vs = true_vp / %.2f  (lid LVZ x=%.0f z=%.2f amp=%+.2f km/s on Vp; "
        "below LVZ x=%.0f z=%.2f amp=%+.2f km/s on Vp)"
        % (
            g.KAPPA,
            LID_LVZ["x0"],
            LID_LVZ["z0"],
            LID_LVZ["amp"],
            BEL_LVZ["x0"],
            BEL_LVZ["z0"],
            BEL_LVZ["amp"],
        )
    )
    print(
        f"init rec_vs_lid vs true  lid {region_rms(xs, zs, i_vs, t_vs, lid=True):.4f}  "
        f"below {region_rms(xs, zs, i_vs, t_vs, lid=False):.4f}"
    )
    if s_vs is not None:
        print(
            f"start_vs vs true         lid {region_rms(xs, zs, s_vs, t_vs, lid=True):.4f}  "
            f"below {region_rms(xs, zs, s_vs, t_vs, lid=False):.4f}"
        )

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6), facecolor="w", layout="constrained")
    panels = (
        (axes[0], t_vs, "RdYlBu_r", *VCLIM, "真 Vs  true_vp/1.73"),
        (
            axes[1],
            i_vs,
            "RdYlBu_r",
            *VCLIM,
            f"初值 rec_vs_lid\nrec_vp/1.73 + {g.HOT_DV:g}（盖层已被 PPS 改过）",
        ),
        (
            axes[2],
            dlt,
            "RdBu_r",
            -DCLIM,
            DCLIM,
            f"初值 - 真\nlid {region_rms(xs, zs, i_vs, t_vs, lid=True):.3f}  "
            f"below {region_rms(xs, zs, i_vs, t_vs, lid=False):.3f}",
        ),
    )
    last_v = last_d = None
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
            last_v = im
        _style(ax, title)
    axes[0].set_ylabel("深度 (km)")
    fig.colorbar(last_v, ax=axes[0:2].ravel().tolist(), shrink=0.78, label="Vs (km/s)")
    fig.colorbar(last_d, ax=axes[2], shrink=0.78, label="dVs (km/s)")
    fig.suptitle("面下反演所用初值 vs 真 Vs    点划线=转换面    红=初值偏快")
    out = HERE / "check_vs_init.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
