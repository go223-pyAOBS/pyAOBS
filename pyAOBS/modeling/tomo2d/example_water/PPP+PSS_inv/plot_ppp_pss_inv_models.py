#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PPP+PSS：Vp 图用 ps_inv 的 plot_models，Vs 图叠 PSS 射线。"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ps_inv"))
sys.path.insert(0, str(HERE.parent / "pss_inv"))
sys.path.insert(0, str(HERE.parent / "ps_fwd"))
sys.path.insert(0, str(HERE.parent / "converse_fwd"))
from check_ps_fwd import draw_ps_rays  # noqa: E402
from make_pss_inv_case import ZMAX, Z_CONV, illum_x_range, parse_smesh  # noqa: E402
import plot_ps_inv_models as psp  # noqa: E402
import plot_pss_inv_models as pss  # noqa: E402


def _zmax(rays) -> float:
    if not rays:
        return 0.0
    return max(max(zs) for _, zs in rays if zs)


def plot_rec_vs_true_rays(rec_path, true_path, rec_rays, recs, true_rays, geom_path, out_png):
    xs, zs, vrec = parse_smesh(rec_path)
    _, _, vtrue = parse_smesh(true_path)
    rec_g, tru_g = psp._grid(xs, zs, vrec), psp._grid(xs, zs, vtrue)
    obs, shots = psp.parse_geom(geom_path)
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    vnorm = Normalize(vmin=pss.VS_VMIN, vmax=pss.VS_VMAX)
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2), facecolor="w", layout="constrained")
    panels = (
        (axes[0], rec_g, rec_rays, f"收回  zmax={_zmax(rec_rays):.2f} km"),
        (axes[1], tru_g, true_rays, f"真值  zmax={_zmax(true_rays):.2f} km"),
    )
    im = None
    for i, (ax, grid, rays, title) in enumerate(panels):
        im = ax.imshow(
            grid, extent=extent, cmap="RdYlBu_r", norm=vnorm,
            aspect="auto", interpolation="nearest", zorder=0,
        )
        if rays and recs:
            draw_ps_rays(ax, rays, recs, thin=len(rays) > 40, codes=(8,))
        psp.draw_geom(ax, obs, shots, legend=(i == 0))
        ax.axhline(Z_CONV, color="0.15", ls="--", lw=1.0, zorder=2)
        ax.set_title(title)
        ax.set_xlim(*illum_x_range())
        ax.set_ylim(ZMAX, 0.0)
        ax.set_xlabel("模型距离 (km)")
        if i == 0:
            ax.set_ylabel("深度 (km)")
            ax.legend(loc="upper right", framealpha=0.9, fontsize=7, ncol=2)
        else:
            ax.tick_params(labelleft=False)
    fig.colorbar(im, ax=axes, shrink=0.72).set_label("Vs (km/s)")
    fig.suptitle("PSS：炮侧盖层 P、面下 S（粉）、台侧盖层 S", fontsize=11)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")


def main() -> int:
    rec = max(
        HERE.glob("out.smesh.*.*"),
        key=lambda p: (int(p.name.split(".")[-2]), int(p.name.split(".")[-1])),
    )
    ppp_rays = psp.parse_rays(HERE / "rays_ppp_rec.dat")
    ppp_recs = psp.parse_syn(HERE / "syn_ppp.dat")
    psp.plot_models(
        HERE / "rec_vp.smesh",
        HERE / "true_vp.smesh",
        HERE / "start_vp.smesh",
        HERE / "check_inv_vp_models.png",
        HERE / "geom_ppp.dat",
        ppp_rays,
        ppp_recs,
        "vp",
    )
    vs_rays = psp.parse_rays(HERE / "rays_rec.dat")
    vs_recs = psp.parse_syn(HERE / "syn_inv.dat")
    pss.plot_models(
        rec,
        HERE / "true_vs.smesh",
        HERE / "start_vs.smesh",
        HERE / "check_inv_models.png",
        HERE / "geom_inv.dat",
        vs_rays,
        vs_recs,
        vp_label="初值Vp",
    )
    true_rays = (
        psp.parse_rays(HERE / "rays_true.dat")
        if (HERE / "rays_true.dat").is_file()
        else []
    )
    plot_rec_vs_true_rays(
        rec,
        HERE / "true_vs.smesh",
        vs_rays,
        vs_recs,
        true_rays,
        HERE / "geom_inv.dat",
        HERE / "check_inv_rays.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
