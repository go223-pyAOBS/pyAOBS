#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""-DD200 (inv_612) vs -DD50。不改原工区。SENS / C2F 关。"""

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
sys.path.insert(0, str(ROOT.parents[1]))
sys.path.insert(0, str(ROOT.parents[1].parent / "water_inv"))
sys.path.insert(0, str(ROOT.parents[1].parent / "ps_fwd"))
import inv_grid as g  # noqa: E402
import make_inv_612 as case612  # noqa: E402
import make_ppp_psp_inv_case as m2  # noqa: E402
import plot_rugged_inv as pr  # noqa: E402
from check_joint import mask_crust, recs_for_draw, ttrms  # noqa: E402
from check_ps_fwd import draw_ps_rays  # noqa: E402
from make_lvz import BEL_LVZ  # noqa: E402

g.z_conv = case612.z_conv

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OFF = HERE / "inv_612"
ON = HERE / "inv_dd50"
XLO, XHI = 10.0, 140.0


def region_rms(xs, zs, a, b, *, lid: bool | None) -> float:
    s = n = 0
    for i, x in enumerate(xs):
        if not (XLO <= x <= XHI):
            continue
        zi = case612.z_conv(x)
        for k, z in enumerate(zs):
            if z < g.H - 1e-9:
                continue
            if lid is True and z >= zi - 1e-9:
                continue
            if lid is False and z < zi - 1e-9:
                continue
            d = a[k, i] - b[k, i]
            if np.isnan(d):
                continue
            s += float(d * d)
            n += 1
    return math.sqrt(s / n) if n else float("nan")


def _load_xz(path: Path) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        a = ln.split()
        if len(a) >= 2:
            out.append((float(a[0]), float(a[1])))
    return out


def _z_at(xz: list[tuple[float, float]], x: float) -> float:
    return float(np.interp(x, [p[0] for p in xz], [p[1] for p in xz]))


def _load_rays(ray_p: Path, syn_p: Path):
    if not ray_p.is_file() or not syn_p.is_file():
        return None, None
    return pr.parse_rays(ray_p), recs_for_draw(syn_p)


def _moho_rms(rec, true) -> float:
    s = n = 0
    for x, zt in true:
        if not (XLO <= x <= XHI):
            continue
        d = _z_at(rec, x) - zt
        s += d * d
        n += 1
    return math.sqrt(s / n) if n else float("nan")


def main() -> int:
    xs, zs, t_raw = m2.parse_smesh(OFF / "true_vs.smesh")
    _, _, s_raw = m2.parse_smesh(OFF / "start_vs.smesh")
    _, _, off_raw = m2.parse_smesh(OFF / "rec_vs.smesh")
    _, _, on_raw = m2.parse_smesh(ON / "rec_vs.smesh")
    t_vs = mask_crust(xs, zs, t_raw)
    s_vs = mask_crust(xs, zs, s_raw)
    r_off = mask_crust(xs, zs, off_raw)
    r_on = mask_crust(xs, zs, on_raw)
    true_m = _load_xz(OFF / "moho_true.refl")
    start_m = _load_xz(OFF / "moho.refl")
    off_m = _load_xz(OFF / "rec_moho.refl")
    on_m = _load_xz(ON / "rec_moho.refl")
    print("=== 612  -DD200 vs -DD50  (冻盖层) ===")
    print(
        f"  start   below {region_rms(xs, zs, s_vs, t_vs, lid=False):.4f}"
        f"  moho {_moho_rms(start_m, true_m):.4f} km  z(50)={_z_at(start_m, 50):.2f}"
    )
    print(
        f"  DD200   below {region_rms(xs, zs, r_off, t_vs, lid=False):.4f}"
        f"  moho {_moho_rms(off_m, true_m):.4f} km  z(50)={_z_at(off_m, 50):.2f}"
        f"  vs start {_moho_rms(off_m, start_m):.4f} km"
    )
    print(
        f"  DD50    below {region_rms(xs, zs, r_on, t_vs, lid=False):.4f}"
        f"  moho {_moho_rms(on_m, true_m):.4f} km  z(50)={_z_at(on_m, 50):.2f}"
        f"  vs start {_moho_rms(on_m, start_m):.4f} km"
    )
    print(f"  true z(50)={_z_at(true_m, 50):.2f}")
    syn_t = OFF / "syn_inv.dat"
    print(
        f"  tt DD200  6={ttrms(syn_t, OFF / 'syn_rec.dat', 6):.4f} s  "
        f"12={ttrms(syn_t, OFF / 'syn_rec.dat', 12):.4f} s"
    )
    print(
        f"  tt DD50   6={ttrms(syn_t, ON / 'syn_rec.dat', 6):.4f} s  "
        f"12={ttrms(syn_t, ON / 'syn_rec.dat', 12):.4f} s"
    )
    _, _, vt = m2.parse_smesh(ON / "true_vp.smesh")
    _, _, vr = m2.parse_smesh(ON / "rec_vp.smesh")
    mx = max(abs(a - b) for col_t, col_r in zip(vt, vr) for a, b in zip(col_t, col_r))
    print(f"  Vp DD50  max|d|={mx:.4f}")

    rays_off, recs_off = _load_rays(OFF / "rays_rec.dat", OFF / "syn_rec.dat")
    rays_on, recs_on = _load_rays(ON / "rays_rec.dat", ON / "syn_rec.dat")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), facecolor="w", layout="constrained")
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    xs_m = np.linspace(XLO, XHI, 200)
    last = None
    for i, (ax, arr, mo, rays, recs, title) in enumerate(
        (
            (axes[0], r_off - t_vs, off_m, rays_off, recs_off, "DD200  反−真"),
            (axes[1], r_on - t_vs, on_m, rays_on, recs_on, "DD50  反−真"),
        )
    ):
        im = ax.imshow(
            arr,
            extent=extent,
            cmap="RdBu_r",
            vmin=-0.80,
            vmax=0.80,
            aspect="auto",
            interpolation="nearest",
            zorder=0,
        )
        last = im
        if rays and recs:
            draw_ps_rays(
                ax,
                rays,
                recs,
                thin=True,
                mark_conv=False,
                z_conv=case612.z_conv,
                codes=(6, 12),
            )
        ax.plot(xs_m, [case612.z_conv(x) for x in xs_m], "k-.", lw=0.7, zorder=2)
        ax.plot(xs_m, [g.H] * len(xs_m), "k--", lw=0.5, zorder=2)
        ax.plot([p[0] for p in true_m], [p[1] for p in true_m], "k:", lw=1.2, label="真", zorder=5)
        ax.plot([p[0] for p in start_m], [p[1] for p in start_m], color="0.45", lw=1.0, label="初", zorder=5)
        ax.plot([p[0] for p in mo], [p[1] for p in mo], color="#c44e8a", lw=1.5, label="反", zorder=6)
        ax.plot(list(case612.OBS_XS), [g.H] * len(case612.OBS_XS), "k^", ms=6, zorder=7)
        ax.plot(BEL_LVZ["x0"], BEL_LVZ["z0"], "kx", ms=7, mew=1.3, zorder=8)
        ax.set_xlim(XLO, XHI)
        ax.set_ylim(16, 0)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x (km)")
        if i == 0:
            ax.legend(loc="lower right", fontsize=7, framealpha=0.9, ncol=2)
        elif i == 1:
            ax.legend(loc="lower right", fontsize=6, framealpha=0.88, ncol=2)
    axes[0].set_ylabel("深度 (km)")
    fig.colorbar(last, ax=axes.ravel().tolist(), shrink=0.82, label="ΔVs (km/s)")
    fig.suptitle("612  -DD200 vs -DD50   蓝P/粉S  点线=真莫霍  灰=初  粉=反", fontsize=12)
    out = ON / "check_inv_dd50.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
