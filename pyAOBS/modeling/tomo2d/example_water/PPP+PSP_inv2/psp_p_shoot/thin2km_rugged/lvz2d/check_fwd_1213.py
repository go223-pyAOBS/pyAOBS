#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""12/13 正演验收：射线打莫霍折回，且晚于同路径 6/8。"""

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
sys.path.insert(0, str(ROOT.parents[1].parent / "ps_fwd"))
sys.path.insert(0, str(ROOT.parents[1].parent / "water_inv"))
import inv_grid as g  # noqa: E402
import plot_rugged_inv as pr  # noqa: E402
from check_joint import recs_for_draw  # noqa: E402
from check_ps_fwd import draw_ps_rays  # noqa: E402
from check_water_inv import parse_picks  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

WORK = HERE / "fwd_1213"


def _load_xz(path: Path) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        a = ln.split()
        if len(a) >= 2:
            out.append((float(a[0]), float(a[1])))
    return out


def _z_at(xz: list[tuple[float, float]], x: float) -> float:
    xs = [p[0] for p in xz]
    zs = [p[1] for p in xz]
    return float(np.interp(x, xs, zs))


def _pair_key(p) -> tuple[float, float]:
    # pick: code, rcv_x, rcv_z, t, src_x, ...
    return round(float(p[4]), 3), round(float(p[1]), 3)


def main() -> int:
    syn = WORK / "syn_true.dat"
    rays_p = WORK / "rays_true.dat"
    moho_p = WORK / "moho_true.refl"
    if not syn.is_file() or not rays_p.is_file():
        raise SystemExit(f"missing {syn} or {rays_p}")
    picks = parse_picks(syn.read_text(encoding="utf-8"))
    rays = pr.parse_rays(rays_p)
    recs = recs_for_draw(syn)
    moho = _load_xz(moho_p)
    by: dict[tuple[int, tuple[float, float]], tuple[float, list[float], list[float]]] = {}
    for p, (xs, zs) in zip(picks, rays):
        code = int(p[0])
        by[(code, _pair_key(p))] = (float(p[3]), xs, zs)

    n_ok = n_late = n_bounce = 0
    print("src   rcv    t6      t12     dt12-6   t8      t13     dt13-8  zmax12  zM")
    keys = sorted({k for (_c, k) in by})
    for key in keys:
        t6 = by.get((6, key))
        t12 = by.get((12, key))
        t8 = by.get((8, key))
        t13 = by.get((13, key))
        if not (t6 and t12 and t8 and t13):
            continue
        n_ok += 1
        zmax12 = max(t12[2])
        zmax13 = max(t13[2])
        xb12 = t12[1][int(np.argmax(t12[2]))]
        xb13 = t13[1][int(np.argmax(t13[2]))]
        zm12 = _z_at(moho, xb12)
        zm13 = _z_at(moho, xb13)
        late12 = t12[0] + 1e-3 >= t6[0]
        late13 = t13[0] + 1e-3 >= t8[0]
        hit = abs(zmax12 - zm12) < 0.45 and abs(zmax13 - zm13) < 0.45
        if late12 and late13:
            n_late += 1
        if hit:
            n_bounce += 1
        print(
            f"{key[0]:5.1f} {key[1]:5.1f}  {t6[0]:7.3f} {t12[0]:7.3f} {t12[0]-t6[0]:7.3f}  "
            f"{t8[0]:7.3f} {t13[0]:7.3f} {t13[0]-t8[0]:7.3f}  {zmax12:6.2f} {zm12:5.2f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharex=True, sharey=True)
    xs_l = np.linspace(18.0, 82.0, 80)
    for ax, codes, title in (
        (axes[0], (6, 12), "PSP 6 转折 vs 12 莫霍反射"),
        (axes[1], (8, 13), "PSS 8 转折 vs 13 莫霍反射"),
    ):
        draw_ps_rays(ax, rays, recs, thin=True, mark_conv=True, z_conv=g.z_conv, codes=codes)
        ax.plot(xs_l, [g.z_conv(x) for x in xs_l], "k-.", lw=0.8)
        ax.plot([p[0] for p in moho], [p[1] for p in moho], "k:", lw=1.0)
        ax.plot(xs_l, [g.H] * len(xs_l), "k--", lw=0.6)
        ax.plot(list((30.0, 50.0)), [g.H, g.H], "k^", ms=6)
        ax.set_xlim(18, 82)
        ax.set_ylim(16, 0)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x (km)")
    axes[0].set_ylabel("z (km)")
    fig.tight_layout()
    out = WORK / "check_fwd_1213.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"pairs={n_ok}  later_than_turn={n_late}  bounce_near_moho={n_bounce}")
    print(f"wrote {out}")
    if n_ok == 0 or n_bounce < n_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
