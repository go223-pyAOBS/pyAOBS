#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""10/11 盖层 SS、14/15 水柱 S→P 正演验收。"""

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
sys.path.insert(0, str(ROOT.parents[1].parent / "ps_fwd"))
sys.path.insert(0, str(ROOT.parents[1].parent / "water_inv"))
import inv_grid as g  # noqa: E402
import plot_rugged_inv as pr  # noqa: E402
import make_ppp_psp_inv_case as m2  # noqa: E402
from check_joint import recs_for_draw  # noqa: E402
from check_ps_fwd import draw_ps_rays  # noqa: E402
from check_water_inv import parse_picks  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

WORK = HERE / "fwd_91011"
DT_WATER = 2.0 * g.H / 1.5


def _pair_key(p) -> tuple[float, float]:
    return round(float(p[4]), 3), round(float(p[1]), 3)


def _lid_ss_twt(x_obs: float) -> float:
    xs, zs, vs = m2.parse_smesh(WORK / "true_vs.smesh")
    zc = g.z_conv(x_obs)
    ix = int(np.argmin(np.abs(np.asarray(xs) - x_obs)))
    twt = 0.0
    for j in range(len(zs) - 1):
        z0, z1 = zs[j], zs[j + 1]
        if z1 <= g.H + 1e-9 or z0 >= zc - 1e-9:
            continue
        za = max(z0, g.H)
        zb = min(z1, zc)
        if zb <= za:
            continue
        v = vs[ix][j] if vs[ix][j] > 0.2 else vs[ix][j + 1]
        if v > 3.5:
            continue
        twt += 2.0 * (zb - za) / v
    return twt


def _obs_side_goes_water(rays, recs, code: int, obs_x: float) -> bool:
    for (c, _dx, _t), (xs, zs) in zip(recs, rays):
        if int(c) != code or not xs:
            continue
        if abs(float(xs[-1]) - obs_x) > 0.6:
            continue
        for x, z in zip(xs, zs):
            if abs(x - obs_x) < 2.5 and z < g.H - 0.25:
                return True
        return False
    return False


def main() -> int:
    syn = WORK / "syn_true.dat"
    rays_p = WORK / "rays_true.dat"
    if not syn.is_file() or not rays_p.is_file():
        raise SystemExit(f"missing {syn} or {rays_p}")
    picks = parse_picks(syn.read_text(encoding="utf-8"))
    rays = pr.parse_rays(rays_p)
    recs = recs_for_draw(syn)
    by: dict[tuple[int, tuple[float, float]], float] = {}
    for p in picks:
        by[(int(p[0]), _pair_key(p))] = float(p[3])

    print(f"2H/vw = {DT_WATER:.3f} s")
    print(
        "src   rcv    t7      t10     dSS     twt     t14     dW      "
        "t8      t11     dSS     t15     dW"
    )
    keys = sorted({k for (_c, k) in by})
    n_ok = n_ss = n_w = 0
    dss, dw = [], []
    for key in keys:
        t7, t10, t14 = by.get((7, key)), by.get((10, key)), by.get((14, key))
        t8, t11, t15 = by.get((8, key)), by.get((11, key)), by.get((15, key))
        t6, t9 = by.get((6, key)), by.get((9, key))
        if None in (t7, t10, t14, t8, t11, t15, t6, t9):
            continue
        n_ok += 1
        twt = _lid_ss_twt(key[0])
        d10, d11 = t10 - t7, t11 - t8
        d14, d15, d9 = t14 - t7, t15 - t8, t9 - t6
        dss.extend((d10, d11))
        dw.extend((d14, d15, d9))
        ss_ok = d10 > 0.6 and d11 > 0.6
        w_ok = all(abs(d - DT_WATER) < 0.45 for d in (d9, d14, d15)) and min(d9, d14, d15) > 1.5
        if ss_ok:
            n_ss += 1
        if w_ok:
            n_w += 1
        print(
            f"{key[0]:5.1f} {key[1]:5.1f}  {t7:7.3f} {t10:7.3f} {d10:7.3f} {twt:7.3f}  "
            f"{t14:7.3f} {d14:7.3f}  {t8:7.3f} {t11:7.3f} {d11:7.3f}  {t15:7.3f} {d15:7.3f}"
        )

    water_at_obs = []
    for ox in (30.0, 50.0):
        w10 = _obs_side_goes_water(rays, recs, 10, ox)
        w14 = _obs_side_goes_water(rays, recs, 14, ox)
        water_at_obs.append((ox, w10, w14))
        print(f"OBS {ox:.0f}: type10 water-col near OBS={w10}  type14={w14}")

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.6), sharex=True, sharey=True)
    xs_l = np.linspace(18.0, 82.0, 80)
    panels = (
        (axes[0, 0], (6, 9), "PSP 6 vs 水柱 9"),
        (axes[0, 1], (7, 10), "PPS 7 vs 盖层SS 10"),
        (axes[0, 2], (8, 11), "PSS 8 vs 盖层SS 11"),
        (axes[1, 0], (7, 14), "PPS 7 vs 水柱 14"),
        (axes[1, 1], (8, 15), "PSS 8 vs 水柱 15"),
        (axes[1, 2], (10, 14), "SS 10 vs 水柱 14"),
    )
    for ax, codes, title in panels:
        draw_ps_rays(ax, rays, recs, thin=True, mark_conv=True, z_conv=g.z_conv, codes=codes)
        ax.plot(xs_l, [g.z_conv(x) for x in xs_l], "k-.", lw=0.8)
        ax.plot(xs_l, [g.H] * len(xs_l), "k--", lw=0.6)
        ax.plot(list((30.0, 50.0)), [g.H, g.H], "k^", ms=6)
        ax.set_xlim(18, 82)
        ax.set_ylim(16, 0)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x (km)")
    axes[0, 0].set_ylabel("z (km)")
    axes[1, 0].set_ylabel("z (km)")
    fig.tight_layout()
    out = WORK / "check_fwd_91011.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(
        f"pairs={n_ok}  ss_ok={n_ss}  water_ok={n_w}  "
        f"mean_dSS={np.mean(dss) if dss else np.nan:.3f}  "
        f"mean_dW={np.mean(dw) if dw else np.nan:.3f}"
    )
    print(f"wrote {out}")
    no_ss_water = all(not w10 for _ox, w10, _w14 in water_at_obs)
    has_w14 = all(w14 for _ox, _w10, w14 in water_at_obs)
    if n_ok == 0 or n_ss < n_ok or n_w < n_ok or not no_ss_water or not has_w14:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
