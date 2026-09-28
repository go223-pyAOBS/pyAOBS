#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""每个炮-台对：在 PPP 收回 Vp 的 PPP/PPS 正演路径上，只缩放台侧盖层 Vs，
使正演 PPS-PPP 等于观测 PPS-PPP，再用该时差减真实 PSS 得到 PSP。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "inv_2d"))
sys.path.insert(0, str(HERE.parent))
from corr_psp import (  # noqa: E402
    VelField,
    _leg_time,
    _pick_map,
    _ray_map,
    _rms,
)
import inv_grid as g  # noqa: E402
from check_water_inv import parse_picks  # noqa: E402
from plot_rugged_inv import parse_rays  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def recv_lid_xy(xs, zs):
    """转折最深点之后、海底以上的台侧盖层段。"""
    if len(xs) < 4:
        return None
    i_deep = int(np.argmax(zs))
    ox, oz = [], []
    started = False
    for i in range(i_deep, len(xs)):
        zc = g.z_conv(xs[i])
        near_conv = abs(zs[i] - zc) <= 8e-2
        in_lid = zs[i] > g.H + 1e-3 and zs[i] <= zc + 5e-2
        if not started:
            if near_conv or in_lid:
                started = True
                ox.append(xs[i])
                oz.append(zs[i])
            continue
        ox.append(xs[i])
        oz.append(zs[i])
        if zs[i] <= g.H + 2e-2:
            break
    if len(ox) < 2:
        return None
    return ox, oz


def main() -> int:
    obs_p = HERE / "syn_true_all.dat"
    fwd_p = HERE / "syn_pppvp.dat"
    ray_p = HERE / "rays_pppvp.dat"
    vp_p = HERE / "rec_vp.smesh"
    vs_p = HERE / "vs_ppp.smesh"
    for pth in (obs_p, fwd_p, ray_p, vp_p, vs_p):
        if not pth.is_file():
            raise SystemExit(f"missing {pth} -- run run_corr_psp.sh first")

    vp, vs = VelField(vp_p), VelField(vs_p)
    obs = parse_picks(obs_p.read_text(encoding="utf-8"))
    fwd = parse_picks(fwd_p.read_text(encoding="utf-8"))
    rays = parse_rays(ray_p)
    mo, mf, mr = _pick_map(obs), _pick_map(fwd), _ray_map(fwd, rays)

    rows = []
    n_skip = n_bad_a = 0
    keys0 = [k[1:] for k in mo if k[0] == 0]
    for src, sx in sorted(set(keys0)):
        k0, k6, k7, k8 = ((c, src, sx) for c in (0, 6, 7, 8))
        if not all(k in mo and k in mf and k in mr for k in (k0, k6, k7, k8)):
            n_skip += 1
            continue
        t_ppp_o, t_pps_o, t_pss_o, t_psp_o = (mo[k][3] for k in (k0, k7, k8, k6))
        t_ppp_f, t_pps_f = mf[k0][3], mf[k7][3]
        dt_obs = t_pps_o - t_ppp_o
        dt_fwd = t_pps_f - t_ppp_f
        pps_lid = recv_lid_xy(*mr[k7]) if k7 in mr else None
        ppp_lid = recv_lid_xy(*mr[k0]) if k0 in mr else None
        alpha = float("nan")
        dt_fit = dt_obs
        t_up_s = t_up_p = t_up_s_fit = float("nan")
        if pps_lid is not None and ppp_lid is not None:
            t_up_s = _leg_time(*pps_lid, vs)
            t_up_p = _leg_time(*ppp_lid, vp)
            den = t_up_s + dt_obs - dt_fwd
            if t_up_s >= 0.04 and den >= 0.04:
                alpha = t_up_s / den
                t_up_s_fit = t_up_s / alpha
                dt_fit = dt_fwd - t_up_s + t_up_s_fit
            else:
                n_bad_a += 1
        else:
            n_skip += 1
        psp_corr = t_pss_o - dt_fit
        rows.append(
            dict(
                src=src,
                sx=sx,
                dx=abs(sx - src),
                psp=t_psp_o,
                corr=psp_corr,
                naive=t_pss_o - dt_obs,
                alpha=alpha,
                dt_obs=dt_obs,
                dt_fwd=dt_fwd,
                dt_fit=dt_fit,
                t_up_s=t_up_s,
                t_up_p=t_up_p,
                t_up_s_fit=t_up_s_fit,
            )
        )

    if not rows:
        raise SystemExit("no pairs")

    def col(name):
        return [r[name] for r in rows]

    d_c = [r["corr"] - r["psp"] for r in rows]
    d_n = [r["naive"] - r["psp"] for r in rows]
    a = [r["alpha"] for r in rows if math.isfinite(r["alpha"])]
    print(f"n={len(rows)}  no_lid_path={n_skip}  bad_alpha_den={n_bad_a}  fitted={len(a)}")
    print(
        f"  dt_fit vs dt_obs RMS  {_rms([r['dt_fit']-r['dt_obs'] for r in rows]):.4e} s"
    )
    print(
        f"  PSP = PSS - dt_fit     RMS vs true PSP  {_rms(d_c):.4f} s"
        f"  mean {sum(d_c)/len(d_c):+.4f} s"
    )
    print(
        f"  naive PSS-(PPS-PPP)    RMS vs true PSP  {_rms(d_n):.4f} s"
        f"  mean {sum(d_n)/len(d_n):+.4f} s"
    )
    print(
        f"  lid Vs scale alpha     median {float(np.median(a)):.3f}  "
        f"p10 {float(np.percentile(a,10)):.3f}  p90 {float(np.percentile(a,90)):.3f}"
    )
    n_phys = sum(1 for x in a if 0.4 <= x <= 2.5)
    print(f"  alpha in [0.4, 2.5]: {n_phys}/{len(a)}")
    dx_a = [r["dx"] for r in rows if math.isfinite(r["alpha"])]
    a_plot = a

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), facecolor="w", layout="constrained")
    ax, axr, axa, axp = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    dx = col("dx")
    ax.plot(dx, col("psp"), "o", color="#2ca02c", ms=4.5, label="true PSP")
    ax.plot(dx, col("corr"), "D", color="#c44e8a", ms=4, label="PSS - dt_fit")
    ax.set_xlabel("offset dx (km)")
    ax.set_ylabel("t (s)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title("true PSP vs lid-Vs-matched PPS-PPP correction")

    axr.axhline(0.0, color="0.45", lw=0.8)
    axr.plot(dx, d_c, "D", color="#c44e8a", ms=4, label=f"corr {_rms(d_c):.3f} s")
    axr.set_xlabel("offset dx (km)")
    axr.set_ylabel("t - true PSP (s)")
    axr.grid(True, alpha=0.3)
    axr.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axr.set_title("residual vs true PSP")

    axa.axhline(1.0, color="0.45", lw=0.8)
    axa.plot(dx_a, a_plot, "o", color="#1f77b4", ms=4)
    axa.set_xlabel("offset dx (km)")
    axa.set_ylabel("alpha  (Vs_lid -> alpha*Vs)")
    axa.set_ylim(0.0, 3.5)
    axa.grid(True, alpha=0.3)
    axa.set_title("per-pair receiver-lid Vs scale")

    axp.plot(col("psp"), col("corr"), "D", color="#c44e8a", ms=4)
    lims = [min(col("psp")) - 0.2, max(col("psp")) + 0.2]
    axp.plot(lims, lims, "k--", lw=0.8)
    axp.set_xlabel("true PSP (s)")
    axp.set_ylabel("corrected PSP (s)")
    axp.grid(True, alpha=0.3)
    axp.set_title("1:1  true vs corrected")
    axp.set_aspect("equal", adjustable="box")

    fig.suptitle("PPP Vp paths  scale recv-lid Vs to match obs PPS-PPP  then PSP = PSS - dt")
    out = HERE / "check_corr_psp_lidvs.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
