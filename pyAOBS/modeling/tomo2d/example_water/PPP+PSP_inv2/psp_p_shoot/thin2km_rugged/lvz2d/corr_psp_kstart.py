#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""错误初值 kappa 下，用 PPS-PPP 拟合台侧盖层 Vs，看能收回多少 k。"""

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
from corr_psp import VelField, _leg_time, _pick_map, _ray_map, _rms  # noqa: E402
from corr_psp_lidvs import recv_lid_xy  # noqa: E402
from check_water_inv import parse_picks  # noqa: E402
from plot_rugged_inv import parse_rays  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

K_TRUE = 1.73
K_TRY = (1.50, 1.73, 2.20)


def _paths(k_start: float):
    if abs(k_start - 1.73) < 1e-9:
        return HERE / "syn_pppvp.dat", HERE / "rays_pppvp.dat", HERE / "vs_ppp.smesh"
    tag = f"{k_start:.2f}".replace(".", "")
    return HERE / f"syn_k{tag}.dat", HERE / f"rays_k{tag}.dat", HERE / f"vs_k{tag}.smesh"


def fit_k(k_start: float):
    fwd_p, ray_p, vs_p = _paths(k_start)
    obs_p = HERE / "syn_true_all.dat"
    for p in (fwd_p, ray_p, vs_p, obs_p):
        if not p.is_file():
            raise SystemExit(f"missing {p}")
    vs = VelField(vs_p)
    mo = _pick_map(parse_picks(obs_p.read_text(encoding="utf-8")))
    fwd = parse_picks(fwd_p.read_text(encoding="utf-8"))
    mf = _pick_map(fwd)
    mr = _ray_map(fwd, parse_rays(ray_p))
    rows = []
    n_fail = 0
    for src, sx in sorted(set(k[1:] for k in mo if k[0] == 0)):
        kk = {c: (c, src, sx) for c in (0, 6, 7, 8)}
        if not all(kk[c] in mo and kk[c] in mf for c in (0, 6, 7, 8)):
            n_fail += 1
            continue
        if kk[7] not in mr:
            n_fail += 1
            continue
        pps_lid = recv_lid_xy(*mr[kk[7]])
        t_ppp_o, t_pps_o, t_pss_o, t_psp_o = (
            mo[kk[0]][3],
            mo[kk[7]][3],
            mo[kk[8]][3],
            mo[kk[6]][3],
        )
        dt_obs = t_pps_o - t_ppp_o
        dt_fwd = mf[kk[7]][3] - mf[kk[0]][3]
        alpha = float("nan")
        dt_fit = dt_obs
        if pps_lid is None:
            n_fail += 1
        else:
            t_up_s = _leg_time(*pps_lid, vs)
            den = t_up_s + dt_obs - dt_fwd
            if t_up_s >= 0.04 and den >= 0.04:
                alpha = t_up_s / den
                dt_fit = dt_fwd - t_up_s + t_up_s / alpha
            else:
                n_fail += 1
        rows.append(
            dict(
                dx=abs(sx - src),
                alpha=alpha,
                k_fit=k_start / alpha if math.isfinite(alpha) else float("nan"),
                dt_obs=dt_obs,
                dt_fwd=dt_fwd,
                dt_fit=dt_fit,
                psp=t_psp_o,
                corr=t_pss_o - dt_fit,
            )
        )
    return rows, n_fail


def _stat(a):
    a = np.asarray(a, float)
    return dict(
        n=len(a),
        median=float(np.median(a)),
        mean=float(a.mean()),
        p10=float(np.percentile(a, 10)),
        p90=float(np.percentile(a, 90)),
        lo=float(a.min()),
        hi=float(a.max()),
        rms=float(np.sqrt(np.mean((a - K_TRUE) ** 2))),
    )


def main() -> int:
    colors = {1.50: "#1f77b4", 1.73: "0.35", 2.20: "#c44e8a"}
    marks = {1.50: "o", 1.73: "s", 2.20: "D"}
    fig_k, axk = plt.subplots(figsize=(8.4, 5.0), facecolor="w", layout="constrained")
    axk.axhline(K_TRUE, color="0.3", ls="--", lw=1.0, label=f"true k={K_TRUE}")
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4), facecolor="w", layout="constrained")
    ax, axr, axp = axes
    ax.plot([], [], "o", color="#2ca02c", ms=4.5, label="true PSP")
    first_true = True
    all_rows = {}
    for k0 in K_TRY:
        rows, n_fail = fit_k(k0)
        all_rows[k0] = rows
        kf = [r["k_fit"] for r in rows if math.isfinite(r["k_fit"])]
        st = _stat(kf)
        d = [r["corr"] - r["psp"] for r in rows]
        print(
            f"k_start={k0:.2f}  n={len(rows)}  k_fail={n_fail}  "
            f"k_fit median {st['median']:.3f}  RMS vs 1.73 {st['rms']:.3f}"
        )
        print(
            f"  corr PSP vs true  RMS {_rms(d):.4f} s  mean {sum(d)/len(d):+.4f} s"
        )
        dx = [r["dx"] for r in rows]
        axk.plot(
            [r["dx"] for r in rows if math.isfinite(r["k_fit"])],
            kf,
            marks[k0],
            ms=4,
            color=colors[k0],
            label=f"start {k0:.2f} -> {st['median']:.3f}",
        )
        if first_true:
            ax.plot(dx, [r["psp"] for r in rows], "o", color="#2ca02c", ms=4.5, label="true PSP")
            first_true = False
        ax.plot(
            dx,
            [r["corr"] for r in rows],
            marks[k0],
            ms=3.8,
            color=colors[k0],
            alpha=0.8,
            label=f"corr k0={k0:.2f}",
        )
        axr.plot(
            dx,
            d,
            marks[k0],
            ms=3.8,
            color=colors[k0],
            alpha=0.8,
            label=f"k0={k0:.2f}  {_rms(d):.3f} s",
        )
        axp.plot(
            [r["psp"] for r in rows],
            [r["corr"] for r in rows],
            marks[k0],
            ms=3.8,
            color=colors[k0],
            alpha=0.8,
            label=f"k0={k0:.2f}",
        )
    # pairwise corr identity
    r0 = all_rows[K_TRY[0]]
    for k0 in K_TRY[1:]:
        r1 = all_rows[k0]
        dd = [a["corr"] - b["corr"] for a, b in zip(r0, r1)]
        print(f"  corr[{K_TRY[0]:.2f}] - corr[{k0:.2f}]  RMS {_rms(dd):.4e} s")

    axk.set_xlabel("offset dx (km)")
    axk.set_ylabel("fitted lid k = Vp_PPP / Vs")
    axk.set_ylim(1.2, 2.5)
    axk.grid(True, alpha=0.3)
    axk.legend(loc="best", fontsize=8, framealpha=0.9)
    axk.set_title("PPS-PPP  fit recv-lid Vs   recover k")
    fig_k.savefig(HERE / "check_corr_kstart.png", dpi=140)
    plt.close(fig_k)

    ax.set_xlabel("offset dx (km)")
    ax.set_ylabel("t (s)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
    ax.set_title("true PSP vs corrected PSP")
    axr.axhline(0.0, color="0.45", lw=0.8)
    axr.set_xlabel("offset dx (km)")
    axr.set_ylabel("corr - true PSP (s)")
    axr.grid(True, alpha=0.3)
    axr.legend(loc="upper right", fontsize=7, framealpha=0.9)
    axr.set_title("residual vs true PSP")
    lims = [
        min(r["psp"] for r in r0) - 0.2,
        max(r["psp"] for r in r0) + 0.2,
    ]
    axp.plot(lims, lims, "k--", lw=0.8)
    axp.set_xlabel("true PSP (s)")
    axp.set_ylabel("corrected PSP (s)")
    axp.grid(True, alpha=0.3)
    axp.legend(fontsize=7)
    axp.set_title("1:1")
    axp.set_aspect("equal", adjustable="box")
    fig.suptitle("corrected PSP vs true PSP   different starting k")
    out = HERE / "check_corr_kstart_psp.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {HERE / 'check_corr_kstart.png'}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
