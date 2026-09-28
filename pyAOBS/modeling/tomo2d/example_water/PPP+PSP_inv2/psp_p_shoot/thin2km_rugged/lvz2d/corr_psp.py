#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PPP 收回 Vp 上，用 PPP/PPS 正演路径的台侧盖层走时差，把观测 PSS 校成 PSP。

不是数据域 PSP = PSS - (PPS - PPP)。
观测走时来自真模型；路径和台侧 ΔT 来自 rec_vp + rec_vp/κ 正演。
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
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inv_2d"))
sys.path.insert(0, str(ROOT.parents[1]))
sys.path.insert(0, str(ROOT.parents[1].parent / "water_inv"))
sys.path.insert(0, str(ROOT.parents[1].parent / "ps_fwd"))
import inv_grid as g  # noqa: E402
import make_ppp_psp_inv_case as m2  # noqa: E402
from check_ps_fwd import find_psx_conversion_points  # noqa: E402
from check_water_inv import parse_picks  # noqa: E402
from plot_rugged_inv import parse_rays  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _rms(a) -> float:
    a = [float(x) for x in a if math.isfinite(x)]
    return math.sqrt(sum(v * v for v in a) / len(a)) if a else float("nan")


class VelField:
    def __init__(self, path: Path):
        self.xs, self.zs, vel = m2.parse_smesh(path)
        self.v = np.asarray(vel, float)
        self.xa = np.asarray(self.xs, float)
        self.za = np.asarray(self.zs, float)

    def __call__(self, x: float, z: float) -> float:
        xs, zs, v = self.xa, self.za, self.v
        if x <= xs[0]:
            i0 = 0
        elif x >= xs[-1]:
            i0 = len(xs) - 2
        else:
            i0 = int(np.searchsorted(xs, x) - 1)
            i0 = max(0, min(i0, len(xs) - 2))
        if z <= zs[0]:
            k0 = 0
        elif z >= zs[-1]:
            k0 = len(zs) - 2
        else:
            k0 = int(np.searchsorted(zs, z) - 1)
            k0 = max(0, min(k0, len(zs) - 2))
        x0, x1 = xs[i0], xs[i0 + 1]
        z0, z1 = zs[k0], zs[k0 + 1]
        tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        tz = 0.0 if z1 == z0 else (z - z0) / (z1 - z0)
        v00, v10 = v[i0, k0], v[i0 + 1, k0]
        v01, v11 = v[i0, k0 + 1], v[i0 + 1, k0 + 1]
        return float((1 - tx) * (1 - tz) * v00 + tx * (1 - tz) * v10 + (1 - tx) * tz * v01 + tx * tz * v11)


def _pick_map(picks):
    out = {}
    for p in picks:
        if not math.isfinite(p[3]):
            continue
        out[(int(p[0]), round(p[4], 3), round(p[1], 3))] = p
    return out


def _ray_map(picks, rays):
    out = {}
    n = min(len(picks), len(rays))
    for i in range(n):
        p = picks[i]
        out[(int(p[0]), round(p[4], 3), round(p[1], 3))] = rays[i]
    return out


def _upgoing_xy(xs, zs):
    pts = find_psx_conversion_points(xs, zs, z_conv=g.z_conv, eps=8e-2)
    if pts is None:
        return None
    xc, zc = pts[1]
    best_i, best_d = None, 1e9
    half = max(1, len(xs) // 3)
    for i in range(half, len(xs)):
        d = (xs[i] - xc) ** 2 + (zs[i] - zc) ** 2
        if d < best_d:
            best_d, best_i = d, i
    if best_i is None or best_i >= len(xs) - 2:
        return None
    ox = [xc] + list(xs[best_i + 1 :])
    oz = [zc] + list(zs[best_i + 1 :])
    return ox, oz


def _leg_time(xs, zs, vel_at) -> float:
    t = 0.0
    for i in range(len(xs) - 1):
        ds = math.hypot(xs[i + 1] - xs[i], zs[i + 1] - zs[i])
        if ds < 1e-8:
            continue
        v0, v1 = vel_at(xs[i], zs[i]), vel_at(xs[i + 1], zs[i + 1])
        if v0 < 0.2 or v1 < 0.2:
            continue
        t += ds * 0.5 * (1.0 / v0 + 1.0 / v1)
    return t


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
    n_skip = 0
    keys0 = [k[1:] for k in mo if k[0] == 0]
    for src, sx in sorted(set(keys0)):
        k0, k6, k7, k8 = ((c, src, sx) for c in (0, 6, 7, 8))
        if not all(k in mo and k in mf for k in (k0, k6, k7, k8)):
            n_skip += 1
            continue
        t_ppp_o, t_pps_o, t_pss_o, t_psp_o = (mo[k][3] for k in (k0, k7, k8, k6))
        t_ppp_f, t_psp_f, t_pps_f, t_pss_f = (mf[k][3] for k in (k0, k6, k7, k8))
        dt_obs = t_pps_o - t_ppp_o
        dt_fwd = t_pps_f - t_ppp_f
        naive = t_pss_o - dt_obs
        naive_fwd = t_pss_f - (t_pps_f - t_ppp_f)
        # 观测朴素组合 + 正演路径上 (PSP - 朴素). 正演走时就是路径真实走时.
        path_adj = naive + (t_psp_f - naive_fwd)
        dt_up = float("nan")
        if k0 in mr and k7 in mr:
            ppp_xy = _upgoing_xy(*mr[k0])
            pps_xy = _upgoing_xy(*mr[k7])
            if ppp_xy is not None and pps_xy is not None:
                dt_up = _leg_time(*pps_xy, vs) - _leg_time(*ppp_xy, vp)
        path_only = t_pss_o - dt_up if math.isfinite(dt_up) else float("nan")
        dx = abs(sx - src)
        rows.append(
            dict(
                src=src,
                sx=sx,
                dx=dx,
                psp=t_psp_o,
                naive=naive,
                path_only=path_only,
                path_adj=path_adj,
                psp_fwd=t_psp_f,
                dt_up=dt_up,
                dt_obs=dt_obs,
                dt_fwd=dt_fwd,
                naive_fwd=naive_fwd,
            )
        )

    if not rows:
        raise SystemExit("no matched PPP/PPS/PSS/PSP rows")

    def col(name):
        return [r[name] for r in rows]

    d_naive = [r["naive"] - r["psp"] for r in rows]
    d_adj = [r["path_adj"] - r["psp"] for r in rows]
    d_fwd = [r["psp_fwd"] - r["psp"] for r in rows]
    d_id = [r["naive_fwd"] - r["psp_fwd"] for r in rows]
    d_up = [r["path_only"] - r["psp"] for r in rows if math.isfinite(r["path_only"])]
    print(f"n={len(rows)}  skip={n_skip}")
    print(
        f"  naive  PSS-(PPS-PPP) obs          RMS vs true PSP  {_rms(d_naive):.4f} s"
        f"  mean {sum(d_naive)/len(d_naive):+.4f} s"
    )
    print(
        f"  corr   naive_obs + (PSP_fwd-naive_fwd)  RMS  {_rms(d_adj):.4f} s"
        f"  mean {sum(d_adj)/len(d_adj):+.4f} s"
    )
    print(
        f"  fwd    PSP on PPP Vp /k           RMS vs true PSP  {_rms(d_fwd):.4f} s"
        f"  mean {sum(d_fwd)/len(d_fwd):+.4f} s"
    )
    print(
        f"  ident  naive_fwd - PSP_fwd (path mismatch on model)  RMS {_rms(d_id):.4f} s"
        f"  mean {sum(d_id)/len(d_id):+.4f} s"
    )
    if d_up:
        print(
            f"  lid    PSS - path(tS_up-tP_up)    RMS vs true PSP  {_rms(d_up):.4f} s"
        )

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), facecolor="w", layout="constrained")
    ax, axr, axh, axp = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    dx = col("dx")
    ax.plot(dx, col("psp"), "o", color="#2ca02c", ms=4, label="true PSP")
    ax.plot(dx, col("naive"), "s", color="0.55", ms=3.5, alpha=0.8, label="naive PSS-(PPS-PPP)")
    ax.plot(dx, col("path_adj"), "D", color="#c44e8a", ms=4, label="path-corrected")
    ax.plot(dx, col("psp_fwd"), "^", color="#1f77b4", ms=3.5, alpha=0.7, label="PSP fwd on PPP Vp")
    ax.set_xlabel("offset dx (km)")
    ax.set_ylabel("t (s)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title("true PSP vs corrected PSP")

    axr.axhline(0.0, color="0.45", lw=0.8)
    axr.plot(dx, d_naive, "s", color="0.55", ms=3.5, alpha=0.8, label=f"naive {_rms(d_naive):.3f} s")
    axr.plot(dx, d_adj, "D", color="#c44e8a", ms=4, label=f"path-corr {_rms(d_adj):.3f} s")
    axr.set_xlabel("offset dx (km)")
    axr.set_ylabel("t - true PSP (s)")
    axr.grid(True, alpha=0.3)
    axr.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axr.set_title("residual vs true PSP")

    bins = np.linspace(-1.2, 1.2, 25)
    axh.hist(d_naive, bins=bins, color="0.65", alpha=0.7, label="naive")
    axh.hist(d_adj, bins=bins, color="#c44e8a", alpha=0.55, label="path-corr")
    axh.axvline(0.0, color="0.2", lw=0.8)
    axh.set_xlabel("t - true PSP (s)")
    axh.set_ylabel("count")
    axh.legend(fontsize=8)
    axh.grid(True, alpha=0.3)
    axh.set_title("residual histogram")

    axp.plot(col("psp"), col("path_adj"), "D", color="#c44e8a", ms=4, label="path-corr")
    axp.plot(col("psp"), col("naive"), "s", color="0.55", ms=3.5, alpha=0.7, label="naive")
    lims = [min(col("psp")) - 0.2, max(col("psp")) + 0.2]
    axp.plot(lims, lims, "k--", lw=0.8)
    axp.set_xlabel("true PSP (s)")
    axp.set_ylabel("estimated PSP (s)")
    axp.grid(True, alpha=0.3)
    axp.legend(fontsize=8)
    axp.set_title("1:1  true vs estimated")
    axp.set_aspect("equal", adjustable="box")

    fig.suptitle("PPP Vp model  PSS -> PSP via PPP/PPS paths   not naive PSS-(PPS-PPP)")
    out = HERE / "check_corr_psp.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
