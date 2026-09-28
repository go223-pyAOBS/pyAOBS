#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检验「面下若干公里 Vs 低值」是否落在图论射线转折深度。

真模型是光滑梯度、没有预设低速层。若 rec−start 的负异常深度
与 (zmax − zc) 对齐，就是转折核 smearing，不是地质构造。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "inv_2d"))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1].parent / "ps_inv"))
sys.path.insert(0, str(HERE.parents[1].parent / "water_inv"))
import inv_grid as g  # noqa: E402
import make_ppp_psp_inv_case as m2  # noqa: E402
from check_water_inv import parse_picks  # noqa: E402
from plot_rugged_inv import parse_rays, rays_used_in_inv  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

XLO, XHI = 25.0, 75.0
DZ_BIN = 0.4


def _mean_by_depth(xs, zs, grid, zc_fun):
    """Return (z_below centers, mean, n) for nodes with x in [XLO,XHI], z>=zc."""
    buckets: dict[int, list[float]] = {}
    for i, x in enumerate(xs):
        if not (XLO <= x <= XHI):
            continue
        zi = zc_fun(x)
        for k, z in enumerate(zs):
            if z < zi - 1e-9:
                continue
            ib = int(round((z - zi) / DZ_BIN))
            buckets.setdefault(ib, []).append(float(grid[i][k]))
    if not buckets:
        return np.array([]), np.array([]), np.array([])
    keys = sorted(buckets)
    zcents = np.array([k * DZ_BIN for k in keys])
    means = np.array([float(np.mean(buckets[k])) for k in keys])
    ns = np.array([len(buckets[k]) for k in keys])
    return zcents, means, ns


def _turn_depths(segs, zc_fun):
    out = []
    for rx, rz in segs:
        if not rz:
            continue
        j = int(np.argmax(rz))
        zm = float(rz[j])
        x = float(rx[j])
        out.append(zm - zc_fun(x))
    return np.array(out)


def _load_case(folder: Path, title: str, zc_fun, rec_name="rec_vs.smesh",
               start_name="start_mixed.smesh", ray_name="rays_rec.dat",
               syn_fwd="syn_rec.dat"):
    xs, zs, rec = m2.parse_smesh(folder / rec_name)
    _, _, start = m2.parse_smesh(folder / start_name)
    _, _, true = m2.parse_smesh(folder / "true_mixed.smesh")
    rec_a = np.asarray(rec, float)
    d_start = rec_a - np.asarray(start, float)
    d_true = rec_a - np.asarray(true, float)
    zb, ms, _ = _mean_by_depth(xs, zs, d_start, zc_fun)
    _, mt, _ = _mean_by_depth(xs, zs, d_true, zc_fun)
    segs = []
    rp = folder / ray_name
    if rp.is_file():
        if (folder / "syn_inv.dat").is_file() and (folder / syn_fwd).is_file():
            segs = rays_used_in_inv(rp, folder / syn_fwd, folder / "syn_inv.dat")
        else:
            segs = parse_rays(rp)
    turns = _turn_depths(segs, zc_fun) if segs else np.array([])
    return {
        "title": title,
        "zb": zb,
        "d_start": ms,
        "d_true": mt,
        "turns": turns,
        "nray": len(segs),
    }


def main() -> int:
    cases = [
        _load_case(HERE / "inv_2d", "fast 观测=二维转折", g.z_conv),
        _load_case(HERE / "inv_graph", "fast 观测=图论初至", g.z_conv),
    ]
    hot = HERE / "inv_2d_hot"
    if (hot / "rec_hot.smesh").is_file():
        # 直方图用第一轮（偏快初值）射线，那才是核所在；收回后已跳回贴面。
        start_rays = "rays_hot_start.dat" if (hot / "rays_hot_start.dat").is_file() else "rays_hot.dat"
        start_fwd = "syn_hot_start.dat" if (hot / "syn_hot_start.dat").is_file() else "syn_hot_rec.dat"
        cases.append(_load_case(
            hot, "fast 初值面下+0.35（逼减速）", g.z_conv,
            rec_name="rec_hot.smesh", start_name="start_hot.smesh",
            ray_name=start_rays, syn_fwd=start_fwd,
        ))
    flat = HERE.parents[1]
    # 平底工区射线大量贴面，不放进这张对照，避免冲淡转折核测试。

    fig, axes = plt.subplots(len(cases), 2, figsize=(10.8, 3.6 * len(cases)),
                             facecolor="w", layout="constrained")
    if len(cases) == 1:
        axes = np.array([axes])
    for row, c in enumerate(cases):
        ax, ah = axes[row]
        if len(c["zb"]):
            ax.plot(c["d_start"], c["zb"], "C3-o", ms=4, label="反演 − 初值")
            ax.plot(c["d_true"], c["zb"], "C0-s", ms=3.5, alpha=0.85, label="反演 − 真值")
        ax.axvline(0.0, color="0.4", lw=0.8)
        if len(c["turns"]):
            med = float(np.median(c["turns"]))
            p90 = float(np.percentile(c["turns"], 90))
            ax.axhline(med, color="0.25", ls="--", lw=1.0, label=f"转折中位 {med:.1f} km")
            ax.axhline(p90, color="0.45", ls=":", lw=1.0, label=f"转折 90% {p90:.1f} km")
            i_neg = None
            if len(c["zb"]) and len(c["d_start"]):
                mask = (c["zb"] >= 0.0) & (c["zb"] <= 8.0)
                if np.any(mask):
                    idx = np.where(mask)[0]
                    i_neg = int(idx[np.argmin(c["d_start"][mask])])
                    ax.plot(c["d_start"][i_neg], c["zb"][i_neg], "r*", ms=14, zorder=5,
                            label=f"0–8 km 最负 {c['zb'][i_neg]:.1f} km")
            print(
                f"{c['title']}: nray={c['nray']}  turn med={med:.2f} p90={p90:.2f}"
                + (f"  dVmin at z-zc={c['zb'][i_neg]:.2f}  {c['d_start'][i_neg]:+.3f}"
                   if i_neg is not None else "")
            )
        ax.set_ylim(10.0, -0.2)
        ax.set_xlabel("ΔVs (km/s)")
        ax.set_ylabel("转换面以下 (km)")
        ax.set_title(c["title"])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.92)
        if len(c["turns"]):
            ah.hist(c["turns"], bins=np.arange(-0.2, 12.2, 0.4),
                    color="0.45", edgecolor="0.2", orientation="horizontal")
            ah.axhline(float(np.median(c["turns"])), color="C3", ls="--", lw=1.2)
        ah.set_ylim(10.0, -0.2)
        ah.set_xlabel("射线条数")
        ah.set_title(f"图论转折 zmax−zc   n={c['nray']}")
        ah.grid(True, alpha=0.3)
    fig.suptitle("面下 Vs 低值深度 vs 图论射线转折深度（真模型无预设低速层）")
    out = HERE / "check_vs_lowband.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
