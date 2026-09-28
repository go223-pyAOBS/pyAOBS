# -*- coding: utf-8 -*-
"""过滤 PPS：只保留转换点 C 更靠近 OBS 的那支；重画折合剖面与模型射线。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pyAOBS.modeling.wave2d.elastic2d import Gather
from pyAOBS.modeling.wave2d.io_smesh import (
    interp_z,
    load_xz,
    parse_pickfile,
    parse_rays,
    parse_smesh,
)
from pyAOBS.modeling.wave2d.plot_model_rays import (
    VS_CMAP,
    VS_TICKS,
    VS_XP,
    VS_YP,
    _PiecewiseNorm,
    _draw_rays,
    _style_ifaces,
)
from pyAOBS.modeling.wave2d.run_gather_017 import plot_section
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def _work() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "tomo2d"
        / "example_water"
        / "PPP+PSP_inv2"
        / "psp_p_shoot"
        / "thin2km_rugged"
        / "lvz2d"
        / "inv_612"
    )


def _out() -> Path:
    return _work().parent / "wave_fwd"


def pps_conv_x(
    xs: list[float], zs: list[float], conv: list[tuple[float, float]]
) -> float:
    """PPS 真转换点：路径上连续重合的 conv 钉（shot…→C→…OBS）。

    不可用「从炮点首次碰到转换面」——P 腿穿面再上来时那是下行交点，不是 C。
    """
    n = len(xs)
    for i in range(1, n - 1):
        if abs(xs[i] - xs[i - 1]) > 1e-3 or abs(xs[i] - xs[i + 1]) > 1e-3:
            continue
        if abs(zs[i] - interp_z(conv, xs[i])) < 0.35:
            return float(xs[i])
    # 退化：从 OBS 端回走，第一次贴转换面
    for x, z in zip(reversed(xs), reversed(zs)):
        if abs(z - interp_z(conv, x)) < 0.25:
            return float(x)
    return float("nan")


def filter_picks_c_near_obs(
    picks: list,
    rays: list,
    conv: list[tuple[float, float]],
    *,
    obs_x: float,
    codes_filter=(7,),
    max_dx_obs: float = 12.0,
    keep_early_envelope: bool = True,
) -> tuple[list, list, dict]:
    """PPS：C 须靠近 OBS；同 C 下若 P 腿双支交错，只留较早折合包络。"""
    # 第一遍：C 靠近 OBS
    cand: list[tuple] = []
    other_picks = []
    other_rays = []
    stats = {c: [0, 0, 0] for c in codes_filter}  # keep, drop_C, drop_late
    for p, ray in zip(picks, rays):
        code, rx, rz, tt, src = p
        if code not in codes_filter:
            other_picks.append(p)
            other_rays.append(ray)
            continue
        xs, zs = ray
        if len(xs) < 2 or not np.isfinite(tt) or tt <= 0:
            stats[code][1] += 1
            continue
        cx = pps_conv_x(xs, zs, conv)
        if not np.isfinite(cx) or abs(cx - obs_x) > max_dx_obs:
            stats[code][1] += 1
            continue
        if abs(cx - obs_x) > abs(cx - rx):
            stats[code][1] += 1
            continue
        cand.append((p, ray, cx))

    keep_idx = set(range(len(cand)))
    if keep_early_envelope and cand:
        # 同侧局部最早包络：落在局部最小附近才留（去掉平行晚支）
        vred = 8.0
        rows = []
        for i, (p, ray, cx) in enumerate(cand):
            code, rx, rz, tt, src = p
            tred = tt - abs(rx - src) / vred
            rows.append((i, rx, tred, src))
        for side in (-1, 1):
            side_rows = [r for r in rows if (r[1] - r[3]) * side > 0.05]
            if len(side_rows) < 5:
                continue
            for i, rx, tred, _s in side_rows:
                local = [
                    s[2]
                    for s in side_rows
                    if abs(s[1] - rx) <= 2.0
                ]
                if not local:
                    continue
                if tred > min(local) + 0.06:
                    if i in keep_idx:
                        keep_idx.remove(i)
                        stats[cand[i][0][0]][2] += 1

    keep_picks = list(other_picks)
    keep_rays = list(other_rays)
    for i, (p, ray, _cx) in enumerate(cand):
        if i in keep_idx:
            keep_picks.append(p)
            keep_rays.append(ray)
            stats[p[0]][0] += 1
    return keep_picks, keep_rays, stats


def main() -> int:
    work = _work()
    out = _out()
    obs = 50.0
    conv = load_xz(work / "conv.refl")
    picks = parse_pickfile(out / "syn_obs50_017.dat")
    rays = parse_rays(out / "rays_obs50_017.dat")
    if len(rays) != len(picks):
        print(f"warn nray={len(rays)} npick={len(picks)}")
    fp, fr, st = filter_picks_c_near_obs(
        picks, rays, conv, obs_x=obs, codes_filter=(7,), max_dx_obs=12.0
    )
    print(
        f"PPS keep={st[7][0]} drop_C={st[7][1]} drop_lateP={st[7][2]}  "
        f"total picks {len(fp)}/{len(picks)}"
    )

    # 写过滤后的 syn（便于复查）
    # 保持 tomo2d 行格式粗略写出
    lines = ["1", f"s  {obs:8.3f}     2.000 {sum(1 for p in fp if abs(p[4]-obs)<0.2):4d}"]
    for code, rx, rz, tt, src in fp:
        if abs(src - obs) > 0.2:
            continue
        lines.append(f"r  {rx:8.3f}  {rz:8.3f} {code:4d}  {tt:10.5f}     0.050")
    syn_f = out / "syn_obs50_017_pps_cobs.dat"
    syn_f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # rays：按 keep 顺序写 >
    ray_f = out / "rays_obs50_017_pps_cobs.dat"
    with ray_f.open("w", encoding="utf-8") as f:
        for xs, zs in fr:
            f.write(">\n")
            for x, z in zip(xs, zs):
                f.write(f"{x:.6g} {z:.6g}\n")
    print(f"wrote {syn_f.name} {ray_f.name}")

    z = np.load(out / "gather_obs50_off80_d0.2_f3.npz")
    g = Gather(
        t=z["t"],
        rec_x=z["rec_x"],
        data=z["data"],
        src_x=float(z["src_x"]),
        src_z=2.0,
        delay=float(z["delay"]),
        dt=float(z["dt"]),
        f0=float(z["f0"]),
    )
    png = out / "gather_obs50_off80_d0.2_f3_017_vred8.png"
    plot_section(g, fp, png, 80.0, pclip=98.0, vred=8.0, trace_norm=True)
    print(f"wrote {png}")

    # 模型+射线（0/1 全留，7 已滤）
    xs, zs, vel = parse_smesh(work / "true_vp.smesh")
    sea = load_xz(work / "seafloor.refl")
    vp = np.asarray(vel, dtype=float).T
    zsea = float(sea[0][1]) if sea else 2.0
    for k, zv in enumerate(zs):
        if zv < zsea - 1e-9:
            vp[k, :] = np.nan
    xs_a = np.asarray(xs)
    zs_a = np.asarray(zs)
    extent = (float(xs_a[0]), float(xs_a[-1]), float(zs_a[-1]), float(zs_a[0]))
    zlim = float(zs_a[-1])
    moho = load_xz(work / "moho_true.refl")

    VP_XP = (1.5, 3.0, 5.0, 6.5, 7.2, 8.0, 9.0)
    VP_YP = (0.00, 0.12, 0.28, 0.45, 0.55, 0.78, 1.00)
    VP_CMAP = mcolors.LinearSegmentedColormap.from_list(
        "vp_struct",
        [
            (0.00, "#08306b"),
            (0.15, "#2171b5"),
            (0.35, "#6baed6"),
            (0.5, "#ffffcc"),
            (0.65, "#fdae61"),
            (0.82, "#f03b20"),
            (1.00, "#67001f"),
        ],
    )
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(12.0, 5.2), facecolor="w", layout="constrained")
    im = ax.imshow(
        vp,
        extent=extent,
        cmap=VP_CMAP,
        norm=_PiecewiseNorm(VP_XP, VP_YP),
        aspect="auto",
        interpolation="nearest",
        zorder=0,
    )
    st2 = _draw_rays(
        ax, fp, fr, (0, 1, 7), moho, every=1, bounce_tol=None, mark_bounce=False
    )
    _style_ifaces(ax, work, obs)
    ax.set_xlim(0, 150)
    ax.set_ylim(zlim, 0)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("深度 (km)")
    ax.set_title(
        "Vp + PPP/PmP/PPS   PPS：C≈OBS，且去掉晚到的平行 P 支",
        fontsize=11,
    )
    ax.grid(True, alpha=0.22)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9, ncol=3)
    fig.colorbar(im, ax=ax, shrink=0.9).set_label("Vp (km/s)")
    png2 = out / "model_rays_obs50_017_pps_cobs.png"
    fig.savefig(png2, dpi=150)
    plt.close(fig)
    print(f"wrote {png2}  rays {st2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
