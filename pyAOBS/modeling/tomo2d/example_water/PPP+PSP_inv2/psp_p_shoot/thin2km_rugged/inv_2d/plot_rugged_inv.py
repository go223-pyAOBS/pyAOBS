#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""崎岖两步反演：start / rec / true（混合网格）+ 射线 + 走时拟合。"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "thin2km"))
sys.path.insert(0, str(HERE.parent.parent / "thin2km_incline"))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[2].parent / "ps_inv"))
sys.path.insert(0, str(HERE.parents[2].parent / "water_inv"))
import inv_grid as g  # noqa: E402
import make_ppp_psp_inv_case as m2  # noqa: E402
from check_water_inv import parse_picks  # noqa: E402
from check_ps_inv import plot_ttimes_fit  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

P_COLOR, S_COLOR = "#1f77b4", "#c44e8a"
OBS_XS = (30.0, 40.0, 50.0, 60.0, 70.0)
TRUE_RAY_LABEL = "二维转折"
REC_RAY_LABEL = "图论初至（PSP 当 0）"
REC_MESH = "rec_vs.smesh"
SUPTITLE_MODELS = "崎岖面  观测=二维积分转折枝  反演=图论初至两步"
SUPTITLE_RAYS = "崎岖面射线：收回=图论初至，真值=二维转折（观测）"


def parse_rays(path: Path) -> list[tuple[list[float], list[float]]]:
    segs, xs, zs = [], [], []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith(">"):
            if len(xs) >= 2:
                segs.append((xs, zs))
            xs, zs = [], []
            continue
        a = s.split()
        if len(a) >= 2:
            xs.append(float(a[0]))
            zs.append(float(a[1]))
    if len(xs) >= 2:
        segs.append((xs, zs))
    return segs


def pick_key(p) -> tuple[float, float]:
    return round(float(p[4]), 3), round(float(p[1]), 3)


def rays_used_in_inv(ray_path: Path, syn_fwd: Path, syn_obs: Path):
    segs = parse_rays(ray_path)
    fwd = parse_picks(syn_fwd.read_text(encoding="utf-8"))
    obs = parse_picks(syn_obs.read_text(encoding="utf-8"))
    keep = {pick_key(p) for p in obs}
    out = []
    for p, seg in zip(fwd, segs):
        if pick_key(p) in keep:
            out.append(seg)
    return out


def _is_s(x: float, z: float) -> bool:
    return z > g.z_conv(x) + 1e-3


def draw_iface_rays(ax, segs, *, legend: bool) -> None:
    labeled = set()
    pending_s: list[tuple[list[float], list[float]]] = []
    for rx, rz in segs:
        cx, cz = [rx[0]], [rz[0]]
        cur = _is_s(rx[0], rz[0])
        for j in range(len(rx) - 1):
            s1 = _is_s(0.5 * (rx[j] + rx[j + 1]), 0.5 * (rz[j] + rz[j + 1]))
            if s1 == cur:
                cx.append(rx[j + 1])
                cz.append(rz[j + 1])
                continue
            if len(cx) >= 2:
                if cur:
                    pending_s.append((cx, cz))
                else:
                    kw = {}
                    if legend and "P" not in labeled:
                        kw["label"] = "P（蓝）"
                        labeled.add("P")
                    ax.plot(cx, cz, color=P_COLOR, lw=0.55, alpha=0.75, zorder=3, **kw)
            cx, cz, cur = [rx[j], rx[j + 1]], [rz[j], rz[j + 1]], s1
        if len(cx) >= 2:
            if cur:
                pending_s.append((cx, cz))
            else:
                kw = {}
                if legend and "P" not in labeled:
                    kw["label"] = "P（蓝）"
                    labeled.add("P")
                ax.plot(cx, cz, color=P_COLOR, lw=0.55, alpha=0.75, zorder=3, **kw)
    for sx, sz in pending_s:
        kw = {}
        if legend and "S" not in labeled:
            kw["label"] = "S（粉）"
            labeled.add("S")
        ax.plot(sx, sz, color=S_COLOR, lw=0.7, alpha=0.85, zorder=4, **kw)


def _draw_geom(ax, *, legend: bool) -> None:
    ax.plot(list(OBS_XS), [2.0] * len(OBS_XS), "^", color="k", ms=6, zorder=7,
            label="OBS" if legend else None)


def _panel(ax, grid, extent, title, rays=None, *, legend=False):
    ax.imshow(grid, extent=extent, cmap="RdYlBu_r", vmin=1.2, vmax=6.2,
              aspect="auto", interpolation="nearest", zorder=0)
    xs = [x for x in np.linspace(18, 82, 80)]
    ax.plot(xs, [2.0] * len(xs), "k--", lw=0.7, zorder=2)
    ax.plot(xs, [g.z_conv(x) for x in xs], "k-.", lw=0.9, zorder=2)
    if rays:
        draw_iface_rays(ax, rays, legend=legend)
    _draw_geom(ax, legend=legend)
    ax.set_xlim(18, 82)
    ax.set_ylim(16, 0)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if legend:
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9)


def _shoot_true_psp():
    import make_rugged_2d_inv as mk  # noqa: WPS433

    md = mk.inc.Model("fast", mk.CRUST_TRUE)
    rays = mk.shoot_codes(md, (6,))
    return [(r.xs, r.zs) for r in rays]


def main() -> int:
    xs, zs, true_m = m2.parse_smesh(HERE / "true_mixed.smesh")
    _, _, start_m = m2.parse_smesh(HERE / "start_mixed.smesh")
    rec_p = HERE / REC_MESH if not Path(REC_MESH).is_absolute() else Path(REC_MESH)
    if not rec_p.is_file():
        rec_p = HERE / "rec_vs.smesh"
    if not rec_p.is_file():
        rec_p = HERE / "start_mixed.smesh"
    _, _, rec_m = m2.parse_smesh(rec_p)
    gtrue = np.asarray(true_m, float).T
    gstart = np.asarray(start_m, float).T
    grec = np.asarray(rec_m, float).T
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))

    rec_rays = []
    if (HERE / "rays_rec.dat").is_file() and (HERE / "syn_rec.dat").is_file():
        rec_rays = rays_used_in_inv(
            HERE / "rays_rec.dat", HERE / "syn_rec.dat", HERE / "syn_inv.dat",
        )
        print(f"graph rec rays used in inv: {len(rec_rays)}")
    true_rays = _shoot_true_psp()
    print(f"true rays ({TRUE_RAY_LABEL}): {len(true_rays)}")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), facecolor="w", layout="constrained")
    _panel(axes[0], gstart, extent, "start mixed")
    _panel(axes[1], grec, extent, f"rec mixed  {REC_RAY_LABEL} n={len(rec_rays)}", rec_rays, legend=True)
    _panel(axes[2], gtrue, extent, f"true mixed  {TRUE_RAY_LABEL} n={len(true_rays)}", true_rays)
    axes[0].set_ylabel("深度 (km)")
    fig.suptitle(SUPTITLE_MODELS)
    out = HERE / "check_inv_models.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), facecolor="w", layout="constrained")
    _panel(axes[0], grec, extent, f"收回  {REC_RAY_LABEL}  n={len(rec_rays)}", rec_rays, legend=True)
    _panel(axes[1], gtrue, extent, f"真值  {TRUE_RAY_LABEL}  n={len(true_rays)}", true_rays)
    axes[0].set_ylabel("深度 (km)")
    axes[0].set_xlabel("模型距离 (km)")
    axes[1].set_xlabel("模型距离 (km)")
    fig.suptitle(SUPTITLE_RAYS)
    out_r = HERE / "check_inv_rays.png"
    fig.savefig(out_r, dpi=140)
    plt.close(fig)
    print(f"wrote {out_r}")

    if (HERE / "syn_ppp.dat").is_file() and (HERE / "syn_ppp_start.dat").is_file():
        rec = parse_picks((HERE / "syn_ppp_rec.dat").read_text(encoding="utf-8")) if (HERE / "syn_ppp_rec.dat").is_file() else None
        plot_ttimes_fit(
            parse_picks((HERE / "syn_ppp.dat").read_text(encoding="utf-8")),
            parse_picks((HERE / "syn_ppp_start.dat").read_text(encoding="utf-8")),
            rec,
            HERE / "check_inv_vp_ttimes.png",
            show=False,
        )
        print(f"wrote {HERE / 'check_inv_vp_ttimes.png'}")
    if (HERE / "syn_inv.dat").is_file() and (HERE / "syn_start.dat").is_file():
        rec = parse_picks((HERE / "syn_rec.dat").read_text(encoding="utf-8")) if (HERE / "syn_rec.dat").is_file() else None
        plot_ttimes_fit(
            parse_picks((HERE / "syn_inv.dat").read_text(encoding="utf-8")),
            parse_picks((HERE / "syn_start.dat").read_text(encoding="utf-8")),
            rec,
            HERE / "check_inv_ttimes.png",
            show=False,
        )
        print(f"wrote {HERE / 'check_inv_ttimes.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
