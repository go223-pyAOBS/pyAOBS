# -*- coding: utf-8 -*-
"""真 Vs + 多震相射线（OBS=50）。不改 inv_*。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pyAOBS.modeling.wave2d.io_smesh import load_xz, parse_pickfile, parse_rays, parse_smesh
from pyAOBS.modeling.wave2d.phases import PHASE_STYLE, SHOT_XS, write_geom

VS_XP = (1.00, 2.20, 3.40, 4.35, 5.05)
VS_YP = (0.00, 0.26, 0.30, 0.70, 1.00)
VS_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "vs_struct",
    [
        (0.00, "#08306b"),
        (0.13, "#2b8cbe"),
        (0.26, "#7fcdbb"),
        (0.30, "#ffffcc"),
        (0.50, "#fdae61"),
        (0.70, "#f03b20"),
        (0.88, "#bd0026"),
        (1.00, "#67001f"),
    ],
)
VS_TICKS = (1.2, 1.6, 2.0, 3.5, 3.8, 4.1, 4.35, 4.6, 4.8, 5.0)
BEL_LVZ = (58.0, 8.00)


class _PiecewiseNorm(mcolors.Normalize):
    def __init__(self, xp, yp):
        self.xp = np.asarray(xp, dtype=float)
        self.yp = np.asarray(yp, dtype=float)
        super().__init__(vmin=float(self.xp[0]), vmax=float(self.xp[-1]), clip=True)

    def __call__(self, value, clip=None):
        x = np.ma.asarray(value, dtype=float)
        y = np.interp(x.filled(self.xp[0]), self.xp, self.yp, left=0.0, right=1.0)
        return np.ma.array(y, mask=np.ma.getmaskarray(x))

    def inverse(self, value):
        return np.interp(np.asarray(value, dtype=float), self.yp, self.xp)


def _work_default() -> Path:
    return (
        HERE.parent
        / "tomo2d"
        / "example_water"
        / "PPP+PSP_inv2"
        / "psp_p_shoot"
        / "thin2km_rugged"
        / "lvz2d"
        / "inv_612"
    )


def _out_default() -> Path:
    return _work_default().parent / "wave_fwd"


def _vs_masked(work: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, zs, vel = parse_smesh(work / "true_vs.smesh")
    sea = load_xz(work / "seafloor.refl")
    arr = np.asarray(vel, dtype=float).T
    zsea = float(sea[0][1]) if sea else 2.0
    for k, z in enumerate(zs):
        if z < zsea - 1e-9:
            arr[k, :] = np.nan
    return np.asarray(xs), np.asarray(zs), arr


def _style_ifaces(ax, work: Path, src_x: float) -> None:
    sea = load_xz(work / "seafloor.refl")
    conv = load_xz(work / "conv.refl")
    moho = load_xz(work / "moho_true.refl")
    ax.plot([p[0] for p in sea], [p[1] for p in sea], "k--", lw=0.7, zorder=3)
    ax.plot([p[0] for p in conv], [p[1] for p in conv], "k-.", lw=0.8, zorder=3)
    ax.plot(
        [p[0] for p in moho],
        [p[1] for p in moho],
        color="w",
        ls="-",
        lw=2.6,
        zorder=5.5,
    )
    ax.plot(
        [p[0] for p in moho],
        [p[1] for p in moho],
        color="k",
        ls="-",
        lw=1.4,
        zorder=6,
        label="莫霍",
    )
    ax.plot([src_x], [sea[0][1] if sea else 2.0], "k^", ms=8, zorder=7)
    ax.plot([BEL_LVZ[0]], [BEL_LVZ[1]], "kx", ms=7, mew=1.3, zorder=8)


def _bounce_ok(xs: list[float], zs: list[float], moho: list[tuple[float, float]], tol: float) -> bool:
    """最深点应落在莫霍附近（反射），否则视为穿面坏射线。"""
    from pyAOBS.modeling.wave2d.io_smesh import interp_z

    i = int(np.argmax(zs))
    return abs(float(zs[i]) - interp_z(moho, float(xs[i]))) <= tol


def _draw_rays(
    ax,
    picks,
    rays,
    codes: tuple[int, ...],
    moho: list[tuple[float, float]],
    *,
    every: int = 1,
    bounce_tol: float | None = None,
    mark_bounce: bool = False,
) -> dict[int, tuple[int, int]]:
    """返回 {code: (kept, skipped)}。bounce_tol 非空时过滤穿莫霍射线。"""
    drawn: set[int] = set()
    stats: dict[int, list[int]] = {c: [0, 0] for c in codes}
    bounce_xy: dict[int, list[tuple[float, float]]] = {c: [] for c in codes}
    for (code, rx, _rz, tt, src), (xs, zs) in zip(picks, rays):
        if code not in codes:
            continue
        if not np.isfinite(tt) or tt <= 0:
            continue
        if every > 1:
            ix = int(min(range(len(SHOT_XS)), key=lambda i: abs(SHOT_XS[i] - rx)))
            if ix % every:
                continue
        if bounce_tol is not None and not _bounce_ok(xs, zs, moho, bounce_tol):
            stats[code][1] += 1
            continue
        stats[code][0] += 1
        col, lab = PHASE_STYLE[code]
        kw = dict(color=col, lw=0.85, alpha=0.88, zorder=5)
        if code not in drawn:
            kw["label"] = lab
            drawn.add(code)
        ax.plot(xs, zs, **kw)
        if mark_bounce:
            i = int(np.argmax(zs))
            bounce_xy[code].append((float(xs[i]), float(zs[i])))
    if mark_bounce:
        for code, pts in bounce_xy.items():
            if not pts:
                continue
            col, _lab = PHASE_STYLE[code]
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                "o",
                ms=3.5,
                mfc="none",
                mew=1.0,
                color=col,
                zorder=8,
            )
    return {c: (a[0], a[1]) for c, a in stats.items()}


def plot_model_rays(
    work: Path,
    syn: Path,
    ray_p: Path,
    out: Path,
    *,
    src_x: float = 50.0,
    show_all: bool = False,
) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    xs, zs, vs = _vs_masked(work)
    picks = parse_pickfile(syn)
    rays = parse_rays(ray_p)
    moho = load_xz(work / "moho_true.refl")
    if len(rays) != len(picks):
        print(f"warn: nray={len(rays)} npick={len(picks)}")
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    zlim = float(zs[-1])
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 8.6), facecolor="w", layout="constrained", sharex=True)
    last = None
    # 上：只画莫霍反射；PmP 远偏移常穿面，用 bounce_tol 滤掉
    # 下：转折/盖层（本来就会进更深，不滤）
    bt = None if show_all else 0.15
    pmp_title = (
        "PmP（1）：全部射线（含穿面）"
        if show_all
        else "PmP（1）：仅保留贴面射线；远偏移穿面已滤掉"
    )
    panels = (
        (axes[0], (12, 13), "莫霍反射  12 PSP-Moho  13 PSS-Moho（圆点=最深点）", 1, bt, True),
        (axes[1], (1,), pmp_title, 1, bt, True),
    )
    for ax, codes, title, every, bounce_tol, mark_b in panels:
        last = ax.imshow(
            vs,
            extent=extent,
            cmap=VS_CMAP,
            norm=_PiecewiseNorm(VS_XP, VS_YP),
            aspect="auto",
            interpolation="nearest",
            zorder=0,
        )
        st = _draw_rays(
            ax,
            picks,
            rays,
            codes,
            moho,
            every=every,
            bounce_tol=bounce_tol,
            mark_bounce=mark_b,
        )
        _style_ifaces(ax, work, src_x)
        ax.set_xlim(0.0, 150.0)
        ax.set_ylim(zlim, 0.0)
        ax.set_ylabel("深度 (km)")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.22)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9, ncol=2)
        for c, (keep, skip) in st.items():
            if keep or skip:
                print(f"  panel codes={codes}  {c}: keep={keep} skip_penetrate={skip}")
    axes[1].set_xlabel("x (km)")
    cbar = fig.colorbar(last, ax=axes.ravel().tolist(), shrink=0.78)
    cbar.set_ticks(VS_TICKS)
    cbar.set_label("Vs (km/s)   蓝绿=盖层  黄橙=地壳  红=地幔")
    fig.suptitle(
        f"OBS={src_x:.0f} km  反射射线 vs 莫霍   白黑粗线=莫霍  点划=转换面  虚线=海底",
        fontsize=12,
    )
    fig.savefig(out, dpi=140)
    plt.close(fig)
    out2 = out.with_name(f"model_turning_obs{int(src_x)}.png")
    fig2, ax2 = plt.subplots(figsize=(11.2, 4.4), facecolor="w", layout="constrained")
    ax2.imshow(
        vs,
        extent=extent,
        cmap=VS_CMAP,
        norm=_PiecewiseNorm(VS_XP, VS_YP),
        aspect="auto",
        interpolation="nearest",
        zorder=0,
    )
    _draw_rays(ax2, picks, rays, (0, 6, 7, 8), moho, every=2, bounce_tol=None, mark_bounce=False)
    _style_ifaces(ax2, work, src_x)
    ax2.set_xlim(0.0, 150.0)
    ax2.set_ylim(zlim, 0.0)
    ax2.set_xlabel("x (km)")
    ax2.set_ylabel("深度 (km)")
    ax2.set_title("转折/盖层（本来可进地幔梯度）  0/6/7/8", fontsize=10)
    ax2.grid(True, alpha=0.22)
    ax2.legend(loc="lower right", fontsize=7, framealpha=0.9, ncol=2)
    fig2.savefig(out2, dpi=140)
    plt.close(fig2)
    print(f"wrote {out2}")


def main() -> int:
    p = argparse.ArgumentParser(description="真模型 + 多震相射线图")
    p.add_argument("--work", type=Path, default=_work_default())
    p.add_argument("--out", type=Path, default=_out_default())
    p.add_argument("--obs", type=float, default=50.0)
    p.add_argument("--show-all", action="store_true", help="不过滤穿莫霍射线")
    p.add_argument("--write-geom-only", action="store_true")
    args = p.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    geom = out / "geom_obs50.dat"
    n = write_geom(geom, obs_xs=(args.obs,))
    print(f"wrote {geom} nrec={n}")
    if args.write_geom_only:
        return 0
    syn = out / "syn_obs50.dat"
    ray_p = out / "rays_obs50.dat"
    if not syn.is_file() or not ray_p.is_file():
        raise SystemExit(f"missing {syn} or {ray_p}  (先跑 run_ray_phases.sh)")
    png = out / f"model_rays_obs{int(args.obs)}.png"
    plot_model_rays(args.work, syn, ray_p, png, src_x=args.obs, show_all=args.show_all)
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
