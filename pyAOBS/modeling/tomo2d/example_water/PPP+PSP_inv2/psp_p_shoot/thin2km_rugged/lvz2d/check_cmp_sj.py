#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单场公平对比：旧联合 vs 新策略。Vp/Vs RMS + rec−true 图。"""

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
import inv_grid as g  # noqa: E402
import plot_rugged_inv as pr  # noqa: E402
from check_joint import DCLIM, OBS_XS, recs_for_draw, region_rms, ttrms  # noqa: E402
from check_lvz import load_vp_vs  # noqa: E402
from check_ps_fwd import draw_ps_rays  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

XLO, XHI = 18.0, 82.0
PHASES = ((0, "PPP"), (1, "PmP"), (6, "PSP"), (7, "PPS"), (8, "PSS"))
MOHO_XZ: list[tuple[float, float]] = []


def _load_xz(path: Path) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    if not path.is_file():
        return out
    for ln in path.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) >= 2:
            out.append((float(p[0]), float(p[1])))
    return out


def _z_at(xz: list[tuple[float, float]], x: float) -> float:
    if not xz:
        return float("nan")
    return float(np.interp(x, [p[0] for p in xz], [p[1] for p in xz]))


def _moho_rms(rec_xz: list[tuple[float, float]], true_xz: list[tuple[float, float]]) -> float:
    if not rec_xz or not true_xz:
        return float("nan")
    s = n = 0
    for x, zt in true_xz:
        if not (XLO <= x <= XHI):
            continue
        d = _z_at(rec_xz, x) - zt
        s += d * d
        n += 1
    return math.sqrt(s / n) if n else float("nan")


def region_rms_band(xs, zs, a, b, zlo_fn, zhi_fn) -> float:
    s = n = 0
    for i, x in enumerate(xs):
        if not (XLO <= x <= XHI):
            continue
        zlo, zhi = zlo_fn(x), zhi_fn(x)
        for k, z in enumerate(zs):
            if z < zlo - 1e-9 or z >= zhi - 1e-9:
                continue
            d = a[k, i] - b[k, i]
            if np.isnan(d):
                continue
            s += float(d * d)
            n += 1
    return math.sqrt(s / n) if n else float("nan")


def _print_v(lab, xs, zs, rec, true):
    print(
        f"  {lab:22s}  lid {region_rms(xs, zs, rec, true, lid=True):.4f}  "
        f"below {region_rms(xs, zs, rec, true, lid=False):.4f}  "
        f"crust {region_rms(xs, zs, rec, true, lid=None):.4f}"
    )
    if MOHO_XZ:
        print(
            f"  {lab:22s}  conv-moho {region_rms_band(xs, zs, rec, true, g.z_conv, lambda x: _z_at(MOHO_XZ, x)):.4f}  "
            f"below-moho {region_rms_band(xs, zs, rec, true, lambda x: _z_at(MOHO_XZ, x), lambda _x: 1e9):.4f}"
        )


def _load_rays(folder: Path, name: str):
    rp, sp = folder / f"rays_{name}.dat", folder / f"syn_{name}.dat"
    if rp.is_file() and sp.is_file():
        return pr.parse_rays(rp), recs_for_draw(sp)
    return [], []


def _style(ax, title, rec_xz=None, start_xz=None, rays=None, recs=None, codes=None, *, mark_conv=False, legend=False):
    if rays and recs:
        draw_ps_rays(
            ax, rays, recs, thin=True, mark_conv=mark_conv, z_conv=g.z_conv, codes=codes,
        )
        if legend:
            ax.legend(loc="upper right", framealpha=0.88, fontsize=6, ncol=2)
    xs_l = np.linspace(XLO, XHI, 80)
    ax.plot(xs_l, [g.z_conv(x) for x in xs_l], "k-.", lw=0.8, zorder=2)
    if MOHO_XZ:
        ax.plot([p[0] for p in MOHO_XZ], [p[1] for p in MOHO_XZ], "k:", lw=0.9, zorder=2)
    if start_xz:
        ax.plot([p[0] for p in start_xz], [p[1] for p in start_xz], color="0.45", ls="--", lw=0.9, zorder=3)
    if rec_xz:
        ax.plot([p[0] for p in rec_xz], [p[1] for p in rec_xz], color="C3", lw=1.1, zorder=4)
    ax.plot(xs_l, [g.H] * len(xs_l), "k--", lw=0.6, zorder=2)
    ax.plot(list(OBS_XS), [g.H] * len(OBS_XS), "k^", ms=5, zorder=7)
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(16, 0)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)


def _tt(folder: Path, label: str) -> None:
    obs, start_p, rec_p = folder / "syn_inv.dat", folder / "syn_start.dat", folder / "syn_rec.dat"
    if not (obs.is_file() and start_p.is_file() and rec_p.is_file()):
        return
    print(f"{label}  t RMS  start → rec")
    for code, name in PHASES:
        print(
            f"  {name:3s}({code})  {ttrms(obs, start_p, code):.4f} → {ttrms(obs, rec_p, code):.4f}"
        )


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "cmp_sj"
    work = HERE / tag
    fj, fs = work / "joint", work / "strat"
    true_moho = work / "moho_true.refl"
    start_moho = work / "moho.refl"
    MOHO_XZ.clear()
    MOHO_XZ.extend(_load_xz(true_moho if true_moho.is_file() else start_moho))
    start_xz = _load_xz(start_moho)
    j_moho = _load_xz(fj / "rec_moho.refl")
    s_moho = _load_xz(fs / "rec_moho.refl")
    j_st_rays, j_st_recs = _load_rays(fj, "start")
    j_rays, j_recs = _load_rays(fj, "rec")
    s_st_rays, s_st_recs = _load_rays(fs, "start")
    s_rays, s_recs = _load_rays(fs, "rec")
    if not j_st_rays:
        j_st_rays, j_st_recs = s_st_rays, s_st_recs
    xs, zs, t_vp, s_vp, _, t_vs, _, _ = load_vp_vs(HERE)
    _, _, _, _, j_vp, _, _, j_vs = load_vp_vs(fj)
    _, _, _, _, s_rec_vp, _, _, s_vs = load_vp_vs(fs)
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))

    print(f"公平：单场 -M start_vp -k1.73  无 -U  {tag}  同一 -I8")
    if MOHO_XZ:
        print("Moho z RMS vs true")
        print(f"  start  {_moho_rms(start_xz, MOHO_XZ):.4f} km")
        print(f"  joint  {_moho_rms(j_moho, MOHO_XZ):.4f} km")
        print(f"  strat  {_moho_rms(s_moho, MOHO_XZ):.4f} km")
    print("Vp rec vs true")
    _print_v("start", xs, zs, s_vp, t_vp)
    _print_v("joint", xs, zs, j_vp, t_vp)
    _print_v("strat", xs, zs, s_rec_vp, t_vp)
    print("Vs rec vs true")
    if t_vs is not None:
        start_vs = work / "start_vs.smesh"
        if start_vs.is_file():
            import make_ppp_psp_inv_case as m2
            from check_joint import mask_crust

            _, _, sv = m2.parse_smesh(start_vs)
            sv = mask_crust(xs, zs, sv)
            _print_v("start Vp/k", xs, zs, sv, t_vs)
        if j_vs is not None:
            _print_v("joint", xs, zs, j_vs, t_vs)
        if s_vs is not None:
            _print_v("strat", xs, zs, s_vs, t_vs)
    _tt(fj, "joint")
    _tt(fs, "strat")
    for folder, lab in ((fj, "joint"), (fs, "strat")):
        hold_t, hold_r = folder / "syn_holdout_true.dat", folder / "syn_holdout_rec.dat"
        if hold_t.is_file() and hold_r.is_file():
            print(f"{lab}  holdout t RMS  (full syn_obs, 含未参与反演的 PSP)")
            for code, name in PHASES:
                print(f"  {name:3s}({code})  {ttrms(hold_t, hold_r, code):.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.6), facecolor="w", layout="constrained")
    codes_p, codes_s = (0, 1), (6, 7, 8)
    panels = (
        (axes[0, 0], s_vp - t_vp, f"start−true Vp\nlid {region_rms(xs,zs,s_vp,t_vp,lid=True):.3f}  below {region_rms(xs,zs,s_vp,t_vp,lid=False):.3f}", start_xz, None, j_st_rays, j_st_recs, codes_p, False, False),
        (axes[0, 1], j_vp - t_vp, f"联合 rec−true Vp\nlid {region_rms(xs,zs,j_vp,t_vp,lid=True):.3f}  below {region_rms(xs,zs,j_vp,t_vp,lid=False):.3f}", None, j_moho, j_rays, j_recs, codes_p, False, False),
        (axes[0, 2], s_rec_vp - t_vp, f"新策略 rec−true Vp\nlid {region_rms(xs,zs,s_rec_vp,t_vp,lid=True):.3f}  below {region_rms(xs,zs,s_rec_vp,t_vp,lid=False):.3f}", None, s_moho, s_rays, s_recs, codes_p, False, False),
        (
            axes[1, 0],
            (j_vs - s_vs) if j_vs is not None and s_vs is not None else t_vp * np.nan,
            "联合−策略  Vs（红=联合更快）",
            None,
            None,
            [],
            [],
            codes_s,
            False,
            False,
        ),
        (
            axes[1, 1],
            (j_vs - t_vs) if j_vs is not None and t_vs is not None else t_vp * np.nan,
            (
                f"联合 rec−true Vs\nlid {region_rms(xs,zs,j_vs,t_vs,lid=True):.3f}  "
                f"below {region_rms(xs,zs,j_vs,t_vs,lid=False):.3f}"
                if j_vs is not None and t_vs is not None
                else "联合 无 rec_vs"
            ),
            None,
            j_moho,
            j_rays,
            j_recs,
            codes_s,
            True,
            True,
        ),
        (
            axes[1, 2],
            (s_vs - t_vs) if s_vs is not None and t_vs is not None else t_vp * np.nan,
            (
                f"新策略 rec−true Vs\nlid {region_rms(xs,zs,s_vs,t_vs,lid=True):.3f}  "
                f"below {region_rms(xs,zs,s_vs,t_vs,lid=False):.3f}"
                if s_vs is not None and t_vs is not None
                else "策略 无 rec_vs"
            ),
            None,
            s_moho,
            s_rays,
            s_recs,
            codes_s,
            True,
            False,
        ),
    )
    last = None
    for ax, grid, title, st_xz, rec_xz, rays, recs, codes, mark_conv, legend in panels:
        im = ax.imshow(
            grid,
            extent=extent,
            cmap="RdBu_r",
            vmin=-DCLIM,
            vmax=DCLIM,
            aspect="auto",
            interpolation="nearest",
            zorder=0,
        )
        last = im
        _style(
            ax, title, rec_xz=rec_xz, start_xz=st_xz,
            rays=rays, recs=recs, codes=codes, mark_conv=mark_conv, legend=legend,
        )
    fig.colorbar(last, ax=axes.ravel().tolist(), shrink=0.55, label="ΔV (km/s)")
    if "0178" in tag or "pmp" in tag:
        phase_lab = "0+1+7+8（可动PmP，无PSP拾取）"
    elif "078" in tag:
        phase_lab = "0+7+8（无PSP拾取）"
    else:
        phase_lab = "0+6+7+8"
    fig.suptitle(
        f"单场公平  {phase_lab}  -k1.73  -I8  -SD20 -TD1    上：Vp+PPP/PmP    下：Vs+PPS/PSS    "
        "点线=真Moho  灰虚=初值  红实=收回"
    )
    out = work / "check_cmp_sj.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
