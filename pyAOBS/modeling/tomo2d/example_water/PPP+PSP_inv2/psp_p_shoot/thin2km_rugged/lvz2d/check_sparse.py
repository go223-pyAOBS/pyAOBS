#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""稀疏真 PSP holdout：A / A+稀疏PSP / 校正插值 / 全 PSP(B)。"""

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
import inv_grid as g  # noqa: E402
from check_joint import DCLIM, OBS_XS, PHASES, mask_crust, region_rms, ttrms  # noqa: E402
from check_lvz import load_vp_vs  # noqa: E402
from check_water_inv import parse_picks  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

XLO, XHI = 18.0, 82.0


def _rms(a) -> float:
    a = [float(x) for x in a if math.isfinite(x)]
    return math.sqrt(sum(v * v for v in a) / len(a)) if a else float("nan")


def load_keep(path: Path):
    keep, hold = set(), set()
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        _isrc, _irec, sx, rx, _dx, flag = ln.split()
        key = (round(float(sx), 3), round(float(rx), 3))
        (keep if int(flag) else hold).add(key)
    return keep, hold


def _pmap(picks, code: int):
    out = {}
    for p in picks:
        if int(p[0]) != code or not math.isfinite(p[3]):
            continue
        out[(round(p[4], 3), round(p[1], 3))] = float(p[3])
    return out


def interp_residual(xy_k, val_k, xy_q):
    xy_k = np.asarray(xy_k, float)
    val_k = np.asarray(val_k, float)
    xy_q = np.asarray(xy_q, float)
    if len(xy_k) == 0:
        return np.full(len(xy_q), np.nan)
    try:
        from scipy.interpolate import RBFInterpolator

        pred = RBFInterpolator(
            xy_k, val_k, kernel="thin_plate_spline", smoothing=1e-3
        )(xy_q)
        return np.asarray(pred, float)
    except Exception:
        out = np.empty(len(xy_q), float)
        for i, q in enumerate(xy_q):
            d2 = np.sum((xy_k - q) ** 2, axis=1)
            w = 1.0 / np.maximum(d2, 1e-6)
            out[i] = float(np.dot(w, val_k) / w.sum())
        return out


def match_rows(true_picks, keep):
    m0, m6, m7, m8 = (_pmap(true_picks, c) for c in (0, 6, 7, 8))
    keys = sorted(set(m0) & set(m6) & set(m7) & set(m8))
    rows = []
    for src, sx in keys:
        t_ppp, t_psp, t_pps, t_pss = m0[(src, sx)], m6[(src, sx)], m7[(src, sx)], m8[(src, sx)]
        corr = t_pss - (t_pps - t_ppp)
        dx = abs(sx - src)
        xm = 0.5 * (sx + src)
        is_keep = (src, sx) in keep
        rows.append(
            dict(
                src=src,
                sx=sx,
                dx=dx,
                xm=xm,
                psp=t_psp,
                corr=corr,
                keep=is_keep,
            )
        )
    return rows


def attach_fwd(rows, pred_picks):
    m6 = _pmap(pred_picks, 6)
    for r in rows:
        r["fwd"] = m6.get((r["src"], r["sx"]), float("nan"))


def _print_vs(lab, xs, zs, rec, true):
    print(
        f"  {lab:18s}  lid {region_rms(xs, zs, rec, true, lid=True):.4f}  "
        f"below {region_rms(xs, zs, rec, true, lid=False):.4f}  "
        f"crust {region_rms(xs, zs, rec, true, lid=None):.4f}"
    )


def _style(ax, title):
    xs_l = np.linspace(XLO, XHI, 80)
    ax.plot(xs_l, [g.z_conv(x) for x in xs_l], "k-.", lw=0.8, zorder=2)
    ax.plot(xs_l, [g.H] * len(xs_l), "k--", lw=0.6, zorder=2)
    ax.plot(list(OBS_XS), [g.H] * len(OBS_XS), "k^", ms=5, zorder=7)
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(16, 0)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)


def main() -> int:
    fa, fb, fs = HERE / "path_a", HERE / "path_b", HERE / "path_s"
    keep_p = fs / "keep_pairs.txt"
    if not keep_p.is_file():
        raise SystemExit("missing path_s/keep_pairs.txt — 先跑 make_sparse_holdout.py")
    keep, hold = load_keep(keep_p)

    true_p = HERE / "syn_true_all.dat"
    if not true_p.is_file():
        true_p = fa / "syn_holdout_true.dat"
    true_picks = parse_picks(true_p.read_text(encoding="utf-8"))
    rows = match_rows(true_picks, keep)
    if not rows:
        raise SystemExit("no matched PPP/PPS/PSS/PSP rows")

    for folder, key in ((fa, "fwd_a"), (fb, "fwd_b"), (fs, "fwd_s")):
        hp = folder / "syn_holdout_rec.dat"
        if hp.is_file():
            attach_fwd(rows, parse_picks(hp.read_text(encoding="utf-8")))
            for r in rows:
                r[key] = r.get("fwd", float("nan"))
        else:
            for r in rows:
                r[key] = float("nan")

    kr = [r for r in rows if r["keep"]]
    hr = [r for r in rows if not r["keep"]]
    xy_k = [(r["xm"], r["dx"]) for r in kr]
    d_corr_k = [r["psp"] - r["corr"] for r in kr]
    xy_h = [(r["xm"], r["dx"]) for r in hr]
    d_corr_h = interp_residual(xy_k, d_corr_k, xy_h) if hr else np.array([])
    for r, d in zip(hr, d_corr_h):
        r["corr_i"] = r["corr"] + float(d)
    for r in kr:
        r["corr_i"] = r["psp"]

    if hr and all(math.isfinite(r["fwd_s"]) for r in kr + hr):
        d_s_k = [r["psp"] - r["fwd_s"] for r in kr]
        d_s_h = interp_residual(xy_k, d_s_k, xy_h)
        for r, d in zip(hr, d_s_h):
            r["fwd_s_i"] = r["fwd_s"] + float(d)
        for r in kr:
            r["fwd_s_i"] = r["psp"]
    else:
        for r in rows:
            r["fwd_s_i"] = float("nan")

    print(f"pairs  keep {len(kr)}  holdout {len(hr)}  all {len(rows)}")
    print("holdout PSP vs true  (smaller is better)")
    methods = (
        ("corr  PSS-(PPS-PPP)", [r["corr"] for r in hr]),
        ("corr + d interp", [r["corr_i"] for r in hr]),
        ("A  PSS-only fwd", [r["fwd_a"] for r in hr]),
        ("S  A+sparsePSP fwd", [r["fwd_s"] for r in hr]),
        ("S fwd + r interp", [r["fwd_s_i"] for r in hr]),
        ("B  full PSP fwd", [r["fwd_b"] for r in hr]),
    )
    for lab, pred in methods:
        d = [p - r["psp"] for p, r in zip(pred, hr) if math.isfinite(p)]
        print(f"  {lab:22s}  n={len(d):3d}  RMS {_rms(d):.4f} s")

    xs, zs, _, _, _, t_vs, _, _ = load_vp_vs(fa)
    print("Vs rec vs true")
    for lab, folder in (("A", fa), ("S", fs), ("B", fb)):
        if (folder / "rec_vs.smesh").is_file():
            _, _, _, _, _, t_vs_f, _, r_vs = load_vp_vs(folder)
            _print_vs(lab, xs, zs, r_vs, t_vs_f)
        else:
            print(f"  {lab:18s}  no rec_vs")

    for lab, folder, phases in (
        ("A", fa, ((0, "PPP"), (7, "PPS"), (8, "PSS"))),
        ("S", fs, ((6, "PSP"), (8, "PSS"))),
        ("B", fb, ((6, "PSP"),)),
    ):
        hold_t, hold_r = folder / "syn_holdout_true.dat", folder / "syn_holdout_rec.dat"
        if hold_t.is_file() and hold_r.is_file():
            print(f"{lab}  holdout all vs true")
            for code, name in PHASES:
                print(f"  {name:3s}({code})  {ttrms(hold_t, hold_r, code):.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.8), facecolor="w", layout="constrained")
    _, _, _, _, _, t_vs_a, _, r_vs_a = load_vp_vs(fa)
    r_vs_s = r_vs_b = None
    if (fs / "rec_vs.smesh").is_file():
        _, _, _, _, _, _, _, r_vs_s = load_vp_vs(fs)
    if (fb / "rec_vs.smesh").is_file():
        _, _, _, _, _, t_vs_b, _, r_vs_b = load_vp_vs(fb)
    else:
        t_vs_b = t_vs_a
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))

    def _vs_title(name, rec):
        if rec is None:
            return f"{name} 无 rec_vs"
        return (
            f"{name} rec-true Vs\n"
            f"lid {region_rms(xs, zs, rec, t_vs_a, lid=True):.3f}  "
            f"below {region_rms(xs, zs, rec, t_vs_a, lid=False):.3f}"
        )

    vs_panels = (
        (axes[0, 0], r_vs_a, _vs_title("A  仅PSS", r_vs_a)),
        (axes[0, 1], r_vs_s, _vs_title("S  PSS+15%PSP", r_vs_s)),
        (axes[0, 2], r_vs_b, _vs_title("B  全PSP", r_vs_b)),
    )
    last = None
    for ax, rec, title in vs_panels:
        grid = (rec - t_vs_a) if rec is not None else t_vs_a * np.nan
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
        _style(ax, title)
    fig.colorbar(last, ax=axes[0, :].ravel().tolist(), shrink=0.82, label="dVs (km/s)")
    axes[0, 0].set_ylabel("深度 (km)")

    axm = axes[1, 0]
    axm.plot([r["sx"] for r in hr], [r["src"] for r in hr], "o", color="0.75", ms=4, label="holdout")
    axm.plot([r["sx"] for r in kr], [r["src"] for r in kr], "D", color="#c44e8a", ms=5, label="keep PSP")
    axm.set_xlabel("台 x (km)")
    axm.set_ylabel("炮 x (km)")
    axm.set_title(f"保留 {len(kr)} / {len(rows)} 对")
    axm.grid(True, alpha=0.3)
    axm.legend(fontsize=8, loc="best")

    axr = axes[1, 1]
    axr.axhline(0.0, color="0.45", lw=0.8)
    series = (
        ("corr", [r["corr"] - r["psp"] for r in hr], "0.55", "s"),
        ("corr+d", [r["corr_i"] - r["psp"] for r in hr], "#ff7f0e", "D"),
        ("A fwd", [r["fwd_a"] - r["psp"] for r in hr], "#1f77b4", "o"),
        ("S fwd", [r["fwd_s"] - r["psp"] for r in hr], "#c44e8a", "^"),
        ("S+r", [r["fwd_s_i"] - r["psp"] for r in hr], "#2ca02c", "v"),
        ("B fwd", [r["fwd_b"] - r["psp"] for r in hr], "#8c564b", "P"),
    )
    for lab, d, c, m in series:
        if not any(math.isfinite(v) for v in d):
            continue
        axr.plot(
            [r["dx"] for r in hr],
            d,
            m,
            color=c,
            ms=4,
            alpha=0.85,
            label=f"{lab} {_rms(d):.3f}s",
        )
    axr.set_xlabel("offset dx (km)")
    axr.set_ylabel("pred - true PSP (s)")
    axr.grid(True, alpha=0.3)
    axr.legend(fontsize=7, loc="best", ncol=2)
    axr.set_title("holdout PSP 残差")

    ax1 = axes[1, 2]
    lims = [r["psp"] for r in hr]
    lo, hi = min(lims) - 0.3, max(lims) + 0.3
    ax1.plot([lo, hi], [lo, hi], "k--", lw=0.7)
    for lab, pred, c, m in (
        ("corr+d", [r["corr_i"] for r in hr], "#ff7f0e", "D"),
        ("S fwd", [r["fwd_s"] for r in hr], "#c44e8a", "^"),
        ("S+r", [r["fwd_s_i"] for r in hr], "#2ca02c", "v"),
        ("B", [r["fwd_b"] for r in hr], "#8c564b", "P"),
    ):
        if not any(math.isfinite(v) for v in pred):
            continue
        ax1.plot(lims, pred, m, color=c, ms=4, alpha=0.85, label=lab)
    ax1.set_xlabel("true PSP (s)")
    ax1.set_ylabel("predicted (s)")
    ax1.set_title("holdout 1:1")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.invert_yaxis()
    ax1.invert_xaxis()

    fig.suptitle(
        "稀疏 PSP holdout    A: 仅PSS    S: PSS+15%真PSP    B: 全PSP    橙=校正插值"
    )
    out = HERE / "check_sparse.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
