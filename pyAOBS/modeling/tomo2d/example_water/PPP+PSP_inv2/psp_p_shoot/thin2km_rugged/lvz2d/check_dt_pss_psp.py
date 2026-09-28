#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PPP+PPS 收回模型上正演 PSS-PSP，对真 PSS-PSP。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inv_2d"))
sys.path.insert(0, str(ROOT.parents[1]))
sys.path.insert(0, str(ROOT.parents[1].parent / "water_inv"))
from check_sparse import _rms  # noqa: E402
from check_water_inv import parse_picks  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _pmap(picks, code: int):
    out = {}
    for p in picks:
        if int(p[0]) != code or not math.isfinite(p[3]):
            continue
        out[(round(p[4], 3), round(p[1], 3))] = float(p[3])
    return out


def _rows(true_picks, pred_picks):
    tm = {c: _pmap(true_picks, c) for c in (0, 6, 7, 8)}
    pm = {c: _pmap(pred_picks, c) for c in (0, 6, 7, 8)}
    keys = sorted(set(tm[6]) & set(tm[8]) & set(pm[6]) & set(pm[8]) & set(tm[0]) & set(tm[7]))
    rows = []
    for src, sx in keys:
        dt_true = tm[8][(src, sx)] - tm[6][(src, sx)]
        dt_fwd = pm[8][(src, sx)] - pm[6][(src, sx)]
        dt_pps = tm[7][(src, sx)] - tm[0][(src, sx)]
        dt_pps_f = (
            pm[7][(src, sx)] - pm[0][(src, sx)]
            if (src, sx) in pm[0] and (src, sx) in pm[7]
            else float("nan")
        )
        rows.append(
            dict(
                src=src,
                sx=sx,
                dx=abs(sx - src),
                dt_true=dt_true,
                dt_fwd=dt_fwd,
                dt_pps=dt_pps,
                dt_pps_f=dt_pps_f,
            )
        )
    return rows


def _stat(ds):
    ds = [float(x) for x in ds if math.isfinite(x)]
    if not ds:
        return "n=0"
    mean = sum(ds) / len(ds)
    mx = max(ds, key=abs)
    return f"n={len(ds):3d}  RMS {_rms(ds):.4f} s  mean {mean:+.4f} s  maxabs {abs(mx):.4f} s"


def main() -> int:
    true_p = HERE / "syn_true_all.dat"
    srcs = (
        ("PPP Vp + PPS/PPP Vs", HERE / "syn_lidvs_all.dat"),
        ("PPP Vp + Vp/1.73", HERE / "syn_pppvp.dat"),
        ("A  PSS below", HERE / "path_a" / "syn_holdout_rec.dat"),
        ("S  sparse PSP", HERE / "path_s" / "syn_holdout_rec.dat"),
        ("B  full PSP", HERE / "path_b" / "syn_holdout_rec.dat"),
    )
    true_picks = parse_picks(true_p.read_text(encoding="utf-8"))
    print("dT = PSS - PSP   (fwd - true)")
    lid_rows = None
    for lab, path in srcs:
        if not path.is_file():
            print(f"  {lab:22s}  missing {path.name}")
            continue
        pred = parse_picks(path.read_text(encoding="utf-8"))
        rows = _rows(true_picks, pred)
        if lid_rows is None and "PPS/PPP" in lab:
            lid_rows = rows
        print(f"  {lab:22s}  {_stat([r['dt_fwd'] - r['dt_true'] for r in rows])}")
        if "PPS/PPP" in lab:
            print(f"    vs  obs PPS-PPP         {_stat([r['dt_pps'] - r['dt_true'] for r in rows])}")
            print(
                f"    fwd (PPS-PPP)-(PSS-PSP) {_stat([r['dt_pps_f'] - r['dt_fwd'] for r in rows])}"
            )
            bins = ((0, 10), (10, 20), (20, 30), (30, 99))
            for lo, hi in bins:
                br = [r for r in rows if lo <= r["dx"] < hi]
                print(
                    f"    offset [{lo},{hi})  {_stat([r['dt_fwd'] - r['dt_true'] for r in br])}"
                )

    if not lid_rows:
        raise SystemExit("missing syn_lidvs_all.dat — run run_fwd_lidvs.sh")

    _plot_dt(lid_rows, HERE / "check_dt_pss_psp.png", "PPP Vp + PPS/PPP Vs  正演 PSS-PSP  vs  真 PSS-PSP")
    far = [r for r in lid_rows if r["dx"] > 20.0]
    _plot_dt(
        far,
        HERE / "check_dt_pss_psp_far.png",
        f"offset > 20 km  n={len(far)}  正演 PSS-PSP vs 真值",
    )
    return 0


def _plot_dt(rows, out: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0), facecolor="w", layout="constrained")
    dx = [r["dx"] for r in rows]
    ax = axes[0]
    ax.plot(dx, [r["dt_true"] for r in rows], "o", color="#2ca02c", ms=5, label="true PSS-PSP")
    ax.plot(dx, [r["dt_fwd"] for r in rows], "D", color="#c44e8a", ms=4.5, label="fwd on PPP+PPS")
    ax.plot(dx, [r["dt_pps"] for r in rows], "s", color="0.55", ms=4, alpha=0.8, label="obs PPS-PPP")
    ax.set_xlabel("offset dx (km)")
    ax.set_ylabel("dT (s)")
    ax.set_title("PSS-PSP")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    axr = axes[1]
    d_lid = [r["dt_fwd"] - r["dt_true"] for r in rows]
    d_pps = [r["dt_pps"] - r["dt_true"] for r in rows]
    axr.axhline(0.0, color="0.45", lw=0.8)
    axr.plot(dx, d_pps, "s", color="0.55", ms=4.5, alpha=0.8, label=f"obs PPS-PPP {_rms(d_pps):.3f}s")
    axr.plot(dx, d_lid, "D", color="#c44e8a", ms=5, label=f"fwd PPP+PPS {_rms(d_lid):.3f}s")
    axr.set_xlabel("offset dx (km)")
    axr.set_ylabel("dT - true (s)")
    axr.set_title("residual")
    axr.grid(True, alpha=0.3)
    axr.legend(fontsize=8)

    ax1 = axes[2]
    tt = [r["dt_true"] for r in rows]
    lo, hi = min(tt) - 0.05, max(tt) + 0.05
    ax1.plot([lo, hi], [lo, hi], "k--", lw=0.7)
    ax1.plot(tt, [r["dt_fwd"] for r in rows], "D", color="#c44e8a", ms=5, label="fwd PPP+PPS")
    ax1.plot(tt, [r["dt_pps"] for r in rows], "s", color="0.55", ms=4, alpha=0.8, label="obs PPS-PPP")
    ax1.set_xlabel("true PSS-PSP (s)")
    ax1.set_ylabel("predicted (s)")
    ax1.set_title("1:1")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    fig.suptitle(title)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
