#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""面下 Vs 真/假 对 PSS-PSP 时差的影响。盖层都是 PPS/PPP 收回。"""

from __future__ import annotations

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
from check_dt_pss_psp import _rows, _stat  # noqa: E402
from check_joint import mask_crust, region_rms  # noqa: E402
from check_lvz import load_vp_vs  # noqa: E402
from check_sparse import _rms  # noqa: E402
from check_water_inv import parse_picks  # noqa: E402
import inv_grid as g  # noqa: E402
import make_ppp_psp_inv_case as m2  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _vs_rms(path: Path, t_vs, xs, zs) -> str:
    _, _, vel = m2.parse_smesh(path)
    rec = mask_crust(xs, zs, vel)
    return (
        f"lid {region_rms(xs, zs, rec, t_vs, lid=True):.4f}  "
        f"below {region_rms(xs, zs, rec, t_vs, lid=False):.4f}"
    )


def main() -> int:
    true_picks = parse_picks((HERE / "syn_true_all.dat").read_text(encoding="utf-8"))
    srcs = (
        ("lid rec / below start", HERE / "syn_lidvs_all.dat", "#c44e8a", "D"),
        ("lid rec / below TRUE", HERE / "syn_truebelow_all.dat", "#2ca02c", "o"),
        ("true Vs all", HERE / "syn_truevs_all.dat", "#1f77b4", "^"),
    )
    packed = []
    print("dT = PSS-PSP  (fwd - true)")
    for lab, path, col, mk in srcs:
        if not path.is_file():
            print(f"  {lab:24s}  missing {path.name}")
            continue
        rows = _rows(true_picks, parse_picks(path.read_text(encoding="utf-8")))
        packed.append((lab, rows, col, mk))
        print(f"  {lab:24s}  all      {_stat([r['dt_fwd'] - r['dt_true'] for r in rows])}")
        far = [r for r in rows if r["dx"] > 20.0]
        print(f"  {lab:24s}  >20 km   {_stat([r['dt_fwd'] - r['dt_true'] for r in far])}")

    xs, zs, _, _, _, t_vs, _, _ = load_vp_vs(HERE / "path_a")
    print("Vs vs true")
    for lab, p in (
        ("rec_vs_lid", HERE / "path_a" / "rec_vs_lid.smesh"),
        ("lid+truebelow", HERE / "vs_lid_truebelow.smesh"),
        ("true_vs", HERE / "true_vs.smesh"),
    ):
        if p.is_file():
            print(f"  {lab:16s}  {_vs_rms(p, t_vs, xs, zs)}")

    if len(packed) < 2:
        raise SystemExit("need syn_lidvs_all.dat and syn_truebelow_all.dat")

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1), facecolor="w", layout="constrained")
    ax, axr, axf = axes
    for lab, rows, col, mk in packed:
        ax.plot(
            [r["dx"] for r in rows],
            [r["dt_fwd"] for r in rows],
            mk,
            color=col,
            ms=3.5,
            alpha=0.85,
            label=lab,
        )
    ax.plot(
        [r["dx"] for r in packed[0][1]],
        [r["dt_true"] for r in packed[0][1]],
        "x",
        color="0.25",
        ms=3.5,
        label="true",
    )
    ax.set_xlabel("offset dx (km)")
    ax.set_ylabel("PSS-PSP (s)")
    ax.set_title("dT")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    axr.axhline(0.0, color="0.45", lw=0.8)
    axf.axhline(0.0, color="0.45", lw=0.8)
    for lab, rows, col, mk in packed:
        d = [r["dt_fwd"] - r["dt_true"] for r in rows]
        axr.plot(
            [r["dx"] for r in rows],
            d,
            mk,
            color=col,
            ms=3.5,
            alpha=0.85,
            label=f"{lab} {_rms(d):.3f}s",
        )
        far = [r for r in rows if r["dx"] > 20.0]
        df = [r["dt_fwd"] - r["dt_true"] for r in far]
        axf.plot(
            [r["dx"] for r in far],
            df,
            mk,
            color=col,
            ms=5,
            label=f"{lab} {_rms(df):.3f}s",
        )
    axr.set_xlabel("offset dx (km)")
    axr.set_ylabel("dT - true (s)")
    axr.set_title("residual  all")
    axr.grid(True, alpha=0.3)
    axr.legend(fontsize=7)
    axf.set_xlabel("offset dx (km)")
    axf.set_ylabel("dT - true (s)")
    axf.set_title("residual  offset > 20 km")
    axf.grid(True, alpha=0.3)
    axf.legend(fontsize=7)

    fig.suptitle("盖层=PPS/PPP收回   面下=热初值 vs 真Vs    对 PSS-PSP 时差")
    out = HERE / "check_dt_truebelow.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
