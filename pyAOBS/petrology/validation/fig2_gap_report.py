"""Quantify Fig.2 reproduction gap: ours vs KKHS02 proxy anchors."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.fc.cdat_library import FIG2_PAPER_SAMPLE, fig2_primary_melt, list_samples
from petrology.fc.wl1990 import simulate_crystallization_path

PAPER = {
    0.0: {"vp": 7.57, "fo": 90, "an": 80, "ol": 100, "cpx": 0},
    0.15: {"fo": 88, "an": 77, "ol": 68, "cpx": 1},
    0.25: {"vp": 7.55, "fo": 85, "an": 73, "ol": 55, "cpx": 3},
    0.45: {"fo": 81, "an": 69, "ol": 47, "cpx": 8},
    0.50: {"vp": 7.42, "ol": 47, "cpx": 8},
    0.60: {"fo": 72, "an": 62, "ol": 40, "cpx": 15},
    0.70: {"vp": 7.34, "fo": 63, "an": 57, "ol": 35, "cpx": 20},
    0.80: {"vp": 7.31, "fo": 58, "an": 54, "ol": 33, "cpx": 20},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="KKHS02 Fig.2 gap report (fc_100, fig2 backend)")
    parser.add_argument(
        "--sample",
        type=str,
        default=FIG2_PAPER_SAMPLE,
        help="CDAT catalog sample (paper anchors assume kinzler1997_morb_primary)",
    )
    parser.add_argument("--data", type=Path, default=None, help="CDAT file or catalog id")
    parser.add_argument("--list-samples", action="store_true")
    args = parser.parse_args()

    if args.list_samples:
        for s in list_samples():
            print(f"{s.id:28s}  {s.name}  [{', '.join(s.tags)}]")
        return

    primary = fig2_primary_melt(sample_id=args.sample, data_path=args.data)
    if primary["id"] != FIG2_PAPER_SAMPLE:
        print(
            f"Note: PAPER anchors are for {FIG2_PAPER_SAMPLE}; "
            f"running with {primary['id']} ({primary['name']}).\n"
        )

    f = np.array(sorted(PAPER.keys()))
    st = simulate_crystallization_path(
        primary_melt_oxides_wt=primary["oxides_wt_percent"],
        f_grid=f,
        path="fc_100",
        mineral_backend="fig2",
        cipw_backend="fallback",
        fig2_ab_calibrate=True,
    )
    by_f = {s.f_solid: s for s in st}

    print(f"=== KKHS02 Fig.2 gap analysis — {primary['name']} (fc_100, fig2 backend) ===\n")
    rows = []
    for fv in f:
        s = by_f[fv]
        ref = PAPER[fv]
        row = {"F": fv}
        if "vp" in ref:
            row["dVp"] = s.vp_cumulate_km_s - ref["vp"]
        if "fo" in ref:
            row["dFo"] = s.cum_fo_pct - ref["fo"]
        if "an" in ref:
            row["dAn"] = s.cum_an_pct - ref["an"]
        if "ol" in ref:
            row["dOl%"] = 100 * s.cum_ol - ref["ol"]
        if "cpx" in ref:
            row["dCpx%"] = 100 * s.cum_cpx - ref["cpx"]
        rows.append(row)

    def _rms(key: str) -> float:
        vals = [r[key] for r in rows if key in r]
        return float(np.sqrt(np.mean(np.array(vals) ** 2)))

    print(f"{'F':>4}  {'dVp':>6}  {'dFo':>5}  {'dAn':>5}  {'dOl%':>5}  {'dCpx%':>6}")
    for r in rows:
        print(
            f"{r['F']:4.2f}  "
            f"{r.get('dVp', float('nan')):6.3f}  "
            f"{r.get('dFo', float('nan')):5.1f}  "
            f"{r.get('dAn', float('nan')):5.1f}  "
            f"{r.get('dOl%', float('nan')):5.1f}  "
            f"{r.get('dCpx%', float('nan')):6.1f}"
        )

    print("\nRMS misfit:")
    for k, label in [
        ("dVp", "Vp (km/s)"),
        ("dFo", "Fo"),
        ("dAn", "An"),
        ("dOl%", "cum Ol (%)"),
        ("dCpx%", "cum Cpx (%)"),
    ]:
        print(f"  {label:12s}: {_rms(k):.3f}")

    print("\n--- Root cause hints ---")
    s25 = by_f[0.25]
    print(f"F=0.25 inc modes: Ol={s25.inc_ol:.2f} Pl={s25.inc_pl:.2f} Cpx={s25.inc_cpx:.2f}")
    melt = by_f[0.25].melt_oxides_wt
    an_m = 100 * melt["CaO"] / (melt["CaO"] + melt["Na2O"])
    print(f"F=0.25 inc An={s25.an_pct:.1f} (paper inc ~72); melt An_m≈{an_m:.0f}%")
    s15 = by_f[0.15]
    print(f"F=0.15 cum Cpx={100*s15.cum_cpx:.1f}% (paper ~1%) — Cpx too early")


if __name__ == "__main__":
    main()
