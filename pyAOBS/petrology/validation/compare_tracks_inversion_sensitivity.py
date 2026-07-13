"""
Track-level sensitivity: how Repro vs Modern (Pbar, Fbar, Vp) differ at fixed Tp, chi.

Complements ``compare_tracks_melting.py`` with a coarse Fig.3 eq.(1) context and
optional ``reproduce_fig3`` RMS summary.

Usage::

  py -3.11 petrology/validation/compare_tracks_inversion_sensitivity.py
  py -3.11 petrology/validation/compare_tracks_inversion_sensitivity.py --fig3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _column(tp: float, chi: float, b: float, *, backend: str):
    from petrology.melting.heterogeneous import forward_heterogeneous_column

    chem = "pmelts_klb1" if backend == "pymelt" else "kinzler1997"
    return forward_heterogeneous_column(
        tp_c=tp,
        b_km=b,
        chi=chi,
        pyroxenite_frac=0.0,
        lithology_backend=backend,
        peridotite_lith="katz_lherzolite",
        peridotite_chemistry=chem if backend == "pymelt" else None,
        compute_norm_vp=True,
    )


def main() -> None:
    from petrology.vp_regression import predict_vp_km_s

    parser = argparse.ArgumentParser(description="Repro vs Modern inversion sensitivity")
    parser.add_argument("--tp", type=float, default=1450.0)
    parser.add_argument("--chi", type=float, default=8.0)
    parser.add_argument("--b", type=float, default=0.0)
    parser.add_argument("--fig3", action="store_true", help="Print reproduce_fig3 RMS summary")
    args = parser.parse_args()

    tp, chi, b = float(args.tp), float(args.chi), float(args.b)
    repro = _column(tp, chi, b, backend="native")
    modern = _column(tp, chi, b, backend="pymelt")

    print(f"# Inversion drivers (Tp={tp:g} C, chi={chi:g}, b={b:g} km)\n")
    hdr = f"{'track':<10} {'P0':>6} {'Pf':>6} {'Pbar':>6} {'Fbar':>6} {'H_km':>6} {'Vp_eq1':>8} {'Vp_norm':>8}"
    print(hdr)
    print("-" * len(hdr))
    for tag, col in (("repro", repro), ("modern", modern)):
        vn = col.vp_bulk_norm_km_s
        vn_s = f"{vn:8.3f}" if vn is not None else "     n/a"
        print(
            f"{tag:<10} {col.p0_gpa:6.2f} {col.pf_gpa:6.2f} {col.pbar_gpa:6.2f} "
            f"{col.fbar:6.3f} {col.h_km:6.1f} {col.vp_bulk_eq1_km_s:8.3f} {vn_s}"
        )

    vp_r = predict_vp_km_s(repro.pbar_gpa, repro.fbar)
    vp_m = predict_vp_km_s(modern.pbar_gpa, modern.fbar)
    print(f"\ndPbar = {modern.pbar_gpa - repro.pbar_gpa:+.2f} GPa")
    print(f"dFbar = {modern.fbar - repro.fbar:+.3f}")
    print(f"dVp_eq1 (modern - repro) = {vp_m - vp_r:+.3f} km/s")
    if repro.vp_bulk_norm_km_s is not None and modern.vp_bulk_norm_km_s is not None:
        print(f"dVp_norm (modern - repro) = {modern.vp_bulk_norm_km_s - repro.vp_bulk_norm_km_s:+.3f} km/s")

    print(
        "\nNote: KKHS02 inversion uses delta Vp vs MORB; a +0.3 km/s column Vp shift"
        "\n      can move inferred Tp by tens of C — see compare_tracks_hvp_sensitivity.py"
    )

    if args.fig3:
        from petrology.validation.reproduce_fig3 import run as run_fig3

        print("\n# Fig.3 eq.(1) catalog regression (reproduce_fig3)\n")
        summary = run_fig3()
        print(f"  n={summary['n_points']} RMS={summary['rms_km_s']:.3f} km/s")
        print(f"  mean residual={summary['mean_residual_km_s']:+.3f} km/s")
        print(f"  paper MORB Vp pred={summary['paper_eq1_morb_km_s']:.3f} km/s")
        if "eq1_catalog_bias_km_s" in summary:
            print(f"  catalog eq.(1) bias={summary['eq1_catalog_bias_km_s']:+.3f} km/s")
            print(f"  calibrated RMS={summary['rms_calibrated_km_s']:.3f} km/s")
            print(f"  calibrated MORB Vp={summary['paper_eq1_morb_calibrated_km_s']:.3f} km/s")


if __name__ == "__main__":
    main()
