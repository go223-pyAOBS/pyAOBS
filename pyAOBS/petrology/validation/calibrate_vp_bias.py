"""
Calibrate Vp bias for Repro / Modern tracks vs KKHS02 Step-2 bulk bounds.

Usage::

  py -3.11 petrology/validation/calibrate_vp_bias.py
  py -3.11 petrology/validation/calibrate_vp_bias.py --tp 1425 --chi 10
  py -3.11 petrology/validation/calibrate_vp_bias.py --sweep-feasible --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.melting.vp_bias_calibrate import (
    auto_calibrate_track_bias,
    calibrate_track_at_tp_chi,
    sweep_feasible_vp_bias,
)


def _print_rec(rec) -> None:
    b = rec.bounds
    print(
        f"  {rec.track}: raw Vp={rec.vp_raw_km_s:.3f}  "
        f"bias={rec.vp_bias_km_s:+.3f}  -> Vp={rec.vp_calibrated_km_s:.3f} km/s"
    )
    print(
        f"    ref Tp={rec.tp_c:.0f} chi={rec.chi:g}  "
        f"bounds=[{b.v_bulk_lower_km_s:.3f}, {b.v_bulk_upper_km_s:.3f}]  target={rec.target}"
    )


def _sweep_feasible(
    *,
    track: str,
    melting_engine: str,
    lithology_backend: str,
    vp_mode: str,
    peridotite_chemistry: str | None,
    v_lc: float,
    h_obs: float,
    b_km: float,
    tp_range: tuple[float, float],
    tp_step: float,
    chi_values: list[float],
    bias_values: list[float],
    h_tolerance: float,
    n_isentropic_steps: int,
) -> tuple[float, int]:
    return sweep_feasible_vp_bias(
        melting_engine=melting_engine,
        lithology_backend=lithology_backend,
        vp_mode=vp_mode,
        peridotite_chemistry=peridotite_chemistry,
        v_lc_obs_km_s=v_lc,
        h_obs_km=h_obs,
        b_km=b_km,
        tp_range_c=tp_range,
        tp_step_c=tp_step,
        chi_values=chi_values,
        bias_values=bias_values,
        h_tolerance_km=h_tolerance,
        n_isentropic_steps=n_isentropic_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Vp bias for HVP scans")
    parser.add_argument("--h", type=float, default=30.0)
    parser.add_argument("--vlc", type=float, default=7.0)
    parser.add_argument("--b", type=float, default=0.0)
    parser.add_argument("--tp", type=float, default=1425.0, help="Reference Tp for point calibration")
    parser.add_argument("--chi", type=float, default=10.0, help="Reference chi for point calibration")
    parser.add_argument(
        "--target",
        choices=("bound_mid", "bound_upper", "bound_lower"),
        default="bound_mid",
    )
    parser.add_argument("--margin", type=float, default=0.02, help="Margin inside bound (km/s)")
    parser.add_argument("--sweep-feasible", action="store_true", help="Sweep bias for max feasible count")
    parser.add_argument(
        "--strategy",
        choices=("point", "sweep"),
        default="point",
        help="Bias recommendation for CLI hints (sweep uses grid max feasible)",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("-o", type=Path, default=None, help="Write JSON recommendations")
    args = parser.parse_args()

    v_lc, h_obs, b_km = float(args.vlc), float(args.h), float(args.b)
    tp_ref, chi_ref = float(args.tp), float(args.chi)

    print(f"# Vp bias calibration (V_LC={v_lc:g} km/s, H={h_obs:g} km, ref Tp={tp_ref:g} chi={chi_ref:g})\n")

    tracks = [
        ("repro_reebox", "reebox", "native", "norm", None),
        ("modern_reebox", "reebox", "pymelt", "norm", "pmelts_klb1"),
        ("repro_linear", "kinzler_linear", "native", "eq1", None),
    ]

    recs = []
    for name, engine, backend, vp_mode, chem in tracks:
        rec = calibrate_track_at_tp_chi(
            track=name,
            tp_c=tp_ref,
            chi=chi_ref,
            v_lc_obs_km_s=v_lc,
            b_km=b_km,
            melting_engine=engine,
            lithology_backend=backend,
            vp_mode=vp_mode,
            peridotite_chemistry=chem,
            target=args.target,
            margin_km_s=float(args.margin),
            n_isentropic_steps=48,
        )
        recs.append(rec)
        _print_rec(rec)

    out = {
        "v_lc_obs_km_s": v_lc,
        "h_obs_km": h_obs,
        "reference_tp_c": tp_ref,
        "reference_chi": chi_ref,
        "target": args.target,
        "tracks": {
            r.track: {
                "vp_bias_km_s": r.vp_bias_km_s,
                "vp_raw_km_s": r.vp_raw_km_s,
                "vp_calibrated_km_s": r.vp_calibrated_km_s,
                "bounds_lower": r.bounds.v_bulk_lower_km_s,
                "bounds_upper": r.bounds.v_bulk_upper_km_s,
            }
            for r in recs
        },
    }

    if args.sweep_feasible:
        print("\n## Bias sweep (max feasible, full H+Vp bounds)\n")
        tp_step = 25.0 if args.quick else 15.0
        tp_range = (1200.0, 1500.0)
        chi_values = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        bias_grid = [round(float(x), 2) for x in np.arange(-1.0, -0.04, 0.05)]

        sweep_out = {}
        for name, engine, backend, vp_mode, chem in tracks[:2]:
            best_bias, best_n = _sweep_feasible(
                track=name,
                melting_engine=engine,
                lithology_backend=backend,
                vp_mode=vp_mode,
                peridotite_chemistry=chem,
                v_lc=v_lc,
                h_obs=h_obs,
                b_km=b_km,
                tp_range=tp_range,
                tp_step=tp_step,
                chi_values=chi_values,
                bias_values=bias_grid,
                h_tolerance=3.0,
                n_isentropic_steps=48,
            )
            sweep_out[name] = {"best_bias_km_s": best_bias, "n_feasible": best_n}
            print(f"  {name}: best bias={best_bias:+.2f}  n_feasible={best_n}")

        out["sweep_feasible"] = sweep_out

    if args.o:
        args.o.parent.mkdir(parents=True, exist_ok=True)
        args.o.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nWrote {args.o}")

    print("\n# Use with compare_tracks_hvp_sensitivity.py:")
    if args.sweep_feasible and "sweep_feasible" in out:
        sr = out["sweep_feasible"]["repro_reebox"]["best_bias_km_s"]
        sm = out["sweep_feasible"]["modern_reebox"]["best_bias_km_s"]
        print(f"  --auto-bias   # default strategy=sweep")
        print(f"  --repro-vp-bias {sr:+.2f} --modern-vp-bias {sm:+.2f}")
    else:
        rr = out["tracks"]["repro_reebox"]["vp_bias_km_s"]
        mr = out["tracks"]["modern_reebox"]["vp_bias_km_s"]
        print(f"  --auto-bias --auto-bias-strategy point")
        print(f"  --repro-vp-bias {rr:+.3f} --modern-vp-bias {mr:+.3f}")


if __name__ == "__main__":
    main()
