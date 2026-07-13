"""
Greenland transect anchor scan (KKHS02 Fig.15c discussion).

Observation anchor: H ~ 30 km, V_LC ~ 7.0 km/s, b = 0.

Runs REEBOX-core (Tp, chi) grid + optional Phi sweep; compares legacy kinzler_linear.

  python petrology/validation/greenland_anchor_scan.py
  python petrology/validation/greenland_anchor_scan.py --no-phi --h-tol 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"


def main() -> None:
    parser = argparse.ArgumentParser(description="Greenland anchor: H + V_LC scan (Modern REEBOX track)")
    parser.add_argument("--v-lc", type=float, default=7.0)
    parser.add_argument("--h-km", type=float, default=30.0)
    parser.add_argument("--b-km", type=float, default=0.0)
    parser.add_argument("--h-tol", type=float, default=5.0, help="|H-H_obs| tolerance (km); use 3 for strict")
    parser.add_argument("--tp-min", type=float, default=1200.0)
    parser.add_argument("--tp-max", type=float, default=1450.0)
    parser.add_argument("--tp-step", type=float, default=25.0)
    parser.add_argument("--no-phi", action="store_true", help="Skip Phi sweep")
    parser.add_argument("--legacy", action="store_true", help="Also run kinzler_linear comparison")
    parser.add_argument(
        "--lithology-preset",
        type=str,
        default=None,
        help="pyMelt lithology preset (e.g. greenland_kg1, lip_klb1_hydrous)",
    )
    parser.add_argument("--out-dir", type=Path, default=FIG_DIR)
    args = parser.parse_args()

    lith_kw: dict = {}
    if args.lithology_preset:
        from petrology.melting.pymelt_lithology_adapter import resolve_lithology_col_kwargs

        lith_kw = resolve_lithology_col_kwargs(lithology_preset=args.lithology_preset)

    import importlib.util

    def _load_run(name: str, filename: str):
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).resolve().parent / filename
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.run

    scan_run = _load_run("scan_hvp_lip", "scan_hvp_lip.py")
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("Greenland anchor — REEBOX-core (Tp, chi)")
    print(f"  H={args.h_km:.0f} km  V_LC={args.v_lc:.2f} km/s  b={args.b_km:.0f} km")
    print("=" * 64)
    scan_run(
        v_lc_obs_km_s=args.v_lc,
        h_obs_km=args.h_km,
        b_km=args.b_km,
        tp_range_c=(args.tp_min, args.tp_max),
        tp_step_c=args.tp_step,
        h_tolerance_km=args.h_tol,
        melting_engine="reebox",
        refine_norm_vp=8,
        export_csv=out / "greenland_scan_reebox.csv",
        save_figure=out / "greenland_scan_reebox.png",
        **lith_kw,
    )

    if not args.no_phi:
        phi_run = _load_run("sweep_phi_lip", "sweep_phi_lip.py")

        print("\n" + "=" * 64)
        print("Greenland anchor — Phi sweep (REEBOX)")
        print("=" * 64)
        phi_run(
            v_lc_obs_km_s=args.v_lc,
            h_obs_km=args.h_km,
            b_km=args.b_km,
            tp_range_c=(args.tp_min, min(args.tp_max, 1400.0)),
            tp_step_c=args.tp_step,
            h_tolerance_km=args.h_tol,
            melting_engine="reebox",
            save_figure=out / "greenland_phi_sweep.png",
            **lith_kw,
        )

    if args.legacy:
        print("\n" + "=" * 64)
        print("Greenland anchor — legacy kinzler_linear (reproduction track)")
        print("=" * 64)
        scan_run(
            v_lc_obs_km_s=args.v_lc,
            h_obs_km=args.h_km,
            b_km=args.b_km,
            tp_range_c=(1250.0, 1400.0),
            tp_step_c=10.0,
            h_tolerance_km=min(args.h_tol, 3.0),
            melting_engine="kinzler_linear",
            delta_vp_engine="wl1990",
            refine_norm_vp=0,
            require_bulk_in_bounds=False,
            save_figure=out / "greenland_scan_kkhs02.png",
        )


if __name__ == "__main__":
    main()
