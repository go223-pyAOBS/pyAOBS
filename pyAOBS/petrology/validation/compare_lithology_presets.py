"""
Compare Greenland-like anchor feasibility across lithology presets.

  python petrology/validation/compare_lithology_presets.py
  python petrology/validation/compare_lithology_presets.py --presets lip_default,greenland_kg1,lip_ball
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
    parser = argparse.ArgumentParser(description="Compare lithology presets for H + V_LC scan")
    parser.add_argument("--v-lc", type=float, default=7.0)
    parser.add_argument("--h-km", type=float, default=30.0)
    parser.add_argument("--b-km", type=float, default=0.0)
    parser.add_argument("--pyroxenite-frac", type=float, default=0.10)
    parser.add_argument("--h-tol", type=float, default=5.0)
    parser.add_argument("--tp-min", type=float, default=1200.0)
    parser.add_argument("--tp-max", type=float, default=1400.0)
    parser.add_argument("--tp-step", type=float, default=25.0)
    parser.add_argument("--chi", type=str, default="4,6,8,10,12,16")
    parser.add_argument("--presets", type=str, default="", help="Comma list; default = all presets + native")
    parser.add_argument("--no-native", action="store_true")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=FIG_DIR / "compare_lithology_presets.png",
    )
    args = parser.parse_args()

    from petrology.gui.preset_compare import compare_lithology_presets, format_preset_compare_table

    preset_list = [x.strip() for x in args.presets.split(",") if x.strip()] or None
    chi_vals = tuple(float(x) for x in args.chi.split(",") if x.strip())

    print("=" * 72)
    print(f"Lithology preset compare — H={args.h_km:.0f} km  V_LC={args.v_lc:.2f} km/s  Φ={args.pyroxenite_frac:.2f}")
    print("=" * 72)

    rows = compare_lithology_presets(
        v_lc_obs_km_s=args.v_lc,
        h_obs_km=args.h_km,
        b_km=args.b_km,
        pyroxenite_frac=args.pyroxenite_frac,
        presets=preset_list,
        tp_range_c=(args.tp_min, args.tp_max),
        tp_step_c=args.tp_step,
        chi_values=chi_vals,
        h_tolerance_km=args.h_tol,
        include_native=not args.no_native,
        verbose=True,
    )
    print(format_preset_compare_table(rows))

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable ({exc})")
        return

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45 * len(rows))))
    names = [r.preset.replace("__native__", "native") for r in rows]
    counts = [r.n_feasible for r in rows]
    colors = ["#2ca02c" if c > 0 else "#c44e52" for c in counts]
    ax.barh(names, counts, color=colors)
    ax.set_xlabel("Feasible (Tp, chi) grid points")
    ax.set_title(f"H={args.h_km:.0f} km, V_LC={args.v_lc:.2f} km/s, |dH|<={args.h_tol} km")
    ax.invert_yaxis()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"\nSaved {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
