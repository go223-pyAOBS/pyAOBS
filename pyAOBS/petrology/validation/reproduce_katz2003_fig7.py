"""
Reproduce Katz et al. (2003) Fig. 7 — experimental F(P,T) vs Katz / Hirschmann solidus.

  python petrology/validation/reproduce_katz2003_fig7.py
  python petrology/validation/reproduce_katz2003_fig7.py --build-data --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.validation._katz_cli import default_fig_path, write_figure

FIG_DEFAULT = default_fig_path(7)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Katz (2003) Fig. 7")
    parser.add_argument("-o", "--out", type=Path, default=FIG_DEFAULT)
    parser.add_argument("--p-max", type=float, default=9.0, help="Max pressure (GPa)")
    parser.add_argument("--t-min", type=float, default=1000.0)
    parser.add_argument("--t-max", type=float, default=2000.0)
    parser.add_argument("--f-cmap-max", type=float, default=0.5, help="Colorbar F max")
    parser.add_argument(
        "--build-data",
        action="store_true",
        help="Rebuild fig7/experiments.csv from Fig.6 + Jaques before plotting",
    )
    parser.add_argument("--no-experiments", action="store_true", help="Solidus curves only")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.build_data:
        from petrology.validation.import_katz2003_fig7_data import build_fig7_experiments

        build_fig7_experiments()

    from petrology.melting.katz2003 import (
        build_katz2003_fig7_figure,
        katz2003_fig7_near_hirschmann_stats,
        load_katz2003_fig7_experiments,
    )

    fig = build_katz2003_fig7_figure(
        p_max_gpa=args.p_max,
        t_min_c=args.t_min,
        t_max_c=args.t_max,
        f_cmap_max=args.f_cmap_max,
        show_experiments=not args.no_experiments,
    )
    rows = load_katz2003_fig7_experiments()
    stats = katz2003_fig7_near_hirschmann_stats(rows)
    print(f"  Experiments plotted: {len(rows)}")
    print(
        f"  ±{stats['dt_c']:.0f}°C of Hirschmann solidus: n={stats['n_near']:.0f}, "
        f"mean F={stats['mean_f_wt_pct']:.1f} wt%, F=0: {stats['n_zero']:.0f}"
    )
    print("  (Paper: n=29, mean F=10 wt%, 5 with F=0 — full ESI Table S2 pending)")

    write_figure(fig, args.out, show=args.show)


if __name__ == "__main__":
    main()
