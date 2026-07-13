"""
Reproduce Katz et al. (2003) Fig. 3: solidus for different bulk H2O (T–P).

Paper bulk H2O (wt%): 0, 0.05, 0.1, 0.3 and water-saturated solidus.

  python petrology/validation/reproduce_katz2003_fig3.py
  python petrology/validation/reproduce_katz2003_fig3.py --bulk-wt 0.05,0.1,0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.validation._katz_cli import default_fig_path, parse_float_list, write_figure

FIG_DEFAULT = default_fig_path(3)
def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Katz (2003) Fig. 3")
    parser.add_argument("-o", "--out", type=Path, default=FIG_DEFAULT)
    parser.add_argument("--p-max", type=float, default=8.0)
    parser.add_argument("--t-min", type=float, default=850.0, help="T axis min (°C); hydrous solidus reaches ~916°C")
    parser.add_argument("--t-max", type=float, default=2000.0)
    parser.add_argument(
        "--bulk-wt",
        default="0.05,0.1,0.3",
        help="Bulk source H2O in wt%% (comma-separated); 0 wt%% dry curve always plotted",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    from petrology.melting.katz2003 import build_katz2003_fig3_figure, katz2003_hydrous_solidus_t

    bulk = parse_float_list(args.bulk_wt)
    fig = build_katz2003_fig3_figure(
        p_max_gpa=args.p_max,
        t_min_c=args.t_min,
        t_max_c=args.t_max,
        bulk_h2o_wt_pct=bulk,
    )
    print("  Curves: 0 wt% (dry), " + ", ".join(f"{x:g} wt%" for x in bulk) + ", water saturated")
    for pg in (1.0, 3.0, 6.0):
        print(
            f"  P={pg:g} GPa solidus T (°C): dry={katz2003_hydrous_solidus_t(pg):.0f}  "
            + "  ".join(
                f"{b:g}%={katz2003_hydrous_solidus_t(pg, bulk_h2o_wt_pct=b):.0f}"
                for b in bulk
            )
            + f"  sat={katz2003_hydrous_solidus_t(pg, saturated=True):.0f}"
        )

    write_figure(fig, args.out, show=args.show)


if __name__ == "__main__":
    main()
