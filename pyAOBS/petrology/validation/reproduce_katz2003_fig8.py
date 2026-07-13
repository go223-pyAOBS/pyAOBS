"""
Reproduce Katz et al. (2003) Fig. 8 — eq. (16) ΔT(X) and eq. (17) X_sat(P) calibration.

  python petrology/validation/reproduce_katz2003_fig8.py
  python petrology/validation/reproduce_katz2003_fig8.py --no-calibration --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.validation._katz_cli import default_fig_path, write_figure

FIG_DEFAULT = default_fig_path(8)
def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Katz (2003) Fig. 8")
    parser.add_argument("-o", "--out", type=Path, default=FIG_DEFAULT)
    parser.add_argument("--x-max", type=float, default=60.0, help="Panel (a) X_H2O max (wt%)")
    parser.add_argument("--dt-max", type=float, default=900.0, help="Panel (a) delta T max (C)")
    parser.add_argument("--p-max", type=float, default=8.0, help="Panel (b) P max (GPa)")
    parser.add_argument("--xsat-max", type=float, default=50.0, help="Panel (b) X_sat max (wt%)")
    parser.add_argument("--no-calibration", action="store_true", help="Model curves only")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    from petrology.melting.katz2003 import (
        build_katz2003_fig8_figure,
        katz2003_table2_hydrous_constants,
    )

    const = katz2003_table2_hydrous_constants()
    fig = build_katz2003_fig8_figure(
        x_h2o_max=args.x_max,
        delta_t_max=args.dt_max,
        p_max_gpa=args.p_max,
        x_sat_max=args.xsat_max,
        show_calibration=not args.no_calibration,
    )
    print(
        f"  Table 2: K={const['K']:g}, gamma={const['gamma']:g}, "
        f"chi1={const['chi1']:g}, chi2={const['chi2']:g}, lambda={const['plam']:g}"
    )

    write_figure(fig, args.out, show=args.show)


if __name__ == "__main__":
    main()
