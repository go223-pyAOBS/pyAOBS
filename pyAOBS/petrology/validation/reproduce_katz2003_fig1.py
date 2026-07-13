"""
Reproduce Katz et al. (2003) Fig. 1: anhydrous solidus, lherzolite liquidus, liquidus.

Uses Table 2 coefficients via ``petrology.melting.katz2003`` (eq. 4, 5, 10).

For batch export prefer::

  python petrology/validation/katz2003_workflow.py --only katz_fig1

  python petrology/validation/reproduce_katz2003_fig1.py
  python petrology/validation/reproduce_katz2003_fig1.py --show
  python petrology/validation/reproduce_katz2003_fig1.py -o petrology/figures/katz2003_fig1.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.validation._katz_cli import default_fig_path, write_figure

FIG_DEFAULT = default_fig_path(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Katz (2003) Fig. 1")
    parser.add_argument("-o", "--out", type=Path, default=FIG_DEFAULT)
    parser.add_argument("--p-max", type=float, default=8.0, help="Max pressure (GPa), y-axis bottom")
    parser.add_argument("--t-min", type=float, default=1100.0, help="Temperature axis min (°C)")
    parser.add_argument("--t-max", type=float, default=2000.0, help="Temperature axis max (°C)")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    from petrology.melting.katz2003 import (
        build_katz2003_fig1_figure,
        katz2003_lherzolite_liquidus,
        katz2003_liquidus,
        katz2003_solidus,
    )

    fig = build_katz2003_fig1_figure(
        p_max_gpa=args.p_max,
        t_min_c=args.t_min,
        t_max_c=args.t_max,
    )

    for pg in (0.0, 1.0, 2.0, 3.0, 4.0):
        print(
            f"  P={pg:.1f} GPa: Tsol={katz2003_solidus(pg):.1f}  "
            f"T_lliq={katz2003_lherzolite_liquidus(pg):.1f}  "
            f"T_liq={katz2003_liquidus(pg):.1f} °C"
        )

    write_figure(fig, args.out, show=args.show)


if __name__ == "__main__":
    main()
