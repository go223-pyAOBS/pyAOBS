"""
Reproduce Katz et al. (2003) Fig. 2: isobaric anhydrous F(T) curves.

Batch export: ``python petrology/validation/katz2003_workflow.py --only katz_fig2``

  python petrology/validation/reproduce_katz2003_fig2.py
  python petrology/validation/reproduce_katz2003_fig2.py --cpx-mass 0.17
  python petrology/validation/reproduce_katz2003_fig2.py --pressures 1,2,3 --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.validation._katz_cli import default_fig_path, parse_float_list, write_figure

FIG_DEFAULT = default_fig_path(2)
def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Katz (2003) Fig. 2")
    parser.add_argument("-o", "--out", type=Path, default=FIG_DEFAULT)
    parser.add_argument("--pressures", default="1,2,3", help="Isobaric pressures (GPa), comma-separated")
    parser.add_argument("--t-min", type=float, default=1000.0)
    parser.add_argument("--t-max", type=float, default=2000.0)
    parser.add_argument(
        "--cpx-mass",
        type=float,
        default=0.15,
        help="Modal cpx mass fraction (0.15 paper; errata 2023: 0.17)",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    from petrology.melting.katz2003 import (
        build_katz2003_fig2_figure,
        katz2003_fcpx_out,
        katz2003_tcpx_out,
    )

    pressures = parse_float_list(args.pressures)
    fig = build_katz2003_fig2_figure(
        pressures_gpa=pressures,
        t_min_c=args.t_min,
        t_max_c=args.t_max,
        cpx_mass=args.cpx_mass,
    )
    for pg in pressures:
        fc = katz2003_fcpx_out(p_gpa=pg, cpx_mass=args.cpx_mass)
        tc = katz2003_tcpx_out(p_gpa=pg, cpx_mass=args.cpx_mass)
        print(f"  P={pg:g} GPa: F_cpx-out={fc:.3f}  T_cpx-out={tc:.1f} °C")

    write_figure(fig, args.out, show=args.show)


if __name__ == "__main__":
    main()
