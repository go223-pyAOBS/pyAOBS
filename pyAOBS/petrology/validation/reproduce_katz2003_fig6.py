"""
Reproduce Katz et al. (2003) Fig. 6 — validation figure (model + Jaques 1980 experiments).

Acceptance scope: four isobaric panels (0/1/1.5/3 GPa); panels (b)(c) show
Jaques & Green (1980) Pyrolite + Tinaquillo from experiments.csv. Sufficient
to verify katz2003 Table 2 melting code against published F(T) constraints.

  python petrology/validation/reproduce_katz2003_fig6.py
  python petrology/validation/reproduce_katz2003_fig6.py --show
  python petrology/validation/reproduce_katz2003_fig6.py --no-experiments
  python petrology/validation/reproduce_katz2003_fig6.py --all-studies
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.validation._katz_cli import default_fig_path, parse_float_list, write_figure

FIG_DEFAULT = default_fig_path(6)
def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Katz (2003) Fig. 6 (model + experiments)")
    parser.add_argument("-o", "--out", type=Path, default=FIG_DEFAULT)
    parser.add_argument(
        "--pressures",
        default="0,1,1.5,3",
        help="Isobaric panel pressures (GPa), comma-separated (four values)",
    )
    parser.add_argument(
        "--cpx-mass",
        type=float,
        default=0.15,
        help="Modal cpx mass fraction (Fig. 6 caption: 0.15)",
    )
    parser.add_argument("--f-max", type=float, default=0.7)
    parser.add_argument(
        "--no-experiments",
        action="store_true",
        help="Draw model curves only (no scatter from experiments.csv)",
    )
    parser.add_argument(
        "--all-studies",
        action="store_true",
        help="Plot every non-verified=no row in experiments.csv (default: Jaques 1980 only)",
    )
    parser.add_argument(
        "--study-ids",
        default="",
        help="Comma-separated study_id filter (overrides default JG1980_HP)",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    from petrology.melting.katz2003 import (
        build_katz2003_fig6_figure,
        katz2003_fcpx_out,
        katz2003_solidus,
        katz2003_tcpx_out,
    )

    pressures = parse_float_list(args.pressures)
    if len(pressures) != 4:
        parser.error("Fig. 6 expects exactly four pressures (four panels)")

    if args.study_ids.strip():
        study_ids = tuple(s.strip() for s in args.study_ids.split(",") if s.strip())
    elif args.all_studies:
        study_ids = None
    else:
        study_ids = ("JG1980_HP", "JG1980_TQ")

    fig = build_katz2003_fig6_figure(
        pressures_gpa=pressures,
        cpx_mass=args.cpx_mass,
        f_max=args.f_max,
        show_experiments=not args.no_experiments,
        experiment_study_ids=study_ids,
    )
    if not args.no_experiments:
        print(
            "Fig.6 validation: Katz (2003) model + Jaques & Green (1980) "
            f"({', '.join(study_ids) if study_ids else 'all studies'})"
        )
    for pg in pressures:
        fc = katz2003_fcpx_out(p_gpa=pg, cpx_mass=args.cpx_mass)
        tc = katz2003_tcpx_out(p_gpa=pg, cpx_mass=args.cpx_mass)
        print(
            f"  P={pg:g} GPa: Tsol={katz2003_solidus(pg):.1f} °C  "
            f"F_cpx-out={fc:.3f}  T_cpx-out={tc:.1f} °C"
        )

    write_figure(fig, args.out, show=args.show)


if __name__ == "__main__":
    main()
