"""
Reproduce Katz et al. (2003) Fig. 4: isobaric hydrous F(T) at 1 GPa.

Paper bulk H2O (wt%): 0, 0.02, 0.05, 0.1, 0.3 — 0.3 wt% is saturated at the solidus.

  python petrology/validation/reproduce_katz2003_fig4.py
  python petrology/validation/reproduce_katz2003_fig4.py --p-gpa 1 --bulk-wt 0,0.02,0.05,0.1,0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.validation._katz_cli import default_fig_path, parse_float_list, write_figure

FIG_DEFAULT = default_fig_path(4)
def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Katz (2003) Fig. 4")
    parser.add_argument("-o", "--out", type=Path, default=FIG_DEFAULT)
    parser.add_argument("--p-gpa", type=float, default=1.0)
    parser.add_argument("--t-min", type=float, default=900.0)
    parser.add_argument("--t-max", type=float, default=1400.0)
    parser.add_argument("--f-max", type=float, default=0.4, help="Y-axis melt fraction limit")
    parser.add_argument(
        "--bulk-wt",
        default="0,0.02,0.05,0.1,0.3",
        help="Bulk source H2O in wt%% (comma-separated)",
    )
    parser.add_argument(
        "--cpx-mass",
        type=float,
        default=0.15,
        help="Modal cpx mass fraction (0.15 paper; errata 2023: 0.17)",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    from petrology.melting.katz2003 import (
        build_katz2003_fig4_figure,
        katz2003_hydrous_solidus_t,
        katz2003_melt_fraction_hydrous,
        katz2003_x_melt_h2o_at_f,
        katz2003_x_sat,
    )

    bulk = parse_float_list(args.bulk_wt)
    fig = build_katz2003_fig4_figure(
        p_gpa=args.p_gpa,
        t_min_c=args.t_min,
        t_max_c=args.t_max,
        f_max=args.f_max,
        bulk_h2o_wt_pct=bulk,
        cpx_mass=args.cpx_mass,
    )
    print(f"  P = {args.p_gpa:g} GPa")
    xsat = katz2003_x_sat(args.p_gpa)
    print(f"  X_sat @ P = {xsat:.2f} wt%")
    for b in bulk:
        ts = katz2003_hydrous_solidus_t(args.p_gpa, bulk_h2o_wt_pct=b)
        x0 = katz2003_x_melt_h2o_at_f(bulk_h2o_wt_pct=b, f=0.0, p_gpa=args.p_gpa)
        sat_tag = " (saturated at solidus)" if b > 0 and x0 >= xsat - 1e-9 else ""
        f_mid = katz2003_melt_fraction_hydrous(args.p_gpa, ts + 50.0, bulk_h2o_wt_pct=b)
        print(
            f"  bulk {b:g} wt%: T_solidus={ts:.0f}°C  x(F=0)={x0:.2f} wt%{sat_tag}  "
            f"F(T_s+50°C)={f_mid:.3f}"
        )

    write_figure(fig, args.out, show=args.show)


if __name__ == "__main__":
    main()
