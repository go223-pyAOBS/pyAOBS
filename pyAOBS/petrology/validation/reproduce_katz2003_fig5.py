"""
Reproduce Katz et al. (2003) Fig. 5: F vs bulk / melt H2O at fixed P, T.

  (a) F vs bulk H2O (wt%) — compare to Gaetani & Grove (1998) Fig. 13a
  (b) F vs dissolved H2O in melt (D_H2O = 0.01)

Default: P = 1.5 GPa, T = 1200/1250/1300/1350°C, modal cpx = 17 wt%.

  python petrology/validation/reproduce_katz2003_fig5.py
  python petrology/validation/reproduce_katz2003_fig5.py --temperatures 1200,1250,1300,1350
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.validation._katz_cli import default_fig_path, parse_float_list, write_figure

FIG_DEFAULT = default_fig_path(5)
def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Katz (2003) Fig. 5")
    parser.add_argument("-o", "--out", type=Path, default=FIG_DEFAULT)
    parser.add_argument("--p-gpa", type=float, default=1.5)
    parser.add_argument(
        "--temperatures",
        default="1200,1250,1300,1350",
        help="Isotherms (°C), comma-separated",
    )
    parser.add_argument("--bulk-max", type=float, default=0.5, help="Bulk H2O sweep max (wt%%)")
    parser.add_argument("--bulk-x-max", type=float, default=0.25, help="Panel (a) x-axis max (wt%%)")
    parser.add_argument("--f-max-a", type=float, default=0.25, help="Panel (a) y-axis max")
    parser.add_argument("--melt-x-max", type=float, default=5.0, help="Panel (b) x-axis max (wt%%)")
    parser.add_argument("--f-max-b", type=float, default=0.2, help="Panel (b) y-axis max")
    parser.add_argument("--cpx-mass", type=float, default=0.17)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    import numpy as np

    from petrology.melting.katz2003 import build_katz2003_fig5_figure, katz2003_fig5_melt_vs_water

    temps = parse_float_list(args.temperatures)
    fig = build_katz2003_fig5_figure(
        p_gpa=args.p_gpa,
        temperatures_c=temps,
        bulk_h2o_max_wt_pct=args.bulk_max,
        cpx_mass=args.cpx_mass,
        bulk_x_max=args.bulk_x_max,
        f_max_a=args.f_max_a,
        melt_x_max=args.melt_x_max,
        f_max_b=args.f_max_b,
    )
    print(f"  P = {args.p_gpa:g} GPa  cpx = {100 * args.cpx_mass:g} wt%  T = {', '.join(f'{t:g}' for t in temps)} °C")

    bulk = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    for t_c in temps:
        b, f, x = katz2003_fig5_melt_vs_water(
            p_gpa=args.p_gpa,
            t_c=t_c,
            bulk_h2o_wt_pct=bulk,
            cpx_mass=args.cpx_mass,
        )
        print(f"  T={t_c:g}°C: F(dry)={f[0]:.3f}  F@0.2%={f[3]:.3f}  F@0.5%={f[-1]:.3f}")

    write_figure(fig, args.out, show=args.show)


if __name__ == "__main__":
    main()
