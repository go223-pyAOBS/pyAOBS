"""
Reproduce Katz et al. (2003) Fig. 9: dry melting parameterization comparison.

  python petrology/validation/reproduce_katz2003_fig9.py
  python petrology/validation/reproduce_katz2003_fig9.py --show
  python petrology/validation/reproduce_katz2003_fig9.py --models katz2003,mckenzie1988,pmelts2002
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.validation._katz_cli import default_fig_path, write_figure

FIG_DEFAULT = default_fig_path(9)
def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Katz (2003) Fig. 9")
    parser.add_argument("-o", "--out", type=Path, default=FIG_DEFAULT)
    parser.add_argument("--t-min", type=float, default=1200.0)
    parser.add_argument("--t-max", type=float, default=1800.0)
    parser.add_argument("--f-max", type=float, default=0.5)
    parser.add_argument(
        "--cpx-mass",
        type=float,
        default=None,
        help="Modal cpx for Katz curve (default: KATZ_CPX_MASS=0.17)",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model keys (default: paper = MB88 + pMELTS)",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Include Katz (2003) overlay (FIG9_EXTENDED_MODEL_ORDER)",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    from petrology.melting.katz2003 import KATZ_CPX_MASS, build_katz2003_fig9_figure
    from petrology.melting.katz2003_fig9_models import (
        FIG9_EXTENDED_MODEL_ORDER,
        FIG9_MODEL_ORDER,
        FIG9_PAPER_MODEL_ORDER,
        build_fig9_dry_melt_models,
    )

    model_keys: tuple[str, ...] | None = None
    if args.extended:
        model_keys = FIG9_EXTENDED_MODEL_ORDER
    if args.models.strip():
        model_keys = tuple(k.strip() for k in args.models.split(",") if k.strip())

    fig = build_katz2003_fig9_figure(
        t_min_c=args.t_min,
        t_max_c=args.t_max,
        f_max=args.f_max,
        cpx_mass=args.cpx_mass,
        model_keys=model_keys,
    )
    models = build_fig9_dry_melt_models(cpx_mass=args.cpx_mass)
    keys = model_keys or FIG9_MODEL_ORDER
    for key in keys:
        m = models[key]
        tag = "" if m.verified == "yes" else f" [{m.verified}]"
        print(f"  {key}{tag}: {m.label}")

    write_figure(fig, args.out, show=args.show)


if __name__ == "__main__":
    main()
