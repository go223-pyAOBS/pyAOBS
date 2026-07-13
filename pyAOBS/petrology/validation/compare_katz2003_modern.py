"""
Compare Katz (2003) native (Table 2) vs Modern LIP pyMelt katz_lherzolite for Fig. 1–6.

Modern here = peridotite melting backend used in LIP GUI (``pymelt`` + ``katz_lherzolite``),
not the full REEBOX column forward model.

  python petrology/validation/compare_katz2003_modern.py
  python petrology/validation/compare_katz2003_modern.py --only fig2,fig6 --show
  python petrology/validation/compare_katz2003_modern.py --katz-cpx-mass 0.17
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
OUT_NAMES = {
    "fig1": "compare_katz_modern_fig1.png",
    "fig2": "compare_katz_modern_fig2.png",
    "fig3": "compare_katz_modern_fig3.png",
    "fig4": "compare_katz_modern_fig4.png",
    "fig5": "compare_katz_modern_fig5.png",
    "fig6": "compare_katz_modern_fig6.png",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay Katz native vs Modern pyMelt for Fig. 1–6"
    )
    parser.add_argument("-o", "--out-dir", type=Path, default=FIG_DIR)
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated: fig1,fig2,...,fig6 (default: all)",
    )
    parser.add_argument(
        "--katz-cpx-mass",
        type=float,
        default=0.15,
        help="Mcpx for native Katz curves in Fig. 2/4/6 (pyMelt fixed at 0.17)",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    from petrology.melting import katz_modern_compare as cmp
    from petrology.melting.katz_modern_compare import COMPARE_BUILDERS

    keys = list(COMPARE_BUILDERS.keys())
    if args.only.strip():
        keys = [k.strip() for k in args.only.split(",") if k.strip()]

    cmp.print_compare_summary(katz_cpx_mass=args.katz_cpx_mass)
    print()

    import matplotlib.pyplot as plt

    args.out_dir.mkdir(parents=True, exist_ok=True)
    last_fig = None
    for key in keys:
        builder = COMPARE_BUILDERS[key]
        if key in ("fig2", "fig4", "fig6"):
            fig = builder(katz_cpx_mass=args.katz_cpx_mass)
        else:
            fig = builder()
        out = args.out_dir / OUT_NAMES[key]
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Wrote {out}")
        last_fig = fig
        if not args.show:
            plt.close(fig)

    if args.show and last_fig is not None:
        plt.show()


if __name__ == "__main__":
    main()
