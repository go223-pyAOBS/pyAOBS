"""
Reproduce KKHS02 Fig.12a (standard H–Vp diagram) from digitized paper curves.

Usage::

    py -3.11 petrology/validation/reproduce_fig12a_digitized.py
    py -3.11 petrology/validation/reproduce_fig12a_digitized.py --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.validation.fig12a_background import (
    DEFAULT_HVP_DIGITIZED,
    H_LIM,
    VP_LIM,
    draw_fig12a_background,
    parse_hvp_digitized,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"


def run(
    *,
    data_file: Path = DEFAULT_HVP_DIGITIZED,
    save_figure: Path | None = None,
    show: bool = False,
    h_lim: tuple[float, float] = H_LIM,
    vp_lim: tuple[float, float] = VP_LIM,
) -> dict:
    if not data_file.is_file():
        raise FileNotFoundError(data_file)

    all_series = parse_hvp_digitized(data_file)
    summary = {
        "n_series": len(all_series),
        "n_track": sum(1 for s in all_series if s.kind == "track"),
        "n_tp": sum(1 for s in all_series if s.kind == "tp"),
        "data_file": str(data_file),
    }
    print(f"Loaded {summary['n_series']} curves from {data_file.name}")
    print(f"  tracks: {summary['n_track']}, Tp contours: {summary['n_tp']}")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib required: {exc}") from exc

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    draw_fig12a_background(ax, all_series, h_lim=h_lim, vp_lim=vp_lim)
    ax.set_xlabel("Igneous crustal thickness $H$ (km)")
    ax.set_ylabel(r"Bulk crustal $V_p$ (km/s) @ 600 MPa, 400°C")
    ax.set_title("KKHS02 Fig.12a — standard H–Vp diagram (digitized + smooth fit)")
    ax.grid(True, ls=":", lw=0.35, alpha=0.4)
    fig.tight_layout()

    if save_figure:
        save_figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_figure, dpi=160)
        print(f"Saved {save_figure}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fig.12a from digitized H-Vp curves")
    parser.add_argument("--data", type=Path, default=DEFAULT_HVP_DIGITIZED)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=FIG_DIR / "fig12a_standard_hvp_digitized.png",
    )
    parser.add_argument("--h-min", type=float, default=H_LIM[0])
    parser.add_argument("--h-max", type=float, default=H_LIM[1])
    parser.add_argument("--vp-min", type=float, default=VP_LIM[0])
    parser.add_argument("--vp-max", type=float, default=VP_LIM[1])
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    run(
        data_file=args.data,
        save_figure=args.output,
        show=args.show,
        h_lim=(args.h_min, args.h_max),
        vp_lim=(args.vp_min, args.vp_max),
    )


if __name__ == "__main__":
    main()
