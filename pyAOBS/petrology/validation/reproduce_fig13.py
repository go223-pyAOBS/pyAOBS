"""KKHS02 Fig.13-style H–Vp sensitivity (melting-law variants)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.gui.hvp_plot import (
    FIG12_CHI1_B_VALUES_KM,
    FIG12_H_LIM_KM,
    FIG12_SOLID_CHI_VALUES,
    FIG12_TP_CONTOUR_INTERVAL_C,
    FIG12_VP_LIM_KM_S,
    build_fig12_linear_curves,
    plot_hvp_fig12_style,
)
from petrology.melting_laws import FIG12_MELTING, FIG13A_MELTING, FIG13B_MELTING, MeltingLaw


def run(
    *,
    tp_range_c: tuple[float, float] = (1200.0, 1600.0),
    tp_step_c: float = 10.0,
    vp_bias_km_s: float = 0.0,
    save_figure: Path | None = None,
    show: bool = False,
) -> dict:
    tp_vals = np.arange(tp_range_c[0], tp_range_c[1] + tp_step_c * 0.5, tp_step_c)

    panels: tuple[tuple[str, MeltingLaw, str], ...] = (
        ("(a) Fig.12 baseline — (dF/dP)_S = 12%/GPa", FIG12_MELTING, "12%/GPa"),
        ("(b) Fig.13a — (dF/dP)_S = 16%/GPa", FIG13A_MELTING, "16%/GPa"),
        ("(c) Fig.13b — Asimow (1997) 3-stage F(ΔP)", FIG13B_MELTING, "Asimow 3-stage"),
    )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0), sharey=True)
    summary: dict = {"panels": []}

    for ax, (title, law, tag) in zip(axes, panels):
        main, chi1_b = build_fig12_linear_curves(
            tp_values_c=tp_vals,
            solid_chi_values=FIG12_SOLID_CHI_VALUES,
            chi1_b_values_km=FIG12_CHI1_B_VALUES_KM,
            melting_law=law,
            vp_bias_km_s=vp_bias_km_s,
        )
        plot_hvp_fig12_style(
            ax,
            main,
            chi1_b_lines=chi1_b,
            tp_range_c=tp_range_c,
            tp_contour_interval_c=FIG12_TP_CONTOUR_INTERVAL_C,
            label_fontsize=7.0,
            legend_prefix="",
            use_paper_axis_limits=True,
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlim(*FIG12_H_LIM_KM)
        ax.set_ylim(*FIG12_VP_LIM_KM_S)
        if main:
            hs = [p.h_km for p in main]
            vs = [p.vp_bulk_km_s for p in main]
            summary["panels"].append(
                {
                    "tag": tag,
                    "n": len(main),
                    "h_km": (float(min(hs)), float(max(hs))),
                    "vp_km_s": (float(min(vs)), float(max(vs))),
                }
            )

    axes[0].set_ylabel("Bulk Vp (km/s) @ 600 MPa, 400°C")
    for ax in axes:
        ax.set_xlabel("Igneous crustal thickness H (km)")
    fig.suptitle(
        "KKHS02 Fig.13 — melting-law sensitivity (same curves as Fig.12a)",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()

    if save_figure:
        save_figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_figure, dpi=150, bbox_inches="tight")
        print(f"Saved {save_figure}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    for row in summary.get("panels", []):
        print(
            f"{row['tag']}: n={row['n']}  H={row['h_km'][0]:.1f}–{row['h_km'][1]:.1f} km  "
            f"Vp={row['vp_km_s'][0]:.3f}–{row['vp_km_s'][1]:.3f} km/s"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce KKHS02 Fig.13 H–Vp sensitivity")
    parser.add_argument("--tp-min", type=float, default=1200.0)
    parser.add_argument("--tp-max", type=float, default=1600.0)
    parser.add_argument("--tp-step", type=float, default=10.0)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures" / "reproduce_fig13_mvp.png",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    run(
        tp_range_c=(args.tp_min, args.tp_max),
        tp_step_c=args.tp_step,
        vp_bias_km_s=0.0,
        save_figure=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
