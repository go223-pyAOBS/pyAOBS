"""Reproduce H–Vp diagram (KKHS02 Fig.12-style computed track)."""

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
    FIG12_SOLID_CHI_VALUES,
    build_fig12_linear_curves,
    plot_hvp_fig12_style,
)


def run(
    *,
    b_km: float = 0.0,
    tp_range_c: tuple[float, float] = (1200.0, 1600.0),
    tp_step_c: float = 10.0,
    tp_contour_interval_c: float = 50.0,
    chi_values: tuple[float, ...] = FIG12_SOLID_CHI_VALUES,
    chi1_b_values_km: tuple[float, ...] = FIG12_CHI1_B_VALUES_KM,
    dfdp_per_gpa: float = 0.12,
    vp_bias_km_s: float = 0.0,
    h_lim_km: tuple[float, float] = (3.0, 35.0),
    vp_lim_km_s: tuple[float, float] = (6.9, 7.35),
    save_figure: Path | None = None,
    show: bool = False,
) -> dict:
    del b_km  # solids always at b=0; chi1 family uses chi1_b_values_km
    tp_vals = np.arange(tp_range_c[0], tp_range_c[1] + tp_step_c * 0.5, tp_step_c)
    res_main, chi1_b_pairs = build_fig12_linear_curves(
        tp_values_c=tp_vals,
        solid_chi_values=chi_values,
        chi1_b_values_km=chi1_b_values_km,
        dfdp_per_gpa=dfdp_per_gpa,
        vp_bias_km_s=vp_bias_km_s,
    )
    res_chi1_b = [p for _b, pts in chi1_b_pairs for p in pts]
    res = list(res_main) + res_chi1_b
    if len(res_main) == 0:
        raise RuntimeError("No valid H-Vp points produced")

    hs = np.array([r.h_km for r in res], dtype=float)
    vs = np.array([r.vp_bulk_km_s for r in res], dtype=float)
    fs = np.array([r.fbar for r in res], dtype=float)

    summary = {
        "n_points": len(res),
        "h_range_km": (float(hs.min()), float(hs.max())),
        "vp_range_km_s": (float(vs.min()), float(vs.max())),
        "fbar_range": (float(fs.min()), float(fs.max())),
        "chi_values": [float(x) for x in chi_values],
        "chi1_b_values_km": [float(x) for x in chi1_b_values_km],
        "tp_contour_interval_c": float(tp_contour_interval_c),
        "dfdp_per_gpa": float(dfdp_per_gpa),
        "vp_bias_km_s": float(vp_bias_km_s),
    }

    print(f"H-Vp points: {summary['n_points']}")
    print(f"H range: {summary['h_range_km'][0]:.2f} – {summary['h_range_km'][1]:.2f} km")
    print(f"Vp range: {summary['vp_range_km_s'][0]:.3f} – {summary['vp_range_km_s'][1]:.3f} km/s")
    print(f"Fbar range: {summary['fbar_range'][0]:.3f} – {summary['fbar_range'][1]:.3f}")
    print(f"Applied dF/dP: {summary['dfdp_per_gpa']:.3f} 1/GPa")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable ({exc}) — skipping figure")
        return summary

    # Temporarily override paper axis constants used by plot helper
    import petrology.gui.hvp_plot as hvp_plot

    old_h, old_v = hvp_plot.FIG12_H_LIM_KM, hvp_plot.FIG12_VP_LIM_KM_S
    hvp_plot.FIG12_H_LIM_KM = h_lim_km
    hvp_plot.FIG12_VP_LIM_KM_S = vp_lim_km_s
    try:
        fig, ax = plt.subplots(figsize=(7.4, 5.2))
        plot_hvp_fig12_style(
            ax,
            res_main,
            chi1_b_lines=chi1_b_pairs,
            tp_range_c=tp_range_c,
            tp_contour_interval_c=tp_contour_interval_c,
            label_fontsize=7.0,
            use_paper_axis_limits=True,
        )
        ax.set_xlabel("Igneous crustal thickness H (km)")
        ax.set_ylabel("Bulk crustal Vp (km/s) @ 600 MPa, 400°C")
        ax.set_title("H–Vp diagram (eq.(1); method from KKHS02 Fig.12)")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()

        if save_figure:
            save_figure.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_figure, dpi=150)
            print(f"Saved {save_figure}")
        if show:
            plt.show()
        else:
            plt.close(fig)
    finally:
        hvp_plot.FIG12_H_LIM_KM = old_h
        hvp_plot.FIG12_VP_LIM_KM_S = old_v
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce H–Vp diagram (KKHS02 Fig.12-style eq.(1) track)"
    )
    parser.add_argument("--b-km", type=float, default=0.0, help="Unused (solids at b=0)")
    parser.add_argument("--tp-min", type=float, default=1200.0)
    parser.add_argument("--tp-max", type=float, default=1600.0)
    parser.add_argument("--tp-step", type=float, default=10.0)
    parser.add_argument(
        "--tp-contour-interval",
        type=float,
        default=50.0,
        help="Mantle potential temperature contour interval in °C (paper uses 50°C)",
    )
    parser.add_argument(
        "--chi",
        type=str,
        default="1,2,4,8",
        help="Comma-separated chi values for b=0 curves",
    )
    parser.add_argument(
        "--chi1-b",
        type=str,
        default="10,20,30",
        help="Comma-separated b values (km) for chi=1 dashed curves",
    )
    parser.add_argument(
        "--dfdp",
        type=float,
        default=0.12,
        help="Mean dF/dP used by the active-upwelling model (1/GPa)",
    )
    parser.add_argument("--h-min", type=float, default=3.0, help="X-axis lower limit (km)")
    parser.add_argument("--h-max", type=float, default=35.0, help="X-axis upper limit (km)")
    parser.add_argument("--vp-min", type=float, default=6.9, help="Y-axis lower limit (km/s)")
    parser.add_argument("--vp-max", type=float, default=7.35, help="Y-axis upper limit (km/s)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures" / "reproduce_fig12_mvp.png",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    chi_values = tuple(float(x) for x in args.chi.split(",") if x.strip())
    chi1_b_values = tuple(float(x) for x in args.chi1_b.split(",") if x.strip())
    run(
        b_km=args.b_km,
        tp_range_c=(args.tp_min, args.tp_max),
        tp_step_c=args.tp_step,
        tp_contour_interval_c=args.tp_contour_interval,
        chi_values=chi_values,
        chi1_b_values_km=chi1_b_values,
        dfdp_per_gpa=args.dfdp,
        vp_bias_km_s=0.0,
        h_lim_km=(args.h_min, args.h_max),
        vp_lim_km_s=(args.vp_min, args.vp_max),
        save_figure=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
