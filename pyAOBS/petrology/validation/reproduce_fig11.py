"""
KKHS02 Fig.11 — active upwelling melting vs Tp (§4 standard model).

Panels (a)–(d): b = 0, several χ (active upwelling), left column 4×1.
Panels (e)–(h): χ = 1 passive upwelling, several b, right column 4×1.

Panels d/h: V_bulk from eq.(1) with §4 ``P_bar``, ``F_bar`` (same as
:func:`petrology.active_upwelling.solve_active_upwelling`).

Usage::

    py -3.11 petrology/validation/reproduce_fig11.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.active_upwelling import sweep_hvp
from petrology.melting_laws import FIG11_MELTING

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

# Fig.11 caption (standard model): linear melting 12%/GPa
FIG11_DFDP = FIG11_MELTING.dfdp_per_gpa
FIG11_CHI_ACTIVE = (1.0, 2.0, 4.0, 8.0)
FIG11_B_PASSIVE = (0.0, 10.0, 20.0, 30.0)

# Paper Fig.11 axis ranges (4 rows × 2 cols: active | passive)
FIG11_TP_LIM = (1250.0, 1600.0)
FIG11_YLIM = {
    "pbar_gpa": (0.0, 3.5),
    "fbar": (0.0, 0.25),
    "h_km_active": (0.0, 80.0),
    "h_km_passive": (0.0, 25.0),
    "vp_bulk_km_s": (6.9, 7.4),
}
FIG11_ROW_SPECS = (
    ("pbar_gpa", r"$\bar P$ (GPa)", "pbar_gpa", "pbar_gpa", "pbar_gpa"),
    ("fbar", r"$\bar F$", "fbar", "fbar", "fbar"),
    ("h_km", r"$H$ (km)", "h_km_active", "h_km_passive", "h_km"),
    ("vp_bulk_km_s", r"$V_{\mathrm{bulk}}$ (km/s)", "vp_bulk_km_s", "vp_bulk_km_s", "vp_bulk_km_s"),
)
FIG11_PANEL_LABELS = ("a", "b", "c", "d", "e", "f", "g", "h")
# Paper Fig.11 panel shape: plot width ≈ 1.5 × height (matplotlib box_aspect = height/width)
FIG11_PANEL_WIDTH_OVER_HEIGHT = 1.8


def _set_fig11_panel_box_aspect(ax) -> None:
    ax.set_box_aspect(1.0 / FIG11_PANEL_WIDTH_OVER_HEIGHT)


def run(
    *,
    tp_range_c: tuple[float, float] = FIG11_TP_LIM,
    tp_step_c: float = 5.0,
    dfdp_per_gpa: float = FIG11_DFDP,
    vp_bias_km_s: float = 0.0,
    save_figure: Path | None = None,
    show: bool = False,
) -> dict:
    tp_vals = np.arange(tp_range_c[0], tp_range_c[1] + tp_step_c * 0.5, tp_step_c)

    active_rows: dict[float, list] = {}
    for chi in FIG11_CHI_ACTIVE:
        active_rows[chi] = sweep_hvp(
            tp_values_c=tp_vals,
            chi_values=[chi],
            b_km=0.0,
            dfdp_per_gpa=dfdp_per_gpa,
            vp_bias_km_s=vp_bias_km_s,
        )

    passive_rows: dict[float, list] = {}
    for b_km in FIG11_B_PASSIVE:
        passive_rows[b_km] = sweep_hvp(
            tp_values_c=tp_vals,
            chi_values=[1.0],
            b_km=b_km,
            dfdp_per_gpa=dfdp_per_gpa,
            vp_bias_km_s=vp_bias_km_s,
        )

    summary = {
        "dfdp_per_gpa": dfdp_per_gpa,
        "tp_range_c": tp_range_c,
        "n_tp": len(tp_vals),
        "chi_active": list(FIG11_CHI_ACTIVE),
        "b_passive_km": list(FIG11_B_PASSIVE),
        "vp_model": "eq.(1) with P_bar, F_bar",
    }
    print(
        f"Fig.11 sweep: dF/dP={dfdp_per_gpa:.2f}/GPa, Tp={tp_range_c[0]:.0f}–{tp_range_c[1]:.0f}°C, "
        f"V_bulk=eq.(1)(P_bar, F_bar)"
    )

    def _y_value(r, attr: str) -> float:
        return float(getattr(r, attr))

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable ({exc})")
        return summary

    fig, axes = plt.subplots(4, 2, figsize=(10.0, 9.0), sharex=True)

    for row, (attr, ylab, ylim_active_key, ylim_passive_key, _) in enumerate(FIG11_ROW_SPECS):
        ax_l = axes[row, 0]
        for chi in FIG11_CHI_ACTIVE:
            pts = sorted(active_rows[chi], key=lambda r: r.tp_c)
            if not pts:
                continue
            ax_l.plot(
                [r.tp_c for r in pts],
                [_y_value(r, attr) for r in pts],
                lw=1.8,
                label=f"χ={chi:g}",
            )
        ax_l.set_ylabel(ylab)
        ax_l.set_xlim(*FIG11_TP_LIM)
        ax_l.set_ylim(*FIG11_YLIM[ylim_active_key])
        _set_fig11_panel_box_aspect(ax_l)
        ax_l.grid(True, ls=":", lw=0.4, alpha=0.45)
        ax_l.text(
            0.03,
            0.97,
            FIG11_PANEL_LABELS[row],
            transform=ax_l.transAxes,
            fontsize=11,
            va="top",
            fontweight="bold",
        )
        if row == 0:
            ax_l.legend(fontsize=7, loc="lower right")
            ax_l.set_title("b = 0 (active upwelling)", fontsize=9)
        if row == len(FIG11_ROW_SPECS) - 1:
            ax_l.set_xlabel(r"$T_p$ (°C)")

        ax_r = axes[row, 1]
        for b_km in FIG11_B_PASSIVE:
            pts = sorted(passive_rows[b_km], key=lambda r: r.tp_c)
            if not pts:
                continue
            ax_r.plot(
                [r.tp_c for r in pts],
                [_y_value(r, attr) for r in pts],
                lw=1.8,
                label=f"b={b_km:g} km",
            )
        ax_r.set_ylabel(ylab)
        ax_r.set_xlim(*FIG11_TP_LIM)
        ax_r.set_ylim(*FIG11_YLIM[ylim_passive_key])
        _set_fig11_panel_box_aspect(ax_r)
        ax_r.grid(True, ls=":", lw=0.4, alpha=0.45)
        ax_r.text(
            0.03,
            0.97,
            FIG11_PANEL_LABELS[row + 4],
            transform=ax_r.transAxes,
            fontsize=11,
            va="top",
            fontweight="bold",
        )
        if row == 0:
            ax_r.legend(fontsize=7, loc="lower right")
            ax_r.set_title("χ = 1 (passive upwelling)", fontsize=9)
        if row == len(FIG11_ROW_SPECS) - 1:
            ax_r.set_xlabel(r"$T_p$ (°C)")

    fig.suptitle(
        f"KKHS02 Fig.11 — linear melting ({dfdp_per_gpa*100:.0f}%/GPa), "
        r"$V_{\mathrm{bulk}}$ = eq.(1)($\bar P$, $\bar F$)",
        fontsize=11,
    )
    fig.tight_layout()

    if save_figure:
        save_figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_figure, dpi=150)
        print(f"Saved {save_figure}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce KKHS02 Fig.11")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=FIG_DIR / "fig11_active_upwelling_tp_sweep.png",
    )
    parser.add_argument("--dfdp", type=float, default=FIG11_DFDP, help="(dF/dP)_S in 1/GPa")
    parser.add_argument("--vp-bias", type=float, default=0.0)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    run(
        dfdp_per_gpa=args.dfdp,
        vp_bias_km_s=args.vp_bias,
        save_figure=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
