"""
Fig.11 panels (d) and (h) only — eq.(1) vs Holbrook linear on the same §4 paths.

Overlays GetData digitized paper curves when available.

Usage::

    py -3.11 petrology/validation/reproduce_fig11_dh_compare.py
    py -3.11 petrology/validation/reproduce_fig11_dh_compare.py --show
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.active_upwelling import sweep_hvp
from petrology.melting_laws import FIG11_MELTING
from petrology.vp_regression import predict_v_bulk_fig11_km_s, predict_v_bulk_km_s

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

FIG11_TP_LIM = (1250.0, 1600.0)
FIG11_VP_YLIM = (6.9, 7.4)
FIG11_CHI_ACTIVE = (1.0, 2.0, 4.0, 8.0)
FIG11_B_PASSIVE = (0.0, 10.0, 20.0, 30.0)
FIG11_PANEL_WIDTH_OVER_HEIGHT = 1.5

DIGITIZED_D = DATA_DIR / "ScreenShot_2026-07-04_124209_264.txt"
DIGITIZED_H = DATA_DIR / "ScreenShot_2026-07-04_124221_433.txt"

# chi / b keys present in digitized files (subset of full sweeps)
DIGITIZED_D_KEYS = ("X=1", "x=8")
DIGITIZED_H_KEYS = ("b=0", "b=20", "b=30")


def parse_getdata(path: Path) -> dict[str, list[tuple[float, float]]]:
    curves: dict[str, list[tuple[float, float]]] = {}
    key: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("Generated"):
            continue
        if re.match(r"^[xXb=\w]+$", line.replace(" ", "")):
            key = line.rstrip()
            curves[key] = []
            continue
        if key is None:
            continue
        parts = line.split()
        if len(parts) >= 2:
            curves[key].append((float(parts[0]), float(parts[1])))
    return curves


def _set_panel_box_aspect(ax) -> None:
    ax.set_box_aspect(1.0 / FIG11_PANEL_WIDTH_OVER_HEIGHT)


def _chi_color(chi: float) -> str:
    idx = FIG11_CHI_ACTIVE.index(chi) if chi in FIG11_CHI_ACTIVE else 0
    return f"C{idx}"


def _b_color(b_km: float) -> str:
    idx = FIG11_B_PASSIVE.index(b_km) if b_km in FIG11_B_PASSIVE else 0
    return f"C{idx}"


def _plot_digitized(
    ax,
    curves: dict[str, list[tuple[float, float]]],
    keys: tuple[str, ...],
    *,
    label_prefix: str,
) -> None:
    for key in keys:
        pts = curves.get(key)
        if not pts:
            continue
        tp = [p[0] for p in sorted(pts, key=lambda x: x[0])]
        v = [p[1] for p in sorted(pts, key=lambda x: x[0])]
        ax.plot(
            tp,
            v,
            "o",
            ms=4.5,
            mfc="0.15",
            mew=0.6,
            mec="0.15",
            zorder=5,
            label=f"paper {label_prefix}{key.split('=')[-1]}",
        )


def plot_panel_d(
    ax,
    active_rows: dict[float, list],
    *,
    digitized: dict[str, list[tuple[float, float]]] | None,
) -> None:
    for chi in FIG11_CHI_ACTIVE:
        pts = sorted(active_rows[chi], key=lambda r: r.tp_c)
        if not pts:
            continue
        tp = [r.tp_c for r in pts]
        col = _chi_color(chi)
        v_eq1 = [predict_v_bulk_km_s(r.pbar_gpa, r.fbar) for r in pts]
        v_hol = [predict_v_bulk_fig11_km_s(r.pbar_gpa, r.fbar) for r in pts]
        ax.plot(tp, v_eq1, "-", color=col, lw=1.8, label=f"eq.(1) χ={chi:g}")
        ax.plot(
            tp,
            v_hol,
            "--",
            color=col,
            lw=1.8,
            dashes=(5, 2),
            label=f"Holbrook χ={chi:g}",
        )

    if digitized:
        _plot_digitized(ax, digitized, DIGITIZED_D_KEYS, label_prefix="χ=")

    ax.set_xlim(*FIG11_TP_LIM)
    ax.set_ylim(*FIG11_VP_YLIM)
    ax.set_xlabel(r"$T_p$ (°C)")
    ax.set_ylabel(r"$V_{\mathrm{bulk}}$ (km/s)")
    ax.set_title(r"(d) Bulk crustal velocity, $b=0$ — eq.(1) vs Holbrook")
    ax.grid(True, ls=":", lw=0.4, alpha=0.45)
    _set_panel_box_aspect(ax)
    ax.legend(fontsize=6.5, loc="lower right", ncol=2, framealpha=0.92)


def plot_panel_h(
    ax,
    passive_rows: dict[float, list],
    *,
    digitized: dict[str, list[tuple[float, float]]] | None,
) -> None:
    for b_km in FIG11_B_PASSIVE:
        pts = sorted(passive_rows[b_km], key=lambda r: r.tp_c)
        if not pts:
            continue
        tp = [r.tp_c for r in pts]
        col = _b_color(b_km)
        v_eq1 = [predict_v_bulk_km_s(r.pbar_gpa, r.fbar) for r in pts]
        v_hol = [predict_v_bulk_fig11_km_s(r.pbar_gpa, r.fbar) for r in pts]
        ax.plot(tp, v_eq1, "-", color=col, lw=1.8, label=f"eq.(1) b={b_km:g} km")
        ax.plot(
            tp,
            v_hol,
            "--",
            color=col,
            lw=1.8,
            dashes=(5, 2),
            label=f"Holbrook b={b_km:g} km",
        )

    if digitized:
        _plot_digitized(ax, digitized, DIGITIZED_H_KEYS, label_prefix="b=")

    ax.set_xlim(*FIG11_TP_LIM)
    ax.set_ylim(*FIG11_VP_YLIM)
    ax.set_xlabel(r"$T_p$ (°C)")
    ax.set_ylabel(r"$V_{\mathrm{bulk}}$ (km/s)")
    ax.set_title(r"(h) Bulk crustal velocity, $\chi=1$ — eq.(1) vs Holbrook")
    ax.grid(True, ls=":", lw=0.4, alpha=0.45)
    _set_panel_box_aspect(ax)
    ax.legend(fontsize=6.5, loc="lower right", ncol=2, framealpha=0.92)


def run(
    *,
    tp_range_c: tuple[float, float] = FIG11_TP_LIM,
    tp_step_c: float = 5.0,
    save_d: Path | None = None,
    save_h: Path | None = None,
    save_combined: Path | None = None,
    show: bool = False,
    use_digitized: bool = True,
) -> None:
    tp_vals = np.arange(tp_range_c[0], tp_range_c[1] + tp_step_c * 0.5, tp_step_c)
    dfdp = FIG11_MELTING.dfdp_per_gpa

    active_rows: dict[float, list] = {}
    for chi in FIG11_CHI_ACTIVE:
        active_rows[chi] = sweep_hvp(
            tp_values_c=tp_vals,
            chi_values=[chi],
            b_km=0.0,
            dfdp_per_gpa=dfdp,
        )

    passive_rows: dict[float, list] = {}
    for b_km in FIG11_B_PASSIVE:
        passive_rows[b_km] = sweep_hvp(
            tp_values_c=tp_vals,
            chi_values=[1.0],
            b_km=b_km,
            dfdp_per_gpa=dfdp,
        )

    dig_d = parse_getdata(DIGITIZED_D) if use_digitized and DIGITIZED_D.is_file() else None
    dig_h = parse_getdata(DIGITIZED_H) if use_digitized and DIGITIZED_H.is_file() else None

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib required: {exc}") from exc

    if save_d or show:
        fig_d, ax_d = plt.subplots(figsize=(6.0, 4.0))
        plot_panel_d(ax_d, active_rows, digitized=dig_d)
        fig_d.tight_layout()
        if save_d:
            save_d.parent.mkdir(parents=True, exist_ok=True)
            fig_d.savefig(save_d, dpi=150)
            print(f"Saved {save_d}")
        if show:
            plt.show()
        else:
            plt.close(fig_d)

    if save_h or show:
        fig_h, ax_h = plt.subplots(figsize=(6.0, 4.0))
        plot_panel_h(ax_h, passive_rows, digitized=dig_h)
        fig_h.tight_layout()
        if save_h:
            save_h.parent.mkdir(parents=True, exist_ok=True)
            fig_h.savefig(save_h, dpi=150)
            print(f"Saved {save_h}")
        if show:
            plt.show()
        else:
            plt.close(fig_h)

    if save_combined:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
        plot_panel_d(axes[0], active_rows, digitized=dig_d)
        plot_panel_h(axes[1], passive_rows, digitized=dig_h)
        fig.suptitle(
            r"KKHS02 Fig.11 d/h — same $\bar P$, $\bar F$ from §4; "
            r"solid = eq.(1), dashed = Holbrook, dots = digitized paper",
            fontsize=10,
        )
        fig.tight_layout()
        save_combined.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_combined, dpi=150)
        print(f"Saved {save_combined}")
        if not show:
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fig.11 panels d/h: eq.(1) vs Holbrook")
    parser.add_argument(
        "--output-d",
        type=Path,
        default=FIG_DIR / "fig11_panel_d_vp_compare.png",
    )
    parser.add_argument(
        "--output-h",
        type=Path,
        default=FIG_DIR / "fig11_panel_h_vp_compare.png",
    )
    parser.add_argument(
        "--output-combined",
        type=Path,
        default=FIG_DIR / "fig11_panel_dh_vp_compare.png",
    )
    parser.add_argument("--no-combined", action="store_true")
    parser.add_argument("--no-digitized", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    run(
        save_d=args.output_d,
        save_h=args.output_h,
        save_combined=None if args.no_combined else args.output_combined,
        show=args.show,
        use_digitized=not args.no_digitized,
    )


if __name__ == "__main__":
    main()
