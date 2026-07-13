"""
KKHS02 regression panel — mantle melts in (P, F) with eq.(1) Vp field + contours.

Paper-style layout (Fig.4a):
  - X: pressure 0.5 → 7 GPa
  - Y: melt fraction 0 (top) → 0.6 (bottom)
  - Grayscale fill: predicted Vp 6.8–7.6 km/s (darker = faster)
  - Solid contours: 7.2–7.6 km/s @ 0.1 km/s spacing
  - Published melts: large open circles; supplementary: small filled circles

Usage::

    py -3.11 petrology/validation/reproduce_fig04a.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.data.load_catalog import load_melt_catalog
from petrology.norm_velocity import norm_velocity_from_record
from petrology.vp_regression import load_eq1, predict_vp_km_s

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

# Paper panel defaults
P_LIM_GPA = (0.5, 7.0)
F_LIM = (0.0, 0.6)
VP_COLOR_LIM = (6.8, 7.6)
CONTOUR_LEVELS = (7.2, 7.3, 7.4, 7.5, 7.6)


def _is_supplementary(rec: dict) -> bool:
    return str(rec.get("melt_style", "")).lower() == "calculated"


def run(
    *,
    p_lim_gpa: tuple[float, float] = P_LIM_GPA,
    f_lim: tuple[float, float] = F_LIM,
    vp_color_lim: tuple[float, float] = VP_COLOR_LIM,
    contour_levels: tuple[float, ...] = CONTOUR_LEVELS,
    cipw_backend: str = "auto",
    mineral_backend: str = "auto",
    save_figure: Path | None = None,
    show: bool = False,
) -> dict:
    eq1 = load_eq1()
    rows = [r for r in load_melt_catalog() if r.get("include_in_regression")]

    pub_p, pub_f = [], []
    sup_p, sup_f = [], []
    for rec in rows:
        p = float(rec["P_melt_GPa"])
        f = float(rec["F_melt"])
        if _is_supplementary(rec):
            sup_p.append(p)
            sup_f.append(f)
        else:
            pub_p.append(p)
            pub_f.append(f)

    pub_p = np.asarray(pub_p)
    pub_f = np.asarray(pub_f)
    sup_p = np.asarray(sup_p)
    sup_f = np.asarray(sup_f)

    p_grid = np.linspace(p_lim_gpa[0], p_lim_gpa[1], 200)
    f_grid = np.linspace(f_lim[0], f_lim[1], 200)
    pp, ff = np.meshgrid(p_grid, f_grid)
    vp_pred = np.vectorize(lambda p, f: predict_vp_km_s(p, f, eq1=eq1))(pp, ff)

    summary = {
        "n_published": int(len(pub_p)),
        "n_supplementary": int(len(sup_p)),
        "p_lim_gpa": p_lim_gpa,
        "f_lim": f_lim,
        "vp_color_lim_km_s": vp_color_lim,
        "contour_levels_km_s": list(contour_levels),
        "cipw_backend": cipw_backend,
        "mineral_backend": mineral_backend,
    }

    print(f"Regression catalog: {summary['n_published']} published + {summary['n_supplementary']} supplementary")
    print(f"Panel: P={p_lim_gpa[0]:g}–{p_lim_gpa[1]:g} GPa, F={f_lim[0]:g}–{f_lim[1]:g} (0 at top)")
    print(f"Grayscale Vp: {vp_color_lim[0]:.1f}–{vp_color_lim[1]:.1f} km/s")
    print(f"Contours: {', '.join(f'{v:.1f}' for v in contour_levels)}")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except Exception as exc:
        print(f"matplotlib unavailable ({exc})")
        return summary

    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    norm = Normalize(vmin=vp_color_lim[0], vmax=vp_color_lim[1])
    n_fill = int((vp_color_lim[1] - vp_color_lim[0]) / 0.02) + 1
    fill_levels = np.linspace(vp_color_lim[0], vp_color_lim[1], n_fill)
    cf = ax.contourf(
        pp,
        ff,
        vp_pred,
        levels=fill_levels,
        cmap="Greys",
        norm=norm,
        extend="both",
        zorder=1,
    )

    levels = list(contour_levels)
    levels_low = [v for v in levels if v < 7.5]
    levels_high = [v for v in levels if v >= 7.5]

    label_bbox = dict(
        boxstyle="square,pad=0.25",
        facecolor="white",
        edgecolor="0.35",
        linewidth=0.6,
    )

    def _style_contour_labels(texts) -> None:
        for t in texts:
            t.set_fontsize(7)
            t.set_bbox(label_bbox)

    if levels_low:
        cs_low = ax.contour(
            pp,
            ff,
            vp_pred,
            levels=levels_low,
            colors="0.12",
            linewidths=0.9,
            zorder=3,
        )
        _style_contour_labels(
            ax.clabel(cs_low, inline=True, fmt="%.1f", colors="0.05", inline_spacing=4)
        )

    if levels_high:
        # White halo so 7.5 / 7.6 stay visible on dark (high-Vp) fill
        ax.contour(
            pp,
            ff,
            vp_pred,
            levels=levels_high,
            colors="0.15",
            linewidths=2.4,
            zorder=3,
        )
        cs_high = ax.contour(
            pp,
            ff,
            vp_pred,
            levels=levels_high,
            colors="white",
            linewidths=1.05,
            zorder=4,
        )
        _style_contour_labels(
            ax.clabel(cs_high, inline=True, fmt="%.1f", colors="0.05", inline_spacing=4)
        )

    if len(sup_p):
        ax.scatter(
            sup_p,
            sup_f,
            s=20,
            c="0.72",
            edgecolors="0.25",
            linewidths=0.35,
            label=f"supplementary (n={len(sup_p)})",
            zorder=5,
        )
    if len(pub_p):
        ax.scatter(
            pub_p,
            pub_f,
            s=105,
            facecolors="none",
            edgecolors="0.08",
            linewidths=1.25,
            label=f"published (n={len(pub_p)})",
            zorder=6,
        )

    ax.set_xlim(p_lim_gpa)
    ax.set_ylim(f_lim)
    ax.invert_yaxis()
    ax.set_xlabel("Pressure of melting (GPa)")
    ax.set_ylabel("Fraction of melting, $F$")
    ax.set_title("KKHS02 eq.(1) — $V_p(P,F)$ @ 600 MPa, 400°C")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, edgecolor="0.7")

    cb = fig.colorbar(cf, ax=ax, shrink=0.88, pad=0.02)
    cb.set_label("$V_p$ predicted (km/s)")
    cb.set_ticks(np.arange(vp_color_lim[0], vp_color_lim[1] + 0.05, 0.2))

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
    parser = argparse.ArgumentParser(description="KKHS02 Fig.4a — P–F Vp grayscale + contours")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=FIG_DIR / "fig04a_regression_pf_contours.png",
    )
    parser.add_argument("--p-min", type=float, default=P_LIM_GPA[0])
    parser.add_argument("--p-max", type=float, default=P_LIM_GPA[1])
    parser.add_argument("--f-min", type=float, default=F_LIM[0])
    parser.add_argument("--f-max", type=float, default=F_LIM[1])
    parser.add_argument("--vp-min", type=float, default=VP_COLOR_LIM[0])
    parser.add_argument("--vp-max", type=float, default=VP_COLOR_LIM[1])
    parser.add_argument(
        "--contours",
        type=str,
        default=None,
        help="Comma-separated Vp contour levels (default 7.2–7.6 step 0.1)",
    )
    parser.add_argument(
        "--cipw-backend",
        choices=("auto", "pyrolite", "fallback"),
        default="auto",
    )
    parser.add_argument(
        "--mineral-backend",
        choices=("auto", "burnman", "empirical", "sb1994_fig2ol"),
        default="auto",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.contours:
        levels = tuple(float(x.strip()) for x in args.contours.split(",") if x.strip())
    else:
        levels = CONTOUR_LEVELS
    run(
        p_lim_gpa=(args.p_min, args.p_max),
        f_lim=(args.f_min, args.f_max),
        vp_color_lim=(args.vp_min, args.vp_max),
        contour_levels=levels,
        cipw_backend=args.cipw_backend,
        mineral_backend=args.mineral_backend,
        save_figure=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
