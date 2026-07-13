"""Sallarès et al. (2005) H–Vp diagram: 2D wet melting + KKHS02 eq.(1) curves + GVP observations."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from petrology.hvp.fig12a_background import H_LIM, VP_LIM
from petrology.melting.sallares2005 import (
    FIG10_PANELS,
    SallaresMeltingResult,
    sweep_sallares_hvp,
)
from petrology.reference.galapagos_sallares_observations import (
    FIG10_PANEL_LABELS,
    GALAPAGOS_SALLARES_OBSERVATIONS,
    GalapagosProfilePoint,
)


def _sort_curve(points: list[SallaresMeltingResult], key: str) -> list[SallaresMeltingResult]:
    return sorted(points, key=lambda r: getattr(r, key))


def plot_sallares_observations(
    ax,
    observations: Iterable[GalapagosProfilePoint] | None = None,
    *,
    label_fontsize: float = 8.0,
) -> None:
    obs = list(observations or GALAPAGOS_SALLARES_OBSERVATIONS)
    thick = [p for p in obs if p.thick_crest]
    thin = [p for p in obs if not p.thick_crest]

    if thick:
        ax.errorbar(
            [p.h_km for p in thick],
            [p.v_lc_km_s for p in thick],
            yerr=[p.v_sigma_km_s for p in thick],
            fmt="s",
            mfc="white",
            mec="#c0392b",
            mew=1.2,
            ms=8,
            capsize=3,
            zorder=8,
            label="观测 厚壳脊部 (Layer 3)",
        )
        for p in thick:
            ax.annotate(
                p.profile_id,
                (p.h_km, p.v_lc_km_s),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=label_fontsize - 1,
                color="#922b21",
            )

    if thin:
        ax.errorbar(
            [p.h_km for p in thin],
            [p.v_lc_km_s for p in thin],
            yerr=[p.v_sigma_km_s for p in thin],
            fmt="o",
            mfc="#3498db",
            mec="0.2",
            mew=1.0,
            ms=7,
            capsize=3,
            zorder=7,
            label="观测 薄壳 / NOC",
        )
        for p in thin:
            ax.annotate(
                p.profile_id,
                (p.h_km, p.v_lc_km_s),
                textcoords="offset points",
                xytext=(4, -10),
                fontsize=label_fontsize - 1,
                color="#2471a3",
            )


def plot_sallares_model_curves(
    ax,
    results: list[SallaresMeltingResult],
    *,
    tp_contour_step_c: float = 50.0,
    label_fontsize: float = 8.0,
) -> None:
    """Thin = constant Tp (varying X); thick = constant X (varying Tp) — Fig. 10 style."""
    if not results:
        return

    x_vals = sorted({r.upwelling_x for r in results})
    tp_vals = sorted({r.tp_c for r in results})

    # Thick: constant upwelling X
    for i, x in enumerate(x_vals):
        pts = _sort_curve([r for r in results if abs(r.upwelling_x - x) < 1e-6], "tp_c")
        if len(pts) < 2:
            continue
        color = plt_cmap(i, len(x_vals))
        ax.plot(
            [p.h_km for p in pts],
            [p.vp_bulk_km_s for p in pts],
            color=color,
            lw=2.2,
            alpha=0.85,
            zorder=4,
            label=f"X={x:g}" if i < 4 else None,
        )

    # Thin: constant Tp (every tp_contour_step_c)
    tp_show = [t for t in tp_vals if abs(t - tp_vals[0]) < 1e-6 or (t - tp_vals[0]) % tp_contour_step_c < 1e-6]
    if len(tp_show) < 2:
        tp_show = tp_vals[:: max(1, len(tp_vals) // 5)]

    for j, tp in enumerate(tp_show):
        pts = _sort_curve([r for r in results if abs(r.tp_c - tp) < 1e-6], "upwelling_x")
        if len(pts) < 2:
            continue
        ax.plot(
            [p.h_km for p in pts],
            [p.vp_bulk_km_s for p in pts],
            color="0.55",
            lw=0.9,
            ls="--",
            alpha=0.55,
            zorder=3,
            label=f"Tp={tp:.0f}°C" if j < 3 else None,
        )


def plt_cmap(i: int, n: int) -> str:
    """Simple discrete colors without matplotlib colormap import at module level."""
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    return palette[i % len(palette)]


def plot_sallares_hvp_diagram(
    ax,
    *,
    panel_key: str = "10a_ref",
    tp_range_c: tuple[float, float] = (1150.0, 1450.0),
    tp_step_c: float = 25.0,
    x_values: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 40.0),
    vp_bias_km_s: float = 0.0,
    h_lim: tuple[float, float] = H_LIM,
    vp_lim: tuple[float, float] = (6.85, 7.50),
    show_observations: bool = True,
    label_fontsize: float = 8.0,
) -> list[SallaresMeltingResult]:
    """
    Draw Sallarès (2005) Fig. 10-style H–Vp: eq.(1) model curves + GVP Layer 3 points.

    Model: 2D wet melting (eq. 8–10) → (F̄, Z̄) → KKHS02 eq.(1) V_bulk.
    """
    params = FIG10_PANELS.get(panel_key, FIG10_PANELS["10a_ref"])
    tp_vals = np.arange(tp_range_c[0], tp_range_c[1] + tp_step_c * 0.5, tp_step_c)
    results = sweep_sallares_hvp(
        tp_values_c=tp_vals,
        x_values=list(x_values),
        params=params,
        vp_bias_km_s=vp_bias_km_s,
    )
    plot_sallares_model_curves(ax, results, label_fontsize=label_fontsize)
    if show_observations:
        plot_sallares_observations(ax, label_fontsize=label_fontsize)

    panel_label = FIG10_PANEL_LABELS.get(panel_key, panel_key)
    ax.set_xlim(h_lim)
    ax.set_ylim(vp_lim)
    ax.set_xlabel("Crustal thickness H (km)")
    ax.set_ylabel(r"$V_{\mathrm{p}}$ from eq.(1) / Layer 3 obs. (km/s) @ 400°C, 600 MPa")
    ax.set_title(f"Sallarès et al. (2005) Galápagos — {panel_label}")
    ax.grid(True, ls=":", lw=0.35, alpha=0.4)
    ax.legend(loc="upper left", fontsize=label_fontsize, framealpha=0.92, ncol=1)
    return results
