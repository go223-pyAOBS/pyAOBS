"""KKHS02 Fig.15 composite: (a) V_LC profile, (b) crustal thickness, (c) H–Vp."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from petrology.hvp.fig15c_plot import plot_fig12a_model_overlays
from petrology.seismic.transect import TransectWindow

# (a) / (b) shorter; (c) H–Vp panel taller
FIG15_PANEL_HEIGHT_RATIOS = (0.72, 0.72, 1.68)

if TYPE_CHECKING:
    from petrology.imodel_bridge.export_contract import CrustObservation


def _sorted_windows(windows: list[TransectWindow]) -> list[TransectWindow]:
    return sorted(windows, key=lambda w: float(w.distance_km))


def plot_fig15a_vlc_profile(
    ax,
    windows: list[TransectWindow],
    *,
    transect_label: str = "transect",
    window_half_width_km: float = 10.0,
    distance_step_km: float = 10.0,
    n_mc: int = 100,
) -> None:
    """Panel (a): harmonic-mean V_LC vs distance with MC uncertainty."""
    wins = _sorted_windows(windows)
    if not wins:
        return

    x = [w.distance_km for w in wins]
    y = [w.v_lc_km_s for w in wins]
    yerr = [w.v_lc_sigma_km_s for w in wins]
    win_km = 2.0 * float(window_half_width_km)

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o-",
        mfc="white",
        mec="0.15",
        mew=1.0,
        ms=5,
        lw=1.0,
        capsize=3,
        zorder=4,
    )
    ax.set_ylabel(r"$V_{\mathrm{LC}}$ (km/s)")
    ax.set_title("(a)", loc="left", fontsize=10, fontweight="bold")
    ax.grid(True, ls=":", lw=0.35, alpha=0.45)
    ax.text(
        0.98,
        0.98,
        r"$V$ @ 600 MPa & 400°C",
        transform=ax.transAxes,
        fontsize=7.5,
        va="top",
        ha="right",
        color="0.35",
    )
    ax.text(
        0.02,
        0.03,
        (
            f"Harmonic mean lower-crust velocity of {transect_label} "
            f"at {distance_step_km:g} km intervals with a {win_km:g} km wide "
            f"averaging window, using {int(n_mc)} Monte Carlo ensembles."
        ),
        transform=ax.transAxes,
        fontsize=6.5,
        va="bottom",
        ha="left",
        color="0.35",
        wrap=True,
    )


def plot_fig15b_thickness_profile(
    ax,
    windows: list[TransectWindow],
    *,
    transect_label: str = "transect",
) -> None:
    """
    Panel (b): left axis thickness (km); right axis lower-crust fraction.

    - Whole crust H: black solid line + markers
    - Lower crust thickness: gray solid line + markers
    - Thickness fraction h_lower/H: dashed line on right axis
    """
    wins = _sorted_windows(windows)
    if not wins:
        return

    x = np.array([w.distance_km for w in wins], dtype=float)
    h_whole = np.array([w.h_whole_km for w in wins], dtype=float)
    h_lower = np.array([w.h_lower_km for w in wins], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(h_whole > 0.0, h_lower / h_whole, np.nan)
    if wins[0].f_lower is not None:
        frac = np.array(
            [float(w.f_lower) if w.f_lower is not None else (hl / h if h > 0 else np.nan)
             for w, hl, h in zip(wins, h_lower, h_whole)],
            dtype=float,
        )

    ax.plot(
        x,
        h_whole,
        "-o",
        color="0.1",
        lw=1.4,
        ms=4.5,
        mfc="white",
        mec="0.1",
        mew=1.0,
        label="Whole crust",
        zorder=4,
    )
    ax.plot(
        x,
        h_lower,
        "-o",
        color="0.55",
        lw=1.3,
        ms=4.5,
        mfc="white",
        mec="0.55",
        mew=1.0,
        label="Lower crust",
        zorder=3,
    )

    ax_r = ax.twinx()
    ax_r.plot(
        x,
        frac,
        "--",
        color="0.35",
        lw=1.2,
        dashes=(5, 3),
        label="Lower-crust fraction",
        zorder=2,
    )
    ax_r.set_ylabel(r"$H_{\mathrm{LC}}/H$")
    frac_hi = float(np.nanmax(frac)) if np.any(np.isfinite(frac)) else 1.0
    ax_r.set_ylim(0.0, min(1.05, max(0.35, frac_hi + 0.08)))

    ax.set_ylabel(r"$H$ & $H_{\mathrm{LC}}$ (km)")
    ax.set_xlabel("Distance along profile (km)")
    ax.set_title("(b)", loc="left", fontsize=10, fontweight="bold")
    ax.grid(True, ls=":", lw=0.35, alpha=0.45)
    lines_l, labels_l = ax.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax.legend(
        lines_l + lines_r,
        labels_l + labels_r,
        fontsize=7,
        loc="upper right",
        framealpha=0.92,
    )
    ax.text(
        0.02,
        0.03,
        f"Whole crustal (solid) and lower crustal thickness along {transect_label}.",
        transform=ax.transAxes,
        fontsize=6.5,
        va="bottom",
        ha="left",
        color="0.35",
    )
    lo = float(min(h_whole.min(), h_lower.min(), 0.0))
    hi = float(max(h_whole.max(), h_lower.max()))
    pad = 0.08 * max(hi - lo, 1.0)
    ax.set_ylim(max(0.0, lo - pad), hi + pad)


def plot_fig15_composite_on_axes(
    axes: tuple,
    *,
    windows: list[TransectWindow],
    observation: CrustObservation | None = None,
    transect_label: str = "transect",
    window_half_width_km: float = 10.0,
    distance_step_km: float = 10.0,
    n_mc: int = 100,
    delta_vp_max_km_s: float = 0.15,
    h_min_km: float = 15.0,
    all_series=None,
) -> None:
    """Draw Fig.15 (a)(b)(c) on three axes `(ax_a, ax_b, ax_c)`."""
    ax_a, ax_b, ax_c = axes
    if windows:
        plot_fig15a_vlc_profile(
            ax_a,
            windows,
            transect_label=transect_label,
            window_half_width_km=window_half_width_km,
            distance_step_km=distance_step_km,
            n_mc=n_mc,
        )
        plot_fig15b_thickness_profile(ax_b, windows, transect_label=transect_label)
    else:
        ax_a.set_title("(a)", loc="left", fontsize=10, fontweight="bold")
        ax_b.set_title("(b)", loc="left", fontsize=10, fontweight="bold")
        ax_a.set_ylabel(r"$V_{\mathrm{LC}}$ (km/s)")
        ax_a.text(
            0.98,
            0.98,
            r"$V$ @ 600 MPa & 400°C",
            transform=ax_a.transAxes,
            fontsize=7.5,
            va="top",
            ha="right",
            color="0.35",
        )
        ax_b.set_ylabel(r"$H$ & $H_{\mathrm{LC}}$ (km)")
        ax_a.text(0.5, 0.5, "No transect windows", ha="center", va="center", transform=ax_a.transAxes)
        ax_b.text(0.5, 0.5, "No transect windows", ha="center", va="center", transform=ax_b.transAxes)

    plot_fig12a_model_overlays(
        ax_c,
        observation=observation,
        windows=windows or None,
        all_series=all_series,
        delta_vp_max_km_s=delta_vp_max_km_s,
        h_min_km=h_min_km,
        show_legend=True,
    )
    ax_c.set_title("(c)", loc="left", fontsize=10, fontweight="bold")
    ax_c.set_xlabel("Igneous crustal thickness H (km)")
    ax_c.set_ylabel(r"Mean $V_{\mathrm{p}}$ (km/s)")
    ax_c.grid(True, ls=":", lw=0.35, alpha=0.35)


def create_fig15_composite_figure(
    windows: list[TransectWindow],
    *,
    observation: CrustObservation | None = None,
    fig12a_file: Path | None = None,
    transect_label: str = "transect",
    window_half_width_km: float = 10.0,
    distance_step_km: float = 10.0,
    n_mc: int = 100,
    delta_vp_max_km_s: float = 0.15,
    h_min_km: float = 15.0,
    figsize: tuple[float, float] = (8.2, 10.8),
):
    """Return `(fig, (ax_a, ax_b, ax_c))` for Fig.15 layout."""
    import matplotlib.pyplot as plt

    from petrology.hvp.fig12a_background import DEFAULT_HVP_DIGITIZED, parse_hvp_digitized

    series = parse_hvp_digitized(fig12a_file or DEFAULT_HVP_DIGITIZED)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 1, height_ratios=list(FIG15_PANEL_HEIGHT_RATIOS), hspace=0.34)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharex=ax_a)
    ax_c = fig.add_subplot(gs[2])
    plt.setp(ax_a.get_xticklabels(), visible=False)

    plot_fig15_composite_on_axes(
        (ax_a, ax_b, ax_c),
        windows=windows,
        observation=observation,
        transect_label=transect_label,
        window_half_width_km=window_half_width_km,
        distance_step_km=distance_step_km,
        n_mc=n_mc,
        delta_vp_max_km_s=delta_vp_max_km_s,
        h_min_km=h_min_km,
        all_series=series,
    )
    fig.subplots_adjust(top=0.97, bottom=0.06, left=0.11, right=0.88)
    return fig, (ax_a, ax_b, ax_c)


def save_fig15_composite_figure(
    windows: list[TransectWindow],
    path: Path | str,
    *,
    observation: CrustObservation | None = None,
    fig12a_file: Path | None = None,
    transect_label: str = "transect",
    window_half_width_km: float = 10.0,
    distance_step_km: float = 10.0,
    n_mc: int = 100,
    delta_vp_max_km_s: float = 0.15,
    h_min_km: float = 15.0,
) -> Path:
    import matplotlib.pyplot as plt

    path = Path(path)
    fig, _axes = create_fig15_composite_figure(
        windows,
        observation=observation,
        fig12a_file=fig12a_file,
        transect_label=transect_label,
        window_half_width_km=window_half_width_km,
        distance_step_km=distance_step_km,
        n_mc=n_mc,
        delta_vp_max_km_s=delta_vp_max_km_s,
        h_min_km=h_min_km,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
