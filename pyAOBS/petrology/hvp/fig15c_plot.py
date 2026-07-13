"""Fig.15c-style overlay of transect windows on Fig.12a."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from petrology.hvp.fig12a_background import (
    DEFAULT_HVP_DIGITIZED,
    H_LIM,
    VP_LIM,
    draw_fig12a_background,
    parse_hvp_digitized,
)
from petrology.seismic.transect import TransectWindow

if TYPE_CHECKING:
    from petrology.imodel_bridge.export_contract import CrustObservation


def plot_transect_windows_on_ax(
    ax,
    windows: list[TransectWindow],
    *,
    delta_vp_max_km_s: float = 0.15,
    h_min_km: float = 15.0,
    max_read_bands: int = 3,
) -> None:
    """Overlay Fig.15c-style points/bands on an existing Fig.12a axes."""
    thick = [w for w in windows if w.thick_crust]
    thin = [w for w in windows if not w.thick_crust]

    if thick:
        ax.errorbar(
            [w.h_whole_km for w in thick],
            [w.v_lc_km_s for w in thick],
            yerr=[w.v_lc_sigma_km_s for w in thick],
            fmt="o",
            mfc="white",
            mec="0.15",
            mew=1.1,
            ms=7,
            capsize=3,
            zorder=6,
            label=f"transect thick (H>{h_min_km:g} km)",
        )
        for w in thick[: max(0, int(max_read_bands))]:
            ax.plot(
                [w.h_whole_km, w.h_whole_km],
                [w.v_bulk_lower_km_s, w.v_bulk_upper_km_s],
                color="#3498db",
                lw=1.2,
                alpha=0.35,
                zorder=5,
            )
        ax.plot(
            [],
            [],
            color="#3498db",
            lw=1.2,
            alpha=0.6,
            label=f"Step-2 read band (ΔVp≤{delta_vp_max_km_s:.2f})",
        )

    if thin:
        ax.errorbar(
            [w.h_whole_km for w in thin],
            [w.v_lc_km_s for w in thin],
            yerr=[w.v_lc_sigma_km_s for w in thin],
            fmt="o",
            mfc="none",
            mec="0.55",
            mew=1.0,
            ms=7,
            capsize=3,
            zorder=4,
            label=f"transect thin (H≤{h_min_km:g} km, MC σ)",
        )


def plot_fig12a_model_overlays(
    ax,
    *,
    observation: CrustObservation | None = None,
    windows: list[TransectWindow] | None = None,
    all_series=None,
    h_lim: tuple[float, float] = H_LIM,
    vp_lim: tuple[float, float] = VP_LIM,
    delta_vp_max_km_s: float = 0.15,
    h_min_km: float = 15.0,
    show_legend: bool = True,
    max_read_bands: int = 3,
    show_observation_read_band: bool = True,
) -> None:
    """
    Standard Fig.12a background + optional single observation and/or transect windows.

    Fig.12a (单点) 与 Fig.15c (沿迹多窗) 共用同一底图，仅叠加层不同。
    """
    if all_series is None:
        all_series = parse_hvp_digitized(DEFAULT_HVP_DIGITIZED)
    draw_fig12a_background(ax, all_series, h_lim=h_lim, vp_lim=vp_lim, show_legend=show_legend)

    if observation is not None:
        from petrology.imodel_bridge.fig12a_overlay import crust_observation_to_point
        from petrology.hvp.observation_overlay import plot_observation_on_hvp

        pt = crust_observation_to_point(observation, h_min_km=h_min_km)
        plot_observation_on_hvp(
            ax,
            pt,
            delta_vp_max_km_s=delta_vp_max_km_s,
            h_min_km=h_min_km,
            show_read_band=show_observation_read_band,
        )

    if windows:
        plot_transect_windows_on_ax(
            ax,
            windows,
            delta_vp_max_km_s=delta_vp_max_km_s,
            h_min_km=h_min_km,
            max_read_bands=max_read_bands,
        )

    ax.set_xlim(h_lim)
    ax.set_ylim(vp_lim)


def plot_fig15c_on_fig12a(
    ax,
    windows: list[TransectWindow],
    *,
    all_series=None,
    h_lim: tuple[float, float] = H_LIM,
    vp_lim: tuple[float, float] = VP_LIM,
    delta_vp_max_km_s: float = 0.15,
    h_min_km: float = 15.0,
    show_legend: bool = True,
    max_read_bands: int = 3,
) -> None:
    """Draw digitized Fig.12a + Fig.15c open circles (thick/thin crust)."""
    if all_series is None:
        all_series = parse_hvp_digitized(DEFAULT_HVP_DIGITIZED)
    draw_fig12a_background(ax, all_series, h_lim=h_lim, vp_lim=vp_lim, show_legend=show_legend)
    plot_transect_windows_on_ax(
        ax,
        windows,
        delta_vp_max_km_s=delta_vp_max_km_s,
        h_min_km=h_min_km,
        max_read_bands=max_read_bands,
    )
    ax.set_xlim(h_lim)
    ax.set_ylim(vp_lim)


def save_fig15c_figure(
    windows: list[TransectWindow],
    path: Path | str,
    *,
    fig12a_file: Path | None = None,
    delta_vp_max_km_s: float = 0.15,
    h_min_km: float = 15.0,
    transect_label: str = "transect",
    window_half_width_km: float = 10.0,
    distance_step_km: float = 10.0,
    n_mc: int = 100,
    composite: bool = True,
) -> Path:
    """Save Fig.15 (a–c) composite or legacy panel-(c) only."""
    if composite:
        from petrology.hvp.fig15_composite import save_fig15_composite_figure

        return save_fig15_composite_figure(
            windows,
            path,
            fig12a_file=fig12a_file,
            transect_label=transect_label,
            window_half_width_km=window_half_width_km,
            distance_step_km=distance_step_km,
            n_mc=n_mc,
            delta_vp_max_km_s=delta_vp_max_km_s,
            h_min_km=h_min_km,
        )

    import matplotlib.pyplot as plt

    path = Path(path)
    series = parse_hvp_digitized(fig12a_file or DEFAULT_HVP_DIGITIZED)
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    plot_fig15c_on_fig12a(
        ax,
        windows,
        all_series=series,
        delta_vp_max_km_s=delta_vp_max_km_s,
        h_min_km=h_min_km,
    )
    ax.set_xlabel("Igneous crustal thickness H (km)")
    ax.set_ylabel(r"Mean $V_{\mathrm{p}}$ (km/s)")
    ax.set_title("Fig.15c — (H, V_LC) on standard H–Vp diagram (Fig.12a)")
    ax.grid(True, ls=":", lw=0.35, alpha=0.35)
    if windows:
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.92)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
