"""Observation overlay on Fig.12a (Fig.15c style read bands)."""

from __future__ import annotations

from dataclasses import dataclass

from matplotlib.axes import Axes

DEFAULT_THICK_CRUST_H_KM = 15.0
DEFAULT_DELTA_VP_MAX_KM_S = 0.15


@dataclass(frozen=True)
class ObservationPoint:
    h_km: float
    v_lc_km_s: float
    v_lc_sigma_km_s: float | None = None
    thick_crust: bool | None = None
    label: str | None = None


def bulk_read_band(
    v_lc_km_s: float,
    *,
    delta_vp_max_km_s: float = DEFAULT_DELTA_VP_MAX_KM_S,
) -> tuple[float, float]:
    """Step-2 read band: V_bulk ∈ [V_LC − ΔVp, V_LC]."""
    v_up = float(v_lc_km_s)
    v_lo = v_up - float(delta_vp_max_km_s)
    return v_lo, v_up


def is_thick_crust(h_km: float, *, h_min_km: float = DEFAULT_THICK_CRUST_H_KM) -> bool:
    return float(h_km) > float(h_min_km)


def plot_observation_on_hvp(
    ax: Axes,
    obs: ObservationPoint,
    *,
    delta_vp_max_km_s: float = DEFAULT_DELTA_VP_MAX_KM_S,
    h_min_km: float = DEFAULT_THICK_CRUST_H_KM,
    show_read_band: bool = True,
    color: str | None = None,
    marker: str | None = None,
    draw_guides: bool = True,
) -> None:
    """Plot anchor (H, V_LC) with optional Step-2 vertical read band.

    ``color`` / ``marker`` 用于多点叠投时区分；默认厚壳金星 / 薄壳空心圆。
    """
    thick = obs.thick_crust if obs.thick_crust is not None else is_thick_crust(obs.h_km, h_min_km=h_min_km)
    face = color or ("gold" if thick else "none")
    edge = "0.15" if thick else (color or "0.55")
    mk = marker or ("*" if thick else "o")
    ms_scatter = 90 if thick else 55
    ms_err = 12 if thick else 8

    if thick and show_read_band:
        v_lo, v_up = bulk_read_band(obs.v_lc_km_s, delta_vp_max_km_s=delta_vp_max_km_s)
        ax.plot(
            [obs.h_km, obs.h_km],
            [v_lo, v_up],
            color=color or "#3498db",
            lw=1.4,
            alpha=0.45,
            zorder=5,
        )

    if obs.v_lc_sigma_km_s is not None and obs.v_lc_sigma_km_s > 0:
        ax.errorbar(
            [obs.h_km],
            [obs.v_lc_km_s],
            yerr=[obs.v_lc_sigma_km_s],
            fmt=mk,
            mfc=face if face != "none" else "white",
            mec=edge,
            mew=1.0,
            ms=ms_err,
            capsize=3,
            zorder=7,
            label=obs.label or ("观测 (厚壳)" if thick else "观测 (薄壳)"),
            color=edge if face == "none" else face,
        )
    else:
        ax.scatter(
            [obs.h_km],
            [obs.v_lc_km_s],
            s=ms_scatter,
            facecolors=face if face != "none" else "none",
            edgecolors=edge,
            linewidths=1.0 if thick else 1.2,
            marker=mk,
            zorder=7,
            label=obs.label or ("观测 (厚壳)" if thick else "观测 (薄壳)"),
        )

    if draw_guides:
        ax.axhline(obs.v_lc_km_s, color="0.55", ls=":", lw=0.7, alpha=0.6, zorder=3)
        ax.axvline(obs.h_km, color="0.55", ls="--", lw=0.7, alpha=0.6, zorder=3)
