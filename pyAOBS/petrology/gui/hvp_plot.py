"""KKHS02 Fig.12-style H–Vp diagram helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from matplotlib import cm as mpl_cm
from matplotlib.axes import Axes

# ``reproduce_fig12.py`` defaults (H–Vp / linear Step-4 track)
# KKHS02 §4 / Fig.11 standard linear productivity (same for Fig.12-style H–Vp).
FIG12_DFDP_PER_GPA = 0.12
FIG12_TP_CONTOUR_INTERVAL_C = 50.0

# Axis limits / χ families shared with paper Fig.12 layout (constants only).
from petrology.hvp.fig12a_background import (  # noqa: E402
    CHI1_B_DASHED,
    CHI_SOLID,
    H_LIM,
    VP_LIM,
)

FIG12_SOLID_CHI_VALUES = CHI_SOLID
FIG12_CHI1_B_VALUES_KM = CHI1_B_DASHED
FIG12_H_LIM_KM = H_LIM
FIG12_VP_LIM_KM_S = VP_LIM
from petrology.hvp.observation_overlay import ObservationPoint, plot_observation_on_hvp
from petrology.hvp.track_registry import (
    HVP_DISPLAY_BOTH,
    HVP_DISPLAY_CHOICES,
    HVP_DISPLAY_DEFAULT,
    HVP_DISPLAY_FIG12,
    HVP_DISPLAY_HVP,
    HVP_DISPLAY_MODERN,
)

__all__ = [
    "FIG12_DFDP_PER_GPA",
    "HVP_DISPLAY_BOTH",
    "HVP_DISPLAY_CHOICES",
    "HVP_DISPLAY_DEFAULT",
    "HVP_DISPLAY_FIG12",
    "HVP_DISPLAY_HVP",
    "HVP_DISPLAY_MODERN",
    "ObservationPoint",
    "build_modern_hvp_curves",
    "modern_hvp_from_scan_grid",
    "plot_hvp_scan_panel",
    "plot_observation_on_hvp",
]


class HvpPlotPoint(Protocol):
    tp_c: float
    chi: float
    h_km: float
    vp_bulk_km_s: float


@dataclass(frozen=True)
class HvpCurvePoint:
    tp_c: float
    chi: float
    h_km: float
    vp_bulk_km_s: float
    b_km: float = 0.0
    fbar: float = 0.0
    pbar_gpa: float = 0.0


def _tp_contour_levels(
    points: Sequence[HvpPlotPoint],
    *,
    tp_range_c: tuple[float, float] | None,
    interval_c: float,
) -> np.ndarray:
    uniq_tp = np.array(sorted({p.tp_c for p in points}), dtype=float)
    if uniq_tp.size == 0:
        return np.array([], dtype=float)
    if tp_range_c is None:
        tp_lo, tp_hi = float(uniq_tp[0]), float(uniq_tp[-1])
    else:
        tp_lo, tp_hi = tp_range_c
    start = float(np.ceil(tp_lo / interval_c) * interval_c)
    stop = float(np.floor(tp_hi / interval_c) * interval_c)
    if stop < start:
        return np.array([], dtype=float)
    return np.arange(start, stop + 0.5 * interval_c, interval_c)


def default_tp_contour_interval_c(points: Sequence[HvpPlotPoint]) -> float:
    """Paper uses 50 °C; coarser grids use a multiple of the Tp step."""
    tp_u = sorted({p.tp_c for p in points})
    if len(tp_u) < 2:
        return FIG12_TP_CONTOUR_INTERVAL_C
    step = float(tp_u[1] - tp_u[0])
    span = float(tp_u[-1] - tp_u[0])
    if span <= 120.0:
        return max(step * 2.0, 25.0)
    return FIG12_TP_CONTOUR_INTERVAL_C


def sweep_fig12_linear_hvp(
    *,
    tp_values_c: Sequence[float],
    chi_values: Sequence[float],
    b_km: float = 0.0,
    dfdp_per_gpa: float = FIG12_DFDP_PER_GPA,
    melting_law: object | None = None,
    vp_bias_km_s: float = 0.0,
) -> list[HvpCurvePoint]:
    """KKHS02 Step-4 linear active upwelling (``reproduce_fig12.py``).

    ``vp_bias_km_s`` is ignored (always 0): production uses raw eq.(1).
    """
    from petrology.active_upwelling import sweep_hvp, sweep_hvp_law

    del vp_bias_km_s
    if melting_law is not None:
        res = sweep_hvp_law(
            tp_values_c=tp_values_c,
            chi_values=chi_values,
            b_km=b_km,
            melting_law=melting_law,
            vp_bias_km_s=0.0,
        )
    else:
        res = sweep_hvp(
            tp_values_c=tp_values_c,
            chi_values=chi_values,
            b_km=b_km,
            dfdp_per_gpa=dfdp_per_gpa,
            vp_bias_km_s=0.0,
        )
    return [
        HvpCurvePoint(
            tp_c=r.tp_c,
            chi=r.chi,
            h_km=r.h_km,
            vp_bulk_km_s=r.vp_bulk_km_s,
            b_km=float(b_km),
            fbar=r.fbar,
            pbar_gpa=r.pbar_gpa,
        )
        for r in res
    ]


def sweep_modern_hvp_grid(
    *,
    tp_values_c: Sequence[float],
    chi_values: Sequence[float],
    col_kw: dict,
    b_km: float = 0.0,
    pyroxenite_frac: float = 0.10,
    melting_engine: str = "reebox",
) -> list[HvpCurvePoint]:
    """Modern REEBOX / LIP forward grid (eq.(1) raw Vp, no Fig.12 bias).

    Points that cannot form a melting column (e.g. kinzler_linear with
    ``b/30 ≥ P0``) are skipped, matching ``sweep_hvp_law``.
    """
    from petrology.melting.column import forward_melting_column

    pts: list[HvpCurvePoint] = []
    for tp in tp_values_c:
        for chi in chi_values:
            try:
                col = forward_melting_column(
                    tp_c=float(tp),
                    chi=float(chi),
                    b_km=float(b_km),
                    pyroxenite_frac=float(pyroxenite_frac),
                    melting_engine=melting_engine,
                    **col_kw,
                )
            except (ValueError, RuntimeError):
                continue
            pts.append(
                HvpCurvePoint(
                    tp_c=float(tp),
                    chi=float(chi),
                    h_km=col.h_km,
                    vp_bulk_km_s=col.vp_bulk_eq1_km_s,
                    b_km=float(b_km),
                    fbar=col.fbar,
                    pbar_gpa=col.pbar_gpa,
                )
            )
    return pts


def build_modern_hvp_curves(
    *,
    tp_values_c: Sequence[float],
    col_kw: dict,
    solid_chi_values: Sequence[float] = FIG12_SOLID_CHI_VALUES,
    chi1_b_values_km: Sequence[float] = FIG12_CHI1_B_VALUES_KM,
    pyroxenite_frac: float = 0.10,
    melting_engine: str = "reebox",
    b_km_solid: float = 0.0,
) -> tuple[list[HvpCurvePoint], list[tuple[float, list[HvpCurvePoint]]]]:
    """
    Modern REEBOX curves in the same Fig.12 layout:

    - coloured solids: χ ∈ solid_chi_values at ``b_km_solid`` (default 0)
    - grey dashes: χ = 1 at b = 10, 20, 30 km
    """
    main = sweep_modern_hvp_grid(
        tp_values_c=tp_values_c,
        chi_values=solid_chi_values,
        col_kw=col_kw,
        b_km=b_km_solid,
        pyroxenite_frac=pyroxenite_frac,
        melting_engine=melting_engine,
    )
    chi1_b: list[tuple[float, list[HvpCurvePoint]]] = []
    for b_val in chi1_b_values_km:
        pts = sweep_modern_hvp_grid(
            tp_values_c=tp_values_c,
            chi_values=[1.0],
            col_kw=col_kw,
            b_km=float(b_val),
            pyroxenite_frac=pyroxenite_frac,
            melting_engine=melting_engine,
        )
        if pts:
            chi1_b.append((float(b_val), pts))
    return main, chi1_b


def modern_hvp_from_scan_grid(
    modern_points: Sequence[HvpPlotPoint],
    *,
    col_kw: dict,
    pyroxenite_frac: float = 0.10,
    melting_engine: str = "reebox",
    include_chi1_b_family: bool = True,
    chi1_b_values_km: Sequence[float] = FIG12_CHI1_B_VALUES_KM,
    solid_chi_values: Sequence[float] | None = None,
) -> tuple[list[HvpCurvePoint], list[tuple[float, list[HvpCurvePoint]]]]:
    """Build Modern Fig.12-style curves on the Tp (and χ) grid of a scan.

    Solid χ curves reuse scan points when ``solid_chi_values`` is None; otherwise
    they are recomputed on the scan Tp grid. The χ=1,b family is always computed
    via ``forward_melting_column`` (scan is typically only at one b).
    """
    if not modern_points:
        return [], []
    tp_u = sorted({p.tp_c for p in modern_points})
    if solid_chi_values is None:
        # Reuse scan H–Vp samples as solid χ tracks (same Tp–χ grid as the scan).
        main = [
            HvpCurvePoint(
                tp_c=float(p.tp_c),
                chi=float(p.chi),
                h_km=float(p.h_km),
                vp_bulk_km_s=float(p.vp_bulk_km_s),
                b_km=float(getattr(p, "b_km", 0.0) or 0.0),
                fbar=float(getattr(p, "fbar", 0.0) or 0.0),
                pbar_gpa=float(getattr(p, "pbar_gpa", 0.0) or 0.0),
            )
            for p in modern_points
        ]
        chi1_b: list[tuple[float, list[HvpCurvePoint]]] = []
        if include_chi1_b_family:
            for b_val in chi1_b_values_km:
                pts = sweep_modern_hvp_grid(
                    tp_values_c=tp_u,
                    chi_values=[1.0],
                    col_kw=col_kw,
                    b_km=float(b_val),
                    pyroxenite_frac=pyroxenite_frac,
                    melting_engine=melting_engine,
                )
                if pts:
                    chi1_b.append((float(b_val), pts))
        return main, chi1_b

    return build_modern_hvp_curves(
        tp_values_c=tp_u,
        col_kw=col_kw,
        solid_chi_values=solid_chi_values,
        chi1_b_values_km=chi1_b_values_km if include_chi1_b_family else (),
        pyroxenite_frac=pyroxenite_frac,
        melting_engine=melting_engine,
    )


def build_fig12_linear_curves(
    *,
    tp_values_c: Sequence[float],
    solid_chi_values: Sequence[float] = FIG12_SOLID_CHI_VALUES,
    chi1_b_values_km: Sequence[float] = FIG12_CHI1_B_VALUES_KM,
    dfdp_per_gpa: float = FIG12_DFDP_PER_GPA,
    melting_law: object | None = None,
    vp_bias_km_s: float = 0.0,
) -> tuple[list[HvpCurvePoint], list[tuple[float, list[HvpCurvePoint]]]]:
    """
    KKHS02 Fig.12 curve sets (``reproduce_fig12.py`` layout):

    - coloured solids: χ ∈ {1,2,4,8} at b = 0
    - grey dashes: χ = 1 at b = 10, 20, 30 km
    """
    main = sweep_fig12_linear_hvp(
        tp_values_c=tp_values_c,
        chi_values=solid_chi_values,
        b_km=0.0,
        dfdp_per_gpa=dfdp_per_gpa,
        melting_law=melting_law,
        vp_bias_km_s=vp_bias_km_s,
    )
    chi1_b: list[tuple[float, list[HvpCurvePoint]]] = []
    for b_val in chi1_b_values_km:
        pts = sweep_fig12_linear_hvp(
            tp_values_c=tp_values_c,
            chi_values=[1.0],
            b_km=float(b_val),
            dfdp_per_gpa=dfdp_per_gpa,
            melting_law=melting_law,
            vp_bias_km_s=vp_bias_km_s,
        )
        if pts:
            chi1_b.append((float(b_val), pts))
    return main, chi1_b


def linear_hvp_from_scan_grid(
    modern_points: Sequence[HvpPlotPoint],
    *,
    b_km: float = 0.0,
    dfdp_per_gpa: float = FIG12_DFDP_PER_GPA,
    vp_bias_km_s: float = 0.0,
    include_chi1_b_family: bool = True,
) -> tuple[list[HvpCurvePoint], list[tuple[float, list[HvpCurvePoint]]]]:
    """Build Fig.12 linear curves on the Tp grid of a Modern scan."""
    if not modern_points:
        return [], []
    tp_u = sorted({p.tp_c for p in modern_points})
    main, chi1_b = build_fig12_linear_curves(
        tp_values_c=tp_u,
        dfdp_per_gpa=dfdp_per_gpa,
        vp_bias_km_s=vp_bias_km_s,
    )
    if not include_chi1_b_family:
        chi1_b = []
    return main, chi1_b


def _plot_chi1_b_dashed_family(
    ax: Axes,
    chi1_b_lines: Sequence[tuple[float, Sequence[HvpPlotPoint]]],
    *,
    legend_prefix: str = "",
) -> None:
    for j, (b_val, line) in enumerate(chi1_b_lines):
        ordered = sorted(line, key=lambda x: x.tp_c)
        if len(ordered) < 2:
            continue
        shade = 0.15 + 0.18 * (j / max(len(chi1_b_lines) - 1, 1))
        ax.plot(
            [p.h_km for p in ordered],
            [p.vp_bulk_km_s for p in ordered],
            lw=1.1,
            ls="--",
            color=str(shade),
            label=f"{legend_prefix}χ=1, b={b_val:g}",
            zorder=2,
        )


def _isotherm_points_at_tp(
    points: Sequence[HvpPlotPoint],
    chi1_b_lines: Sequence[tuple[float, Sequence[HvpPlotPoint]]] | None,
    tp_use: float,
) -> list[HvpPlotPoint]:
    """Collect all H–Vp samples at one Tp (solid χ + optional χ=1,b family)."""
    pts = [p for p in points if abs(p.tp_c - tp_use) < 1e-6]
    if chi1_b_lines:
        for _b, line in chi1_b_lines:
            pts.extend(p for p in line if abs(p.tp_c - tp_use) < 1e-6)
    # Sort by H so the isotherm crosses the χ-family left→right.
    # HvpScanPoint has no b_km; HvpCurvePoint / active-upwelling rows may.
    pts = sorted(
        pts,
        key=lambda x: (x.h_km, x.chi, float(getattr(x, "b_km", 0.0) or 0.0)),
    )
    return pts


def _visible_isotherm_xy(
    h: list[float],
    v: list[float],
    h_lim: tuple[float, float],
    v_lim: tuple[float, float],
) -> tuple[list[float], list[float]]:
    """Keep isotherm samples inside the axis box; interpolate edge crossings.

    High-Tp / high-χ points often fall outside the paper H–Vp window; without
    clipping, isotherms appear to stop mid-family even though χ-curves continue.
    """
    if len(h) < 2:
        return [], []

    def inside(x: float, y: float) -> bool:
        return h_lim[0] <= x <= h_lim[1] and v_lim[0] <= y <= v_lim[1]

    def edge_hit(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float] | None:
        dx, dy = x1 - x0, y1 - y0
        best_t = None
        for lim, a0, da in (
            (h_lim[0], x0, dx),
            (h_lim[1], x0, dx),
            (v_lim[0], y0, dy),
            (v_lim[1], y0, dy),
        ):
            if abs(da) < 1e-15:
                continue
            t = (lim - a0) / da
            if 0.0 < t < 1.0:
                xi, yi = x0 + t * dx, y0 + t * dy
                if inside(xi, yi) or (
                    h_lim[0] - 1e-9 <= xi <= h_lim[1] + 1e-9
                    and v_lim[0] - 1e-9 <= yi <= v_lim[1] + 1e-9
                ):
                    if best_t is None or t < best_t:
                        best_t = t
        if best_t is None:
            return None
        return (x0 + best_t * dx, y0 + best_t * dy)

    out_h: list[float] = []
    out_v: list[float] = []
    for i in range(len(h)):
        x, y = h[i], v[i]
        if inside(x, y):
            if i > 0 and not inside(h[i - 1], v[i - 1]):
                hit = edge_hit(h[i - 1], v[i - 1], x, y)
                if hit is not None:
                    out_h.append(hit[0])
                    out_v.append(hit[1])
            out_h.append(x)
            out_v.append(y)
        elif i > 0 and inside(h[i - 1], v[i - 1]):
            hit = edge_hit(h[i - 1], v[i - 1], x, y)
            if hit is not None:
                out_h.append(hit[0])
                out_v.append(hit[1])
    return out_h, out_v


def plot_hvp_fig12_style(
    ax: Axes,
    points: Sequence[HvpPlotPoint],
    *,
    chi1_b_lines: Sequence[tuple[float, Sequence[HvpPlotPoint]]] | None = None,
    tp_range_c: tuple[float, float] | None = None,
    tp_contour_interval_c: float | None = None,
    label_fontsize: float = 7.0,
    chi_legend: bool = True,
    legend_prefix: str = "",
    use_paper_axis_limits: bool = False,
) -> None:
    """
    Draw H–Vp diagram (KKHS02 Fig.12-style computed track):

    - solid coloured lines: constant χ at b = 0, Tp varies along curve
    - grey dashed lines: constant-Tp isotherms crossing the χ-family
    - optional dashed χ = 1 family at b = 10, 20, 30 km
    """
    if not points:
        return

    chi_uniq = sorted({p.chi for p in points})
    interval = (
        tp_contour_interval_c
        if tp_contour_interval_c is not None
        else default_tp_contour_interval_c(points)
    )
    uniq_tp = np.array(sorted({p.tp_c for p in points}), dtype=float)

    h_vals = [p.h_km for p in points]
    v_vals = [p.vp_bulk_km_s for p in points]
    if chi1_b_lines:
        for _, line in chi1_b_lines:
            h_vals.extend(p.h_km for p in line)
            v_vals.extend(p.vp_bulk_km_s for p in line)
    h_pad = max(0.8, 0.04 * (max(h_vals) - min(h_vals) + 1e-6))
    v_pad = max(0.02, 0.04 * (max(v_vals) - min(v_vals) + 1e-6))
    if use_paper_axis_limits:
        h_lim = FIG12_H_LIM_KM
        v_lim = FIG12_VP_LIM_KM_S
    else:
        h_lim = (min(h_vals) - h_pad, max(h_vals) + h_pad)
        v_lim = (min(v_vals) - v_pad, max(v_vals) + v_pad)

    for tp in _tp_contour_levels(points, tp_range_c=tp_range_c, interval_c=interval):
        tp_use = float(uniq_tp[np.argmin(np.abs(uniq_tp - tp))])
        pts = _isotherm_points_at_tp(points, chi1_b_lines, tp_use)
        if len(pts) < 2:
            continue
        h = [p.h_km for p in pts]
        v = [p.vp_bulk_km_s for p in pts]
        hs, vs = _visible_isotherm_xy(h, v, h_lim, v_lim)
        if len(hs) < 2:
            # still draw full polyline; axes clip — keeps continuity across χ
            hs, vs = h, v
        ax.plot(hs, vs, color="0.55", lw=1.0, ls="--", alpha=0.9, zorder=1, clip_on=True)
        i_lab = min(max(int(0.55 * (len(hs) - 1)), 0), len(hs) - 1)
        x_lab = float(np.clip(hs[i_lab], h_lim[0] + 0.3, h_lim[1] - 0.3))
        y_lab = float(np.clip(vs[i_lab], v_lim[0] + 0.005, v_lim[1] - 0.005))
        ax.text(
            x_lab,
            y_lab,
            f"{int(round(tp_use))}°C",
            fontsize=label_fontsize,
            color="0.35",
            zorder=2,
            clip_on=True,
        )

    for i, chi in enumerate(chi_uniq):
        line = sorted([p for p in points if p.chi == chi], key=lambda x: x.tp_c)
        if len(line) < 2:
            continue
        color = mpl_cm.plasma(i / max(len(chi_uniq) - 1, 1))
        label = f"{legend_prefix}χ={chi:g}, b=0" if chi_legend else None
        ax.plot(
            [p.h_km for p in line],
            [p.vp_bulk_km_s for p in line],
            lw=1.5,
            color=color,
            label=label,
            zorder=3,
        )

    if chi1_b_lines:
        _plot_chi1_b_dashed_family(ax, chi1_b_lines, legend_prefix=legend_prefix)

    if use_paper_axis_limits:
        ax.set_xlim(*FIG12_H_LIM_KM)
        ax.set_ylim(*FIG12_VP_LIM_KM_S)
    else:
        ax.set_xlim(*h_lim)
        ax.set_ylim(*v_lim)


def _auto_vp_ylim(*point_sets: Sequence[HvpPlotPoint]) -> tuple[float, float]:
    v_all: list[float] = []
    for pts in point_sets:
        v_all.extend(p.vp_bulk_km_s for p in pts)
    if not v_all:
        return 6.85, 7.55
    v_pad = max(0.03, 0.04 * (max(v_all) - min(v_all) + 1e-6))
    return min(v_all) - v_pad, max(v_all) + v_pad


def plot_hvp_dual_track(
    ax: Axes,
    modern_points: Sequence[HvpPlotPoint],
    linear_points: Sequence[HvpPlotPoint],
    *,
    chi1_b_lines: Sequence[tuple[float, Sequence[HvpPlotPoint]]] | None = None,
    modern_chi1_b_lines: Sequence[tuple[float, Sequence[HvpPlotPoint]]] | None = None,
    tp_range_c: tuple[float, float] | None = None,
    tp_contour_interval_c: float | None = None,
    label_fontsize: float = 7.0,
    vp_bias_km_s: float = 0.0,
    use_paper_axis_limits: bool = False,
) -> None:
    """Overlay linear Fig.12 track and Modern REEBOX with matching line kinds.

    Both tracks use plasma χ colours, grey χ=1,b dashes, and grey Tp isotherms.
    Modern solids are dotted (vs linear solid) so the two families stay readable.
    """
    del vp_bias_km_s  # production always uses raw eq.(1); keep arg for API compat
    plot_hvp_fig12_style(
        ax,
        linear_points,
        chi1_b_lines=chi1_b_lines,
        tp_range_c=tp_range_c,
        tp_contour_interval_c=tp_contour_interval_c,
        label_fontsize=label_fontsize,
        chi_legend=True,
        legend_prefix="",
        use_paper_axis_limits=use_paper_axis_limits,
    )

    if modern_points:
        chi_uniq = sorted({p.chi for p in modern_points})
        for i, chi in enumerate(chi_uniq):
            line = sorted([p for p in modern_points if p.chi == chi], key=lambda x: x.tp_c)
            if len(line) < 2:
                continue
            color = mpl_cm.plasma(i / max(len(chi_uniq) - 1, 1))
            ax.plot(
                [p.h_km for p in line],
                [p.vp_bulk_km_s for p in line],
                color=color,
                lw=1.8,
                ls=":",
                alpha=0.85,
                zorder=4,
            )
        if modern_chi1_b_lines:
            _plot_chi1_b_dashed_family(
                ax, modern_chi1_b_lines, legend_prefix="M "
            )
        # Modern Tp isotherms (same grey dashed style; slightly lighter)
        interval = (
            tp_contour_interval_c
            if tp_contour_interval_c is not None
            else default_tp_contour_interval_c(modern_points)
        )
        uniq_tp = np.array(sorted({p.tp_c for p in modern_points}), dtype=float)
        if use_paper_axis_limits:
            h_lim = FIG12_H_LIM_KM
            v_lim = FIG12_VP_LIM_KM_S
        else:
            h_vals = [p.h_km for p in modern_points]
            v_vals = [p.vp_bulk_km_s for p in modern_points]
            if modern_chi1_b_lines:
                for _, line in modern_chi1_b_lines:
                    h_vals.extend(p.h_km for p in line)
                    v_vals.extend(p.vp_bulk_km_s for p in line)
            h_pad = max(0.8, 0.04 * (max(h_vals) - min(h_vals) + 1e-6))
            v_pad = max(0.02, 0.04 * (max(v_vals) - min(v_vals) + 1e-6))
            h_lim = (min(h_vals) - h_pad, max(h_vals) + h_pad)
            v_lim = (min(v_vals) - v_pad, max(v_vals) + v_pad)
        for tp in _tp_contour_levels(modern_points, tp_range_c=tp_range_c, interval_c=interval):
            tp_use = float(uniq_tp[np.argmin(np.abs(uniq_tp - tp))])
            pts = _isotherm_points_at_tp(modern_points, modern_chi1_b_lines, tp_use)
            if len(pts) < 2:
                continue
            h = [p.h_km for p in pts]
            v = [p.vp_bulk_km_s for p in pts]
            hs, vs = _visible_isotherm_xy(h, v, h_lim, v_lim)
            if len(hs) < 2:
                hs, vs = h, v
            ax.plot(hs, vs, color="0.55", lw=1.0, ls="--", alpha=0.45, zorder=1.5, clip_on=True)

        ax.plot([], [], color="0.35", lw=1.5, ls="-", label="H–Vp 线性（实线）")
        ax.plot([], [], color=mpl_cm.plasma(0.5), lw=1.8, ls=":", label="Modern（点线）")

    extra: list[Sequence[HvpPlotPoint]] = [modern_points, linear_points]
    if chi1_b_lines:
        for _, line in chi1_b_lines:
            extra.append(line)
    if modern_chi1_b_lines:
        for _, line in modern_chi1_b_lines:
            extra.append(line)
    if not use_paper_axis_limits:
        v_lo, v_hi = _auto_vp_ylim(*extra)
        ax.set_ylim(v_lo, v_hi)


def plot_hvp_scan_panel(
    ax: Axes,
    modern_points: Sequence[HvpPlotPoint],
    *,
    display_mode: str = HVP_DISPLAY_DEFAULT,
    b_km: float = 0.0,
    tp_range_c: tuple[float, float] | None = None,
    tp_contour_interval_c: float | None = None,
    label_fontsize: float = 7.0,
    vp_bias_km_s: float = 0.0,
    include_chi1_b_family: bool = True,
    use_paper_axis_limits: bool = False,
    observation: ObservationPoint | None = None,
    delta_vp_max_km_s: float = 0.15,
    thick_crust_h_min_km: float = 15.0,
    show_observation_read_band: bool = True,
    linear_cache: tuple[list[HvpCurvePoint], list[tuple[float, list[HvpCurvePoint]]]]
    | None = None,
    modern_cache: tuple[list[HvpCurvePoint], list[tuple[float, list[HvpCurvePoint]]]]
    | None = None,
) -> None:
    """Draw H–Vp panel for LIP GUI according to display mode.

    Pass ``linear_cache`` / ``modern_cache`` from previous builders so mode
    switches do not recompute tracks (keeps UI responsive). Modern mode uses
    the same Fig.12 line kinds: plasma χ solids, grey χ=1,b dashes, Tp isotherms.
    """
    del b_km  # H–Vp solids always at b = 0; χ = 1 dashes use FIG12_CHI1_B_VALUES_KM
    del vp_bias_km_s  # always raw eq.(1)

    if display_mode in (HVP_DISPLAY_HVP, HVP_DISPLAY_FIG12, HVP_DISPLAY_BOTH):
        if linear_cache is not None:
            linear, chi1_b = linear_cache
            if not include_chi1_b_family:
                chi1_b = []
        else:
            linear, chi1_b = linear_hvp_from_scan_grid(
                modern_points,
                vp_bias_km_s=0.0,
                include_chi1_b_family=include_chi1_b_family,
            )
    else:
        linear, chi1_b = [], []

    modern_main = modern_points
    modern_chi1_b: list[tuple[float, list[HvpCurvePoint]]] = []
    if display_mode in (HVP_DISPLAY_MODERN, HVP_DISPLAY_BOTH):
        if modern_cache is not None:
            modern_main, modern_chi1_b = modern_cache
            if not include_chi1_b_family:
                modern_chi1_b = []
        # else: solids from scan points only (no χ=1,b until cache is provided)

    if display_mode in (HVP_DISPLAY_HVP, HVP_DISPLAY_FIG12):
        plot_hvp_fig12_style(
            ax,
            linear,
            chi1_b_lines=chi1_b,
            tp_range_c=tp_range_c,
            tp_contour_interval_c=tp_contour_interval_c or FIG12_TP_CONTOUR_INTERVAL_C,
            label_fontsize=label_fontsize,
            legend_prefix="",
            use_paper_axis_limits=use_paper_axis_limits,
        )
    elif display_mode == HVP_DISPLAY_BOTH:
        plot_hvp_dual_track(
            ax,
            modern_main,
            linear,
            chi1_b_lines=chi1_b,
            modern_chi1_b_lines=modern_chi1_b or None,
            tp_range_c=tp_range_c,
            tp_contour_interval_c=tp_contour_interval_c or FIG12_TP_CONTOUR_INTERVAL_C,
            label_fontsize=label_fontsize,
            vp_bias_km_s=0.0,
            use_paper_axis_limits=use_paper_axis_limits,
        )
    else:
        plot_hvp_fig12_style(
            ax,
            modern_main,
            chi1_b_lines=modern_chi1_b or None,
            tp_range_c=tp_range_c,
            tp_contour_interval_c=tp_contour_interval_c or FIG12_TP_CONTOUR_INTERVAL_C,
            label_fontsize=label_fontsize,
            use_paper_axis_limits=use_paper_axis_limits,
        )

    if observation is not None:
        plot_observation_on_hvp(
            ax,
            observation,
            delta_vp_max_km_s=delta_vp_max_km_s,
            h_min_km=thick_crust_h_min_km,
            show_read_band=show_observation_read_band,
        )
