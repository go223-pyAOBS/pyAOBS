"""
Schematic: passive vs active melting + thickness H and bulk velocity Vp.

Run:
  python petrology/validation/plot_active_passive_schematic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Polygon
import numpy as np

from petrology.active_upwelling import solve_active_upwelling
from petrology.fractionation import delta_vp_km_s
from petrology.gui.hvp_plot import (
    FIG12_CHI1_B_VALUES_KM,
    FIG12_DFDP_PER_GPA,
    FIG12_H_LIM_KM,
    FIG12_SOLID_CHI_VALUES,
    FIG12_TP_CONTOUR_INTERVAL_C,
    FIG12_VP_LIM_KM_S,
    HvpCurvePoint,
    build_fig12_linear_curves,
    build_modern_hvp_curves,
    plot_hvp_fig12_style,
)
from petrology.invert import bulk_vp_bounds
from petrology.melting.column import forward_melting_column
from petrology.melting.lithology import heterogeneous_source
from petrology.melting.pymelt_lithology_adapter import resolve_lithology_col_kwargs
from petrology.melting.reebox_geometry import ReeboxColumnGeometry, build_reebox_column
from petrology.vp_regression import predict_vp_km_s

OUT = Path(__file__).resolve().parents[1] / "figures" / "active_passive_melting_schematic.png"


def schematic_png_path() -> Path:
    return OUT


def ensure_schematic_png() -> Path:
    """Return PNG path, generating it if missing."""
    if not OUT.is_file():
        main()
    return OUT

# Illustrative Greenland-style anchor for panel (f)
V_LC_OBS = 7.0
H_OBS = 30.0
F_LOWER = 0.5
F_SOLID = 0.75
P_FC_MPA = 400.0

# Fig.12-style H–Vp grid (panel e): match reproduce_fig12 / paper axis window
TP_GRID_HVP = np.arange(1200.0, 1600.0 + 0.5, 25.0)
CHI_GRID_HVP = FIG12_SOLID_CHI_VALUES
# Coarser grids for other schematic panels
TP_GRID = np.arange(1200.0, 1425.0, 25.0)
CHI_GRID = (1.0, 2.0, 4.0, 8.0, 12.0, 16.0)
FT_TP_SHOW = (1250.0, 1325.0, 1400.0)


def _active_weight(p: np.ndarray, p_max: float, p_min: float, chi: float) -> np.ndarray:
    if chi <= 1.0:
        return np.zeros_like(p)
    lam = chi - 1.0
    span = max(p_max - p_min, 1e-9)
    w_norm = (p_max - p) / span
    return lam * np.exp(-w_norm / 0.5)


def _f_profile(p: np.ndarray, p0: float, pf: float, f_max: float) -> np.ndarray:
    """Illustrative F(P) along the longest flow line (linear ramp)."""
    f = np.zeros_like(p)
    mask = (p <= p0) & (p >= pf)
    f[mask] = f_max * (p0 - p[mask]) / max(p0 - pf, 1e-9)
    return f


def _tc_profile(f: np.ndarray, rho: float = 3.3) -> np.ndarray:
    scale = 1.0 / (rho * 9.81 * 1e3)
    return scale * f / np.maximum(1.0 - f, 1e-9)


def _vp_grid(p_gpa: np.ndarray, f_melt: np.ndarray) -> np.ndarray:
    vp = np.empty_like(p_gpa, dtype=float)
    it = np.nditer(p_gpa, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        i = it.multi_index
        vp[i] = predict_vp_km_s(float(p_gpa[i]), float(f_melt[i]))
        it.iternext()
    return vp


def _build_hvp_grid(col_kw: dict) -> tuple[list[HvpCurvePoint], list[tuple[float, list[HvpCurvePoint]]]]:
    """Modern LIP track in Fig.12 layout: χ solids + χ=1,b dashes (eq.(1) raw)."""
    return build_modern_hvp_curves(
        tp_values_c=TP_GRID_HVP,
        col_kw=col_kw,
        solid_chi_values=CHI_GRID_HVP,
        chi1_b_values_km=FIG12_CHI1_B_VALUES_KM,
        pyroxenite_frac=0.10,
        melting_engine="reebox",
    )


def _build_hvp_fig12_curves() -> tuple[list[HvpCurvePoint], list[tuple[float, list[HvpCurvePoint]]]]:
    """Fig.12 linear track + χ=1 dashed b-family (same as reproduce_fig12)."""
    return build_fig12_linear_curves(
        tp_values_c=TP_GRID_HVP,
        solid_chi_values=FIG12_SOLID_CHI_VALUES,
        chi1_b_values_km=FIG12_CHI1_B_VALUES_KM,
        dfdp_per_gpa=FIG12_DFDP_PER_GPA,
        vp_bias_km_s=0.0,
    )


def _bulk_solidus_t(p_gpa: np.ndarray, lithologies: list) -> np.ndarray:
    return np.array([min(L.solidus_gpa(float(p)) for L in lithologies) for p in p_gpa])


def _forward_point(tp_c: float, chi: float, *, col_kw: dict) -> tuple[float, float, float, float]:
    col = forward_melting_column(
        tp_c=float(tp_c),
        chi=float(chi),
        b_km=0.0,
        pyroxenite_frac=0.10,
        melting_engine="reebox",
        **col_kw,
    )
    return col.h_km, col.vp_bulk_eq1_km_s, col.pbar_gpa, col.fbar


def _liths_from_kw(col_kw: dict, phi: float = 0.10) -> list:
    return heterogeneous_source(
        pyroxenite_frac=phi,
        backend=col_kw.get("lithology_backend", "pymelt"),
        peridotite_key=col_kw.get("peridotite_lith", "katz_lherzolite"),
        pyroxenite_key=col_kw.get("pyroxenite_lith", "pertermann_g2"),
        peridotite_h2o_wt=col_kw.get("peridotite_h2o_wt", 0.0),
        pyroxenite_h2o_wt=col_kw.get("pyroxenite_h2o_wt", 0.0),
    )


def _plot_ft_panel(
    ax,
    *,
    ref_tp: float,
    ref_chi: float,
    liths: list,
    forward_mode: bool = False,
    rg=None,
    highlight_fbar: float | None = None,
    highlight_fmax: float | None = None,
) -> None:
    """F vs T along REEBOX isentropic paths for several Tp (fixed chi)."""
    if forward_mode and rg is not None:
        t_path = np.array([s.t_c for s in rg.path.steps], dtype=float)
        f_path = np.array([rg.path.f_bulk_at(s) for s in rg.path.steps], dtype=float)
        ax.plot(t_path, f_path, color="#c0392b", lw=2.4, zorder=4, label=rf"$T_p$={ref_tp:.0f}$^\circ$C")
        ax.scatter([t_path[0]], [f_path[0]], s=40, c="#c0392b", edgecolors="white", zorder=5)
        ax.scatter([t_path[-1]], [f_path[-1]], s=40, c="#c0392b", marker="s", edgecolors="white", zorder=5)
        if highlight_fmax is not None:
            ax.scatter(
                [t_path[-1]],
                [highlight_fmax],
                s=65,
                c="#8e44ad",
                marker="*",
                edgecolors="white",
                zorder=6,
                label=rf"$F_{{\max}}$={highlight_fmax:.3f}",
            )
        if highlight_fbar is not None:
            ax.axhline(
                highlight_fbar,
                color="#e67e22",
                ls="--",
                lw=1.3,
                zorder=3,
                label=rf"$\bar F$={highlight_fbar:.3f}",
            )
        ax.set_xlabel(r"温度 $T$ ($^\circ$C)  $\leftarrow$ 浅        深 $\rightarrow$")
        ax.set_ylabel("熔融程度 F")
        ax.set_xlim(t_path.max() + 25, t_path.min() - 25)
        ax.legend(fontsize=7, loc="lower right")
        ax.text(
            0.03,
            0.97,
            rf"正演联动 · $\chi$={ref_chi:g}；圆点=$P_0$，方点=$P_f$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc"),
        )
        return

    colors = ("#95a5a6", "#e67e22", "#c0392b")
    t_all: list[float] = []
    for tp, col in zip(FT_TP_SHOW, colors):
        rg = build_reebox_column(
            liths,
            tp_c=float(tp),
            b_km=0.0,
            chi=float(ref_chi),
            n_isentropic_steps=80,
        )
        t_path = np.array([s.t_c for s in rg.path.steps], dtype=float)
        f_path = np.array([rg.path.f_bulk_at(s) for s in rg.path.steps], dtype=float)
        t_all.extend(t_path.tolist())
        lw = 2.4 if abs(tp - ref_tp) < 1e-6 else 1.3
        zorder = 4 if abs(tp - ref_tp) < 1e-6 else 2
        ax.plot(
            t_path,
            f_path,
            color=col,
            lw=lw,
            zorder=zorder,
            label=rf"$T_p$={tp:.0f}$^\circ$C",
        )
        ax.scatter([t_path[0]], [f_path[0]], s=35, c=col, edgecolors="white", zorder=zorder + 1)
        ax.scatter([t_path[-1]], [f_path[-1]], s=35, c=col, marker="s", edgecolors="white", zorder=zorder + 1)

    t_min, t_max = min(t_all), max(t_all)
    ax.set_xlabel(r"温度 $T$ ($^\circ$C)  $\leftarrow$ 浅        深 $\rightarrow$")
    ax.set_ylabel("熔融程度 F")
    ax.set_xlim(t_max + 25, t_min - 25)
    ax.legend(fontsize=7, loc="lower right")
    ax.text(
        0.03,
        0.97,
        rf"固定 $\chi$={ref_chi:g}；圆点=$P_0$，方点=$P_f$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc"),
    )


def _plot_hvp_fig12_panel(
    ax,
    *,
    linear_pts: list[HvpCurvePoint],
    chi1_b_lines: list[tuple[float, list[HvpCurvePoint]]],
    modern_pts: list[HvpCurvePoint],
    modern_chi1_b_lines: list[tuple[float, list[HvpCurvePoint]]],
    ref_tp: float,
    ref_chi: float,
    h_ref: float,
    vp_ref: float,
    fbar_ref: float,
) -> None:
    """Fig.12-style H–Vp (paper axis) for both linear and Modern tracks."""
    from matplotlib import cm as mpl_cm

    plot_hvp_fig12_style(
        ax,
        linear_pts,
        chi1_b_lines=chi1_b_lines,
        tp_range_c=(float(TP_GRID_HVP[0]), float(TP_GRID_HVP[-1])),
        tp_contour_interval_c=FIG12_TP_CONTOUR_INTERVAL_C,
        label_fontsize=6.5,
        use_paper_axis_limits=True,
    )

    # Modern: same plasma χ colours / grey χ=1,b dashes / Tp isotherms (dotted solids)
    chi_uniq = sorted({p.chi for p in modern_pts})
    for i, chi in enumerate(chi_uniq):
        line = sorted([p for p in modern_pts if p.chi == chi], key=lambda x: x.tp_c)
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
            clip_on=True,
        )
    if modern_chi1_b_lines:
        for j, (b_val, line) in enumerate(modern_chi1_b_lines):
            ordered = sorted(line, key=lambda x: x.tp_c)
            if len(ordered) < 2:
                continue
            shade = 0.15 + 0.18 * (j / max(len(modern_chi1_b_lines) - 1, 1))
            ax.plot(
                [p.h_km for p in ordered],
                [p.vp_bulk_km_s for p in ordered],
                lw=1.2,
                ls="--",
                color=str(shade),
                alpha=0.7,
                zorder=3.5,
                clip_on=True,
            )
    ax.plot([], [], color=mpl_cm.plasma(0.5), lw=1.8, ls=":", label="Modern（点线）")

    lin_ref = solve_active_upwelling(
        tp_c=float(ref_tp),
        b_km=0.0,
        chi=float(ref_chi),
        dfdp_per_gpa=FIG12_DFDP_PER_GPA,
        vp_bias_km_s=0.0,
    )

    d_vp = delta_vp_km_s(F_SOLID, bulk_vp_km_s=vp_ref, p_fc_mpa=P_FC_MPA)
    bounds = bulk_vp_bounds(V_LC_OBS, f_lower=F_LOWER, delta_vp_fc_km_s=d_vp)

    ax.axvspan(H_OBS - 3, H_OBS + 3, color="#2ecc71", alpha=0.10, zorder=0)
    ax.axhspan(bounds.v_bulk_lower_km_s, bounds.v_bulk_upper_km_s, color="#9b59b6", alpha=0.12, zorder=0)
    ax.axhline(V_LC_OBS, color="#8e44ad", ls=":", lw=1.0, alpha=0.7, zorder=0)
    ax.scatter([H_OBS], [V_LC_OBS], s=90, c="#2ecc71", marker="D", edgecolors="#1e8449", zorder=6)
    ax.scatter(
        [lin_ref.h_km],
        [lin_ref.vp_bulk_km_s],
        s=55,
        c="#34495e",
        marker="o",
        edgecolors="white",
        zorder=6,
        label=rf"线性 Step-4 ($T_p$={ref_tp:.0f}, $\chi$={ref_chi:g})",
    )
    ax.scatter(
        [h_ref],
        [vp_ref],
        s=80,
        c="#f39c12",
        marker="*",
        edgecolors="#333333",
        zorder=6,
        label=rf"Modern 参考点 ($\bar F$≈{fbar_ref:.2f})",
        clip_on=False,
    )

    ax.set_xlabel(r"$H$ (km)")
    ax.set_ylabel(r"$V_{\mathrm{bulk}}$ (km/s)")
    ax.set_xlim(*FIG12_H_LIM_KM)
    ax.set_ylim(*FIG12_VP_LIM_KM_S)
    ax.legend(fontsize=6, loc="lower right")
    ax.text(
        0.03,
        0.97,
        "轴范围 = Fig.12 论文窗 "
        + rf"$H\in[{FIG12_H_LIM_KM[0]:g},{FIG12_H_LIM_KM[1]:g}]$ km, "
        + rf"$V\in[{FIG12_VP_LIM_KM_S[0]:g},{FIG12_VP_LIM_KM_S[1]:g}]$ km/s"
        + "\n实线/灰虚等温线/χ=1 虚线族：线性 Step-4（α=12%/GPa）"
        + "\n点线（同色 χ）+ χ=1 虚线族：Modern REEBOX",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        bbox=dict(boxstyle="round,pad=0.25", fc="#fff9e6", ec="#e67e22"),
    )


def _plot_tp_panel(
    ax,
    *,
    ref_tp: float,
    ref_chi: float,
    col_kw: dict,
    rg,
    phi: float = 0.10,
    forward_mode: bool = False,
    highlight_pbar_gpa: float | None = None,
) -> None:
    """T–P: Katz solidus, adiabats, isentropic melt path."""
    liths = _liths_from_kw(col_kw, phi=phi)
    p_axis = np.linspace(0.05, 3.2, 250)
    ts = _bulk_solidus_t(p_axis, liths)
    ax.fill_between(p_axis, 900, ts, color="#ecf0f1", alpha=0.6, label="固相线以下（未熔）")
    ax.plot(p_axis, ts, "k-", lw=2.0, label="bulk 固相线（Katz/pyMelt）")

    if forward_mode:
        tp_show = (float(ref_tp),)
        colors = ("#c0392b",)
    else:
        tp_show = (1250.0, ref_tp, 1400.0)
        colors = ("#95a5a6", "#e67e22", "#c0392b")
    for tp, col in zip(tp_show, colors):
        t_ad = float(tp) - 30.0 * p_axis
        lw = 2.2 if tp == ref_tp else 1.0
        ls = "-" if tp == ref_tp else ":"
        ax.plot(p_axis, t_ad, color=col, lw=lw, ls=ls, alpha=0.9)
        ax.text(
            2.95,
            float(tp) - 30.0 * 2.95 + 15,
            rf"$T_p$={tp:.0f}$^\circ$C",
            fontsize=7.5,
            color=col,
            ha="right",
        )

    path = rg.path
    p_path = np.array([s.p_gpa for s in path.steps], dtype=float)
    t_path = np.array([s.t_c for s in path.steps], dtype=float)
    t_p0 = float(t_path[0])
    ax.plot(p_path, t_path, color="#2980b9", lw=2.5, label="等熵减压路径（REEBOX）")
    ax.scatter([rg.p0_gpa], [t_p0], s=70, c="#27ae60", zorder=5, edgecolors="white")
    ax.annotate(
        r"$P_0$：绝热线 $\cap$ 固相线",
        xy=(rg.p0_gpa, t_p0),
        xytext=(rg.p0_gpa + 0.35, t_p0 + 80),
        fontsize=7.5,
        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.0),
    )
    ax.scatter([rg.pf_gpa], [t_path[-1]], s=55, c="#8e44ad", zorder=5, edgecolors="white")
    ax.annotate(
        r"$P_f$（浅端）",
        xy=(rg.pf_gpa, t_path[-1]),
        xytext=(rg.pf_gpa + 0.45, t_path[-1] + 60),
        fontsize=7.5,
        arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.0),
    )
    if highlight_pbar_gpa is not None:
        p_bar = float(highlight_pbar_gpa)
        t_bar = float(np.interp(p_bar, p_path[::-1], t_path[::-1])) if len(p_path) > 1 else t_path[0]
        ax.axvline(p_bar, color="#e67e22", ls="--", lw=1.2, alpha=0.85, zorder=4)
        ax.scatter([p_bar], [t_bar], s=45, c="#e67e22", marker="D", edgecolors="white", zorder=6)
        ax.text(
            p_bar + 0.08,
            t_bar + 40,
            rf"$\bar P$={p_bar:.2f} GPa",
            fontsize=7,
            color="#d35400",
        )

    ax.set_xlabel(r"压力 $P$ (GPa)  $\leftarrow$ 深        浅 $\rightarrow$")
    ax.set_ylabel(r"温度 $T$ ($^\circ$C)")
    ax.set_xlim(3.2, 0.05)
    ax.set_ylim(1050, 1680)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.text(
        0.98,
        0.04,
        (
            (r"正演联动 · 同验证→模式图 (c)" + "\n" if forward_mode else "")
            + r"绝热线：$T = T_p - 30\,P$ (McKenzie & Bickle 1988)" + "\n"
            + rf"$T_p$={ref_tp:.0f}$^\circ$C，$\chi$={ref_chi:g}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc"),
    )


def lithologies_from_col_kwargs(col_kw: dict, *, phi: float) -> list:
    return _liths_from_kw(col_kw, phi=phi)


def build_forward_geometry(
    *,
    tp_c: float,
    chi: float,
    b_km: float,
    phi: float,
    col_kw: dict,
    melting_engine: str,
    p0_gpa: float,
    pf_gpa: float,
    h_km: float,
    n_isentropic_steps: int = 64,
) -> ReeboxColumnGeometry:
    """Build T–P / F–T path geometry aligned with a forward run."""
    from petrology.active_upwelling import adiabat_mb88_c
    from petrology.melting.isentropic import IsentropicPath, IsentropicStep

    liths = lithologies_from_col_kwargs(col_kw, phi=phi)
    if melting_engine == "reebox":
        return build_reebox_column(
            liths,
            tp_c=float(tp_c),
            b_km=float(b_km),
            chi=float(chi),
            n_isentropic_steps=int(n_isentropic_steps),
        )

    dfdp = float(col_kw.get("dfdp_per_gpa", FIG12_DFDP_PER_GPA))
    steps: list[IsentropicStep] = []
    for p in np.linspace(float(p0_gpa), float(pf_gpa), int(n_isentropic_steps) + 1):
        f_val = float(np.clip(dfdp * (float(p0_gpa) - float(p)), 0.0, 0.995))
        steps.append(
            IsentropicStep(
                p_gpa=float(p),
                t_c=float(adiabat_mb88_c(float(tp_c), float(p))),
                f_by_name={"column": f_val},
                df_dp_by_name={},
                dtdp_c_per_gpa=-30.0,
            )
        )
    path = IsentropicPath(
        tp_c=float(tp_c),
        p0_gpa=float(p0_gpa),
        pf_gpa=float(pf_gpa),
        steps=steps,
        u0_by_name={"column": 1.0},
    )
    return ReeboxColumnGeometry(
        p0_gpa=float(p0_gpa),
        pf_gpa=float(pf_gpa),
        p_floor_gpa=float(pf_gpa),
        h_km=float(h_km),
        path=path,
    )


def plot_schematic_panel_c(
    ax,
    *,
    ref_tp: float,
    ref_chi: float,
    col_kw: dict,
    rg,
    phi: float = 0.10,
    forward_mode: bool = False,
    highlight_pbar_gpa: float | None = None,
) -> None:
    _plot_tp_panel(
        ax,
        ref_tp=ref_tp,
        ref_chi=ref_chi,
        col_kw=col_kw,
        rg=rg,
        phi=phi,
        forward_mode=forward_mode,
        highlight_pbar_gpa=highlight_pbar_gpa,
    )


def plot_schematic_panel_d(
    ax,
    *,
    ref_tp: float,
    ref_chi: float,
    liths: list,
    forward_mode: bool = False,
    rg=None,
    highlight_fbar: float | None = None,
    highlight_fmax: float | None = None,
) -> None:
    _plot_ft_panel(
        ax,
        ref_tp=ref_tp,
        ref_chi=ref_chi,
        liths=liths,
        forward_mode=forward_mode,
        rg=rg,
        highlight_fbar=highlight_fbar,
        highlight_fmax=highlight_fmax,
    )


def _setup_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "figure.dpi": 150,
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavusans",
        }
    )


def main(
    *,
    ref_tp: float | None = None,
    ref_chi: float | None = None,
    phi: float | None = None,
    col_kw: dict | None = None,
) -> None:
    _setup_rcparams()

    if col_kw is None:
        col_kw = resolve_lithology_col_kwargs(
            lithology_backend="pymelt",
            lithology_preset="greenland_kg1",
        )

    ref_tp = 1325.0 if ref_tp is None else float(ref_tp)
    ref_chi = 8.0 if ref_chi is None else float(ref_chi)
    phi_val = 0.10 if phi is None else float(phi)
    chi_pass, chi_act = 1.0, 8.0

    liths = _liths_from_kw(col_kw, phi=phi_val)
    hvp_linear, hvp_chi1_b = _build_hvp_fig12_curves()
    hvp_modern, hvp_modern_chi1_b = _build_hvp_grid(col_kw)
    rg = build_reebox_column(
        liths,
        tp_c=float(ref_tp),
        b_km=0.0,
        chi=float(ref_chi),
        n_isentropic_steps=80,
    )
    p0, pf = float(rg.p0_gpa), float(rg.pf_gpa)
    f_max = float(max(rg.path.f_bulk_at(s) for s in rg.path.steps))
    p = np.linspace(p0, pf, 400)
    f = _f_profile(p, p0, pf, f_max)
    tc = _tc_profile(f)
    w_pass = 1.0 + _active_weight(p, p0, pf, chi_pass)
    w_act = 1.0 + _active_weight(p, p0, pf, chi_act)

    h_ref, vp_ref, pbar_ref, fbar_ref = _forward_point(ref_tp, ref_chi, col_kw=col_kw)

    fig = plt.figure(figsize=(12, 19.5))
    gs = fig.add_gridspec(
        5, 2, height_ratios=[1.0, 0.92, 0.95, 1.0, 1.05], hspace=0.40, wspace=0.28
    )

    # --- Panel A: column cross-section (plan view) ---
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_title("(a) 上涌柱平面示意：柱心 vs 柱缘")
    ax0.set_xlim(-1.2, 1.2)
    ax0.set_ylim(-0.2, 2.2)
    ax0.set_aspect("equal")
    ax0.axis("off")

    for x, lw, alpha in [(-0.55, 1.0, 0.35), (0.0, 2.2, 0.9), (0.55, 1.0, 0.35)]:
        ax0.add_patch(
            FancyArrowPatch(
                (x, 0.0),
                (x, 1.55),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=lw,
                color="#444444",
                alpha=alpha,
            )
        )

    ax0.add_patch(
        mpatches.Rectangle(
            (-1.1, 1.55), 2.2, 0.35, facecolor="#d9d9d9", edgecolor="#666666", linewidth=1.2
        )
    )
    ax0.text(0, 1.72, "岩石圈盖层 b", ha="center", va="center", fontsize=8)

    tri = Polygon(
        [[-0.75, 0.15], [0.75, 0.15], [0.0, 1.55]],
        closed=True,
        facecolor="#ffe6b3",
        edgecolor="#cc7a00",
        alpha=0.55,
        linewidth=1.5,
    )
    ax0.add_patch(tri)
    ax0.text(0, 0.95, "三角熔融带\n(减压路径域)", ha="center", va="center", fontsize=8, color="#8a4b00")
    ax0.scatter([0], [0.95], s=180, c="#e74c3c", alpha=0.35, zorder=3)
    ax0.text(0, 0.55, r"柱心" + "\n(高 $P_0$ 端)", ha="center", va="center", fontsize=8, color="#c0392b")
    ax0.text(-0.85, 0.45, "柱缘\n(短流道)", ha="center", va="center", fontsize=7.5, color="#555555")
    ax0.text(0.85, 0.45, "柱缘\n(短流道)", ha="center", va="center", fontsize=7.5, color="#555555")
    ax0.text(
        -1.05,
        -0.05,
        r"被动 $\chi \approx 1$：" + "\n各流道权重相近",
        fontsize=7.5,
        color="#2c3e50",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#aaaaaa"),
    )
    ax0.text(
        0.35,
        -0.05,
        r"主动 $\chi \gg 1$：" + "\n熔融通量向柱心偏置",
        fontsize=7.5,
        color="#c0392b",
        bbox=dict(boxstyle="round,pad=0.25", fc="#fff5f5", ec="#e74c3c"),
    )
    ax0.text(
        0.0,
        2.05,
        rf"源区 $T_p$={ref_tp:.0f}$^\circ$C（反演参数）",
        ha="center",
        fontsize=7.5,
        color="#d35400",
        bbox=dict(boxstyle="round,pad=0.2", fc="#fef5e7", ec="#e67e22"),
    )

    # --- Panel B: F(P) triangular domain ---
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_title("(b) F–P 三角熔融域（Langmuir 1992 型）")
    ax1.set_ylabel("熔融程度 F")

    for frac in np.linspace(0.15, 1.0, 6):
        p_start = pf + frac * (p0 - pf)
        p_line = np.linspace(p_start, pf, 80)
        f_line = f_max * (p_start - p_line) / max(p_start - pf, 1e-9)
        ax1.plot(p_line, f_line, color="#7f8c8d", lw=0.8, alpha=0.55)

    ax1.plot(p, f, color="#c0392b", lw=2.2, label="最长流道（柱心）")
    ax1.fill_between(p, 0, f, color="#f5b041", alpha=0.25)
    ax1.axvline(p0, color="#2c3e50", ls="--", lw=1.0)
    ax1.axvline(pf, color="#2c3e50", ls="--", lw=1.0)
    ax1.text(p0 - 0.08, 0.01, r"$P_0$" + "\n(固相线/深)", fontsize=8, ha="right")
    ax1.text(pf + 0.08, 0.01, r"$P_f$" + "\n(浅端)", fontsize=8, ha="left")
    ax1.text(1.4, f_max * 0.55, "三角域\n(近分离熔融)", fontsize=8, color="#8a4b00")
    ax1.set_xlim(p0 + 0.15, pf - 0.1)  # 左=高压=深，右=低压=浅（勿在 invert 后再 set_xlim）
    ax1.set_ylim(0, f_max * 1.15)
    ax1.set_xlabel(r"压力 $P$ (GPa)  $\leftarrow$ 深        浅 $\rightarrow$")
    ax1.legend(loc="upper right", fontsize=8)

    # --- Panel C: T-P (Tp, adiabat, solidus, isentropic path) ---
    ax_tp = fig.add_subplot(gs[1, :])
    ax_tp.set_title(r"(c) $T$–$P$：潜在温度 $T_p$、绝热线与 Katz 固相线 $\rightarrow$ $P_0$")
    _plot_tp_panel(ax_tp, ref_tp=ref_tp, ref_chi=ref_chi, col_kw=col_kw, rg=rg)

    # --- Panel D: F-T isentropic paths ---
    ax_ft = fig.add_subplot(gs[2, 0])
    ax_ft.set_title(r"(d) $F$–$T$：等熵减压路径（Katz $F(P,T)$）")
    _plot_ft_panel(ax_ft, ref_tp=ref_tp, ref_chi=ref_chi, liths=liths)

    # --- Panel E: KKHS02 Fig.12-style H–V ---
    ax_hvp = fig.add_subplot(gs[2, 1])
    ax_hvp.set_title(
        r"(e) H–$V_{\mathrm{bulk}}$：Fig.12 风格（论文轴）线性轨 vs Modern"
    )
    _plot_hvp_fig12_panel(
        ax_hvp,
        linear_pts=hvp_linear,
        chi1_b_lines=hvp_chi1_b,
        modern_pts=hvp_modern,
        modern_chi1_b_lines=hvp_modern_chi1_b,
        ref_tp=ref_tp,
        ref_chi=ref_chi,
        h_ref=h_ref,
        vp_ref=vp_ref,
        fbar_ref=fbar_ref,
    )

    # --- Panel F: passive H integration ---
    ax2 = fig.add_subplot(gs[3, 0])
    ax2.set_title(r"(f) 被动 ($\chi=1$)：标准三角积分 $\rightarrow$ H")
    ax2.set_ylabel("单位厚度贡献 tc(P) × 权重")

    integrand_pass = tc * w_pass
    ax2.fill_between(p, 0, integrand_pass, color="#3498db", alpha=0.35)
    ax2.plot(p, integrand_pass, color="#2471a3", lw=2)
    ax2.plot(p, tc, color="#95a5a6", ls=":", lw=1.2, label=r"tc(P) $\propto F/(1-F)$")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.text(
        1.35,
        integrand_pass.max() * 0.55,
        r"$\int tc \cdot (1+w)\, dP$" + "\n" + r"$\approx$ 火成壳厚度 H",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#3498db"),
    )
    ax2.set_xlim(p0 + 0.15, pf - 0.1)
    ax2.set_xlabel(r"压力 $P$ (GPa)  $\leftarrow$ 深        浅 $\rightarrow$")

    # --- Panel G: active H integration ---
    ax3 = fig.add_subplot(gs[3, 1])
    ax3.set_title(rf"(g) 主动 ($\chi={chi_act:g}$)：柱心加权 $\rightarrow$ H 增大")
    ax3.set_ylabel("单位厚度贡献 tc(P) × 权重")

    integrand_act = tc * w_act
    ax3.fill_between(p, 0, integrand_pass, color="#3498db", alpha=0.15, label="被动积分")
    ax3.fill_between(p, 0, integrand_act, color="#e74c3c", alpha=0.35, label="主动积分")
    ax3.plot(p, integrand_pass, color="#2471a3", lw=1.2, ls="--")
    ax3.plot(p, integrand_act, color="#c0392b", lw=2)
    ax3.legend(fontsize=8, loc="upper left")

    h_pass = float(np.trapezoid(integrand_pass, p))
    h_act = float(np.trapezoid(integrand_act, p))
    ax3.annotate(
        "",
        xy=(p0 - 0.05, integrand_act.max() * 0.92),
        xytext=(p0 - 0.05, integrand_pass.max() * 0.92),
        arrowprops=dict(arrowstyle="<->", color="#8e44ad", lw=1.5),
    )
    ax3.text(
        p0 - 0.12,
        integrand_act.max() * 0.92,
        rf"$\Delta H$" + f"\n(示意 {100 * (h_act / h_pass - 1):.0f}%)",
        fontsize=8,
        color="#8e44ad",
        ha="right",
        va="center",
    )
    ax3.text(
        1.35,
        integrand_act.max() * 0.35,
        r"$w(P) = (\chi-1)\cdot\exp(-\cdot)$" + "\n" + r"在 $P \approx P_0$（柱心）最大",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff5f5", ec="#e74c3c"),
    )
    ax3.set_xlim(p0 + 0.15, pf - 0.1)
    ax3.set_xlabel(r"压力 $P$ (GPa)  $\leftarrow$ 深        浅 $\rightarrow$")

    # --- Panel H: Step 1 — Pbar, Fbar -> V_bulk (KKHS02 eq. 1) ---
    ax4 = fig.add_subplot(gs[4, 0])
    ax4.set_title(r"(h) Step 1：$\bar P$, $\bar F$ $\rightarrow$ $V_{\mathrm{bulk}}$（KKHS02 方程 (1)）")

    p_grid = np.linspace(0.55, 1.45, 60)
    f_grid = np.linspace(0.05, 0.78, 60)
    pg, fg = np.meshgrid(p_grid, f_grid)
    vp_grid = _vp_grid(pg, fg)
    levels = np.arange(7.15, 7.58, 0.04)
    cf = ax4.contourf(pg, fg, vp_grid, levels=levels, cmap="YlGnBu", alpha=0.85)
    cs = ax4.contour(pg, fg, vp_grid, levels=levels[::2], colors="#555555", linewidths=0.5, alpha=0.7)
    ax4.clabel(cs, fmt="%.2f", fontsize=6.5)
    cbar = fig.colorbar(cf, ax=ax4, fraction=0.046, pad=0.02)
    cbar.set_label(r"$V_{\mathrm{bulk}}$ (km/s)")

    ax4.set_xlabel(r"平均熔融压力 $\bar P$ (GPa)")
    ax4.set_ylabel(r"平均熔融程度 $\bar F$")

    # Passive vs active RMC (Fbar) at same Tp, chi
    for rmc, marker, color, label in (
        ("passive_langmuir", "s", "#2471a3", r"被动 RMC ($\bar F \approx F_{\max}/2$)"),
        ("active_langmuir", "o", "#c0392b", r"主动 RMC (LIP 默认)"),
    ):
        col = forward_melting_column(
            tp_c=ref_tp,
            chi=ref_chi,
            b_km=0.0,
            pyroxenite_frac=0.10,
            melting_engine="reebox",
            rmc_mode=rmc,
            **col_kw,
        )
        ax4.scatter(
            col.pbar_gpa,
            col.fbar,
            s=55,
            marker=marker,
            c=color,
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
            label=label,
        )

    ax4.scatter(
        pbar_ref,
        fbar_ref,
        s=90,
        marker="*",
        c="#f39c12",
        edgecolors="#333333",
        linewidths=0.8,
        zorder=6,
        label=rf"正演 ($T_p$={ref_tp:.0f}, $\chi$={ref_chi:g})",
    )
    ax4.annotate(
        rf"$V_{{\mathrm{{bulk}}}}$={vp_ref:.2f} km/s",
        xy=(pbar_ref, fbar_ref),
        xytext=(pbar_ref + 0.18, fbar_ref - 0.12),
        fontsize=7.5,
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0),
    )
    ax4.legend(fontsize=7, loc="lower right")
    ax4.text(
        0.58,
        0.74,
        "RMC 汇集熔体\n→ $(\\bar P, \\bar F)$\n→ KKHS02 eq.(1)",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888888", alpha=0.9),
    )

    # --- Panel I: inversion readout (Step 2 band on Fig.12) ---
    ax5 = fig.add_subplot(gs[4, 1])
    ax5.set_title(r"(i) 反演读图：观测 $(H, V_{\mathrm{LC}})$ 与 Step 2")
    ax5.axis("off")

    d_vp_ref = delta_vp_km_s(F_SOLID, bulk_vp_km_s=vp_ref, p_fc_mpa=P_FC_MPA)
    bounds = bulk_vp_bounds(V_LC_OBS, f_lower=F_LOWER, delta_vp_fc_km_s=d_vp_ref)
    lin_diag = solve_active_upwelling(
        tp_c=float(ref_tp),
        b_km=0.0,
        chi=float(ref_chi),
        dfdp_per_gpa=FIG12_DFDP_PER_GPA,
        vp_bias_km_s=0.0,
    )
    feasible = [
        p
        for p in hvp_modern
        if abs(p.h_km - H_OBS) <= 3.0
        and bounds.v_bulk_lower_km_s <= p.vp_bulk_km_s <= bounds.v_bulk_upper_km_s
    ]

    lines = [
        r"$\bullet$ 图 (e) 为两套 **Step 4 计算轨**（均经 Step 1 eq.(1)，$\bar F$ 来源不同）：",
        rf"  – 线性 Step-4：$\bar F$≈{lin_diag.fbar:.2f} $\rightarrow$ $V_p$≈{lin_diag.vp_bulk_km_s:.2f} km/s（eq.(1) 修正 b1，bias=0）",
        rf"  – Modern REEBOX：$\bar F$≈{fbar_ref:.2f} $\rightarrow$ $V_p$≈{vp_ref:.2f} km/s（bias=0）",
        r"  $\Rightarrow$ Modern 偏高主因是 Katz 真实 $\bar F$ 大于线性 12%/GPa 假设",
        r"  （图 (e) **不含** Fig.12a digitized 印刷背景）",
        "",
        rf"观测读图：$H_{{\mathrm{{obs}}}}$={H_OBS:.0f} km，$V_{{\mathrm{{LC}}}}$={V_LC_OBS:.1f} km/s",
        rf"Step 2 bulk 区间：[{bounds.v_bulk_lower_km_s:.3f}, {bounds.v_bulk_upper_km_s:.3f}] km/s",
        "",
        rf"Modern 粗网格可行点：{len(feasible)} / {len(hvp_modern)}",
        "（Modern 整体高于线性 Step-4 — 预期，非 eq.(1) 错误）",
    ]

    ax5.text(
        0.04,
        0.96,
        "\n".join(lines),
        transform=ax5.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        linespacing=1.45,
        bbox=dict(boxstyle="round,pad=0.45", fc="#f8f9fa", ec="#bdc3c7"),
    )

    # Step 2 arrow schematic
    ax5.add_patch(mpatches.FancyBboxPatch((0.08, 0.08), 0.36, 0.22, boxstyle="round,pad=0.06", fc="#d6eaf8", ec="#2471a3", transform=ax5.transAxes))
    ax5.text(0.26, 0.19, r"$V_{\mathrm{bulk}}$", transform=ax5.transAxes, ha="center", fontsize=9)
    ax5.annotate(
        "",
        xy=(0.50, 0.19),
        xytext=(0.46, 0.19),
        xycoords=ax5.transAxes,
        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.2),
    )
    ax5.text(0.48, 0.24, r"$+\Delta V_p$", transform=ax5.transAxes, ha="center", fontsize=8, color="#555555")
    ax5.add_patch(mpatches.FancyBboxPatch((0.52, 0.08), 0.36, 0.22, boxstyle="round,pad=0.06", fc="#e8daef", ec="#8e44ad", transform=ax5.transAxes))
    ax5.text(0.70, 0.19, r"$V_{\mathrm{LC}}$", transform=ax5.transAxes, ha="center", fontsize=9)

    fig.suptitle(
        "被动 vs 主动熔融 — REEBOX / Modern LIP 模式图\n"
        + r"($T_p$/$T$–$P$/$F$–$T$；H–$V$ Fig.12；H：$\chi$；$V$：KKHS02 eq.(1)+Step 2；$F(P,T)$：Katz/pyMelt)",
        fontsize=11,
        y=0.998,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
