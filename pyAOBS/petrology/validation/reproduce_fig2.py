"""Reproduce KKHS02 Figure 2 — proxy layout or W&L crystallization engine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.fc.cdat_library import FIG2_PAPER_SAMPLE, fig2_primary_melt, list_samples
from petrology.fc.wl1990 import (
    CrystallizationPathMode,
    CrystallizationState,
    KdEngine,
    KdMode1990,
    simulate_crystallization_path,
)
from petrology.minerals import Backend
from petrology.norm_velocity import norm_velocity_from_bulk_wt

_FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"


def _interp_curve(f_grid: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    return np.interp(f_grid, x, y)


def _paper_proxy_curves(f: np.ndarray) -> dict[str, np.ndarray]:
    curves = {
        "cum_sol_1kb": _interp_curve(f, [(0.0, 7.57), (0.08, 7.56), (0.25, 7.55), (0.5, 7.42), (0.7, 7.34), (0.8, 7.31)]),
        "cum_sol_8_1kb": _interp_curve(f, [(0.0, 7.55), (0.1, 7.53), (0.25, 7.51), (0.5, 7.38), (0.7, 7.30), (0.8, 7.27)]),
        "eq_sol_1kb": _interp_curve(f, [(0.0, 7.50), (0.2, 7.43), (0.4, 7.35), (0.6, 7.29), (0.8, 7.24)]),
        "inc_sol_1kb": _interp_curve(f, [(0.0, 7.22), (0.22, 7.22), (0.32, 7.20), (0.40, 7.28), (0.55, 7.25), (0.72, 7.10), (0.8, 6.98)]),
        "inc_sol_8_1kb": _interp_curve(f, [(0.0, 7.20), (0.22, 7.20), (0.35, 7.18), (0.55, 7.12), (0.72, 7.02), (0.8, 6.95)]),
        "eq_res_1kb": _interp_curve(f, [(0.0, 7.18), (0.2, 7.10), (0.4, 7.08), (0.6, 7.10), (0.8, 7.12)]),
        "frac_res_1kb": _interp_curve(f, [(0.0, 7.15), (0.2, 7.08), (0.4, 7.04), (0.6, 6.98), (0.8, 6.90)]),
        "frac_res_8_1kb": _interp_curve(f, [(0.0, 7.15), (0.2, 7.00), (0.4, 6.88), (0.6, 6.80), (0.8, 6.70)]),
        "rho_cum_1kb": _interp_curve(f, [(0.0, 3.33), (0.05, 3.34), (0.12, 3.10), (0.30, 2.99), (0.55, 3.00), (0.8, 3.02)]),
        "rho_cum_8_1kb": _interp_curve(f, [(0.0, 3.31), (0.05, 3.33), (0.12, 3.05), (0.30, 2.97), (0.55, 2.98), (0.8, 3.00)]),
        "rho_inc_1kb": _interp_curve(f, [(0.0, 2.72), (0.4, 2.72), (0.6, 2.74), (0.8, 2.80)]),
        "rho_inc_8_1kb": _interp_curve(f, [(0.0, 2.71), (0.4, 2.70), (0.6, 2.73), (0.8, 2.78)]),
        "rho_res_liq_eq": _interp_curve(f, [(0.0, 2.98), (0.2, 2.98), (0.4, 3.00), (0.6, 3.04), (0.8, 3.12)]),
        "rho_res_liq_fc": _interp_curve(f, [(0.0, 2.98), (0.2, 2.95), (0.4, 2.97), (0.6, 2.98), (0.8, 3.00)]),
        "rho_res_liq_8_1kb": _interp_curve(f, [(0.0, 2.98), (0.2, 2.94), (0.4, 2.96), (0.6, 2.97), (0.8, 2.99)]),
        "c_ol": _interp_curve(f, [(0.0, 35), (0.22, 35), (0.40, 12), (0.8, 13)]),
        "c_cpx": _interp_curve(f, [(0.0, 0), (0.22, 0), (0.40, 20), (0.6, 35), (0.8, 36)]),
        "d_ol": _interp_curve(f, [(0.0, 33), (0.15, 32), (0.28, 24), (0.45, 15), (0.8, 15)]),
        "d_cpx": _interp_curve(f, [(0.0, 0), (0.22, 0), (0.28, 10), (0.45, 28), (0.6, 36), (0.8, 38)]),
        "e_ol": _interp_curve(f, [(0.0, 100), (0.08, 82), (0.15, 68), (0.25, 55), (0.4, 47), (0.8, 33)]),
        "e_cpx": _interp_curve(f, [(0.0, 0), (0.15, 1), (0.25, 3), (0.4, 8), (0.6, 15), (0.8, 20)]),
        "f_fo": _interp_curve(f, [(0.0, 90), (0.15, 88), (0.30, 85), (0.45, 81), (0.60, 72), (0.72, 63), (0.8, 58)]),
        "f_an": _interp_curve(f, [(0.0, 80), (0.15, 77), (0.30, 73), (0.45, 69), (0.60, 62), (0.72, 57), (0.8, 54)]),
        "f_di": _interp_curve(f, [(0.0, 90), (0.20, 89), (0.35, 87), (0.45, 86), (0.55, 82), (0.65, 72), (0.8, 58)]),
        "g_fo": _interp_curve(f, [(0.0, 90), (0.15, 88), (0.30, 86), (0.45, 82), (0.60, 74), (0.72, 65), (0.8, 60)]),
        "g_an": _interp_curve(f, [(0.0, 73), (0.25, 72), (0.40, 71), (0.55, 68), (0.65, 64), (0.8, 60)]),
        "g_di": _interp_curve(f, [(0.0, 90), (0.20, 89), (0.35, 87), (0.45, 85), (0.55, 80), (0.65, 70), (0.8, 62)]),
        "h_fo": _interp_curve(f, [(0.0, 88), (0.15, 87), (0.30, 86), (0.45, 85), (0.60, 83), (0.8, 78)]),
        "h_an": _interp_curve(f, [(0.0, 80), (0.20, 78), (0.35, 76), (0.50, 74), (0.65, 71), (0.8, 68)]),
        "h_di": _interp_curve(f, [(0.0, 90), (0.25, 89), (0.45, 88), (0.60, 87), (0.8, 85)]),
    }
    curves["c_pl"] = 100.0 - curves["c_ol"] - curves["c_cpx"]
    curves["d_pl"] = 100.0 - curves["d_ol"] - curves["d_cpx"]
    curves["e_pl"] = 100.0 - curves["e_ol"] - curves["e_cpx"]
    return curves


def _simulate_wl1990(
    f_grid: np.ndarray,
    melt_oxides_wt: dict[str, float],
    *,
    cipw_backend: str,
    mineral_backend: Backend,
    fig2_ab_calibrate: bool = True,
    fig2_ab_anchor_vp: bool = False,
    kd_engine: KdEngine = "heuristic",
    kd_mode_1990: KdMode1990 = "langmuir",
) -> dict[CrystallizationPathMode, list[CrystallizationState]]:
    paths: tuple[CrystallizationPathMode, ...] = ("fc_100", "eq_100", "polybaric_fc")
    return {
        path: simulate_crystallization_path(
            primary_melt_oxides_wt=melt_oxides_wt,
            f_grid=f_grid,
            path=path,
            cipw_backend=cipw_backend,
            mineral_backend=mineral_backend,
            fig2_ab_calibrate=fig2_ab_calibrate,
            fig2_ab_anchor_vp=fig2_ab_anchor_vp,
            kd_engine=kd_engine,
            kd_mode_1990=kd_mode_1990,
        )
        for path in paths
    }


def _plot_proxy(curves: dict[str, np.ndarray], f: np.ndarray, save_figure: Path | None, show: bool) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "serif"
    fig = plt.figure(figsize=(10.2, 8.5))
    gs = fig.add_gridspec(6, 3, width_ratios=[1.65, 0.85, 0.85], hspace=0.60, wspace=0.60)
    ax_a = fig.add_subplot(gs[0:3, 0])
    ax_b = fig.add_subplot(gs[3:6, 0], sharex=ax_a)
    ax_c = fig.add_subplot(gs[0:2, 1], sharex=ax_a)
    ax_d = fig.add_subplot(gs[2:4, 1], sharex=ax_a)
    ax_e = fig.add_subplot(gs[4:6, 1], sharex=ax_a)
    ax_f = fig.add_subplot(gs[0:2, 2], sharex=ax_a)
    ax_g = fig.add_subplot(gs[2:4, 2], sharex=ax_a)
    ax_h = fig.add_subplot(gs[4:6, 2], sharex=ax_a)

    ax_a.plot(f, curves["cum_sol_1kb"], "k-", lw=1.3)
    ax_a.plot(f, curves["cum_sol_8_1kb"], color="k", ls=(0, (6, 2)), lw=1.1)
    ax_a.plot(f, curves["eq_sol_1kb"], color="0.25", ls="-.", lw=1.0)
    ax_a.plot(f, curves["inc_sol_1kb"], color="0.20", lw=1.0)
    ax_a.plot(f, curves["inc_sol_8_1kb"], color="0.35", ls=(0, (3, 2)), lw=0.95)
    ax_a.plot(f, curves["eq_res_1kb"], color="0.55", ls="-", lw=0.9)
    ax_a.plot(f, curves["frac_res_1kb"], color="0.60", ls="--", lw=0.9)
    ax_a.plot(f, curves["frac_res_8_1kb"], color="0.70", ls=":", lw=1.0)
    ax_a.set_title("(a)", loc="left", pad=6)
    ax_a.set_ylabel("P-wave Velocity [km s$^{-1}$]")
    ax_a.set_xlim(0.0, 0.8)
    ax_a.set_ylim(6.7, 7.6)
    ax_a.axhline(7.17, color="0.55", lw=0.6, ls=":", alpha=0.7)

    ax_b.plot(f, curves["rho_cum_1kb"], "k-", lw=1.3)
    ax_b.plot(f, curves["rho_cum_8_1kb"], color="k", ls=(0, (6, 2)), lw=1.1)
    ax_b.plot(f, curves["rho_inc_1kb"], color="0.25", lw=1.0)
    ax_b.plot(f, curves["rho_inc_8_1kb"], color="0.35", ls="--", lw=0.9)
    ax_b.plot(f, curves["rho_res_liq_eq"], color="0.45", ls="-", lw=0.9)
    ax_b.plot(f, curves["rho_res_liq_fc"], color="0.55", ls="--", lw=0.9)
    ax_b.plot(f, curves["rho_res_liq_8_1kb"], color="0.65", ls=":", lw=1.0)
    ax_b.set_title("(b)", loc="left", pad=6)
    ax_b.set_ylabel("Density [kg m$^{-3}$]")
    ax_b.set_xlabel("Solid Fraction")
    ax_b.set_ylim(2.6, 3.4)

    def draw_phase(ax, ol_key: str, pl_key: str, cpx_key: str, title: str) -> None:
        ol, pl, cpx = curves[ol_key], curves[pl_key], curves[cpx_key]
        ax.fill_between(f, 0, ol, color="0.90", edgecolor="0.2", linewidth=0.7)
        ax.fill_between(f, ol, ol + cpx, color="0.82", edgecolor="0.2", linewidth=0.7)
        ax.fill_between(f, ol + cpx, 100, color="0.95", edgecolor="0.2", linewidth=0.7)
        ax.set_title(title, loc="left", pad=2)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Phase")

    draw_phase(ax_c, "c_ol", "c_pl", "c_cpx", "(c)")
    draw_phase(ax_d, "d_ol", "d_pl", "d_cpx", "(d)")
    draw_phase(ax_e, "e_ol", "e_pl", "e_cpx", "(e)")
    ax_e.set_xlabel("Solid Fraction")

    def draw_comp(ax, fo_key: str, an_key: str, di_key: str, title: str) -> None:
        ax.plot(f, curves[fo_key], "k-", lw=1.1)
        ax.plot(f, curves[an_key], color="0.2", ls="--", lw=1.0)
        ax.plot(f, curves[di_key], color="0.35", ls="-.", lw=1.0)
        ax.set_title(title, loc="left", pad=2)
        ax.set_ylim(50, 100)
        ax.set_ylabel("Composition")

    draw_comp(ax_f, "f_fo", "f_an", "f_di", "(f)")
    draw_comp(ax_g, "g_fo", "g_an", "g_di", "(g)")
    draw_comp(ax_h, "h_fo", "h_an", "h_di", "(h)")
    ax_h.set_xlabel("Solid Fraction")

    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h):
        ax.set_xlim(0.0, 0.8)
    fig.tight_layout()
    _save_or_show(fig, save_figure, show)


def _plot_wl1990(
    states: dict[CrystallizationPathMode, list[CrystallizationState]],
    *,
    primary_norm_vp: float,
    sample_label: str,
    save_figure: Path | None,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    fc = states["fc_100"]
    pb = states["polybaric_fc"]
    eq = states["eq_100"]
    f_fc = np.array([s.f_solid for s in fc], dtype=float)
    f_pb = np.array([s.f_solid for s in pb], dtype=float)
    f_eq = np.array([s.f_solid for s in eq], dtype=float)

    plt.rcParams["font.family"] = "serif"
    fig = plt.figure(figsize=(10.2, 8.5))
    gs = fig.add_gridspec(6, 3, width_ratios=[1.65, 0.85, 0.85], hspace=0.60, wspace=0.60)
    ax_a = fig.add_subplot(gs[0:3, 0])
    ax_b = fig.add_subplot(gs[3:6, 0], sharex=ax_a)
    ax_c = fig.add_subplot(gs[0:2, 1], sharex=ax_a)
    ax_d = fig.add_subplot(gs[2:4, 1], sharex=ax_a)
    ax_e = fig.add_subplot(gs[4:6, 1], sharex=ax_a)
    ax_f = fig.add_subplot(gs[0:2, 2], sharex=ax_a)
    ax_g = fig.add_subplot(gs[2:4, 2], sharex=ax_a)
    ax_h = fig.add_subplot(gs[4:6, 2], sharex=ax_a)

    ax_a.plot(f_fc, [s.vp_cumulate_km_s for s in fc], "k-", lw=1.3, label="cum. sol. (1kb)")
    ax_a.plot(f_pb, [s.vp_cumulate_km_s for s in pb], color="k", ls=(0, (6, 2)), lw=1.1, label="cum. sol. (8-1kb)")
    ax_a.plot(
        f_eq,
        [s.vp_eq_solid_km_s if s.vp_eq_solid_km_s is not None else s.vp_cumulate_km_s for s in eq],
        color="0.25",
        ls="-.",
        lw=1.0,
        label="eq. sol. (1kb)",
    )
    ax_a.plot(f_fc, [s.vp_incremental_km_s for s in fc], color="0.20", lw=1.0, label="inc. sol. (1kb)")
    ax_a.plot(f_pb, [s.vp_incremental_km_s for s in pb], color="0.35", ls=(0, (3, 2)), lw=0.95, label="inc. sol. (8-1kb)")
    ax_a.plot(f_eq, [s.vp_residual_norm_km_s for s in eq], color="0.55", ls="-", lw=0.9, label="eq. res. (1kb)")
    ax_a.plot(f_fc, [s.vp_residual_norm_km_s for s in fc], color="0.60", ls="--", lw=0.9, label="frac. res. (1kb)")
    ax_a.plot(f_pb, [s.vp_residual_norm_km_s for s in pb], color="0.70", ls=":", lw=1.0, label="frac. res. (8-1kb)")
    ax_a.set_title(f"(a) W&L engine — {sample_label}", loc="left", pad=6)
    ax_a.set_ylabel("P-wave Velocity [km s$^{-1}$]")
    ax_a.set_xlim(0.0, 0.8)
    ax_a.set_ylim(6.7, 7.8)
    ax_a.axhline(primary_norm_vp, color="0.55", lw=0.6, ls=":", alpha=0.7)
    ax_a.legend(fontsize=7, loc="upper right")

    ax_b.plot(f_fc, [s.rho_cumulate_g_cm3 for s in fc], "k-", lw=1.3)
    ax_b.plot(f_pb, [s.rho_cumulate_g_cm3 for s in pb], color="k", ls=(0, (6, 2)), lw=1.1)
    ax_b.plot(f_fc, [s.rho_incremental_g_cm3 for s in fc], color="0.25", lw=1.0)
    ax_b.plot(f_pb, [s.rho_incremental_g_cm3 for s in pb], color="0.35", ls="--", lw=0.9)
    ax_b.plot(f_eq, [s.rho_residual_liquid_g_cm3 for s in eq], color="0.45", ls="-", lw=0.9)
    ax_b.plot(f_fc, [s.rho_residual_liquid_g_cm3 for s in fc], color="0.55", ls="--", lw=0.9)
    ax_b.plot(f_pb, [s.rho_residual_liquid_g_cm3 for s in pb], color="0.65", ls=":", lw=1.0)
    ax_b.set_title("(b)", loc="left", pad=6)
    ax_b.set_ylabel("Density [kg m$^{-3}$]")
    ax_b.set_xlabel("Solid Fraction")
    ax_b.set_ylim(2.6, 3.5)

    def draw_phase(ax, st: list[CrystallizationState], title: str) -> None:
        ff = np.array([s.f_solid for s in st], dtype=float)
        ol = 100.0 * np.array([s.cum_ol for s in st], dtype=float)
        pl = 100.0 * np.array([s.cum_pl for s in st], dtype=float)
        cpx = 100.0 * np.array([s.cum_cpx for s in st], dtype=float)
        ax.fill_between(ff, 0, ol, color="0.90", edgecolor="0.2", linewidth=0.7)
        ax.fill_between(ff, ol, ol + cpx, color="0.82", edgecolor="0.2", linewidth=0.7)
        ax.fill_between(ff, ol + cpx, ol + cpx + pl, color="0.95", edgecolor="0.2", linewidth=0.7)
        ax.set_title(title, loc="left", pad=2)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Phase")

    draw_phase(ax_c, fc, "(c) FC @100 MPa")
    draw_phase(ax_d, pb, "(d) Polybaric FC")
    draw_phase(ax_e, eq, "(e) EQ @100 MPa")
    ax_e.set_xlabel("Solid Fraction")

    def draw_comp(ax, st: list[CrystallizationState], title: str) -> None:
        ff = np.array([s.f_solid for s in st], dtype=float)
        ax.plot(ff, [s.fo_pct for s in st], "k-", lw=1.1)
        ax.plot(ff, [s.an_pct for s in st], color="0.2", ls="--", lw=1.0)
        ax.plot(ff, [s.di_pct for s in st], color="0.35", ls="-.", lw=1.0)
        ax.set_title(title, loc="left", pad=2)
        ax.set_ylim(50, 100)
        ax.set_ylabel("Composition")

    draw_comp(ax_f, fc, "(f)")
    draw_comp(ax_g, pb, "(g)")
    draw_comp(ax_h, eq, "(h)")
    ax_h.set_xlabel("Solid Fraction")

    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h):
        ax.set_xlim(0.0, 0.8)
    fig.tight_layout()
    _save_or_show(fig, save_figure, show)


def _save_or_show(fig, save_figure: Path | None, show: bool) -> None:
    if save_figure:
        save_figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_figure, dpi=180)
        print(f"Saved {save_figure}")
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        import matplotlib.pyplot as plt

        plt.close(fig)


def _default_output(sample_id: str, engine: str, *, kd_engine: str = "heuristic", fig2_ab: bool = True) -> Path:
    if engine == "proxy":
        return _FIGURES_DIR / "reproduce_fig2_proxy.png"
    if not fig2_ab and kd_engine == "langmuir":
        suffix = "wl_langmuir_hs"
    elif not fig2_ab:
        suffix = f"wl_{kd_engine}"
    elif sample_id == FIG2_PAPER_SAMPLE:
        suffix = "wl1990"
    else:
        suffix = sample_id
    return _FIGURES_DIR / f"reproduce_fig2_{suffix}.png"


def _print_primary_header(primary: dict, *, engine: str) -> None:
    print(f"Sample: {primary['name']} ({primary['id']})")
    if primary.get("description"):
        print(f"  {primary['description']}")
    p_gp = primary.get("P_melt_GPa", float("nan"))
    f_melt = primary.get("F_melt", float("nan"))
    if np.isfinite(p_gp) and np.isfinite(f_melt):
        print(f"  P_melt={p_gp} GPa, F_melt={f_melt}")
    tags = primary.get("tags") or ()
    if tags:
        print(f"  tags: {', '.join(tags)}")
    print(f"  Engine: {engine}")
    print("  Crystallization: wl1990")


def run(
    *,
    engine: str = "wl1990",
    sample: str = FIG2_PAPER_SAMPLE,
    data: Path | None = None,
    f_min: float = 0.0,
    f_max: float = 0.8,
    n_f: int = 81,
    cipw_backend: str = "fallback",
    mineral_backend: Backend = "burnman",
    fig2_ab_calibrate: bool = True,
    fig2_ab_anchor_vp: bool = False,
    kd_engine: KdEngine = "heuristic",
    kd_mode_1990: KdMode1990 = "langmuir",
    save_figure: Path | None = None,
    show: bool = False,
) -> dict:
    primary = fig2_primary_melt(sample_id=sample, data_path=data)
    _print_primary_header(primary, engine=engine)
    print(f"  CIPW backend: {cipw_backend}")
    print(f"  Mineral backend: {mineral_backend}")
    print("  Report state: 100 MPa, 100°C")
    print(f"  Fig.2 paper phases (c-h): {fig2_ab_calibrate}")
    print(f"  Fig.2 anchor Vp (layout): {fig2_ab_anchor_vp}")
    if fig2_ab_calibrate and primary["id"] != FIG2_PAPER_SAMPLE:
        print(
            f"  Warning: paper Fig.2 anchors target {FIG2_PAPER_SAMPLE}; "
            f"panels (a–h) may not match KKHS02 for {primary['id']!r}."
        )
    if not fig2_ab_calibrate:
        if kd_engine == "langmuir":
            print("  Chain: Langmuir STATE (BASALT+langmuir) → HS mixing → CIPW norm Vp")
            print("  (no Fig.2c–h phase anchors; Vp from assemblage + norm physics)")
        print(
            f"  Kd engine: {kd_engine}"
            + (f" (1990 mode={kd_mode_1990})" if kd_engine == "basalt1990" else "")
        )

    f = np.linspace(float(f_min), float(f_max), int(n_f))

    if engine == "proxy":
        curves = _paper_proxy_curves(f)
        _plot_proxy(curves, f, save_figure=save_figure, show=show)
        return {"engine": engine, "sample": primary["id"], "n_points": int(len(f))}

    melt = primary["oxides_wt_percent"]
    states = _simulate_wl1990(
        f,
        melt,
        cipw_backend=cipw_backend,
        mineral_backend=mineral_backend,
        fig2_ab_calibrate=fig2_ab_calibrate,
        fig2_ab_anchor_vp=fig2_ab_anchor_vp,
        kd_engine=kd_engine,
        kd_mode_1990=kd_mode_1990,
    )
    bulk = norm_velocity_from_bulk_wt(
        melt,
        p_pa=100e6,
        t_k=373.15,
        cipw_backend=cipw_backend,
        mineral_backend=mineral_backend,
    )
    primary_vp = float(bulk["vp_km_s"])
    print(f"Primary norm Vp @100 MPa, 100°C: {primary_vp:.3f} km/s (paper ~7.17)")

    for path, st in states.items():
        print(
            f"  {path}: cum Vp {min(s.vp_cumulate_km_s for s in st):.3f}–{max(s.vp_cumulate_km_s for s in st):.3f}; "
            f"residual norm {min(s.vp_residual_norm_km_s for s in st):.3f}–{max(s.vp_residual_norm_km_s for s in st):.3f}"
        )

    _plot_wl1990(
        states,
        primary_norm_vp=primary_vp,
        sample_label=primary["name"],
        save_figure=save_figure,
        show=show,
    )
    return {
        "engine": engine,
        "sample": primary["id"],
        "primary_norm_vp": primary_vp,
        "n_points": int(len(f)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce KKHS02 Figure 2")
    parser.add_argument(
        "--engine",
        choices=("wl1990", "proxy"),
        default="wl1990",
        help="wl1990 = mass-balance crystallization; proxy = paper curve fit",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=FIG2_PAPER_SAMPLE,
        help="CDAT catalog sample id (default: kinzler1997_morb_primary)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="CDAT file path or catalog id (overrides --sample)",
    )
    parser.add_argument("--list-samples", action="store_true", help="List CDAT catalog and exit")
    parser.add_argument("--f-min", type=float, default=0.0)
    parser.add_argument("--f-max", type=float, default=0.8)
    parser.add_argument("--n-f", type=int, default=81)
    parser.add_argument("--cipw-backend", type=str, default="fallback", choices=("pyrolite", "fallback", "auto"))
    parser.add_argument(
        "--fig2-ab",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use KKHS02 Fig.2c-h phase schedule + FC mass balance",
    )
    parser.add_argument(
        "--fig2-ab-anchor-vp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Interpolate sparse Fig.2(a,b) Vp anchors (layout only; default=HS physics)",
    )
    parser.add_argument(
        "--mineral-backend",
        type=str,
        default="fig2",
        choices=("burnman", "empirical", "auto", "fig2", "sb1994", "sb1994_fig2ol"),
        help="sb1994_fig2ol (recommended): S&B + Ol Fig.2 anchor when Ol mass > 80%",
    )
    parser.add_argument(
        "--kd-engine",
        type=str,
        default="heuristic",
        choices=("heuristic", "langmuir", "basalt1990"),
        help="Crystallization backend when --no-fig2-ab",
    )
    parser.add_argument(
        "--kd-mode-1990",
        type=str,
        default="langmuir",
        choices=("1990", "langmuir"),
        help="Kd formula inside basalt1990 STATE (ARHENF vs Langmuir 1992)",
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.list_samples:
        for s in list_samples():
            print(f"{s.id:28s}  {s.name}  [{', '.join(s.tags)}]")
        return

    sample_id = args.sample
    if args.data is not None:
        sample_id = Path(args.data).stem

    output = args.output or _default_output(
        sample_id, args.engine, kd_engine=args.kd_engine, fig2_ab=args.fig2_ab,
    )
    run(
        engine=args.engine,
        sample=args.sample,
        data=args.data,
        f_min=args.f_min,
        f_max=args.f_max,
        n_f=args.n_f,
        cipw_backend=args.cipw_backend,
        mineral_backend=args.mineral_backend,
        fig2_ab_calibrate=args.fig2_ab,
        fig2_ab_anchor_vp=args.fig2_ab_anchor_vp,
        kd_engine=args.kd_engine,
        kd_mode_1990=args.kd_mode_1990,
        save_figure=output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
