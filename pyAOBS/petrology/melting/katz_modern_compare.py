"""
Overlay Katz (2003) Table 2 (``katz2003.py``) vs Modern LIP peridotite melting (pyMelt ``katz_lherzolite``).

Modern track here = the default LIP lithology backend (``pymelt`` + ``katz_lherzolite``),
not the full REEBOX column (no isentropic path / H / Vp).

Fig. 1–6 comparison figures for validation.

Hydrous Fig. 4–5 note (Mcpx matched): native ``katz2003.py`` follows ``lib_k03.f`` / FKATZ —
ΔT depresses **both** solidus and liquidus in the cpx-absent regime (eq. 19 comment in
``lib_k03.f``: "TLIQ ALSO IS EFFECTED BY THE WATER CONTENTS, THEREFORE -DTEM").
pyMelt ``hydrousLithology`` shifts **solidus only** (liquidus stays anhydrous; see pyMelt
Tutorial 2). Solidus T(P) still matches; F(T) and F vs bulk H₂O can differ by ~1–2% F.
"""

from __future__ import annotations

import numpy as np

from typing import Callable

from petrology.melting import katz2003 as katz
from petrology.melting.katz2003 import KATZ_CPX_MASS
from petrology.melting.pymelt_lithology_adapter import instantiate_pymelt_lithology

# pyMelt Katz lithology fixed Mcpx (errata 2023)
_MODERN_CPX_MASS = KATZ_CPX_MASS


def _modern_pm(*, h2o_wt: float = 0.0):
    return instantiate_pymelt_lithology("katz_lherzolite", h2o_wt=float(h2o_wt))


def _modern_f(p_gpa: float, t_c: float, *, h2o_wt: float = 0.0) -> float:
    pm = _modern_pm(h2o_wt=h2o_wt)
    return float(np.clip(pm.F(float(p_gpa), float(t_c)), 0.0, 1.0))


def _modern_tsol(p_gpa: float, *, h2o_wt: float = 0.0) -> float:
    return float(_modern_pm(h2o_wt=h2o_wt).TSolidus(float(p_gpa)))


def _modern_tliq(p_gpa: float, *, h2o_wt: float = 0.0) -> float:
    return float(_modern_pm(h2o_wt=h2o_wt).TLiquidus(float(p_gpa)))


def build_compare_katz_modern_fig1_figure(
    *,
    p_max_gpa: float = 8.0,
    t_min_c: float = 1100.0,
    t_max_c: float = 2000.0,
    n: int = 200,
):
    """Fig. 1 — Katz Table 2 vs pyMelt Katz solidus / liquidus."""
    import matplotlib.pyplot as plt

    p = np.linspace(0.0, float(p_max_gpa), int(n))
    ts_k = np.array([katz.katz2003_solidus(x) for x in p])
    tlh_k = np.array([katz.katz2003_lherzolite_liquidus(x) for x in p])
    tl_k = np.array([katz.katz2003_liquidus(x) for x in p])
    ts_m = np.array([_modern_tsol(x) for x in p])
    tl_m = np.array([_modern_tliq(x) for x in p])

    fig, ax = plt.subplots(figsize=(7.0, 5.4), dpi=100)
    ax.plot(ts_k, p, "k-", lw=2.0, label="Katz Tsol (native)")
    ax.plot(tlh_k, p, color="#1a5276", lw=1.8, ls="--", label="Katz T_lliq (native only)")
    ax.plot(tl_k, p, color="#922b21", lw=2.0, ls="-.", label="Katz T_liq (native)")
    ax.plot(ts_m, p, color="#e74c3c", lw=1.6, ls=":", label="Modern Tsol (pyMelt)")
    ax.plot(tl_m, p, color="#c0392b", lw=1.6, ls=":", label="Modern T_liq (pyMelt)")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Pressure (GPa)")
    ax.set_title(
        "Fig. 1 compare — Katz (2003) native vs Modern pyMelt katz_lherzolite\n"
        f"pyMelt Mcpx={100 * _MODERN_CPX_MASS:g} wt%"
    )
    ax.set_xlim(float(t_min_c), float(t_max_c))
    ax.set_ylim(float(p_max_gpa), 0.0)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def build_compare_katz_modern_fig2_figure(
    *,
    pressures_gpa: tuple[float, ...] = (1.0, 2.0, 3.0),
    t_min_c: float = 1000.0,
    t_max_c: float = 2000.0,
    katz_cpx_mass: float = katz._DEFAULT_CPX_MASS,
    n: int = 400,
):
    """Fig. 2 — isobaric dry F(T); note Mcpx 15% (native fig) vs 17% (pyMelt)."""
    import matplotlib.pyplot as plt

    t = np.linspace(float(t_min_c), float(t_max_c), int(n))
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=100)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(pressures_gpa), 1)))

    for pg, col in zip(pressures_gpa, cmap):
        fk = katz.katz2003_melt_fraction_dry_profile(pg, t, cpx_mass=katz_cpx_mass)
        fm = np.array([_modern_f(pg, float(ti)) for ti in t])
        ax.plot(t, fk, color=col, lw=2.0, label=f"P={pg:g} GPa Katz")
        ax.plot(t, fm, color=col, lw=1.5, ls="--", label=f"P={pg:g} GPa Modern")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Melt fraction, F")
    ax.set_title(
        "Fig. 2 compare — dry F(T)\n"
        f"Katz native Mcpx={100 * katz_cpx_mass:g} wt%  |  Modern pyMelt Mcpx={100 * _MODERN_CPX_MASS:g} wt%"
    )
    ax.set_xlim(float(t_min_c), float(t_max_c))
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left", fontsize=6, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def build_compare_katz_modern_fig3_figure(
    *,
    p_max_gpa: float = 8.0,
    t_min_c: float = 850.0,
    t_max_c: float = 2000.0,
    bulk_h2o_wt_pct: tuple[float, ...] = (0.05, 0.1, 0.3),
    n: int = 200,
):
    """Fig. 3 — hydrous solidus T(P)."""
    import matplotlib.pyplot as plt

    p = np.linspace(0.0, float(p_max_gpa), int(n))
    fig, ax = plt.subplots(figsize=(7.0, 5.4), dpi=100)

    ts_d_k = np.array([katz.katz2003_solidus(x) for x in p])
    ts_d_m = np.array([_modern_tsol(x, h2o_wt=0.0) for x in p])
    ax.plot(ts_d_k, p, "k-", lw=2.0, label="0 wt% Katz")
    ax.plot(ts_d_m, p, "k--", lw=1.4, label="0 wt% Modern")

    ts_sat_k = np.array([katz.katz2003_hydrous_solidus_t(x, saturated=True) for x in p])
    ts_sat_m = np.array([_modern_tsol(x, h2o_wt=katz.katz2003_x_sat(x)) for x in p])
    ax.plot(ts_sat_k, p, color="#555", lw=1.8, ls=":", label="sat Katz")
    ax.plot(ts_sat_m, p, color="#555", lw=1.2, ls="--", label="sat Modern (bulk=X_sat)")

    cmap = plt.cm.plasma(np.linspace(0.25, 0.85, max(len(bulk_h2o_wt_pct), 1)))
    for bulk, col in zip(bulk_h2o_wt_pct, cmap):
        th_k = np.array([katz.katz2003_hydrous_solidus_t(x, bulk_h2o_wt_pct=bulk) for x in p])
        th_m = np.array([_modern_tsol(x, h2o_wt=bulk) for x in p])
        ax.plot(th_k, p, color=col, lw=1.8, label=f"{bulk:g} wt% Katz")
        ax.plot(th_m, p, color=col, lw=1.2, ls="--", label=f"{bulk:g} wt% Modern")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Pressure (GPa)")
    ax.set_title("Fig. 3 compare — hydrous solidus (bulk H₂O wt%)")
    ax.set_xlim(float(t_min_c), float(t_max_c))
    ax.set_ylim(float(p_max_gpa), 0.0)
    ax.legend(loc="upper right", fontsize=6, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def build_compare_katz_modern_fig4_figure(
    *,
    p_gpa: float = 1.0,
    t_min_c: float = 900.0,
    t_max_c: float = 1400.0,
    f_max: float = 0.4,
    bulk_h2o_wt_pct: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1, 0.3),
    katz_cpx_mass: float = katz._DEFAULT_CPX_MASS,
    n: int = 400,
):
    """Fig. 4 — isobaric hydrous F(T) at 1 GPa."""
    import matplotlib.pyplot as plt

    t = np.linspace(float(t_min_c), float(t_max_c), int(n))
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=100)
    hydrous = [b for b in bulk_h2o_wt_pct if b > 0.0]
    cmap = plt.cm.plasma(np.linspace(0.25, 0.85, max(len(hydrous), 1)))
    h_idx = 0

    for bulk in bulk_h2o_wt_pct:
        if bulk <= 0.0:
            col, ls_k, ls_m, label = "k", "-", "--", "0 wt%"
        else:
            col, ls_k, ls_m = cmap[h_idx], "-", "--"
            label = f"{bulk:g} wt%"
            h_idx += 1
        fk = katz.katz2003_melt_fraction_hydrous_profile(
            p_gpa, t, bulk_h2o_wt_pct=float(bulk), cpx_mass=katz_cpx_mass
        )
        fm = np.array([_modern_f(p_gpa, float(ti), h2o_wt=bulk) for ti in t])
        ax.plot(t, fk, color=col, lw=1.8, ls=ls_k, label=f"{label} Katz")
        ax.plot(t, fm, color=col, lw=1.2, ls=ls_m, label=f"{label} Modern")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Melt fraction, F")
    ax.set_title(
        f"Fig. 4 compare — hydrous F(T) at P={p_gpa:g} GPa\n"
        f"Katz Mcpx={100 * katz_cpx_mass:g}%  |  Modern Mcpx={100 * _MODERN_CPX_MASS:g}%\n"
        "ΔF: pyMelt shifts solidus only; native FKATZ also shifts liquidus (lib_k03.f)"
    )
    ax.set_xlim(float(t_min_c), float(t_max_c))
    ax.set_ylim(0.0, float(f_max))
    ax.legend(loc="upper left", fontsize=6, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def build_compare_katz_modern_fig5_figure(
    *,
    p_gpa: float = 1.5,
    temperatures_c: tuple[float, ...] = (1200, 1250, 1300, 1350),
    bulk_h2o_max_wt_pct: float = 0.5,
    n_bulk: int = 120,
):
    """Fig. 5 — F vs bulk H₂O at fixed (P, T)."""
    import matplotlib.pyplot as plt

    bulk = np.linspace(0.0, float(bulk_h2o_max_wt_pct), int(n_bulk))
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=100)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.88, max(len(temperatures_c), 1)))

    for t_c, col in zip(temperatures_c, cmap):
        _, fk, _ = katz.katz2003_fig5_melt_vs_water(p_gpa=p_gpa, t_c=float(t_c), bulk_h2o_wt_pct=bulk)
        fm = np.array([_modern_f(p_gpa, float(t_c), h2o_wt=float(b)) for b in bulk])
        label = f"{t_c:g}°C"
        ax.plot(bulk, fk, color=col, lw=1.8, label=f"{label} Katz")
        ax.plot(bulk, fm, color=col, lw=1.2, ls="--", label=f"{label} Modern")

    ax.set_xlabel(r"Bulk H$_2$O (wt%)")
    ax.set_ylabel("Melt fraction, F")
    ax.set_title(
        f"Fig. 5 compare — F vs bulk H₂O at P={p_gpa:g} GPa\n"
        f"Katz Fig.5 Mcpx=17%  |  Modern pyMelt Mcpx=17%\n"
        "ΔF: pyMelt solidus-only ΔT vs native FKATZ solidus+liquidus ΔT"
    )
    ax.set_xlim(0.0, 0.25)
    ax.set_ylim(0.0, 0.25)
    ax.legend(loc="upper left", fontsize=6, ncol=2, title="T")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def build_compare_katz_modern_fig6_figure(
    *,
    pressures_gpa: tuple[float, ...] = (0.0, 1.0, 1.5, 3.0),
    t_xlim_c: tuple[tuple[float, float], ...] = (
        (1000.0, 1600.0),
        (1150.0, 1500.0),
        (1200.0, 1700.0),
        (1400.0, 1900.0),
    ),
    katz_cpx_mass: float = katz._DEFAULT_CPX_MASS,
    f_max: float = 0.7,
    n: int = 400,
):
    """Fig. 6 — four-panel dry F(T) (same axes as accepted Fig. 6)."""
    import matplotlib.pyplot as plt

    labels = ("(a)", "(b)", "(c)", "(d)")
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5), dpi=100, sharey=True)
    axes_flat = axes.ravel()

    for ax, pg, panel, (t_min, t_max) in zip(axes_flat, pressures_gpa, labels, t_xlim_c):
        t = np.linspace(float(t_min), float(t_max), int(n))
        fk = katz.katz2003_melt_fraction_dry_profile(pg, t, cpx_mass=katz_cpx_mass)
        fm = np.array([_modern_f(pg, float(ti)) for ti in t])
        ax.plot(t, fk, "k-", lw=2.0, label="Katz")
        ax.plot(t, fm, color="#e74c3c", lw=1.6, ls="--", label="Modern")
        ax.set_xlim(float(t_min), float(t_max))
        ax.set_ylim(0.0, float(f_max))
        ax.set_title(f"{panel} P = {pg:g} GPa")
        ax.set_xlabel("Temperature (°C)")
        ax.grid(True, alpha=0.25)

    axes_flat[0].set_ylabel("Melt fraction, F")
    axes_flat[2].set_ylabel("Melt fraction, F")
    axes_flat[0].legend(loc="upper left", fontsize=7)
    fig.suptitle(
        "Fig. 6 compare — dry F(T)\n"
        f"Katz Mcpx={100 * katz_cpx_mass:g}%  |  Modern Mcpx={100 * _MODERN_CPX_MASS:g}%",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    return fig


COMPARE_BUILDERS: dict[str, Callable] = {
    "fig1": build_compare_katz_modern_fig1_figure,
    "fig2": build_compare_katz_modern_fig2_figure,
    "fig3": build_compare_katz_modern_fig3_figure,
    "fig4": build_compare_katz_modern_fig4_figure,
    "fig5": build_compare_katz_modern_fig5_figure,
    "fig6": build_compare_katz_modern_fig6_figure,
}


def max_abs_diff_dry_f(p_gpa: float, t_c: np.ndarray, *, katz_cpx_mass: float) -> float:
    fk = katz.katz2003_melt_fraction_dry_profile(p_gpa, t_c, cpx_mass=katz_cpx_mass)
    fm = np.array([_modern_f(p_gpa, float(ti)) for ti in t_c])
    return float(np.max(np.abs(fk - fm)))


def print_compare_summary(*, katz_cpx_mass: float = katz._DEFAULT_CPX_MASS) -> None:
    """Console spot-check: max |ΔF| and |ΔT| between native Katz and Modern pyMelt."""
    t_scan = np.linspace(1100.0, 1700.0, 121)
    print("Katz (2003) native vs Modern pyMelt katz_lherzolite")
    print(f"  Katz Mcpx={100 * katz_cpx_mass:g}%  Modern Mcpx={100 * _MODERN_CPX_MASS:g}%")
    print()

    for p in (0.0, 1.0, 2.0, 3.0):
        dts = abs(katz.katz2003_solidus(p) - _modern_tsol(p))
        dtl = abs(katz.katz2003_liquidus(p) - _modern_tliq(p))
        df = max_abs_diff_dry_f(p, t_scan, katz_cpx_mass=katz_cpx_mass)
        print(f"  P={p:g} GPa: |ΔTsol|={dts:.3f}°C  |ΔTliq|={dtl:.3f}°C  max|ΔF|={df:.4f}")

    bulk = 0.1
    p = 1.0
    dts_h = abs(katz.katz2003_hydrous_solidus_t(p, bulk_h2o_wt_pct=bulk) - _modern_tsol(p, h2o_wt=bulk))
    t_h = np.linspace(950.0, 1300.0, 71)
    fk = katz.katz2003_melt_fraction_hydrous_profile(p, t_h, bulk_h2o_wt_pct=bulk, cpx_mass=katz_cpx_mass)
    fm = np.array([_modern_f(p, float(ti), h2o_wt=bulk) for ti in t_h])
    df_h = float(np.max(np.abs(fk - fm)))
    print(f"  Hydrous bulk={bulk}% @1GPa: |ΔTsol|={dts_h:.3f}°C  max|ΔF|={df_h:.4f}")
    print()
    print("  Hydrous Fig.4–5: |ΔTsol|≈0 but |ΔF|>0 is expected.")
    print("    native katz2003 = lib_k03.f FKATZ (ΔT on solidus AND liquidus, eq.19)")
    print("    Modern pyMelt   = hydrousLithology (ΔT on solidus only; Tutorial 2)")
