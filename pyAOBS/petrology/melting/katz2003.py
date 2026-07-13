"""
Katz et al. (2003) anhydrous peridotite melting parameterization (Table 2).

Pressure P in GPa, temperature in °C. Coefficients match the reference Fortran
implementation (e.g. mormel ``lib_k03.f``) and pyMelt / ASPECT defaults.

Three T(P) curves (Fig. 1 caption):
  - solidus (eq. 4)
  - lherzolite liquidus T_lliq (eq. 5) — full lherzolite assemblage limit
  - liquidus T_liq (eq. 10) — after opx melting regime

Note: ``lithology.katz2003_peridotite_liquidus`` uses REEBOX-refined eq. (10)
coefficients for the Modern track; use this module for paper Table 2 values.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

_KATZ2003_FIG6_DATA = Path(__file__).resolve().parents[1] / "data" / "katz2003_fig6"
_KATZ2003_FIG7_DATA = Path(__file__).resolve().parents[1] / "data" / "katz2003_fig7"
_KATZ2003_FIG8_DATA = Path(__file__).resolve().parents[1] / "data" / "katz2003_fig8"
_FIG6_PANEL_ORDER = ("a", "b", "c", "d")

# Scatter style keyed by study_id (extend as ESI Table S2 is filled in).
_FIG6_STUDY_STYLE: dict[str, dict[str, Any]] = {
    "JG1980_HP": {
        "marker": "s",
        "color": "#c0392b",
        "ms": 6,
        "label": "Jaques & Green (1980) pyrolite",
    },
    "JG1980_TQ": {
        "marker": "^",
        "color": "#2c3e50",
        "ms": 6,
        "label": "Jaques & Green (1980) Tinaquillo",
    },
    "MM3": {"marker": "o", "color": "#2980b9", "ms": 5, "label": "Baker & Stolper (1994)"},
    "MM3_FG99": {"marker": "^", "color": "#27ae60", "ms": 5, "label": "Falloon et al. (1999)"},
    "KLB1": {"marker": "D", "color": "#8e44ad", "ms": 4, "label": "Takahashi (1986)"},
    "KR4003": {"marker": "v", "color": "#d35400", "ms": 5, "label": "Walter (1998)"},
    "FG87": {"marker": "P", "color": "#16a085", "ms": 5, "label": "Falloon & Green (1988)"},
}
_FIG6_FALLBACK_MARKERS = ("o", "^", "s", "D", "v", "P", "*", "X", "h", "8")

# Katz et al. (2003) Table 2 — P in GPa, T in °C
_SOLIDUS = (1085.7, 132.9, -5.1)  # eq. (4); A1 = Hirschmann (2000) − 35 °C
_HIRSCHMANN2000_SOLIDUS = (1120.661, 132.899, -5.104)  # Hirschmann (2000) G³
_LHERZOLITE_LIQUIDUS = (1475.0, 80.0, -3.2)  # eq. (5)
_LIQUIDUS = (1780.0, 45.0, -2.0)  # eq. (10)
_R_CPX = (0.50, 0.08)  # eq. (6)
_BETA1 = 1.5
_BETA2 = 1.5
# Katz (2003) errata 2023 / pyMelt katz_lherzolite default (17 wt% modal cpx).
KATZ_CPX_MASS = 0.17
_DEFAULT_CPX_MASS = KATZ_CPX_MASS

# Hydrous melting (Table 2, eq. 16–18)
_PK = 43.0
_GAMMA = 0.75
_DH2O = 0.01
_CHI1 = 12.0
_CHI2 = 1.0
_PLAM = 0.6

# Isentropic melting thermodynamics (Table 2, eq. 11–15); not used in F(T) isobaric curves.
_CP_J_KG = 1000.0  # J kg⁻¹ K⁻¹
_ALPHA_S_K = 40.0e-6  # K⁻¹
_ALPHA_F_K = 68.0e-6  # K⁻¹
_RHO_S = 3.3e3  # kg m⁻³
_RHO_F = 2.9e3  # kg m⁻³
_DELTA_S = 300.0  # J kg⁻¹ K⁻¹


def _quad(p_gpa: float, c0: float, c1: float, c2: float) -> float:
    p = float(p_gpa)
    return c0 + c1 * p + c2 * p * p


def katz2003_solidus(p_gpa: float) -> float:
    """Dry peridotite solidus T_s(P), Katz (2003) eq. (4)."""
    return _quad(p_gpa, *_SOLIDUS)


def hirschmann2000_solidus(p_gpa: float) -> float:
    """Anhydrous peridotite solidus T_s(P), Hirschmann (2000) regression (GPa, °C)."""
    return _quad(p_gpa, *_HIRSCHMANN2000_SOLIDUS)


def katz2003_lherzolite_liquidus(p_gpa: float) -> float:
    """Lherzolite multisaturated liquidus T_lliq(P), Katz (2003) eq. (5)."""
    return _quad(p_gpa, *_LHERZOLITE_LIQUIDUS)


def katz2003_liquidus(p_gpa: float) -> float:
    """Full peridotite liquidus T_liq(P), Katz (2003) eq. (10)."""
    return _quad(p_gpa, *_LIQUIDUS)


def katz2003_r_cpx(p_gpa: float) -> float:
    """Cpx reaction coefficient R_cpx(P), eq. (6)."""
    return _R_CPX[0] + _R_CPX[1] * float(p_gpa)


def katz2003_fcpx_out(*, p_gpa: float, cpx_mass: float = _DEFAULT_CPX_MASS) -> float:
    """Melt fraction at cpx exhaustion, eq. (7)."""
    rc = katz2003_r_cpx(p_gpa)
    return float(cpx_mass) / max(rc, 1e-9)


def katz2003_tcpx_out(*, p_gpa: float, cpx_mass: float = _DEFAULT_CPX_MASS) -> float:
    """Temperature at cpx exhaustion along anhydrous isobar (used in eq. 8–10)."""
    ts = katz2003_solidus(p_gpa)
    tlh = katz2003_lherzolite_liquidus(p_gpa)
    fc = katz2003_fcpx_out(p_gpa=p_gpa, cpx_mass=cpx_mass)
    return fc ** (1.0 / _BETA1) * (tlh - ts) + ts


def katz2003_melt_fraction_dry(
    p_gpa: float,
    t_c: float,
    *,
    cpx_mass: float = _DEFAULT_CPX_MASS,
) -> float:
    """Anhydrous melt fraction F(P, T), Katz (2003) eq. (2)–(3) and (8)–(10)."""
    p = float(p_gpa)
    t = float(t_c)
    ts = katz2003_solidus(p)
    if t <= ts:
        return 0.0
    tlh = katz2003_lherzolite_liquidus(p)
    tl = katz2003_liquidus(p)
    td = (t - ts) / max(tlh - ts, 1e-6)
    f = float(np.clip(td**_BETA1, 0.0, 1.0))
    fc = katz2003_fcpx_out(p_gpa=p, cpx_mass=cpx_mass)
    if f <= fc:
        return f
    tc = katz2003_tcpx_out(p_gpa=p, cpx_mass=cpx_mass)
    td2 = (t - tc) / max(tl - tc, 1e-6)
    return float(np.clip(fc + (1.0 - fc) * td2**_BETA2, 0.0, 1.0))


def katz2003_melt_fraction_dry_profile(
    p_gpa: float,
    t_c: np.ndarray | list[float],
    *,
    cpx_mass: float = _DEFAULT_CPX_MASS,
) -> np.ndarray:
    """Vectorized F(T) at fixed P for isobaric melting curves (Fig. 2)."""
    return np.array(
        [katz2003_melt_fraction_dry(p_gpa, float(ti), cpx_mass=cpx_mass) for ti in t_c],
        dtype=float,
    )


def katz2003_t_from_f(
    p_gpa: float,
    f: float,
    *,
    cpx_mass: float = _DEFAULT_CPX_MASS,
    n_iter: int = 64,
) -> float:
    """Invert Katz (2003) dry F(P,T)=f via bisection between solidus and liquidus."""
    f = float(np.clip(f, 0.0, 0.999))
    if f <= 0.0:
        return katz2003_solidus(p_gpa)
    tl = katz2003_liquidus(p_gpa)
    if f >= 0.999:
        return tl
    lo = katz2003_solidus(p_gpa)
    hi = tl
    for _ in range(int(n_iter)):
        mid = 0.5 * (lo + hi)
        fm = katz2003_melt_fraction_dry(p_gpa, mid, cpx_mass=cpx_mass)
        if fm < f:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def katz2003_x_sat(p_gpa: float) -> float:
    """Melt H2O at fluid saturation (wt%), eq. (17)."""
    p = float(p_gpa)
    return _CHI1 * p**_PLAM + _CHI2 * p


def katz2003_delta_t_h2o(x_wt_pct: float) -> float:
    """Solidus depression ΔT (°C); ``x_wt_pct`` = dissolved H2O in melt (wt%), eq. (16)."""
    x = max(float(x_wt_pct), 0.0)
    if x <= 0.0:
        return 0.0
    return _PK * x**_GAMMA


def katz2003_x_melt_h2o_wt_pct(*, bulk_h2o_wt_pct: float, f: float) -> float:
    """
    Dissolved H2O in melt (wt%), Katz (2003) eq. (18).

    ``bulk_h2o_wt_pct`` is bulk source H2O in **wt%** (e.g. 0.1 for 0.1 wt%), not mass fraction.
    """
    f = float(np.clip(f, 0.0, 0.999))
    return float(bulk_h2o_wt_pct) / max(_DH2O + f * (1.0 - _DH2O), 1e-12)


def katz2003_x_melt_h2o_at_f(
    *,
    bulk_h2o_wt_pct: float,
    f: float,
    p_gpa: float,
) -> float:
    """Dissolved melt H2O (wt%) at melt fraction F, capped at ``katz2003_x_sat`` (eq. 17–18)."""
    if bulk_h2o_wt_pct <= 0.0:
        return 0.0
    x = katz2003_x_melt_h2o_wt_pct(bulk_h2o_wt_pct=bulk_h2o_wt_pct, f=f)
    return min(x, katz2003_x_sat(p_gpa))


def katz2003_dtem_h2o_at_f(
    *,
    bulk_h2o_wt_pct: float,
    f: float,
    p_gpa: float,
) -> float:
    """Solidus/liquidus depression ΔT (°C) at melt fraction F (eq. 16 with eq. 18 x)."""
    x = katz2003_x_melt_h2o_at_f(bulk_h2o_wt_pct=bulk_h2o_wt_pct, f=f, p_gpa=p_gpa)
    return katz2003_delta_t_h2o(x)


def _katz2003_melt_fraction_at_dtem(
    p_gpa: float,
    t_c: float,
    dtem: float,
    *,
    cpx_mass: float = _DEFAULT_CPX_MASS,
) -> float:
    """
    Hydrous melt fraction at fixed ΔT (eq. 19, 8–10 with −ΔT on solidus / liquidus).

    Matches ``FKATZ`` in ``lib_k03.f`` once ΔT is known.
    """
    p = float(p_gpa)
    t = float(t_c)
    ts = katz2003_solidus(p)
    tlh = katz2003_lherzolite_liquidus(p)
    tl = katz2003_liquidus(p)
    ts_h = ts - dtem
    if t <= ts_h:
        return 0.0
    td = (t - ts_h) / max(tlh - ts, 1e-6)
    f = float(np.clip(td**_BETA1, 0.0, 1.0))
    fc = katz2003_fcpx_out(p_gpa=p, cpx_mass=cpx_mass)
    if f <= fc:
        return f
    tc = fc ** (1.0 / _BETA1) * (tlh - ts) + ts_h
    td2 = (t - tc) / max(tl - dtem - tc, 1e-6)
    return float(np.clip(fc + (1.0 - fc) * td2**_BETA2, 0.0, 1.0))


def katz2003_melt_fraction_hydrous(
    p_gpa: float,
    t_c: float,
    *,
    bulk_h2o_wt_pct: float = 0.0,
    cpx_mass: float = _DEFAULT_CPX_MASS,
) -> float:
    """
    Isobaric hydrous melt fraction F(P, T) with batch H2O partitioning (eq. 18–19).

    Self-consistent: x(F) from eq. (18), ΔT(x) from eq. (16), F from eq. (19) / (8–10).
    """
    if bulk_h2o_wt_pct <= 0.0:
        return katz2003_melt_fraction_dry(p_gpa, t_c, cpx_mass=cpx_mass)

    p = float(p_gpa)
    t = float(t_c)

    def _fout(f: float) -> float:
        dtem = katz2003_dtem_h2o_at_f(bulk_h2o_wt_pct=bulk_h2o_wt_pct, f=f, p_gpa=p)
        return _katz2003_melt_fraction_at_dtem(p, t, dtem, cpx_mass=cpx_mass)

    def _residual(f: float) -> float:
        return float(f) - _fout(float(f))

    dtem0 = katz2003_dtem_h2o_at_f(bulk_h2o_wt_pct=bulk_h2o_wt_pct, f=0.0, p_gpa=p)
    if t <= katz2003_solidus(p) - dtem0:
        return 0.0

    dtem1 = katz2003_dtem_h2o_at_f(bulk_h2o_wt_pct=bulk_h2o_wt_pct, f=1.0, p_gpa=p)
    if t >= katz2003_liquidus(p) - dtem1:
        return 1.0

    lo, hi = 0.0, 1.0
    r_lo, r_hi = _residual(lo), _residual(hi)
    if abs(r_lo) < 1e-10:
        return 0.0
    if r_lo > 0.0:
        return 0.0
    if r_hi <= 0.0:
        return 1.0

    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if _residual(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def katz2003_melt_fraction_hydrous_profile(
    p_gpa: float,
    t_c: np.ndarray | list[float],
    *,
    bulk_h2o_wt_pct: float = 0.0,
    cpx_mass: float = _DEFAULT_CPX_MASS,
) -> np.ndarray:
    """Vectorized F(T) at fixed P for isobaric hydrous melting curves (Fig. 4)."""
    t_arr = np.asarray(t_c, dtype=float)
    out = np.empty_like(t_arr)
    f_prev = 0.0
    for i, ti in enumerate(t_arr):
        if bulk_h2o_wt_pct <= 0.0:
            out[i] = katz2003_melt_fraction_dry(p_gpa, float(ti), cpx_mass=cpx_mass)
            continue
        # Warm-start bisection from previous T (F increases along isobar).
        p = float(p_gpa)
        t = float(ti)

        def _fout(f: float) -> float:
            dtem = katz2003_dtem_h2o_at_f(bulk_h2o_wt_pct=bulk_h2o_wt_pct, f=f, p_gpa=p)
            return _katz2003_melt_fraction_at_dtem(p, t, dtem, cpx_mass=cpx_mass)

        def _residual(f: float) -> float:
            return float(f) - _fout(float(f))

        dtem0 = katz2003_dtem_h2o_at_f(bulk_h2o_wt_pct=bulk_h2o_wt_pct, f=0.0, p_gpa=p)
        if t <= katz2003_solidus(p) - dtem0:
            out[i] = 0.0
            f_prev = 0.0
            continue
        dtem1 = katz2003_dtem_h2o_at_f(bulk_h2o_wt_pct=bulk_h2o_wt_pct, f=1.0, p_gpa=p)
        if t >= katz2003_liquidus(p) - dtem1:
            out[i] = 1.0
            f_prev = 1.0
            continue

        lo = max(0.0, f_prev - 0.05)
        hi = min(1.0, max(f_prev + 0.05, lo + 1e-6))
        if _residual(lo) > 0.0:
            lo = 0.0
        r_hi = _residual(hi)
        if r_hi < 0.0:
            hi = 1.0
            r_hi = _residual(hi)
        if r_hi <= 0.0:
            out[i] = 1.0
            f_prev = 1.0
            continue
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            if _residual(mid) > 0.0:
                hi = mid
            else:
                lo = mid
        f_prev = 0.5 * (lo + hi)
        out[i] = f_prev
    return out


def katz2003_hydrous_solidus_t(
    p_gpa: float,
    *,
    bulk_h2o_wt_pct: float = 0.0,
    saturated: bool = False,
) -> float:
    """
    Hydrous solidus T(P) for Fig. 3.

    ``bulk_h2o_wt_pct``: bulk H2O in **wt%** (0.05, 0.1, 0.3 in the paper).
    ``saturated=True``: water-saturated solidus (x = X_sat from eq. 17).
    """
    p = float(p_gpa)
    ts = katz2003_solidus(p)
    if saturated:
        x = katz2003_x_sat(p)
    elif bulk_h2o_wt_pct <= 0.0:
        return ts
    else:
        xsat = katz2003_x_sat(p)
        x = katz2003_x_melt_h2o_wt_pct(bulk_h2o_wt_pct=bulk_h2o_wt_pct, f=0.0)
        x = min(x, xsat)
    return ts - katz2003_delta_t_h2o(x)


def build_katz2003_fig1_figure(
    *,
    p_max_gpa: float = 8.0,
    t_min_c: float = 1100.0,
    t_max_c: float = 2000.0,
    n: int = 200,
):
    """
    Katz et al. (2003) Fig. 1 — anhydrous solidus, lherzolite liquidus, liquidus.

    Axes match the paper-style T–P diagram: temperature (°C) horizontal,
    pressure (GPa) vertical increasing downward (0 at top, ``p_max_gpa`` at bottom).

    Returns a matplotlib Figure (does not show or save).
    """
    import matplotlib.pyplot as plt

    p = np.linspace(0.0, float(p_max_gpa), int(n))
    ts = np.array([katz2003_solidus(x) for x in p])
    tlh = np.array([katz2003_lherzolite_liquidus(x) for x in p])
    tl = np.array([katz2003_liquidus(x) for x in p])

    fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=100)
    ax.fill_betweenx(p, ts, tlh, color="#f5e6c8", alpha=0.55, zorder=0)
    ax.plot(ts, p, "k-", lw=2.2, label="Solidus (eq. 4)", zorder=3)
    ax.plot(
        tlh,
        p,
        color="#1a5276",
        lw=2.0,
        ls="--",
        label="Lherzolite liquidus (eq. 5)",
        zorder=3,
    )
    ax.plot(
        tl,
        p,
        color="#922b21",
        lw=2.0,
        ls="-.",
        label="Liquidus (eq. 10)",
        zorder=3,
    )

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Pressure (GPa)")
    ax.set_title(
        "Katz et al. (2003) Fig. 1 — anhydrous peridotite\n"
        "solidus, lherzolite liquidus, liquidus (Table 2)"
    )
    ax.set_xlim(float(t_min_c), float(t_max_c))
    ax.set_ylim(float(p_max_gpa), 0.0)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def build_katz2003_fig2_figure(
    *,
    pressures_gpa: tuple[float, ...] = (1.0, 2.0, 3.0),
    t_min_c: float = 1000.0,
    t_max_c: float = 2000.0,
    cpx_mass: float = _DEFAULT_CPX_MASS,
    n: int = 400,
):
    """
    Katz et al. (2003) Fig. 2 — isobaric anhydrous F(T) at several pressures.

    X: Temperature (°C); Y: melt fraction F (0–1).
    Modal cpx in unmelted rock defaults to 15 wt% (paper caption; errata → 17 wt%).
    """
    import matplotlib.pyplot as plt

    t = np.linspace(float(t_min_c), float(t_max_c), int(n))
    fig, ax = plt.subplots(figsize=(6.8, 5.0), dpi=100)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(pressures_gpa), 1)))

    for pg, col in zip(pressures_gpa, cmap):
        f = katz2003_melt_fraction_dry_profile(pg, t, cpx_mass=cpx_mass)
        ax.plot(t, f, color=col, lw=2.0, label=f"P = {pg:g} GPa")
        fc = katz2003_fcpx_out(p_gpa=pg, cpx_mass=cpx_mass)
        tc = katz2003_tcpx_out(p_gpa=pg, cpx_mass=cpx_mass)
        ax.plot(tc, fc, "o", color=col, ms=5, zorder=4)

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Melt fraction, F")
    ax.set_title(
        "Katz et al. (2003) Fig. 2 — isobaric anhydrous melting\n"
        f"modal cpx = {100.0 * cpx_mass:g} wt%  (● = cpx-out)"
    )
    ax.set_xlim(float(t_min_c), float(t_max_c))
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def load_katz2003_fig6_experiments(
    path: Path | None = None,
    *,
    study_ids: Sequence[str] | None = None,
    include_unverified: bool = False,
) -> list[dict[str, str]]:
    """
    Load Katz Fig.6 experimental (P, T, F) points from ``experiments.csv``.

    By default skips rows with ``verified=no`` unless *include_unverified* is True.
    """
    csv_path = path or (_KATZ2003_FIG6_DATA / "experiments.csv")
    if not csv_path.is_file():
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if study_ids is not None:
        allowed = {s.strip() for s in study_ids if s.strip()}
        rows = [r for r in rows if r.get("study_id", "") in allowed]
    if not include_unverified:
        rows = [r for r in rows if r.get("verified", "").lower() != "no"]
    return rows


def _fig6_style_for_study(study_id: str, index: int) -> dict[str, Any]:
    if study_id in _FIG6_STUDY_STYLE:
        return dict(_FIG6_STUDY_STYLE[study_id])
    marker = _FIG6_FALLBACK_MARKERS[index % len(_FIG6_FALLBACK_MARKERS)]
    return {
        "marker": marker,
        "color": "#555555",
        "ms": 5,
        "label": study_id,
    }


def build_katz2003_fig6_figure(
    *,
    pressures_gpa: tuple[float, ...] = (0.0, 1.0, 1.5, 3.0),
    cpx_mass: float = _DEFAULT_CPX_MASS,
    t_xlim_c: tuple[tuple[float, float], ...] = (
        (1000.0, 1600.0),
        (1150.0, 1500.0),
        (1200.0, 1700.0),
        (1400.0, 1900.0),
    ),
    f_max: float = 0.7,
    n: int = 400,
    show_experiments: bool = True,
    experiments_csv: Path | None = None,
    experiment_study_ids: Sequence[str] | None = ("JG1980_HP", "JG1980_TQ"),
    include_unverified_experiments: bool = False,
):
    """
    Katz et al. (2003) Fig. 6 — model F(T) vs isobaric experimental subsets.

    Four panels at 0, 1, 1.5, and 3 GPa with modal cpx = 15 wt% (Fig. 6 caption).
    Default temperature axes match the paper: (a) 1000–1600, (b) 1150–1500,
    (c) 1200–1700, (d) 1400–1900 °C.

    Experimental scatter comes from ``petrology/data/katz2003_fig6/experiments.csv``.
    By default draws *Jaques & Green (1980)* Table 4 (pyrolite) and Table 5 (Tinaquillo);
    pass ``experiment_study_ids=None`` for every non-``verified=no`` row.
    """
    import matplotlib.pyplot as plt

    if len(pressures_gpa) != len(t_xlim_c):
        raise ValueError("pressures_gpa and t_xlim_c must have the same length")

    labels = ("(a)", "(b)", "(c)", "(d)")
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2), dpi=100, sharey=True)
    axes_flat = axes.ravel()
    panel_to_ax = dict(zip(_FIG6_PANEL_ORDER, axes_flat))

    exp_rows: list[dict[str, str]] = []
    if show_experiments:
        exp_rows = load_katz2003_fig6_experiments(
            experiments_csv,
            study_ids=experiment_study_ids,
            include_unverified=include_unverified_experiments,
        )

    exp_by_panel_study: dict[str, dict[str, list[dict[str, str]]]] = {
        p: {} for p in _FIG6_PANEL_ORDER
    }
    for row in exp_rows:
        panel = row.get("fig6_panel", "")
        sid = row.get("study_id", "")
        if panel not in exp_by_panel_study or not sid:
            continue
        exp_by_panel_study[panel].setdefault(sid, []).append(row)

    plotted_studies: list[str] = []

    for ax, pg, panel, (t_min, t_max) in zip(axes_flat, pressures_gpa, labels, t_xlim_c):
        t = np.linspace(float(t_min), float(t_max), int(n))
        f = katz2003_melt_fraction_dry_profile(pg, t, cpx_mass=cpx_mass)
        fc = katz2003_fcpx_out(p_gpa=pg, cpx_mass=cpx_mass)
        tc = katz2003_tcpx_out(p_gpa=pg, cpx_mass=cpx_mass)

        ax.plot(t, f, "k-", lw=2.0, label="Katz (2003) model", zorder=3)
        ax.plot(tc, fc, "ko", ms=5, zorder=4, label="cpx-out")
        ax.set_xlim(float(t_min), float(t_max))
        ax.set_ylim(0.0, float(f_max))
        ax.set_title(f"{panel} P = {pg:g} GPa")
        ax.set_xlabel("Temperature (°C)")
        ax.grid(True, alpha=0.25)

        panel_key = panel.strip("()")
        for study_idx, (study_id, pts) in enumerate(
            sorted(exp_by_panel_study.get(panel_key, {}).items())
        ):
            style = _fig6_style_for_study(study_id, study_idx)
            t_pts = [float(p["t_c"]) for p in pts]
            f_pts = [float(p["f"]) for p in pts]
            ax.scatter(
                t_pts,
                f_pts,
                marker=style["marker"],
                c=style["color"],
                s=(float(style["ms"]) ** 2) * 1.2,
                edgecolors="k",
                linewidths=0.4,
                zorder=5,
                label=style["label"],
            )
            if study_id not in plotted_studies:
                plotted_studies.append(study_id)

    axes_flat[0].set_ylabel("Melt fraction, F")
    axes_flat[2].set_ylabel("Melt fraction, F")

    handles, leg_labels = axes_flat[0].get_legend_handles_labels()
    seen_labels: set[str] = set()
    unique_handles: list[Any] = []
    unique_labels: list[str] = []
    for h, lab in zip(handles, leg_labels):
        if lab in seen_labels:
            continue
        seen_labels.add(lab)
        unique_handles.append(h)
        unique_labels.append(lab)
    for panel_key in _FIG6_PANEL_ORDER[1:]:
        ax = panel_to_ax[panel_key]
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab in seen_labels:
                continue
            seen_labels.add(lab)
            unique_handles.append(h)
            unique_labels.append(lab)

    ncol = 2 if len(unique_labels) <= 4 else 3
    fig.legend(
        unique_handles,
        unique_labels,
        loc="upper center",
        ncol=ncol,
        fontsize=7,
        bbox_to_anchor=(0.5, 1.02),
    )

    exp_note = ""
    if show_experiments and exp_rows:
        exp_note = f"  ({len(exp_rows)} experimental points)"
    elif show_experiments:
        exp_note = "  (no experimental points loaded)"

    fig.suptitle(
        "Katz et al. (2003) Fig. 6 — anhydrous melting model vs experiments\n"
        f"modal cpx = {100.0 * cpx_mass:g} wt%{exp_note}",
        fontsize=10,
        y=1.08,
    )
    fig.tight_layout()
    return fig


def _mask_tp_for_axes(
    p: np.ndarray,
    t: np.ndarray,
    *,
    t_min_c: float,
    t_max_c: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Hide (P, T) outside the plot window so curves do not draw vertical edge artifacts."""
    p = np.asarray(p, dtype=float)
    t = np.asarray(t, dtype=float)
    ok = (t >= float(t_min_c)) & (t <= float(t_max_c))
    return np.where(ok, p, np.nan), np.where(ok, t, np.nan)


def load_katz2003_fig7_experiments(
    path: Path | None = None,
    *,
    include_unverified: bool = True,
) -> list[dict[str, str]]:
    """Load Katz Fig.7 (P, T, F) experimental database from ``experiments.csv``."""
    csv_path = path or (_KATZ2003_FIG7_DATA / "experiments.csv")
    if not csv_path.is_file():
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not include_unverified:
        rows = [r for r in rows if r.get("verified", "").lower() != "no"]
    return rows


def katz2003_fig7_near_hirschmann_stats(
    experiments: Sequence[dict[str, str]] | None = None,
    *,
    dt_c: float = 10.0,
) -> dict[str, float | int]:
    """
    Experiments within *dt_c* of the Hirschmann (2000) solidus (Katz Fig. 7 caption).

    Returns counts and mean F for annotation / validation.
    """
    rows = list(experiments) if experiments is not None else load_katz2003_fig7_experiments()
    near: list[float] = []
    for row in rows:
        pg = float(row["p_gpa"])
        tc = float(row["t_c"])
        ts_h = hirschmann2000_solidus(pg)
        if abs(tc - ts_h) <= float(dt_c):
            near.append(float(row["f"]))
    n = len(near)
    n_zero = sum(1 for f in near if f <= 0.0)
    mean_f = float(np.mean(near)) if near else float("nan")
    return {
        "n_near": n,
        "n_zero": n_zero,
        "mean_f": mean_f,
        "mean_f_wt_pct": 100.0 * mean_f if n else float("nan"),
        "dt_c": float(dt_c),
    }


def build_katz2003_fig7_figure(
    *,
    p_max_gpa: float = 9.0,
    t_min_c: float = 1000.0,
    t_max_c: float = 2000.0,
    f_cmap_max: float = 0.5,
    show_experiments: bool = True,
    experiments_csv: Path | None = None,
    include_unverified_experiments: bool = True,
    hirschmann_band_c: float = 10.0,
    n_solidus: int = 200,
):
    """
    Katz et al. (2003) Fig. 7 — experimental F in P–T space vs solidus curves.

    Scatter colour encodes melt fraction F (0–1). Overlays:
    Katz (2003) chosen solidus (eq. 4; A1 35 °C below Hirschmann 2000) and
    Hirschmann (2000) solidus (same A2, A3).
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    p = np.linspace(0.0, float(p_max_gpa), int(n_solidus))
    ts_k = np.array([katz2003_solidus(x) for x in p])
    ts_h = np.array([hirschmann2000_solidus(x) for x in p])

    fig, ax = plt.subplots(figsize=(7.4, 5.6), dpi=100)

    exp_rows: list[dict[str, str]] = []
    if show_experiments:
        exp_rows = load_katz2003_fig7_experiments(
            experiments_csv,
            include_unverified=include_unverified_experiments,
        )

    if exp_rows:
        t_pts = np.array([float(r["t_c"]) for r in exp_rows])
        p_pts = np.array([float(r["p_gpa"]) for r in exp_rows])
        f_pts = np.array([float(r["f"]) for r in exp_rows])
        norm = Normalize(vmin=0.0, vmax=float(f_cmap_max))
        cmap = plt.cm.YlOrRd
        sizes = 24.0 + 120.0 * np.clip(f_pts / max(float(f_cmap_max), 1e-9), 0.0, 1.0)
        sc = ax.scatter(
            t_pts,
            p_pts,
            c=f_pts,
            cmap=cmap,
            norm=norm,
            s=sizes,
            edgecolors="k",
            linewidths=0.35,
            zorder=4,
            alpha=0.92,
        )
        cbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            pad=0.02,
            fraction=0.046,
        )
        cbar.set_label("Melt fraction, F")

    py_k, ty_k = _mask_tp_for_axes(p, ts_k, t_min_c=t_min_c, t_max_c=t_max_c)
    py_h, ty_h = _mask_tp_for_axes(p, ts_h, t_min_c=t_min_c, t_max_c=t_max_c)
    ax.plot(
        ty_h,
        py_h,
        color="#7f8c8d",
        lw=1.8,
        ls="--",
        label="Hirschmann (2000) solidus",
        zorder=2,
    )
    ax.plot(
        ty_k,
        py_k,
        "k-",
        lw=2.4,
        label="Katz (2003) solidus (A1 - 35 C)",
        zorder=3,
    )

    if hirschmann_band_c > 0.0 and exp_rows:
        ts_hi = np.array([hirschmann2000_solidus(x) + float(hirschmann_band_c) for x in p])
        ts_lo = np.array([hirschmann2000_solidus(x) - float(hirschmann_band_c) for x in p])
        ax.fill_betweenx(
            p,
            ts_lo,
            ts_hi,
            color="#bdc3c7",
            alpha=0.25,
            zorder=1,
            label=f"±{hirschmann_band_c:g}°C of Hirschmann solidus",
        )

    stats = katz2003_fig7_near_hirschmann_stats(exp_rows, dt_c=hirschmann_band_c)
    if stats["n_near"]:
        note = (
            f"{stats['n_near']:.0f} expts within ±{stats['dt_c']:.0f}°C of "
            f"Hirschmann solidus\n"
            f"mean F = {stats['mean_f_wt_pct']:.0f} wt%; "
            f"{stats['n_zero']:.0f} with F = 0"
        )
        ax.text(
            0.02,
            0.02,
            note,
            transform=ax.transAxes,
            fontsize=7,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="#ccc"),
        )

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Pressure (GPa)")
    ax.set_title(
        "Katz et al. (2003) Fig. 7 — experimental melt fraction in P–T space\n"
        "Anhydrous peridotite; solidus comparison (Table 2 eq. 4 vs Hirschmann 2000)"
    )
    ax.set_xlim(float(t_min_c), float(t_max_c))
    ax.set_ylim(float(p_max_gpa), 0.0)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def build_katz2003_fig3_figure(
    *,
    p_max_gpa: float = 8.0,
    t_min_c: float = 850.0,
    t_max_c: float = 2000.0,
    bulk_h2o_wt_pct: tuple[float, ...] = (0.05, 0.1, 0.3),
    n: int = 200,
):
    """
    Katz et al. (2003) Fig. 3 — solidus for different bulk H2O (T–P).

    Bulk H2O in **wt%**: 0, 0.05, 0.1, 0.3 plus water-saturated solidus (eq. 16–18).
    Default ``t_min_c=850`` (not 1100): hydrous solidus extends to ~916°C at ~1.4 GPa.
    """
    import matplotlib.pyplot as plt

    p = np.linspace(0.0, float(p_max_gpa), int(n))
    ts = np.array([katz2003_solidus(x) for x in p])
    ts_sat = np.array([katz2003_hydrous_solidus_t(x, saturated=True) for x in p])

    fig, ax = plt.subplots(figsize=(7.0, 5.4), dpi=100)
    py, ty = _mask_tp_for_axes(p, ts, t_min_c=t_min_c, t_max_c=t_max_c)
    ax.plot(ty, py, "k-", lw=2.2, label="0 wt%", zorder=2)

    cmap = plt.cm.plasma(np.linspace(0.25, 0.85, max(len(bulk_h2o_wt_pct), 1)))
    for bulk, col in zip(bulk_h2o_wt_pct, cmap):
        th = np.array([katz2003_hydrous_solidus_t(x, bulk_h2o_wt_pct=bulk) for x in p])
        py, ty = _mask_tp_for_axes(p, th, t_min_c=t_min_c, t_max_c=t_max_c)
        ax.plot(ty, py, color=col, lw=1.6, label=f"{bulk:g} wt%", zorder=3)

    ps, tsat = _mask_tp_for_axes(p, ts_sat, t_min_c=t_min_c, t_max_c=t_max_c)
    ax.plot(
        tsat,
        ps,
        color="#1a5276",
        lw=2.4,
        ls="--",
        label="Water saturated",
        zorder=4,
    )

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Pressure (GPa)")
    ax.set_title(
        "Katz et al. (2003) Fig. 3 — solidus for different bulk H$_2$O\n"
        "Solidus depression vs dissolved H$_2$O; bounded by melt saturation (eq. 17)"
    )
    ax.set_xlim(float(t_min_c), float(t_max_c))
    ax.set_ylim(float(p_max_gpa), 0.0)
    ax.legend(loc="upper right", fontsize=7, title="Bulk H$_2$O")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def build_katz2003_fig4_figure(
    *,
    p_gpa: float = 1.0,
    t_min_c: float = 900.0,
    t_max_c: float = 1400.0,
    f_max: float = 0.4,
    bulk_h2o_wt_pct: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1, 0.3),
    cpx_mass: float = _DEFAULT_CPX_MASS,
    n: int = 400,
):
    """
    Katz et al. (2003) Fig. 4 — isobaric hydrous F(T) at 1 GPa (default).

    Bulk H2O in **wt%**. At 1 GPa, 0.3 wt% is fluid-saturated at the solidus (eq. 17–18).
    """
    import matplotlib.pyplot as plt

    t = np.linspace(float(t_min_c), float(t_max_c), int(n))
    fig, ax = plt.subplots(figsize=(6.8, 5.0), dpi=100)

    hydrous = [b for b in bulk_h2o_wt_pct if b > 0.0]
    cmap = plt.cm.plasma(np.linspace(0.25, 0.85, max(len(hydrous), 1)))
    h_idx = 0
    for bulk in bulk_h2o_wt_pct:
        if bulk <= 0.0:
            col, lw, label = "k", 2.2, "0 wt%"
        else:
            col, lw, label = cmap[h_idx], 1.8, f"{bulk:g} wt%"
            h_idx += 1
        f = katz2003_melt_fraction_hydrous_profile(
            p_gpa,
            t,
            bulk_h2o_wt_pct=float(bulk),
            cpx_mass=cpx_mass,
        )
        ax.plot(t, f, color=col, lw=lw, label=label)

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Melt fraction, F")
    ax.set_title(
        f"Katz et al. (2003) Fig. 4 — isobaric melting at P = {p_gpa:g} GPa\n"
        "Different bulk H$_2$O; 0.3 wt% saturated at the solidus"
    )
    ax.set_xlim(float(t_min_c), float(t_max_c))
    ax.set_ylim(0.0, float(f_max))
    ax.legend(loc="upper left", fontsize=8, title="Bulk H$_2$O")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


# Fig. 5 caption (errata): modal cpx = 17 wt% in unmelted solid
_FIG5_CPX_MASS = 0.17


def katz2003_fig5_melt_vs_water(
    *,
    p_gpa: float,
    t_c: float,
    bulk_h2o_wt_pct: np.ndarray | list[float],
    cpx_mass: float = _FIG5_CPX_MASS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    F vs bulk and dissolved melt H2O at fixed (P, T) for Katz (2003) Fig. 5.

    Returns ``bulk_wt``, ``f``, ``x_melt_wt`` (dissolved H2O in melt at equilibrium F).
    """
    bulk = np.asarray(bulk_h2o_wt_pct, dtype=float)
    f = np.array(
        [
            katz2003_melt_fraction_hydrous(
                p_gpa,
                t_c,
                bulk_h2o_wt_pct=float(b),
                cpx_mass=cpx_mass,
            )
            if b > 0.0
            else katz2003_melt_fraction_dry(p_gpa, t_c, cpx_mass=cpx_mass)
            for b in bulk
        ],
        dtype=float,
    )
    x_melt = np.array(
        [
            katz2003_x_melt_h2o_at_f(bulk_h2o_wt_pct=float(b), f=fi, p_gpa=p_gpa)
            if b > 0.0 and fi > 0.0
            else 0.0
            for b, fi in zip(bulk, f)
        ],
        dtype=float,
    )
    return bulk, f, x_melt


def build_katz2003_fig5_figure(
    *,
    p_gpa: float = 1.5,
    temperatures_c: tuple[float, ...] = (1200, 1250, 1300, 1350),
    bulk_h2o_max_wt_pct: float = 0.5,
    n_bulk: int = 120,
    cpx_mass: float = _FIG5_CPX_MASS,
    bulk_x_max: float = 0.25,
    f_max_a: float = 0.25,
    melt_x_max: float = 5.0,
    f_max_b: float = 0.2,
):
    """
    Katz et al. (2003) Fig. 5 — F vs bulk H2O (a) and F vs dissolved melt H2O (b).

    Fixed P = 1.5 GPa; isotherms labelled on curves (1200–1350°C default).
    Modal cpx = 17 wt% (paper errata). Compare panel (a) to Gaetani & Grove (1998) Fig. 13a.
    """
    import matplotlib.pyplot as plt

    bulk = np.linspace(0.0, float(bulk_h2o_max_wt_pct), int(n_bulk))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=100)
    ax_a, ax_b = axes
    cmap = plt.cm.viridis(np.linspace(0.15, 0.88, max(len(temperatures_c), 1)))

    for t_c, col in zip(temperatures_c, cmap):
        bulk_wt, f, x_melt = katz2003_fig5_melt_vs_water(
            p_gpa=p_gpa,
            t_c=float(t_c),
            bulk_h2o_wt_pct=bulk,
            cpx_mass=cpx_mass,
        )
        label = f"{t_c:g}°C"
        ax_a.plot(bulk_wt, f, color=col, lw=1.8, label=label)
        wet = bulk_wt > 0.0
        ax_b.plot(x_melt[wet], f[wet], color=col, lw=1.8, label=label)

    ax_a.set_xlabel(r"Bulk H$_2$O (wt%)")
    ax_a.set_ylabel("Melt fraction, F")
    ax_a.set_title("(a) F vs bulk H$_2$O")
    ax_a.set_xlim(0.0, float(bulk_x_max))
    ax_a.set_ylim(0.0, float(f_max_a))
    ax_a.legend(loc="upper left", fontsize=7, title="T")
    ax_a.grid(True, alpha=0.25)

    ax_b.set_xlabel(r"Dissolved H$_2$O in melt (wt%)")
    ax_b.set_ylabel("Melt fraction, F")
    ax_b.set_title(r"(b) F vs melt H$_2$O  ($D_{H_2O}$ = 0.01)")
    ax_b.set_xlim(0.0, float(melt_x_max))
    ax_b.set_ylim(0.0, float(f_max_b))
    ax_b.grid(True, alpha=0.25)

    fig.suptitle(
        f"Katz et al. (2003) Fig. 5 — melting vs H$_2$O at P = {p_gpa:g} GPa\n"
        f"modal cpx = {100.0 * cpx_mass:g} wt%; compare (a) to Gaetani & Grove (1998) Fig. 13a",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    return fig


def katz2003_table2_hydrous_constants() -> dict[str, float]:
    """Return Katz (2003) Table 2 hydrous melting constants (eq. 16–17)."""
    return {
        "K": _PK,
        "gamma": _GAMMA,
        "chi1": _CHI1,
        "chi2": _CHI2,
        "plam": _PLAM,
        "D_H2O": _DH2O,
    }


def katz2003_table2_isentropic_constants() -> dict[str, float]:
    """Return Katz (2003) Table 2 thermodynamic constants (eq. 11–15)."""
    return {
        "c_P": _CP_J_KG,
        "alpha_s": _ALPHA_S_K,
        "alpha_f": _ALPHA_F_K,
        "rho_s": _RHO_S,
        "rho_f": _RHO_F,
        "DeltaS": _DELTA_S,
    }


def load_katz2003_fig8_delta_t_calibration(
    path: Path | None = None,
) -> list[dict[str, str]]:
    csv_path = path or (_KATZ2003_FIG8_DATA / "delta_t_calibration.csv")
    if not csv_path.is_file():
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_katz2003_fig8_x_sat_calibration(
    path: Path | None = None,
) -> list[dict[str, str]]:
    csv_path = path or (_KATZ2003_FIG8_DATA / "x_sat_calibration.csv")
    if not csv_path.is_file():
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


_FIG8_DELTA_T_STYLE: dict[str, dict[str, Any]] = {
    "HK95": {"marker": "o", "color": "#2980b9", "label": "Hirose & Kawamoto (1995)"},
    "KH97": {"marker": "s", "color": "#27ae60", "label": "Kawamoto & Holloway (1997)"},
    "GG98": {"marker": "^", "color": "#c0392b", "label": "Gaetani & Grove (1998)"},
    "G01": {"marker": "D", "color": "#8e44ad", "label": "Grove (2001)"},
}
_FIG8_XSAT_STYLE: dict[str, dict[str, Any]] = {
    "D95": {"marker": "o", "color": "#d35400", "label": "Dixon et al. (1995)"},
    "MW00": {"marker": "s", "color": "#16a085", "label": "Mysen & Wheeler (2000)"},
}


def build_katz2003_fig8_figure(
    *,
    x_h2o_max: float = 60.0,
    delta_t_max: float = 900.0,
    p_max_gpa: float = 8.0,
    x_sat_max: float = 50.0,
    pk: float = _PK,
    gamma: float = _GAMMA,
    chi1: float = _CHI1,
    chi2: float = _CHI2,
    plam: float = _PLAM,
    show_calibration: bool = True,
    delta_t_csv: Path | None = None,
    x_sat_csv: Path | None = None,
    n_curve: int = 200,
):
    """
    Katz et al. (2003) Fig. 8 — hydrous parameter calibration.

    (a) ΔT vs dissolved melt H₂O (wt%), eq. (16); experiments from HK95, KH97, GG98, G01.
    (b) Melt saturation H₂O vs P (GPa), eq. (17); Dixon et al. (1995) and Mysen & Wheeler (2000).
    """
    import matplotlib.pyplot as plt

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.8, 4.8), dpi=100)

    x = np.linspace(0.0, float(x_h2o_max), int(n_curve))
    dt_model = pk * np.maximum(x, 0.0) ** gamma
    ax_a.plot(
        x,
        dt_model,
        "k-",
        lw=2.2,
        label=rf"Katz (2003) eq. (16): $\Delta T = {pk:g}\,X^{{{gamma:g}}}$",
        zorder=2,
    )

    if show_calibration:
        rows_a = load_katz2003_fig8_delta_t_calibration(delta_t_csv)
        for sid in sorted({r.get("study_id", "") for r in rows_a}):
            pts = [r for r in rows_a if r.get("study_id") == sid]
            if not pts:
                continue
            sty = _FIG8_DELTA_T_STYLE.get(
                sid,
                {"marker": "o", "color": "#555555", "label": sid},
            )
            ax_a.scatter(
                [float(p["x_h2o_wt_pct"]) for p in pts],
                [float(p["delta_t_c"]) for p in pts],
                marker=sty["marker"],
                c=sty["color"],
                s=42,
                edgecolors="k",
                linewidths=0.35,
                label=sty["label"],
                zorder=4,
            )

    ax_a.set_xlabel(r"Dissolved H$_2$O in melt, $X$ (wt%)")
    ax_a.set_ylabel(r"Solidus depression $\Delta T$ (°C)")
    ax_a.set_title("(a) Calibration of eq. (16)")
    ax_a.set_xlim(0.0, float(x_h2o_max))
    ax_a.set_ylim(0.0, float(delta_t_max))
    ax_a.legend(loc="upper left", fontsize=6.5)
    ax_a.grid(True, alpha=0.25)

    p = np.linspace(0.0, float(p_max_gpa), int(n_curve))
    xs_model = chi1 * p**plam + chi2 * p
    ax_b.plot(
        p,
        xs_model,
        "k-",
        lw=2.2,
        label=rf"Katz (2003) eq. (17): $X_{{sat}} = {chi1:g}P^{{{plam:g}}}+{chi2:g}P$",
        zorder=2,
    )

    if show_calibration:
        rows_b = load_katz2003_fig8_x_sat_calibration(x_sat_csv)
        for sid in sorted({r.get("source_id", "") for r in rows_b}):
            pts = [r for r in rows_b if r.get("source_id") == sid]
            if not pts:
                continue
            sty = _FIG8_XSAT_STYLE.get(
                sid,
                {"marker": "o", "color": "#555555", "label": sid},
            )
            ax_b.scatter(
                [float(p["p_gpa"]) for p in pts],
                [float(p["x_sat_wt_pct"]) for p in pts],
                marker=sty["marker"],
                c=sty["color"],
                s=42,
                edgecolors="k",
                linewidths=0.35,
                label=sty["label"],
                zorder=4,
            )

    ax_b.set_xlabel("Pressure (GPa)")
    ax_b.set_ylabel(r"Melt H$_2$O at saturation, $X_{sat}$ (wt%)")
    ax_b.set_title("(b) Calibration of eq. (17)")
    ax_b.set_xlim(0.0, float(p_max_gpa))
    ax_b.set_ylim(0.0, float(x_sat_max))
    ax_b.legend(loc="upper left", fontsize=6.5)
    ax_b.grid(True, alpha=0.25)

    fig.suptitle(
        "Katz et al. (2003) Fig. 8 — hydrous melting parameter calibration (Table 2)",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    return fig


def build_katz2003_fig9_figure(
    *,
    pressures_gpa: tuple[float, ...] = (1.0, 3.0),
    t_min_c: float = 1200.0,
    t_max_c: float = 1800.0,
    f_max: float = 0.5,
    cpx_mass: float | None = None,
    n: int = 400,
    model_keys: tuple[str, ...] | None = None,
):
    """
    Katz et al. (2003) Fig. 9 — dry melting parameterization comparison.

    Default models: McKenzie & Bickle (1988) + pMELTS (paper Fig. 9).
    Pass ``model_keys=FIG9_EXTENDED_MODEL_ORDER`` to overlay Katz (2003).
    """
    import matplotlib.pyplot as plt

    from petrology.melting.katz2003_fig9_models import (
        FIG9_MODEL_ORDER,
        build_fig9_dry_melt_models,
        fig9_curve_style,
        fig9_model_label,
    )

    t = np.linspace(float(t_min_c), float(t_max_c), int(n))
    models = build_fig9_dry_melt_models(cpx_mass=cpx_mass)
    keys = model_keys or FIG9_MODEL_ORDER
    unknown = [k for k in keys if k not in models]
    if unknown:
        raise KeyError(f"unknown Fig. 9 model keys: {unknown}")

    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=100)

    for key in keys:
        model = models[key]
        alpha = 1.0 if model.verified != "partial" else 0.88
        for pg in pressures_gpa:
            sty = fig9_curve_style(key, float(pg))
            f = model.melt_fraction_profile(float(pg), t)
            ax.plot(
                t,
                f,
                color=sty["color"],
                ls=sty["ls"],
                lw=sty["lw"],
                alpha=alpha,
                label=fig9_model_label(key, float(pg)),
                solid_capstyle="round",
            )

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Melt fraction, F")
    ax.set_title(
        "Katz et al. (2003) Fig. 9 — dry melting parameterizations\n"
        "Paper default: MB88 + pMELTS; blue = 1 GPa, green = 3 GPa"
    )
    ax.set_xlim(float(t_min_c), float(t_max_c))
    ax.set_ylim(0.0, float(f_max))
    ax.legend(loc="upper left", fontsize=6.5, ncol=1, framealpha=0.92)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig
