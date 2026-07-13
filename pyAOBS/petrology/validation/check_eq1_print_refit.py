"""
Audit eq.(1) coefficients re-fit to digitized Fig.3 V(P,F) contours.

Compares coefficient sets on the same GetData points:
  1. paper table (korenaga2002_eq1.json)
  2. paper with only b1 sign flipped (b1 = -|b1_paper|)
  3. user L-BFGS report (test1.py)
  4. exact linear least-squares with fixed windows (canonical print-fit)

Writes comparison contour overlay figures.

Usage::

    py -3.11 petrology/validation/check_eq1_print_refit.py
    py -3.11 petrology/validation/check_eq1_print_refit.py --plot
    py -3.11 petrology/validation/check_eq1_print_refit.py --update-json

Dedicated paper vs b1-flip figure (also written by --plot)::

    petrology/figures/eq1_paper_vs_b1neg.png
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.validation.check_eq1_digitized_contours import parse_contour_file
from petrology.vp_regression import (
    eq1_with_printed_coefficients,
    in_eq1_applicable_range,
    load_eq1,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
DEFAULT_CONTOURS = DATA_DIR / "ScreenShot_2026-07-09_105537_962.txt"
PRINT_FIT_JSON = DATA_DIR / "korenaga2002_eq1_print_fit.json"
DEFAULT_FIG = FIG_DIR / "eq1_print_refit_compare.png"

# Window fixed to paper (§2.2)
ALPHA = 0.6
BETA = 8.4
P_T = 1.0
F_T = 0.05

# User-reported L-BFGS (test1.py)
USER_VEC = np.array(
    [
        7.5215,
        -1.7386,
        -0.5441,
        7.7352,
        -0.1109,
        8.836,
        -146.1129,
        -0.3518,
        0.0333,
        0.5207,
        0.0017,
        -0.0393,
        0.0247,
    ],
    dtype=float,
)


@dataclass(frozen=True)
class CoefSet:
    name: str
    short: str
    vec: np.ndarray  # [a0, b0..b5, c0..c5]


def _coef_vec(c: dict) -> np.ndarray:
    return np.array(
        [
            c["a0"],
            c["b0"],
            c["b1"],
            c["b2"],
            c["b3"],
            c["b4"],
            c["b5"],
            c["c0"],
            c["c1"],
            c["c2"],
            c["c3"],
            c["c4"],
            c["c5"],
        ],
        dtype=float,
    )


def paper_vec() -> np.ndarray:
    """Printed (erroneous) table: b1 = +0.55 — for errata comparison only."""
    return _coef_vec(eq1_with_printed_coefficients()["equation_1"]["coefficients"])


def paper_b1neg_vec() -> np.ndarray:
    """Production / corrected: printed table with only b1 -> -|b1|."""
    return _coef_vec(load_eq1()["equation_1"]["coefficients"])


# Shared plot style for all coefficient sets
SET_STYLE: dict[str, dict[str, str]] = {
    "paper": {"color": "#c0392b", "ls": "-", "label": "printed table (b1=+0.55)"},
    "b1neg": {"color": "#e67e22", "ls": ":", "label": "production (b1=-0.55)"},
    "user": {"color": "#27ae60", "ls": "--", "label": "user L-BFGS"},
    "lstsq": {"color": "#2980b9", "ls": "-.", "label": "LSTSQ print-fit"},
}


def window_weights(p: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tp = ALPHA * (p - P_T)
    tf = BETA * (f - F_T)
    wl = 0.25 * (1.0 - np.tanh(tp)) * (1.0 - np.tanh(tf))
    wh = 0.25 * (1.0 + np.tanh(tp)) * (1.0 + np.tanh(tf))
    return wl, wh


def predict_vec(x: np.ndarray, p: np.ndarray, f: np.ndarray) -> np.ndarray:
    a0 = x[0]
    b = x[1:7]
    c = x[7:13]
    wl, wh = window_weights(p, f)
    poly_b = b[0] + b[1] * p + b[2] * f + b[3] * p**2 + b[4] * p * f + b[5] * f**2
    poly_c = c[0] + c[1] * p + c[2] * f + c[3] * p**2 + c[4] * p * f + c[5] * f**2
    return a0 + wl * poly_b + wh * poly_c


def design_matrix(p: np.ndarray, f: np.ndarray) -> np.ndarray:
    wl, wh = window_weights(p, f)
    return np.column_stack(
        [
            np.ones_like(p),
            wl,
            wl * p,
            wl * f,
            wl * p**2,
            wl * p * f,
            wl * f**2,
            wh,
            wh * p,
            wh * f,
            wh * p**2,
            wh * p * f,
            wh * f**2,
        ]
    )


def fit_lstsq(p: np.ndarray, f: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, float]:
    xmat = design_matrix(p, f)
    coef, residuals, _rank, _s = np.linalg.lstsq(xmat, v, rcond=None)
    sse = float(residuals[0]) if len(residuals) else float(np.sum((xmat @ coef - v) ** 2))
    return coef, sse


def contours_to_arrays(
    contours: list[tuple[float, list[tuple[float, float]]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_list: list[float] = []
    f_list: list[float] = []
    v_list: list[float] = []
    for v_lab, pts in contours:
        for p, f in pts:
            p_list.append(p)
            f_list.append(f)
            v_list.append(v_lab)
    return np.asarray(p_list), np.asarray(f_list), np.asarray(v_list)


def residual_metrics(
    x: np.ndarray, p: np.ndarray, f: np.ndarray, v: np.ndarray
) -> dict[str, float]:
    d = predict_vec(x, p, f) - v
    inside = np.array([in_eq1_applicable_range(float(pi), float(fi)) for pi, fi in zip(p, f)])
    di = d[inside]
    return {
        "n": float(d.size),
        "sse": float(np.sum(d**2)),
        "rms": float(np.sqrt(np.mean(d**2))),
        "mean": float(d.mean()),
        "abs_med": float(np.median(np.abs(d))),
        "n_in": float(inside.sum()),
        "rms_in": float(np.sqrt(np.mean(di**2))) if inside.any() else float("nan"),
        "mean_in": float(di.mean()) if inside.any() else float("nan"),
    }


def print_metrics(name: str, m: dict[str, float]) -> None:
    print(f"{name}:")
    print(
        f"  all  n={int(m['n'])}  SSE={m['sse']:.6f}  rms={m['rms']:.5f}  "
        f"mean={m['mean']:+.5f}  |med|={m['abs_med']:.5f}"
    )
    print(
        f"  box  n={int(m['n_in'])}  rms={m['rms_in']:.5f}  mean={m['mean_in']:+.5f}"
    )


def print_coef_table(sets: list[CoefSet]) -> None:
    labels = ["a0"] + [f"b{i}" for i in range(6)] + [f"c{i}" for i in range(6)]
    hdr = f"{'coef':>4}" + "".join(f"{s.short:>14}" for s in sets)
    print(hdr)
    for i, lab in enumerate(labels):
        row = f"{lab:>4}" + "".join(f"{s.vec[i]:14.4f}" for s in sets)
        print(row)
    # highlight b1
    print()
    print("Note: printed b1=+0.55 (errata); production/user/LSTSQ use b1~-0.55.")


def _legend_handles(sets: list[CoefSet]):
    from matplotlib.lines import Line2D

    handles = []
    for s in sets:
        st = SET_STYLE[s.short]
        handles.append(
            Line2D([0], [0], color=st["color"], lw=1.4, ls=st["ls"], label=st["label"])
        )
    handles.append(Line2D([0], [0], color="0.15", lw=1.5, label="digitized"))
    handles.append(Line2D([0], [0], color="#8e44ad", lw=1.3, ls=":", label="applicable box"))
    return handles


def plot_three_way(
    contours: list[tuple[float, list[tuple[float, float]]]],
    sets: list[CoefSet],
    *,
    out: Path,
    show: bool = False,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    levels = [c[0] for c in contours]
    all_p = [p for _, pts in contours for p, _ in pts]
    all_f = [f for _, pts in contours for _, f in pts]
    p_max = max(7.0, max(all_p) * 1.02)
    f_max = max(0.6, max(all_f) * 1.05)

    p_grid = np.linspace(0.05, p_max, 200)
    f_grid = np.linspace(0.005, f_max, 160)
    P, F = np.meshgrid(p_grid, f_grid)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.7), constrained_layout=True)
    ax, axr = axes

    # (a) contour overlay
    for s in sets:
        st = SET_STYLE[s.short]
        V = predict_vec(s.vec, P, F)
        cs = ax.contour(
            P,
            F,
            V,
            levels=levels,
            colors=st["color"],
            linewidths=1.25,
            linestyles=st["ls"],
            zorder=3,
        )
        # label only paper to avoid clutter
        if s.short == "paper":
            ax.clabel(cs, inline=True, fontsize=6.5, fmt="%g")

    for v_lab, pts in contours:
        if not pts:
            continue
        pp = np.array([p for p, _ in pts])
        ff = np.array([f for _, f in pts])
        order = np.argsort(pp)
        ax.plot(pp[order], ff[order], "-", color="0.15", lw=1.5, alpha=0.9, zorder=4)
        ax.scatter(pp, ff, s=14, c="0.15", zorder=5)

    ax.add_patch(
        Rectangle(
            (1.0, 0.02),
            2.0,
            0.18,
            fill=False,
            ec="#8e44ad",
            lw=1.3,
            ls=":",
            zorder=6,
        )
    )
    ax.set_xlim(0.0, p_max)
    ax.set_ylim(f_max, 0.0)
    ax.set_xlabel(r"Mean $P$ (GPa)")
    ax.set_ylabel(r"Mean $F$")
    ax.set_title("(a) Contours vs digitized (incl. paper b1 flip)")
    ax.grid(True, ls=":", lw=0.4, alpha=0.4)
    ax.legend(handles=_legend_handles(sets), fontsize=7.0, loc="lower left", framealpha=0.92)

    # (b) residual bar chart
    p_pts, f_pts, v_pts = contours_to_arrays(contours)
    xpos = np.arange(len(sets))
    width = 0.38
    rms_all = []
    rms_in = []
    for s in sets:
        m = residual_metrics(s.vec, p_pts, f_pts, v_pts)
        rms_all.append(m["rms"])
        rms_in.append(m["rms_in"])

    bars1 = axr.bar(xpos - width / 2, rms_all, width, color="#95a5a6", label="rms all")
    bars2 = axr.bar(xpos + width / 2, rms_in, width, color="#8e44ad", label="rms inside box")
    axr.set_xticks(xpos)
    axr.set_xticklabels([s.name for s in sets], fontsize=7.2, rotation=15, ha="right")
    axr.set_ylabel("rms residual (km/s)")
    axr.set_title("(b) Fit quality on digitized points")
    axr.grid(True, axis="y", ls=":", lw=0.4, alpha=0.5)
    axr.legend(fontsize=7.5, loc="upper right")
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            axr.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"Saved {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out


def plot_paper_vs_b1neg(
    contours: list[tuple[float, list[tuple[float, float]]]],
    paper: np.ndarray,
    b1neg: np.ndarray,
    *,
    out: Path,
    show: bool = False,
) -> Path:
    """Side-by-side P-F contours: paper vs paper with only b1 flipped."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    levels = [c[0] for c in contours]
    all_p = [p for _, pts in contours for p, _ in pts]
    all_f = [f for _, pts in contours for _, f in pts]
    p_max = max(7.0, max(all_p) * 1.02)
    f_max = max(0.6, max(all_f) * 1.05)

    p_grid = np.linspace(0.05, p_max, 240)
    f_grid = np.linspace(0.005, f_max, 200)
    P, F = np.meshgrid(p_grid, f_grid)
    V_paper = predict_vec(paper, P, F)
    V_b1neg = predict_vec(b1neg, P, F)

    p_pts, f_pts, v_pts = contours_to_arrays(contours)
    d_paper = predict_vec(paper, p_pts, f_pts) - v_pts
    d_b1neg = predict_vec(b1neg, p_pts, f_pts) - v_pts
    m_paper = residual_metrics(paper, p_pts, f_pts, v_pts)
    m_b1neg = residual_metrics(b1neg, p_pts, f_pts, v_pts)

    fig = plt.figure(figsize=(12.8, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.85])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    axr = fig.add_subplot(gs[0, 2])

    def _draw_panel(ax, V, *, color: str, title: str, rms: float, mean: float) -> None:
        cs = ax.contour(
            P,
            F,
            V,
            levels=levels,
            colors=color,
            linewidths=1.8,
            linestyles="-",
            zorder=3,
        )
        ax.clabel(cs, inline=True, fontsize=7.5, fmt="%g", colors=color)
        for _v_lab, pts in contours:
            if not pts:
                continue
            pp = np.array([p for p, _ in pts])
            ff = np.array([f for _, f in pts])
            order = np.argsort(pp)
            ax.plot(pp[order], ff[order], "-", color="0.12", lw=2.0, zorder=4)
            ax.scatter(pp, ff, s=22, c="0.12", zorder=5, edgecolors="white", linewidths=0.3)
        ax.add_patch(
            Rectangle(
                (1.0, 0.02),
                2.0,
                0.18,
                fill=False,
                ec="#8e44ad",
                lw=1.4,
                ls="--",
                zorder=6,
            )
        )
        ax.set_xlim(0.0, p_max)
        ax.set_ylim(f_max, 0.0)
        ax.set_xlabel(r"Mean $P$ (GPa)")
        ax.set_ylabel(r"Mean $F$")
        ax.set_title(f"{title}\nrms={rms:.4f} km/s, mean={mean:+.4f}", fontsize=10)
        ax.grid(True, ls=":", lw=0.45, alpha=0.45)
        ax.legend(
            handles=[
                Line2D([0], [0], color=color, lw=2.0, label="eq.(1)"),
                Line2D([0], [0], color="0.12", lw=2.0, label="digitized"),
            ],
            fontsize=8,
            loc="lower left",
            framealpha=0.92,
        )

    _draw_panel(
        ax0,
        V_paper,
        color="#c0392b",
        title=r"(a) paper  ($b_1=+0.55$)",
        rms=m_paper["rms"],
        mean=m_paper["mean"],
    )
    _draw_panel(
        ax1,
        V_b1neg,
        color="#1f77b4",
        title=r"(b) paper $b_1=-|b_1|$  ($b_1=-0.55$)",
        rms=m_b1neg["rms"],
        mean=m_b1neg["mean"],
    )

    # (c) residual at each digitized point, sorted by label V
    order = np.argsort(v_pts)
    x = np.arange(len(v_pts))
    axr.axhline(0.0, color="0.4", lw=0.8)
    axr.plot(
        x,
        d_paper[order],
        "o-",
        color="#c0392b",
        ms=4.5,
        lw=1.2,
        label=rf"paper (rms={m_paper['rms']:.3f})",
    )
    axr.plot(
        x,
        d_b1neg[order],
        "s-",
        color="#1f77b4",
        ms=4.0,
        lw=1.2,
        label=rf"$b_1$ flip (rms={m_b1neg['rms']:.3f})",
    )
    axr.set_xlabel("digitized points (sorted by V label)")
    axr.set_ylabel(r"$V_{\mathrm{eq1}} - V_{\mathrm{label}}$ (km/s)")
    axr.set_title("(c) Point residuals")
    axr.grid(True, ls=":", lw=0.4, alpha=0.5)
    axr.legend(fontsize=7.5, loc="best", framealpha=0.92)

    fig.suptitle(
        r"Eq.(1) ablation: only flip paper $b_1$ sign  (other coefficients unchanged)",
        fontsize=11,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"Saved {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out


def plot_pf_overlay(
    contours: list[tuple[float, list[tuple[float, float]]]],
    sets: list[CoefSet],
    *,
    out: Path,
    show: bool = False,
) -> Path:
    """Dedicated P-F overlay without bar chart."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    levels = [c[0] for c in contours]
    all_p = [p for _, pts in contours for p, _ in pts]
    all_f = [f for _, pts in contours for _, f in pts]
    p_max = max(7.0, max(all_p) * 1.02)
    f_max = max(0.6, max(all_f) * 1.05)
    p_grid = np.linspace(0.05, p_max, 220)
    f_grid = np.linspace(0.005, f_max, 180)
    P, F = np.meshgrid(p_grid, f_grid)

    fig, ax = plt.subplots(figsize=(8.6, 6.4), constrained_layout=True)
    for s in sets:
        st = SET_STYLE[s.short]
        V = predict_vec(s.vec, P, F)
        cs = ax.contour(
            P,
            F,
            V,
            levels=levels,
            colors=st["color"],
            linewidths=1.3,
            linestyles=st["ls"],
        )
        if s.short == "lstsq":
            ax.clabel(cs, inline=True, fontsize=7, fmt="%g")

    for _v_lab, pts in contours:
        if not pts:
            continue
        pp = np.array([p for p, _ in pts])
        ff = np.array([f for _, f in pts])
        order = np.argsort(pp)
        ax.plot(pp[order], ff[order], "-", color="0.1", lw=1.7, zorder=4)
        ax.scatter(pp, ff, s=16, c="0.1", zorder=5)

    ax.add_patch(
        Rectangle((1.0, 0.02), 2.0, 0.18, fill=False, ec="#8e44ad", lw=1.4, ls=":", zorder=6)
    )
    ax.set_xlim(0.0, p_max)
    ax.set_ylim(f_max, 0.0)
    ax.set_xlabel(r"Mean $P$ (GPa)")
    ax.set_ylabel(r"Mean $F$")
    ax.set_title("Eq.(1): paper / b1neg / user / LSTSQ vs digitized Fig.3")
    ax.grid(True, ls=":", lw=0.4, alpha=0.4)
    ax.legend(handles=_legend_handles(sets), fontsize=7.5, loc="lower left", framealpha=0.92)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"Saved {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out


def update_json(lstsq: np.ndarray, b1neg: np.ndarray) -> None:
    payload = {
        "description": (
            "Eq.(1) coefficients re-fit to digitized KKHS02 Fig.3 V(P,F) contours (GetData). "
            "Window parameters fixed to paper values. NOT a replacement for the published table "
            "- use for print-figure alignment only."
        ),
        "source_digitized": "petrology/data/ScreenShot_2026-07-09_105537_962.txt",
        "window_functions": {
            "alpha": ALPHA,
            "beta": BETA,
            "P_t_GPa": P_T,
            "F_t": F_T,
        },
        "printed_table": {
            "comment": "KKHS02 printed table (erroneous b1=+0.55); for comparison only",
            "a0": float(paper_vec()[0]),
            "b": [float(x) for x in paper_vec()[1:7]],
            "c": [float(x) for x in paper_vec()[7:13]],
        },
        "production_corrected": {
            "comment": "Production coeffs in data/korenaga2002_eq1.json (b1=-0.55)",
            "a0": float(b1neg[0]),
            "b": [float(x) for x in b1neg[1:7]],
            "c": [float(x) for x in b1neg[7:13]],
        },
        "user_lbfgs": {
            "comment": "article/复现korenaga/test1.py L-BFGS-B; confirms printed b1 sign error",
            "a0": float(USER_VEC[0]),
            "b": [float(x) for x in USER_VEC[1:7]],
            "c": [float(x) for x in USER_VEC[7:13]],
        },
        "lstsq": {
            "comment": "Exact linear least-squares with fixed windows (canonical print-fit)",
            "a0": float(lstsq[0]),
            "b": [float(x) for x in lstsq[1:7]],
            "c": [float(x) for x in lstsq[7:13]],
        },
    }
    PRINT_FIT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {PRINT_FIT_JSON}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit eq.(1) print-figure coefficient re-fit")
    parser.add_argument("-i", "--input", type=Path, default=DEFAULT_CONTOURS)
    parser.add_argument("--plot", action="store_true", help="Write comparison figures")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_FIG)
    parser.add_argument(
        "--overlay",
        type=Path,
        default=FIG_DIR / "eq1_print_refit_overlay.png",
        help="P-F overlay figure path (all sets)",
    )
    parser.add_argument(
        "--b1neg-fig",
        type=Path,
        default=FIG_DIR / "eq1_paper_vs_b1neg.png",
        help="Dedicated paper vs paper b1=-|b1| comparison figure",
    )
    parser.add_argument("--update-json", action="store_true", help="Rewrite print-fit JSON")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    contours = parse_contour_file(args.input)
    p, f, v = contours_to_arrays(contours)
    print(f"Digitized points: n={len(p)}  from {args.input}")
    print(f"Windows fixed: alpha={ALPHA}, beta={BETA}, Pt={P_T}, Ft={F_T}")
    print()

    lstsq, sse = fit_lstsq(p, f, v)
    paper = paper_vec()
    b1neg = paper_b1neg_vec()
    sets = [
        CoefSet("printed (b1=+0.55)", "paper", paper),
        CoefSet("production (b1=-0.55)", "b1neg", b1neg),
        CoefSet("user L-BFGS", "user", USER_VEC.copy()),
        CoefSet("LSTSQ print-fit", "lstsq", lstsq),
    ]

    print("========== Coefficients ==========")
    print_coef_table(sets)
    print()
    print(f"LSTSQ SSE (from np.linalg.lstsq) = {sse:.8f}")
    print()
    print("========== Residuals on digitized points ==========")
    metrics = {s.short: residual_metrics(s.vec, p, f, v) for s in sets}
    for s in sets:
        print_metrics(s.name, metrics[s.short])
        print()

    # Agreement checks
    print("========== Audit flags ==========")
    print(f"  paper b1 = {paper[2]:+.4f}")
    print(f"  b1neg b1 = {b1neg[2]:+.4f}  (only sign flip)")
    print(f"  user  b1 = {USER_VEC[2]:+.4f}  (d vs paper = {USER_VEC[2] - paper[2]:+.4f})")
    print(f"  LSTSQ b1 = {lstsq[2]:+.4f}  (d vs paper = {lstsq[2] - paper[2]:+.4f})")
    print(
        f"  |user - LSTSQ| max = {np.max(np.abs(USER_VEC - lstsq)):.4f}  "
        f"(a0 d={USER_VEC[0]-lstsq[0]:+.4f}, b1 d={USER_VEC[2]-lstsq[2]:+.4f})"
    )
    print(
        f"  b1neg vs paper: rms {metrics['paper']['rms']:.5f} -> {metrics['b1neg']['rms']:.5f}  "
        f"(box {metrics['paper']['rms_in']:.5f} -> {metrics['b1neg']['rms_in']:.5f})"
    )
    assert metrics["user"]["rms"] < 0.01, "user fit unexpectedly poor"
    assert metrics["lstsq"]["rms"] < 0.01, "LSTSQ fit unexpectedly poor"
    assert USER_VEC[2] < 0 and lstsq[2] < 0 and b1neg[2] < 0, "negative-b1 sets expected"
    assert paper[2] > 0, "printed table must keep erroneous +b1 for comparison"
    # Production = printed with only b1 flipped; must beat printed and nearly match LSTSQ
    assert metrics["b1neg"]["rms"] < metrics["paper"]["rms"], "corrected b1 must beat printed"
    assert metrics["b1neg"]["rms"] < 0.01, "production coeffs must fit digitized contours"
    assert abs(b1neg[2] - (-0.55)) < 1e-9, "production b1 must be -0.55"
    print("  PASS: production b1=-0.55 fits digitized contours; printed +0.55 retained for errata.")

    if args.update_json:
        update_json(lstsq, b1neg)

    if args.plot or args.show:
        plot_three_way(contours, sets, out=args.output, show=False)
        plot_pf_overlay(contours, sets, out=args.overlay, show=False)
        plot_paper_vs_b1neg(
            contours, paper, b1neg, out=args.b1neg_fig, show=args.show
        )


if __name__ == "__main__":
    main()
