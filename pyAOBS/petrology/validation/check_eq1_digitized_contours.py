"""Compare digitized KKHS02 V(P,F) contours vs equation (1).

Input format (GetData / similar)::

    6.9          # contour label = V (km/s)
    P   F        # digitized points on that contour
    ...
    7.0
    ...

Usage::

    py -3.11 petrology/validation/check_eq1_digitized_contours.py
    py -3.11 petrology/validation/check_eq1_digitized_contours.py -i petrology/data/ScreenShot_....txt
    py -3.11 petrology/validation/check_eq1_digitized_contours.py --plot
    py -3.11 petrology/validation/check_eq1_digitized_contours.py --plot --bias -0.18
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.vp_regression import (
    in_eq1_applicable_range,
    load_eq1,
    predict_v_bulk_grid,
    predict_v_bulk_km_s,
)

DEFAULT_INPUT = (
    Path(__file__).resolve().parents[1] / "data" / "ScreenShot_2026-07-09_105537_962.txt"
)
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
DEFAULT_FIG = FIG_DIR / "eq1_digitized_contour_compare.png"


def parse_contour_file(path: Path) -> list[tuple[float, list[tuple[float, float]]]]:
    contours: list[tuple[float, list[tuple[float, float]]]] = []
    cur_v: float | None = None
    cur_pts: list[tuple[float, float]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) == 1:
            if cur_v is not None:
                contours.append((cur_v, cur_pts))
            cur_v = float(parts[0])
            cur_pts = []
        elif len(parts) >= 2:
            cur_pts.append((float(parts[0]), float(parts[1])))
    if cur_v is not None:
        contours.append((cur_v, cur_pts))
    return contours


def _stats(delta: np.ndarray, name: str) -> None:
    if delta.size == 0:
        print(f"{name}: no points")
        return
    print(f"{name}: n={delta.size}")
    print(f"  mean(V_eq1 - V_label) = {delta.mean():+.4f} km/s")
    print(f"  rms                     = {np.sqrt((delta**2).mean()):.4f} km/s")
    print(
        f"  |dV| median / max        = {np.median(np.abs(delta)):.4f} / "
        f"{np.abs(delta).max():.4f} km/s"
    )
    for thr in (0.05, 0.10, 0.15):
        pct = 100.0 * float(np.mean(np.abs(delta) < thr))
        print(f"  |dV| < {thr:.2f} km/s         = {pct:.0f}%")


def plot_comparison(
    contours: list[tuple[float, list[tuple[float, float]]]],
    *,
    bias: float = 0.0,
    out: Path | None = None,
    show: bool = False,
) -> Path:
    """Overlay digitized contours vs eq.(1) isolines; residual scatter."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    eq1 = load_eq1()
    rng = eq1["equation_1"]["applicable_range"]
    p_lo, p_hi = float(rng["P_GPa"][0]), float(rng["P_GPa"][1])
    f_lo, f_hi = float(rng["F"][0]), float(rng["F"][1])

    levels = [c[0] for c in contours]
    all_p = [p for _, pts in contours for p, _ in pts]
    all_f = [f for _, pts in contours for _, f in pts]
    p_max = max(7.0, max(all_p) * 1.02)
    f_max = max(0.6, max(all_f) * 1.05)

    p_grid = np.linspace(0.05, p_max, 200)
    f_grid = np.linspace(0.005, f_max, 160)
    P, F = np.meshgrid(p_grid, f_grid)
    V = predict_v_bulk_grid(P, F, bias=float(bias))

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.6), constrained_layout=True)
    ax, axr = axes

    # --- left: P–F plane ---
    cs = ax.contour(
        P,
        F,
        V,
        levels=levels,
        colors="#c0392b",
        linewidths=1.35,
        linestyles="-",
        zorder=3,
    )
    ax.clabel(cs, inline=True, fontsize=7.5, fmt="%g")

    cmap = plt.get_cmap("viridis")
    n_lev = max(len(levels), 1)
    for i, (v_lab, pts) in enumerate(contours):
        if not pts:
            continue
        color = cmap(0.12 + 0.8 * i / max(n_lev - 1, 1))
        pp = np.array([p for p, _ in pts])
        ff = np.array([f for _, f in pts])
        order = np.argsort(pp)
        ax.plot(pp[order], ff[order], "-", color=color, lw=1.6, alpha=0.95, zorder=4)
        ax.scatter(pp, ff, s=18, c=[color], edgecolors="0.2", linewidths=0.35, zorder=5)

    ax.add_patch(
        Rectangle(
            (p_lo, f_lo),
            p_hi - p_lo,
            f_hi - f_lo,
            fill=False,
            ec="#2980b9",
            lw=1.4,
            ls="--",
            zorder=6,
            label="eq.(1) applicable box",
        )
    )
    ax.set_xlim(0.0, p_max)
    ax.set_ylim(f_max, 0.0)  # paper Fig.3: Mean F increases downward
    ax.set_xlabel(r"Mean $P$ (GPa)")
    ax.set_ylabel(r"Mean $F$")
    bias_txt = f", bias={bias:+.2f}" if bias else ""
    ax.set_title(f"(a) Digitized contours vs eq.(1){bias_txt}")
    ax.grid(True, ls=":", lw=0.4, alpha=0.4)
    legend_handles = [
        Line2D([0], [0], color="#c0392b", lw=1.5, label="eq.(1) isolines"),
        Line2D([0], [0], color="0.35", lw=1.6, label="digitized (GetData)"),
        Line2D([0], [0], color="#2980b9", lw=1.4, ls="--", label="applicable box"),
    ]
    ax.legend(handles=legend_handles, fontsize=7.5, loc="lower left", framealpha=0.92)

    # --- right: residual at digitized points ---
    rows: list[tuple[float, float, float, float, bool]] = []
    for v_lab, pts in contours:
        for p, f in pts:
            v_pred = predict_v_bulk_km_s(p, f) + float(bias)
            rows.append((p, f, v_lab, v_pred - v_lab, in_eq1_applicable_range(p, f)))

    pin = np.array([r[0] for r in rows if r[4]])
    fin = np.array([r[1] for r in rows if r[4]])
    din = np.array([r[3] for r in rows if r[4]])
    pout = np.array([r[0] for r in rows if not r[4]])
    fout = np.array([r[1] for r in rows if not r[4]])
    dout = np.array([r[3] for r in rows if not r[4]])

    vmax = max(0.25, float(np.nanmax(np.abs([*din, *dout]))) if rows else 0.25)
    sc_kw = dict(cmap="coolwarm", vmin=-vmax, vmax=vmax, edgecolors="0.25", linewidths=0.35)

    if pout.size:
        axr.scatter(pout, fout, c=dout, s=36, marker="o", alpha=0.55, zorder=3, **sc_kw)
    if pin.size:
        sc = axr.scatter(pin, fin, c=din, s=55, marker="s", zorder=4, **sc_kw)
    else:
        sc = axr.scatter([], [], c=[], **sc_kw)
    cb = fig.colorbar(sc, ax=axr, shrink=0.88, pad=0.02)
    cb.set_label(r"$V_{\mathrm{eq1}} - V_{\mathrm{label}}$ (km/s)")

    axr.add_patch(
        Rectangle(
            (p_lo, f_lo),
            p_hi - p_lo,
            f_hi - f_lo,
            fill=False,
            ec="#2980b9",
            lw=1.4,
            ls="--",
            zorder=5,
        )
    )
    axr.set_xlim(0.0, p_max)
    axr.set_ylim(f_max, 0.0)
    axr.set_xlabel(r"Mean $P$ (GPa)")
    axr.set_ylabel(r"Mean $F$")
    if pin.size:
        axr.set_title(
            f"(b) Residual at points  "
            f"(inside: mean={din.mean():+.3f}, rms={np.sqrt((din**2).mean()):.3f})"
        )
    else:
        axr.set_title("(b) Residual at digitized points")
    axr.grid(True, ls=":", lw=0.4, alpha=0.4)
    axr.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="#555",
                markersize=7,
                label="inside box",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#555",
                markersize=6,
                alpha=0.55,
                label="outside box",
            ),
        ],
        fontsize=7.5,
        loc="lower left",
        framealpha=0.92,
    )

    out_path = Path(out) if out is not None else DEFAULT_FIG
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print(f"Saved {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Digitized V(P,F) contours vs KKHS02 eq.(1)")
    parser.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--bias",
        type=float,
        default=0.0,
        help=(
            "Add constant to V_eq1 before residual. "
            "With corrected b1=-0.55 default 0 is enough; -0.18 was legacy print-b1 compensation."
        ),
    )
    parser.add_argument("--plot", action="store_true", help="Write comparison figure")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"Figure path (default: {DEFAULT_FIG.name})",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    contours = parse_contour_file(args.input)
    n_tot = sum(len(pts) for _, pts in contours)
    print(f"File: {args.input}")
    print(f"Contours: {[c[0] for c in contours]}  ({n_tot} points)")
    if args.bias:
        print(f"Applied V_eq1 bias = {args.bias:+.3f} km/s")

    eq1 = load_eq1()
    rng = eq1["equation_1"]["applicable_range"]
    print(f"eq.(1) applicable range: P={rng['P_GPa']} GPa, F={rng['F']}")
    print()

    rows: list[tuple[float, float, float, float, float, bool]] = []
    for v_lab, pts in contours:
        for p, f in pts:
            v_pred = predict_v_bulk_km_s(p, f) + float(args.bias)
            inside = in_eq1_applicable_range(p, f)
            rows.append((v_lab, p, f, v_pred, v_pred - v_lab, inside))

    resid = np.array([r[4] for r in rows], dtype=float)
    inside = np.array([r[5] for r in rows], dtype=bool)

    _stats(resid, "ALL digitized points")
    print()
    _stats(resid[inside], "INSIDE eq.(1) box")
    print()
    _stats(resid[~inside], "OUTSIDE applicable range")

    print()
    print("Per-contour (inside box):")
    hdr = f"{'V_lab':>6} {'n_in':>4} {'n_all':>5} {'mean_dV':>9} {'rms':>8} {'|dV|_med':>9}"
    print(hdr)
    for v_lab, _pts in contours:
        sub = [r for r in rows if r[0] == v_lab and r[5]]
        alln = sum(1 for r in rows if r[0] == v_lab)
        if not sub:
            print(f"{v_lab:6.1f} {0:4d} {alln:5d}   (no inside pts)")
            continue
        d = np.array([r[4] for r in sub])
        print(
            f"{v_lab:6.1f} {len(sub):4d} {alln:5d} {d.mean():+9.4f} "
            f"{np.sqrt((d**2).mean()):8.4f} {np.median(np.abs(d)):9.4f}"
        )

    inside_rows = [r for r in rows if r[5]]
    print()
    print("Worst |dV| inside box:")
    print(f"{'V_lab':>6} {'P':>7} {'F':>7} {'V_eq1':>7} {'dV':>8}")
    for r in sorted(inside_rows, key=lambda x: abs(x[4]), reverse=True)[:12]:
        print(f"{r[0]:6.1f} {r[1]:7.3f} {r[2]:7.4f} {r[3]:7.3f} {r[4]:+8.4f}")

    print()
    print("Best |dV| inside box:")
    for r in sorted(inside_rows, key=lambda x: abs(x[4]))[:8]:
        print(f"{r[0]:6.1f} {r[1]:7.3f} {r[2]:7.4f} {r[3]:7.3f} {r[4]:+8.4f}")

    if args.plot or args.show or args.output is not None:
        out = args.output
        if out is None and args.bias:
            out = FIG_DIR / f"eq1_digitized_contour_compare_bias{args.bias:+.2f}.png"
        plot_comparison(contours, bias=float(args.bias), out=out, show=args.show)


if __name__ == "__main__":
    main()
