"""
Quantitative overlay: forward H–Vp tracks vs digitized KKHS02 Fig.12a.

Compares same-label χ (and χ=1,b) curves and Tp isotherms in the H–Vp plane
using Vp residuals along overlapping H, plus a scaled nearest-neighbour
distance.  Primary forward track is the GUI default Fig.12-linear (eq.(1) +
12%/GPa).  Optional Modern REEBOX / kinzler_linear grid for contrast.

Usage::

    py -3.11 petrology/validation/compare_fig12a_forward_vs_digitized.py
    py -3.11 petrology/validation/compare_fig12a_forward_vs_digitized.py --plot
    py -3.11 petrology/validation/compare_fig12a_forward_vs_digitized.py --include-modern --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.gui.hvp_plot import (
    FIG12_CHI1_B_VALUES_KM,
    FIG12_SOLID_CHI_VALUES,
    HvpCurvePoint,
    build_fig12_linear_curves,
    build_modern_hvp_curves,
)
from petrology.hvp.fig12a_background import (
    CHI1_B_DASHED,
    CHI_SOLID,
    DEFAULT_HVP_DIGITIZED,
    H_LIM,
    TP_CONTOURS,
    VP_LIM,
    CurveSeries,
    draw_fig12a_background,
    find_series,
    parse_hvp_digitized,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
DEFAULT_FIG = FIG_DIR / "fig12a_forward_vs_digitized.png"

# Axis-span scales for dimensionless nearest-neighbour distance
_H_SCALE_KM = float(H_LIM[1] - H_LIM[0])  # 32 km
_VP_SCALE_KM_S = float(VP_LIM[1] - VP_LIM[0])  # 0.46 km/s


@dataclass(frozen=True)
class CurveResidual:
    label: str
    kind: str  # "track" | "tp"
    n: int
    h_overlap_km: tuple[float, float]
    mean_dvp_km_s: float  # forward − digitized
    rms_dvp_km_s: float
    med_abs_dvp_km_s: float
    max_abs_dvp_km_s: float
    mean_nn_scaled: float  # mean nearest-neighbour in (H,Vp) / axis span
    rms_nn_scaled: float


@dataclass(frozen=True)
class TrackCompareSummary:
    track_name: str
    n_track_curves: int
    n_tp_curves: int
    pooled_rms_dvp_km_s: float
    pooled_mean_dvp_km_s: float
    pooled_n: int
    best_constant_vp_bias_km_s: float  # dig − fwd mean (add to fwd to match dig)
    rms_after_bias_km_s: float
    curves: tuple[CurveResidual, ...]


def _curve_xy_from_points(pts: list[HvpCurvePoint]) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(pts, key=lambda p: (p.h_km, p.tp_c))
    h = np.array([p.h_km for p in ordered], dtype=float)
    v = np.array([p.vp_bulk_km_s for p in ordered], dtype=float)
    if len(h) < 2:
        return h, v
    # Unique H (keep first) for interp along χ tracks
    uniq_h, idx = np.unique(h, return_index=True)
    return uniq_h, v[idx]


def _isotherm_xy(
    main: list[HvpCurvePoint],
    chi1_b: list[tuple[float, list[HvpCurvePoint]]],
    tp_c: float,
) -> tuple[np.ndarray, np.ndarray]:
    pts = [p for p in main if abs(p.tp_c - tp_c) < 1e-6]
    for _b, line in chi1_b:
        pts.extend(p for p in line if abs(p.tp_c - tp_c) < 1e-6)
    pts = sorted(pts, key=lambda p: (p.h_km, p.chi, float(getattr(p, "b_km", 0.0) or 0.0)))
    if len(pts) < 2:
        return np.array([]), np.array([])
    h = np.array([p.h_km for p in pts], dtype=float)
    v = np.array([p.vp_bulk_km_s for p in pts], dtype=float)
    return h, v


def _residuals_along_h(
    h_dig: np.ndarray,
    v_dig: np.ndarray,
    h_fwd: np.ndarray,
    v_fwd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]] | None:
    if len(h_fwd) < 2 or len(h_dig) < 2:
        return None
    h_lo = float(max(h_dig.min(), h_fwd.min()))
    h_hi = float(min(h_dig.max(), h_fwd.max()))
    if h_hi <= h_lo + 1e-9:
        return None
    mask = (h_dig >= h_lo) & (h_dig <= h_hi)
    hd, vd = h_dig[mask], v_dig[mask]
    if len(hd) < 2:
        return None
    # Forward may not be strictly increasing in H for isotherms; use np.interp
    # after sorting + unique.
    order = np.argsort(h_fwd)
    hf, vf = h_fwd[order], v_fwd[order]
    uniq, idx = np.unique(hf, return_index=True)
    vf = vf[idx]
    v_pred = np.interp(hd, uniq, vf)
    return hd, v_pred - vd, (h_lo, h_hi)


def _nn_scaled_distances(
    h_dig: np.ndarray,
    v_dig: np.ndarray,
    h_fwd: np.ndarray,
    v_fwd: np.ndarray,
) -> np.ndarray:
    """Nearest forward sample to each digitized point, scaled by axis spans."""
    if len(h_fwd) == 0:
        return np.array([])
    out = np.empty(len(h_dig), dtype=float)
    for i, (hd, vd) in enumerate(zip(h_dig, v_dig)):
        dh = (h_fwd - hd) / _H_SCALE_KM
        dv = (v_fwd - vd) / _VP_SCALE_KM_S
        out[i] = float(np.sqrt(np.min(dh * dh + dv * dv)))
    return out


def _curve_residual(
    label: str,
    kind: str,
    h_dig: np.ndarray,
    v_dig: np.ndarray,
    h_fwd: np.ndarray,
    v_fwd: np.ndarray,
) -> CurveResidual | None:
    got = _residuals_along_h(h_dig, v_dig, h_fwd, v_fwd)
    if got is None:
        return None
    _hd, dvp, overlap = got
    nn = _nn_scaled_distances(h_dig, v_dig, h_fwd, v_fwd)
    return CurveResidual(
        label=label,
        kind=kind,
        n=int(dvp.size),
        h_overlap_km=(float(overlap[0]), float(overlap[1])),
        mean_dvp_km_s=float(np.mean(dvp)),
        rms_dvp_km_s=float(np.sqrt(np.mean(dvp * dvp))),
        med_abs_dvp_km_s=float(np.median(np.abs(dvp))),
        max_abs_dvp_km_s=float(np.max(np.abs(dvp))),
        mean_nn_scaled=float(np.mean(nn)) if nn.size else float("nan"),
        rms_nn_scaled=float(np.sqrt(np.mean(nn * nn))) if nn.size else float("nan"),
    )


def compare_forward_to_digitized(
    digitized: list[CurveSeries],
    main: list[HvpCurvePoint],
    chi1_b: list[tuple[float, list[HvpCurvePoint]]],
    *,
    track_name: str,
    solid_chi: tuple[float, ...] = CHI_SOLID,
    chi1_b_values: tuple[float, ...] = CHI1_B_DASHED,
    tp_values: tuple[float, ...] = TP_CONTOURS,
) -> TrackCompareSummary:
    curves: list[CurveResidual] = []
    pooled: list[float] = []

    # χ solids at b=0
    for chi in solid_chi:
        dig = find_series(digitized, chi=float(chi), b_km=0.0)
        fwd_pts = [p for p in main if abs(p.chi - chi) < 1e-9]
        if dig is None or not fwd_pts:
            continue
        hf, vf = _curve_xy_from_points(fwd_pts)
        res = _curve_residual(f"χ={chi:g}, b=0", "track", dig.h_km, dig.vp_km_s, hf, vf)
        if res is not None:
            curves.append(res)
            got = _residuals_along_h(dig.h_km, dig.vp_km_s, hf, vf)
            if got is not None:
                pooled.extend(float(x) for x in got[1])

    # χ=1, b family
    chi1_map = {float(b): pts for b, pts in chi1_b}
    for b_km in chi1_b_values:
        dig = find_series(digitized, chi=1.0, b_km=float(b_km))
        fwd_pts = chi1_map.get(float(b_km))
        if dig is None or not fwd_pts:
            continue
        hf, vf = _curve_xy_from_points(fwd_pts)
        res = _curve_residual(
            f"χ=1, b={b_km:g}", "track", dig.h_km, dig.vp_km_s, hf, vf
        )
        if res is not None:
            curves.append(res)
            got = _residuals_along_h(dig.h_km, dig.vp_km_s, hf, vf)
            if got is not None:
                pooled.extend(float(x) for x in got[1])

    # Tp isotherms
    for tp in tp_values:
        dig = find_series(digitized, tp_c=float(tp))
        hf, vf = _isotherm_xy(main, chi1_b, float(tp))
        if dig is None or len(hf) < 2:
            continue
        res = _curve_residual(f"Tp={tp:.0f}°C", "tp", dig.h_km, dig.vp_km_s, hf, vf)
        if res is not None:
            curves.append(res)
            # Tp residuals not pooled into χ-track bias (different geometry)

    arr = np.array(pooled, dtype=float) if pooled else np.array([])
    if arr.size:
        mean_d = float(np.mean(arr))
        rms = float(np.sqrt(np.mean(arr * arr)))
        # Bias that shifts forward toward digitized: dig − fwd = −mean(fwd−dig)
        bias = -mean_d
        rms_b = float(np.sqrt(np.mean((arr + bias) ** 2)))
    else:
        mean_d = rms = bias = rms_b = float("nan")

    return TrackCompareSummary(
        track_name=track_name,
        n_track_curves=sum(1 for c in curves if c.kind == "track"),
        n_tp_curves=sum(1 for c in curves if c.kind == "tp"),
        pooled_rms_dvp_km_s=rms,
        pooled_mean_dvp_km_s=mean_d,
        pooled_n=int(arr.size),
        best_constant_vp_bias_km_s=bias,
        rms_after_bias_km_s=rms_b,
        curves=tuple(curves),
    )


def build_linear_forward(
    *,
    tp_min: float = 1200.0,
    tp_max: float = 1600.0,
    tp_step: float = 5.0,
    dfdp_per_gpa: float = 0.12,
) -> tuple[list[HvpCurvePoint], list[tuple[float, list[HvpCurvePoint]]]]:
    tp = np.arange(tp_min, tp_max + 0.5 * tp_step, tp_step)
    return build_fig12_linear_curves(
        tp_values_c=tp,
        solid_chi_values=FIG12_SOLID_CHI_VALUES,
        chi1_b_values_km=FIG12_CHI1_B_VALUES_KM,
        dfdp_per_gpa=dfdp_per_gpa,
    )


def build_modern_forward(
    *,
    tp_min: float = 1200.0,
    tp_max: float = 1600.0,
    tp_step: float = 25.0,
    melting_engine: str = "kinzler_linear",
    lithology_backend: str = "native",
    pyroxenite_frac: float = 0.0,
) -> tuple[list[HvpCurvePoint], list[tuple[float, list[HvpCurvePoint]]]]:
    from petrology.melting.pymelt_lithology_adapter import resolve_lithology_col_kwargs

    col_kw = resolve_lithology_col_kwargs(lithology_backend=lithology_backend)
    tp = np.arange(tp_min, tp_max + 0.5 * tp_step, tp_step)
    return build_modern_hvp_curves(
        tp_values_c=tp,
        col_kw=col_kw,
        solid_chi_values=FIG12_SOLID_CHI_VALUES,
        chi1_b_values_km=FIG12_CHI1_B_VALUES_KM,
        pyroxenite_frac=pyroxenite_frac,
        melting_engine=melting_engine,
    )


def _print_summary(summary: TrackCompareSummary) -> None:
    print(f"\n## {summary.track_name}")
    print(
        f"  χ/b tracks={summary.n_track_curves}  Tp isotherms={summary.n_tp_curves}  "
        f"pooled χ-track n={summary.pooled_n}"
    )
    print(
        f"  pooled χ-track: mean(fwd−dig)={summary.pooled_mean_dvp_km_s:+.4f} km/s  "
        f"RMS={summary.pooled_rms_dvp_km_s:.4f} km/s"
    )
    print(
        f"  best constant Vp bias (dig−fwd)={summary.best_constant_vp_bias_km_s:+.4f} km/s  "
        f"RMS after bias={summary.rms_after_bias_km_s:.4f} km/s"
    )
    print(
        f"  {'label':<16} {'n':>3}  {'H overlap':>13}  "
        f"{'mean dVp':>9} {'RMS':>7} {'|dV|med':>8} {'max':>7}  {'nn̄':>6}"
    )
    for c in summary.curves:
        print(
            f"  {c.label:<16} {c.n:3d}  "
            f"[{c.h_overlap_km[0]:4.1f},{c.h_overlap_km[1]:4.1f}]  "
            f"{c.mean_dvp_km_s:+9.4f} {c.rms_dvp_km_s:7.4f} "
            f"{c.med_abs_dvp_km_s:8.4f} {c.max_abs_dvp_km_s:7.4f}  "
            f"{c.mean_nn_scaled:6.3f}"
        )


def plot_overlay(
    digitized: list[CurveSeries],
    summaries: list[tuple[TrackCompareSummary, list[HvpCurvePoint], list[tuple[float, list[HvpCurvePoint]]]]],
    *,
    out: Path,
    h_lim: tuple[float, float] = H_LIM,
    vp_lim: tuple[float, float] = VP_LIM,
) -> Path:
    import matplotlib.pyplot as plt

    n = 1 + len(summaries)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.8), dpi=130, squeeze=False)
    ax0 = axes[0, 0]
    draw_fig12a_background(ax0, digitized, h_lim=h_lim, vp_lim=vp_lim, show_legend=False)
    ax0.set_title("Digitized Fig.12a (print)")
    ax0.set_xlabel("H (km)")
    ax0.set_ylabel(r"$V_p$ (km/s)")

    palette = ("#c0392b", "#1a5276", "#117a65", "#6c3483")
    for i, (summary, main, chi1_b) in enumerate(summaries, start=1):
        ax = axes[0, i]
        # faint digitized
        draw_fig12a_background(ax, digitized, h_lim=h_lim, vp_lim=vp_lim, show_legend=False)
        for line in ax.get_lines():
            line.set_alpha(0.25)
            line.set_linewidth(0.9)
        col = palette[(i - 1) % len(palette)]
        for chi in CHI_SOLID:
            pts = sorted([p for p in main if abs(p.chi - chi) < 1e-9], key=lambda p: p.tp_c)
            if len(pts) < 2:
                continue
            ax.plot(
                [p.h_km for p in pts],
                [p.vp_bulk_km_s for p in pts],
                color=col,
                lw=1.8,
                alpha=0.95,
                zorder=5,
                label=f"fwd χ={chi:g}",
            )
        for b_km, line in chi1_b:
            pts = sorted(line, key=lambda p: p.tp_c)
            if len(pts) < 2:
                continue
            ax.plot(
                [p.h_km for p in pts],
                [p.vp_bulk_km_s for p in pts],
                color=col,
                lw=1.1,
                ls="--",
                alpha=0.85,
                zorder=4,
            )
        ax.set_title(
            f"{summary.track_name}\n"
            f"χ RMS={summary.pooled_rms_dvp_km_s:.3f} km/s  "
            f"bias={summary.best_constant_vp_bias_km_s:+.3f}"
        )
        ax.set_xlabel("H (km)")
        ax.legend(fontsize=6, loc="lower right")

    fig.suptitle(
        "Fig.12a digitized (grey) vs forward H–Vp (colour) — same χ / b labels",
        fontsize=10,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out}")
    return out


def run(
    *,
    data_file: Path = DEFAULT_HVP_DIGITIZED,
    include_modern: bool = False,
    modern_engines: tuple[str, ...] = ("kinzler_linear", "reebox"),
    tp_step_linear: float = 5.0,
    tp_step_modern: float = 25.0,
    plot: bool = False,
    out_fig: Path = DEFAULT_FIG,
    json_out: Path | None = None,
) -> list[TrackCompareSummary]:
    digitized = parse_hvp_digitized(data_file)
    print(f"# Digitized: {data_file.name}  n_series={len(digitized)}")
    print("# Residual = V_fwd(H) − V_dig(H) on overlapping H; pooled over χ/b tracks only.")

    summaries: list[TrackCompareSummary] = []
    plot_bundle: list[
        tuple[TrackCompareSummary, list[HvpCurvePoint], list[tuple[float, list[HvpCurvePoint]]]]
    ] = []

    main_l, chi1_l = build_linear_forward(tp_step=tp_step_linear)
    sum_l = compare_forward_to_digitized(
        digitized, main_l, chi1_l, track_name="fig12_linear"
    )
    summaries.append(sum_l)
    plot_bundle.append((sum_l, main_l, chi1_l))
    _print_summary(sum_l)

    if include_modern:
        for eng in modern_engines:
            main_m, chi1_m = build_modern_forward(
                tp_step=tp_step_modern, melting_engine=eng
            )
            sum_m = compare_forward_to_digitized(
                digitized, main_m, chi1_m, track_name=f"modern:{eng}"
            )
            summaries.append(sum_m)
            plot_bundle.append((sum_m, main_m, chi1_m))
            _print_summary(sum_m)

    print("\n## Verdict notes")
    print(
        "  Digitized = GetData facsimile of printed Fig.12a "
        "(imodel visual background / validation)."
    )
    print(
        "  fig12_linear = GUI default H–Vp (KKHS02 §4 linear F + eq.(1)); "
        "matches digitized χ tracks at digitization noise if RMS ≲ 0.01 km/s."
    )
    print(
        "  Modern REEBOX differs by melting physics (not print error); "
        "use for current petrology, not as a drop-in print facsimile."
    )
    print(
        "  Preferred scientific reference = forward tracks; keep digitized as "
        "print facsimile only (annotate ≈ linear forward)."
    )

    if plot:
        plot_overlay(digitized, plot_bundle, out=out_fig)

    if json_out is not None:
        payload = {
            "data_file": str(data_file),
            "tracks": [
                {
                    **{k: v for k, v in asdict(s).items() if k != "curves"},
                    "curves": [asdict(c) for c in s.curves],
                }
                for s in summaries
            ],
        }
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {json_out}")

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RMS overlay: forward H–Vp vs digitized Fig.12a"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_HVP_DIGITIZED)
    parser.add_argument("--include-modern", action="store_true")
    parser.add_argument(
        "--modern-engine",
        default="kinzler_linear,reebox",
        help="Comma-separated Modern engines when --include-modern "
        "(default: kinzler_linear,reebox)",
    )
    parser.add_argument("--tp-step-linear", type=float, default=5.0)
    parser.add_argument("--tp-step-modern", type=float, default=25.0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_FIG)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    engines = tuple(
        x.strip() for x in str(args.modern_engine).split(",") if x.strip()
    )
    run(
        data_file=args.data,
        include_modern=bool(args.include_modern),
        modern_engines=engines,
        tp_step_linear=float(args.tp_step_linear),
        tp_step_modern=float(args.tp_step_modern),
        plot=bool(args.plot),
        out_fig=args.output,
        json_out=args.json,
    )


if __name__ == "__main__":
    main()
