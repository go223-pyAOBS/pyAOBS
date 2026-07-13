"""
Compare Reproduction vs Modern H–Vp scans: feasible (Tp, chi) region and best-fit shift.

Runs ``scan_hvp_lip`` on a shared (Tp, chi) grid for:
  - ``repro_reebox``  — REEBOX + native Katz Table 2 + Kinzler + norm Vp
  - ``modern_reebox`` — REEBOX + pyMelt Katz + pMELTS chemistry + norm Vp
  - ``repro_linear``  — KKHS02 linear F(P) + eq.(1) Vp (optional, ``--include-linear``)

Usage::

  py -3.11 petrology/validation/compare_tracks_hvp_sensitivity.py
  py -3.11 petrology/validation/compare_tracks_hvp_sensitivity.py --plot
  py -3.11 petrology/validation/compare_tracks_hvp_sensitivity.py --quick --plot
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.melting.hvp_scan import HvpScanPoint, HvpScanResult, scan_hvp_lip
from petrology.melting.lip_scan import _best_pareto_from_result


@dataclass(frozen=True)
class TrackScan:
    name: str
    label: str
    result: HvpScanResult


def _best_point(result: HvpScanResult, *, v_lc: float, h_obs: float) -> HvpScanPoint | None:
    if result.feasible:
        return min(result.feasible, key=lambda p: abs(p.h_match_km))
    return _best_pareto_from_result(result, v_lc_obs=v_lc, h_obs=h_obs)


def _range_mid(rng: tuple[float, float] | None) -> float | None:
    if rng is None:
        return None
    return 0.5 * (float(rng[0]) + float(rng[1]))


def _print_track_summary(track: TrackScan) -> None:
    r = track.result
    rng = r.tp_chi_ranges()
    print(f"\n## {track.name} — {track.label}")
    print(f"  grid={len(r.points)}  feasible={r.n_feasible}")
    if rng["tp_c"]:
        print(f"  Tp range: {rng['tp_c'][0]:.0f} – {rng['tp_c'][1]:.0f} C")
        print(f"  chi range: {rng['chi'][0]:.1f} – {rng['chi'][1]:.1f}")
    else:
        print("  Tp/chi feasible: (none)")
    best = _best_point(r, v_lc=r.v_lc_obs_km_s, h_obs=r.h_obs_km)
    if best:
        print(
            f"  best: Tp={best.tp_c:.0f} C chi={best.chi:g} "
            f"H={best.h_km:.1f} dH={best.h_match_km:+.1f} "
            f"Vp={best.vp_bulk_km_s:.3f} in_bounds={best.bulk_in_bounds}"
        )


def _print_delta(repro: TrackScan, modern: TrackScan) -> None:
    rr = repro.result
    mr = modern.result
    rr_rng = rr.tp_chi_ranges()
    mr_rng = mr.tp_chi_ranges()

    print("\n## Delta (modern_reebox - repro_reebox)\n")

    if rr_rng["tp_c"] and mr_rng["tp_c"]:
        d_tp_lo = mr_rng["tp_c"][0] - rr_rng["tp_c"][0]
        d_tp_hi = mr_rng["tp_c"][1] - rr_rng["tp_c"][1]
        d_tp_mid = (_range_mid(mr_rng["tp_c"]) or 0) - (_range_mid(rr_rng["tp_c"]) or 0)
        print(f"  dTp range: [{d_tp_lo:+.0f}, {d_tp_hi:+.0f}] C  (mid {d_tp_mid:+.0f} C)")
    else:
        print("  dTp range: n/a (one track has no feasible region)")

    if rr_rng["chi"] and mr_rng["chi"]:
        d_chi_lo = mr_rng["chi"][0] - rr_rng["chi"][0]
        d_chi_hi = mr_rng["chi"][1] - rr_rng["chi"][1]
        d_chi_mid = (_range_mid(mr_rng["chi"]) or 0) - (_range_mid(rr_rng["chi"]) or 0)
        print(f"  dchi range: [{d_chi_lo:+.1f}, {d_chi_hi:+.1f}]  (mid {d_chi_mid:+.1f})")
    else:
        print("  dchi range: n/a")

    br = _best_point(rr, v_lc=rr.v_lc_obs_km_s, h_obs=rr.h_obs_km)
    bm = _best_point(mr, v_lc=mr.v_lc_obs_km_s, h_obs=mr.h_obs_km)
    if br and bm:
        print(
            f"  dTp best = {bm.tp_c - br.tp_c:+.0f} C"
            f"   dchi best = {bm.chi - br.chi:+.1f}"
            f"   dVp best = {bm.vp_bulk_km_s - br.vp_bulk_km_s:+.3f} km/s"
        )

    # Grid-cell forward differences on shared (Tp, chi)
    repro_map = {(p.tp_c, p.chi): p for p in rr.points}
    modern_map = {(p.tp_c, p.chi): p for p in mr.points}
    shared = sorted(set(repro_map) & set(modern_map))
    if shared:
        dh = [modern_map[k].h_km - repro_map[k].h_km for k in shared]
        dvp = [modern_map[k].vp_bulk_km_s - repro_map[k].vp_bulk_km_s for k in shared]
        df = [modern_map[k].fbar - repro_map[k].fbar for k in shared]
        print(
            f"  shared grid n={len(shared)}: "
            f"dH [{min(dh):+.1f}, {max(dh):+.1f}] km  "
            f"dVp [{min(dvp):+.3f}, {max(dvp):+.3f}] km/s  "
            f"dFbar [{min(df):+.3f}, {max(df):+.3f}]"
        )


def _run_scan(
    name: str,
    label: str,
    *,
    v_lc: float,
    h_obs: float,
    b_km: float,
    tp_range: tuple[float, float],
    tp_step: float,
    chi_values: list[float],
    melting_engine: str,
    lithology_backend: str,
    vp_mode: str,
    peridotite_chemistry: str | None,
    n_isentropic_steps: int,
    h_tolerance: float,
    vp_bias: float,
    require_bulk_in_bounds: bool,
    delta_vp_engine: str = "auto",
) -> TrackScan:
    use_norm = vp_mode == "norm"
    res = scan_hvp_lip(
        v_lc_obs_km_s=v_lc,
        h_obs_km=h_obs,
        b_km=b_km,
        pyroxenite_frac=0.0,
        tp_range_c=tp_range,
        tp_step_c=tp_step,
        chi_values=chi_values,
        h_tolerance_km=h_tolerance,
        melting_engine=melting_engine,
        lithology_backend=lithology_backend,
        peridotite_lith="katz_lherzolite",
        peridotite_chemistry=peridotite_chemistry,
        use_norm_vp=use_norm,
        vp_mode=vp_mode,
        vp_bias_km_s=vp_bias,
        n_isentropic_steps=n_isentropic_steps,
        verbose=False,
        require_bulk_in_bounds=require_bulk_in_bounds,
        delta_vp_engine=delta_vp_engine,
    )
    return TrackScan(name=name, label=label, result=res)


def _plot_comparison(
    tracks: list[TrackScan],
    *,
    repro: TrackScan,
    modern: TrackScan,
    out: Path,
) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "repro_reebox": "#27ae60",
        "modern_reebox": "#c0392b",
        "repro_linear": "#2471a3",
    }

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), dpi=120)

    ax = axes[0]
    h_obs = repro.result.h_obs_km
    v_lc = repro.result.v_lc_obs_km_s
    for tr in tracks:
        col = colors.get(tr.name, "#333")
        chi_u = sorted({p.chi for p in tr.result.points})
        for chi in chi_u[:5]:
            line = sorted(
                [p for p in tr.result.points if p.chi == chi],
                key=lambda x: x.tp_c,
            )
            if len(line) < 2:
                continue
            ax.plot(
                [p.h_km for p in line],
                [p.vp_bulk_km_s for p in line],
                lw=1.2,
                color=col,
                alpha=0.85,
                label=f"{tr.name} chi={chi:g}",
            )
        for p in tr.result.feasible:
            ax.scatter(p.h_km, p.vp_bulk_km_s, s=22, c=col, edgecolors="k", linewidths=0.3, zorder=4)
    ax.axhline(v_lc, color="#888", ls=":", lw=0.9)
    ax.axvline(h_obs, color="#888", ls="--", lw=0.9)
    ax.scatter([h_obs], [v_lc], s=70, c="k", marker="*", zorder=6, label="obs")
    ax.set_xlabel("H (km)")
    ax.set_ylabel("Vp_bulk (km/s)")
    ax.set_title("(a) H–Vp forward curves")
    ax.legend(fontsize=5.5, loc="best")
    ax.grid(True, alpha=0.25)

    ax2 = axes[1]
    for tr in (repro, modern):
        col = colors.get(tr.name, "#333")
        pts = tr.result.points
        if not pts:
            continue
        tp_u = sorted({p.tp_c for p in pts})
        chi_u = sorted({p.chi for p in pts})
        grid = np.full((len(chi_u), len(tp_u)), np.nan)
        for p in pts:
            grid[chi_u.index(p.chi), tp_u.index(p.tp_c)] = 1.0 if p.feasible else 0.0
        ax2.imshow(
            grid,
            aspect="auto",
            origin="lower",
            alpha=0.45 if tr.name == "repro_reebox" else 0.75,
            cmap="Greens" if tr.name == "repro_reebox" else "Reds",
            vmin=0,
            vmax=1,
            extent=[tp_u[0] - 5, tp_u[-1] + 5, chi_u[0] - 0.5, chi_u[-1] + 0.5],
            label=tr.name,
        )
    ax2.set_xlabel("Tp (C)")
    ax2.set_ylabel("chi")
    ax2.set_title(
        f"(b) Feasible (Tp, chi)\n"
        f"repro n={repro.result.n_feasible}  modern n={modern.result.n_feasible}"
    )

    ax3 = axes[2]
    repro_map = {(p.tp_c, p.chi): p for p in repro.result.points}
    modern_map = {(p.tp_c, p.chi): p for p in modern.result.points}
    shared = sorted(set(repro_map) & set(modern_map))
    if shared:
        dtp = np.array([modern_map[k].tp_c - repro_map[k].tp_c for k in shared])
        dchi = np.array([modern_map[k].chi - repro_map[k].chi for k in shared])
        dvp = np.array([modern_map[k].vp_bulk_km_s - repro_map[k].vp_bulk_km_s for k in shared])
        dh = np.array([modern_map[k].h_km - repro_map[k].h_km for k in shared])
        ax3.scatter(dh, dvp, c=dchi, cmap="viridis", s=28, alpha=0.85)
        ax3.axhline(0, color="#888", lw=0.8)
        ax3.axvline(0, color="#888", lw=0.8)
        ax3.set_xlabel("dH modern − repro (km)")
        ax3.set_ylabel("dVp modern − repro (km/s)")
        ax3.set_title("(c) Grid-cell delta (color = chi)")
        ax3.grid(True, alpha=0.25)
    else:
        ax3.text(0.5, 0.5, "No shared grid", ha="center", va="center", transform=ax3.transAxes)

    obs = repro.result
    fig.suptitle(
        f"H–Vp scan — H={obs.h_obs_km:.0f} km V_LC={obs.v_lc_obs_km_s:.2f} km/s b={obs.b_km:.0f} km",
        fontsize=10,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"\nWrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repro vs Modern HVP scan delta Tp/chi")
    parser.add_argument("--h", type=float, default=30.0, help="Observed igneous thickness (km)")
    parser.add_argument("--vlc", type=float, default=7.0, help="Observed lower-crust Vp (km/s)")
    parser.add_argument("--b", type=float, default=0.0)
    parser.add_argument("--tp-min", type=float, default=1200.0)
    parser.add_argument("--tp-max", type=float, default=1500.0)
    parser.add_argument("--tp-step", type=float, default=None)
    parser.add_argument(
        "--chi",
        default="1,2,4,6,8,10,12",
        help="Comma-separated chi values",
    )
    parser.add_argument("--h-tol", type=float, default=3.0)
    parser.add_argument("--vp-bias", type=float, default=-0.10, help="Fallback bias if not auto/per-track")
    parser.add_argument("--repro-vp-bias", type=float, default=None)
    parser.add_argument("--modern-vp-bias", type=float, default=None)
    parser.add_argument("--linear-vp-bias", type=float, default=None)
    parser.add_argument("--auto-bias", action="store_true", help="Calibrate per-track Vp bias for H+Vp bounds")
    parser.add_argument(
        "--auto-bias-strategy",
        choices=("point", "sweep"),
        default="sweep",
        help="point=ref Tp/chi bound_mid; sweep=max feasible grid (default)",
    )
    parser.add_argument("--ref-tp", type=float, default=1425.0)
    parser.add_argument("--ref-chi", type=float, default=10.0)
    parser.add_argument("--bias-target", choices=("bound_mid", "bound_upper", "bound_lower"), default="bound_mid")
    parser.add_argument("--bias-margin", type=float, default=0.02)
    parser.add_argument("--isentropic-steps", type=int, default=48)
    parser.add_argument("--quick", action="store_true", help="Coarse grid for smoke test")
    parser.add_argument("--include-linear", action="store_true", help="Also scan KKHS02 linear repro")
    parser.add_argument(
        "--h-only",
        action="store_true",
        help="Feasible by |dH| only (ignore Vp bounds; for track-shape comparison)",
    )
    parser.add_argument(
        "--delta-vp-engine",
        choices=("auto", "wl1990", "param"),
        default="auto",
        help="auto=WL FC when melt known; param=legacy MVP formula (fast)",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "-o",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures" / "compare_tracks_hvp_sensitivity.png",
    )
    args = parser.parse_args()

    tp_step = float(args.tp_step) if args.tp_step is not None else (25.0 if args.quick else 15.0)
    chi_values = [float(x.strip()) for x in str(args.chi).split(",") if x.strip()]
    tp_range = (float(args.tp_min), float(args.tp_max))
    v_lc, h_obs, b_km = float(args.vlc), float(args.h), float(args.b)
    require_vp = not bool(args.h_only)

    print(f"# H-Vp track comparison (H={h_obs:g} km, V_LC={v_lc:g} km/s, b={b_km:g} km)")
    if args.h_only:
        print("# Mode: H-only feasible (Vp bounds ignored)\n")
    else:
        print(f"# Grid: Tp={tp_range[0]:.0f}-{tp_range[1]:.0f} step={tp_step:g}  chi={chi_values}\n")

    fallback_bias = float(args.vp_bias)
    repro_bias = float(args.repro_vp_bias) if args.repro_vp_bias is not None else fallback_bias
    modern_bias = float(args.modern_vp_bias) if args.modern_vp_bias is not None else fallback_bias
    linear_bias = float(args.linear_vp_bias) if args.linear_vp_bias is not None else fallback_bias

    if args.auto_bias and not args.h_only:
        from petrology.melting.vp_bias_calibrate import auto_calibrate_track_bias

        strategy = str(args.auto_bias_strategy)
        print(
            f"# Auto Vp bias strategy={strategy}"
            + (f" @ Tp={args.ref_tp:g} chi={args.ref_chi:g} target={args.bias_target}" if strategy == "point" else "")
            + "\n"
        )
        r_rec = auto_calibrate_track_bias(
            track="repro_reebox",
            strategy=strategy,
            tp_c=float(args.ref_tp),
            chi=float(args.ref_chi),
            v_lc_obs_km_s=v_lc,
            h_obs_km=h_obs,
            b_km=b_km,
            melting_engine="reebox",
            lithology_backend="native",
            vp_mode="norm",
            target=args.bias_target,
            margin_km_s=float(args.bias_margin),
            tp_range_c=tp_range,
            tp_step_c=tp_step,
            chi_values=chi_values,
            h_tolerance_km=float(args.h_tol),
            n_isentropic_steps=int(args.isentropic_steps),
        )
        r_mod = auto_calibrate_track_bias(
            track="modern_reebox",
            strategy=strategy,
            tp_c=float(args.ref_tp),
            chi=float(args.ref_chi),
            v_lc_obs_km_s=v_lc,
            h_obs_km=h_obs,
            b_km=b_km,
            melting_engine="reebox",
            lithology_backend="pymelt",
            vp_mode="norm",
            peridotite_chemistry="pmelts_klb1",
            target=args.bias_target,
            margin_km_s=float(args.bias_margin),
            tp_range_c=tp_range,
            tp_step_c=tp_step,
            chi_values=chi_values,
            h_tolerance_km=float(args.h_tol),
            n_isentropic_steps=int(args.isentropic_steps),
        )
        repro_bias = r_rec.vp_bias_km_s
        modern_bias = r_mod.vp_bias_km_s
        if strategy == "sweep":
            print(
                f"  repro_reebox bias={repro_bias:+.2f}  n_feasible={r_rec.n_feasible}"
                f"  (point ref raw {r_rec.point_rec.vp_raw_km_s:.3f})"
            )
            print(
                f"  modern_reebox bias={modern_bias:+.2f}  n_feasible={r_mod.n_feasible}"
                f"  (point ref raw {r_mod.point_rec.vp_raw_km_s:.3f})"
            )
        else:
            pr, pm = r_rec.point_rec, r_mod.point_rec
            print(f"  repro_reebox bias={repro_bias:+.3f}  (raw {pr.vp_raw_km_s:.3f} -> {pr.vp_calibrated_km_s:.3f})")
            print(f"  modern_reebox bias={modern_bias:+.3f}  (raw {pm.vp_raw_km_s:.3f} -> {pm.vp_calibrated_km_s:.3f})")
        if args.include_linear:
            r_lin = auto_calibrate_track_bias(
                track="repro_linear",
                strategy=strategy,
                tp_c=float(args.ref_tp),
                chi=float(args.ref_chi),
                v_lc_obs_km_s=v_lc,
                h_obs_km=h_obs,
                b_km=b_km,
                melting_engine="kinzler_linear",
                lithology_backend="native",
                vp_mode="eq1",
                target=args.bias_target,
                margin_km_s=float(args.bias_margin),
                tp_range_c=tp_range,
                tp_step_c=tp_step,
                chi_values=chi_values,
                h_tolerance_km=float(args.h_tol),
                n_isentropic_steps=int(args.isentropic_steps),
            )
            linear_bias = r_lin.vp_bias_km_s
            pl = r_lin.point_rec
            if strategy == "sweep":
                print(f"  repro_linear bias={linear_bias:+.2f}  n_feasible={r_lin.n_feasible}  (point ref raw {pl.vp_raw_km_s:.3f})")
            else:
                print(f"  repro_linear bias={linear_bias:+.3f}  (raw {pl.vp_raw_km_s:.3f} -> {pl.vp_calibrated_km_s:.3f})")
        print()

    scan_kw = dict(
        v_lc=v_lc,
        h_obs=h_obs,
        b_km=b_km,
        tp_range=tp_range,
        tp_step=tp_step,
        chi_values=chi_values,
        n_isentropic_steps=int(args.isentropic_steps),
        h_tolerance=float(args.h_tol),
        require_bulk_in_bounds=require_vp,
        delta_vp_engine=str(args.delta_vp_engine),
    )

    repro = _run_scan(
        "repro_reebox",
        "REEBOX native Katz + Kinzler + norm Vp",
        melting_engine="reebox",
        lithology_backend="native",
        vp_mode="norm",
        peridotite_chemistry=None,
        vp_bias=repro_bias,
        **scan_kw,
    )
    modern = _run_scan(
        "modern_reebox",
        "REEBOX pyMelt Katz + pMELTS + norm Vp",
        melting_engine="reebox",
        lithology_backend="pymelt",
        vp_mode="norm",
        peridotite_chemistry="pmelts_klb1",
        vp_bias=modern_bias,
        **scan_kw,
    )

    tracks = [repro, modern]
    if args.include_linear:
        linear = _run_scan(
            "repro_linear",
            "KKHS02 linear F(P) + eq.(1) Vp",
            melting_engine="kinzler_linear",
            lithology_backend="native",
            vp_mode="eq1",
            peridotite_chemistry=None,
            vp_bias=linear_bias,
            **scan_kw,
        )
        tracks.insert(0, linear)

    for tr in tracks:
        _print_track_summary(tr)
    _print_delta(repro, modern)

    if args.plot:
        _plot_comparison(tracks, repro=repro, modern=modern, out=args.o)


if __name__ == "__main__":
    main()
