"""
Scan (Tp, chi) feasible region for observed H + V_LC (Modern LIP track).

Example — Greenland-like anchor (KKHS02 Fig.15c discussion):
  H ~ 30 km, V_LC ~ 7.0 km/s, b = 0, chi > 8, Tp ~ 1300 C
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.melting.hvp_scan import scan_hvp_lip
from petrology.melting.pymelt_lithology_adapter import add_lithology_cli, lithology_kwargs_from_args, print_lithology_catalog
from petrology.fc.delta_vp import R3_DELTA_VP_WL_KW


def run(
    *,
    v_lc_obs_km_s: float,
    h_obs_km: float,
    b_km: float = 0.0,
    pyroxenite_frac: float = 0.0,
    tp_range_c: tuple[float, float] = (1200.0, 1500.0),
    tp_step_c: float = 10.0,
    chi_values: tuple[float, ...] = (1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0),
    h_tolerance_km: float = 3.0,
    vlc_tolerance_km_s: float | None = None,
    require_vlc_match: bool = False,
    use_norm_vp: bool = False,
    melting_engine: str = "kinzler_linear",
    vp_bias_km_s: float = -0.10,
    n_isentropic_steps: int = 32,
    verbose: bool = True,
    refine_norm_vp: int = 0,
    delta_vp_engine: str = "wl1990",
    delta_vp_wl_kw: dict | None = None,
    require_bulk_in_bounds: bool = True,
    save_figure: Path | None = None,
    export_csv: Path | None = None,
    show: bool = False,
    lithology_backend: str = "native",
    peridotite_lith: str = "katz_lherzolite",
    pyroxenite_lith: str = "pertermann_g2",
    peridotite_h2o_wt: float = 0.0,
    pyroxenite_h2o_wt: float = 0.0,
    lithology_preset: str | None = None,
) -> dict:
    result = scan_hvp_lip(
        v_lc_obs_km_s=v_lc_obs_km_s,
        h_obs_km=h_obs_km,
        b_km=b_km,
        pyroxenite_frac=pyroxenite_frac,
        tp_range_c=tp_range_c,
        tp_step_c=tp_step_c,
        chi_values=list(chi_values),
        h_tolerance_km=h_tolerance_km,
        vlc_tolerance_km_s=vlc_tolerance_km_s,
        require_vlc_match=require_vlc_match,
        use_norm_vp=use_norm_vp,
        melting_engine=melting_engine,
        vp_bias_km_s=vp_bias_km_s,
        n_isentropic_steps=n_isentropic_steps,
        verbose=verbose,
        refine_norm_vp=refine_norm_vp,
        delta_vp_engine=delta_vp_engine,
        delta_vp_wl_kw=delta_vp_wl_kw,
        require_bulk_in_bounds=require_bulk_in_bounds,
        lithology_backend=lithology_backend,
        peridotite_lith=peridotite_lith,
        pyroxenite_lith=pyroxenite_lith,
        peridotite_h2o_wt=peridotite_h2o_wt,
        pyroxenite_h2o_wt=pyroxenite_h2o_wt,
        lithology_preset=lithology_preset,
    )

    summary = {
        "v_lc_obs_km_s": v_lc_obs_km_s,
        "h_obs_km": h_obs_km,
        "b_km": b_km,
        "pyroxenite_frac": pyroxenite_frac,
        "n_grid": len(result.points),
        "n_feasible": result.n_feasible,
        "tp_chi_ranges": result.tp_chi_ranges(),
        "melting_engine": melting_engine,
    }

    print(f"Observation: H={h_obs_km:.1f} km, V_LC={v_lc_obs_km_s:.3f} km/s, b={b_km:.0f} km")
    print(f"Grid points: {summary['n_grid']}, feasible: {summary['n_feasible']}")
    rng = summary["tp_chi_ranges"]
    if rng["tp_c"]:
        print(f"Feasible Tp: {rng['tp_c'][0]:.0f} – {rng['tp_c'][1]:.0f} C")
        print(f"Feasible chi: {rng['chi'][0]:.1f} – {rng['chi'][1]:.1f}")
    else:
        print("No feasible (Tp, chi) — relax tolerances or adjust V_LC/H.")
        # Diagnostic: closest by H, and bulk-in-bounds subset
        if result.points:
            by_h = min(result.points, key=lambda p: abs(p.h_match_km))
            in_b = [p for p in result.points if p.bulk_in_bounds]
            print(
                f"Closest H: Tp={by_h.tp_c:.0f} chi={by_h.chi:.1f} "
                f"H={by_h.h_km:.1f} Vp={by_h.vp_bulk_km_s:.3f} in_bounds={by_h.bulk_in_bounds}"
            )
            if in_b:
                bi = min(in_b, key=lambda p: abs(p.h_match_km))
                print(
                    f"Best in-bounds: Tp={bi.tp_c:.0f} chi={bi.chi:.1f} "
                    f"H={bi.h_km:.1f} dH={bi.h_match_km:+.1f} Vp={bi.vp_bulk_km_s:.3f}"
                )

    if result.feasible:
        best = min(result.feasible, key=lambda p: p.h_match_km ** 2 + (10 * p.vlc_match_km_s) ** 2)
        print(
            f"Best fit: Tp={best.tp_c:.0f}C chi={best.chi:.1f} "
            f"H={best.h_km:.1f} Vp_bulk={best.vp_bulk_km_s:.3f} "
            f"V_LC_th={best.v_lc_theory_km_s:.3f} Fbar={best.fbar:.3f}"
        )

    if export_csv:
        import csv

        export_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "tp_c", "chi", "pyroxenite_frac", "h_km", "fbar", "pbar_gpa",
            "vp_bulk_km_s", "v_lc_theory_km_s", "bulk_in_bounds", "h_match_km",
            "vlc_match_km_s", "feasible",
        ]
        with export_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for p in result.points:
                w.writerow({k: getattr(p, k) for k in fields})
        print(f"Exported {len(result.points)} rows to {export_csv}")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable ({exc})")
        return summary

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))

    # Panel A: H–Vp_bulk curves (chi families)
    ax = axes[0]
    cmap = plt.cm.plasma
    chi_uniq = sorted({p.chi for p in result.points})
    for i, chi in enumerate(chi_uniq):
        line = sorted([p for p in result.points if p.chi == chi], key=lambda x: x.tp_c)
        if len(line) < 2:
            continue
        h = [p.h_km for p in line]
        v = [p.vp_bulk_km_s for p in line]
        ax.plot(h, v, lw=1.2, color=cmap(i / max(len(chi_uniq) - 1, 1)), label=f"χ={chi:g}")

    ax.axhline(v_lc_obs_km_s, color="0.5", ls=":", lw=0.9, label=f"V_LC obs={v_lc_obs_km_s:.2f}")
    ax.axvline(h_obs_km, color="0.5", ls="--", lw=0.9, label=f"H obs={h_obs_km:.0f}")
    if result.feasible:
        fh = [p.h_km for p in result.feasible]
        fv = [p.vp_bulk_km_s for p in result.feasible]
        ax.scatter(fh, fv, s=28, c="crimson", zorder=5, label=f"feasible ({result.n_feasible})")
    ax.scatter([h_obs_km], [v_lc_obs_km_s], s=60, c="k", marker="*", zorder=6, label="obs")
    ax.set_xlabel("Igneous thickness H (km)")
    ax.set_ylabel("Bulk Vp (km/s)")
    ax.set_title("(a) H–Vp forward curves")
    ax.legend(fontsize=7, loc="best")

    # Panel B: Tp–chi feasible mask
    ax2 = axes[1]
    if result.points:
        tp_u = sorted({p.tp_c for p in result.points})
        chi_u = sorted({p.chi for p in result.points})
        grid = np.full((len(chi_u), len(tp_u)), np.nan)
        for p in result.points:
            it = tp_u.index(p.tp_c)
            ic = chi_u.index(p.chi)
            grid[ic, it] = 1.0 if p.feasible else 0.0
        im = ax2.imshow(
            grid,
            aspect="auto",
            origin="lower",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            extent=[tp_u[0] - 5, tp_u[-1] + 5, chi_u[0] - 0.5, chi_u[-1] + 0.5],
        )
        fig.colorbar(im, ax=ax2, label="feasible")
    ax2.set_xlabel("Tp (C)")
    ax2.set_ylabel("chi")
    ax2.set_title(f"(b) Feasible region (n={result.n_feasible})")

    fig.suptitle("Modern LIP: scan (Tp, chi) vs H + V_LC", fontsize=11)
    fig.tight_layout()

    if save_figure:
        save_figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_figure, dpi=150)
        print(f"Saved {save_figure}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan feasible (Tp, chi) for H + V_LC")
    parser.add_argument("--v-lc", type=float, default=7.0, help="Observed lower-crust Vp (km/s)")
    parser.add_argument("--h-km", type=float, default=30.0, help="Observed igneous thickness (km)")
    parser.add_argument("--b-km", type=float, default=0.0)
    parser.add_argument("--pyroxenite-frac", type=float, default=0.0)
    parser.add_argument("--tp-min", type=float, default=1200.0)
    parser.add_argument("--tp-max", type=float, default=1500.0)
    parser.add_argument("--tp-step", type=float, default=None, help="Default 25 (reebox) or 10 (legacy)")
    parser.add_argument("--chi", type=str, default="1,2,4,6,8,10,12,16")
    parser.add_argument("--h-tol", type=float, default=3.0)
    parser.add_argument("--vlc-tol", type=float, default=None, help="Optional V_LC_theory match (km/s)")
    parser.add_argument("--require-vlc-match", action="store_true")
    parser.add_argument("--melting-engine", choices=("kinzler_linear", "reebox"), default="kinzler_linear")
    parser.add_argument("--norm-vp", action="store_true", help="Full-grid CIPW+BurnMan (very slow)")
    parser.add_argument("--no-norm-vp", action="store_true", help="Force eq.(1) Vp")
    parser.add_argument("--refine-norm", type=int, default=12, help="Refine top-N with norm Vp (0=off)")
    parser.add_argument("--isentropic-steps", type=int, default=32)
    parser.add_argument(
        "--delta-vp-engine",
        choices=("param", "wl1990", "auto"),
        default="wl1990",
        help="ΔVp engine for Step-2 bounds (default wl1990)",
    )
    parser.add_argument(
        "--kd-engine",
        default=R3_DELTA_VP_WL_KW["kd_engine"],
        help="Kd engine when delta-vp-engine=wl1990",
    )
    parser.add_argument(
        "--mineral-backend",
        default=R3_DELTA_VP_WL_KW["mineral_backend"],
        help="Mineral backend when delta-vp-engine=wl1990",
    )
    parser.add_argument(
        "--h-only",
        action="store_true",
        help="Feasible on H match only (ignore bulk Vp bounds)",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--vp-bias", type=float, default=-0.10)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Export grid results to CSV",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures" / "scan_hvp_lip.png",
    )
    parser.add_argument("--show", action="store_true")
    add_lithology_cli(parser)
    args = parser.parse_args()
    if args.list_lithologies:
        print_lithology_catalog()
        return
    chi_vals = tuple(float(x) for x in args.chi.split(",") if x.strip())
    tp_step = args.tp_step
    if tp_step is None:
        tp_step = 25.0 if args.melting_engine == "reebox" else 10.0
    norm_vp = True if args.norm_vp else (False if args.no_norm_vp else False)
    refine = 0 if args.norm_vp else args.refine_norm
    lith_kw = lithology_kwargs_from_args(args)
    wl_kw = {"kd_engine": args.kd_engine, "mineral_backend": args.mineral_backend}
    run(
        v_lc_obs_km_s=args.v_lc,
        h_obs_km=args.h_km,
        b_km=args.b_km,
        pyroxenite_frac=args.pyroxenite_frac,
        tp_range_c=(args.tp_min, args.tp_max),
        tp_step_c=tp_step,
        chi_values=chi_vals,
        h_tolerance_km=args.h_tol,
        vlc_tolerance_km_s=args.vlc_tol,
        require_vlc_match=args.require_vlc_match,
        use_norm_vp=norm_vp,
        melting_engine=args.melting_engine,
        vp_bias_km_s=args.vp_bias,
        n_isentropic_steps=args.isentropic_steps,
        verbose=not args.quiet,
        refine_norm_vp=refine if args.melting_engine == "reebox" else 0,
        delta_vp_engine=args.delta_vp_engine,
        delta_vp_wl_kw=wl_kw,
        require_bulk_in_bounds=not args.h_only,
        save_figure=args.output,
        export_csv=args.csv,
        show=args.show,
        **lith_kw,
    )


if __name__ == "__main__":
    main()
