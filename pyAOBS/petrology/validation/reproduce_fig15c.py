"""
Reproduce KKHS02 Fig.15c — Greenland transect anchor (H + V_LC vs Tp, χ).

For the full observation→overlay pipeline (P–T correction, harmonic mean, MC windows
on Fig.12a), use ``reproduce_fig15_transect.py``.

Default track: kinzler_linear + wl1990 ΔVp (langmuir + sb1994_fig2ol), H-only
feasibility (|H−H_obs| ≤ tol). Step-2 horizontal band is an **interpretation aid** only
(original Fig.15c plots V_LC directly on Fig.12a).

  py -3.11 petrology/validation/reproduce_fig15c.py
  py -3.11 petrology/validation/reproduce_fig15c.py --melting-engine reebox --refine-norm 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.fc.delta_vp import FIG5_P_EVAL_MPA, FIG5_T_EVAL_C, R3_DELTA_VP_WL_KW
from petrology.fc.wl1990 import load_kinzler1997_morb_primary
from petrology.invert import bulk_vp_bounds_from_fractionation
from petrology.melting.hvp_scan import HvpScanResult, scan_hvp_lip
from petrology.melting.pymelt_lithology_adapter import add_lithology_cli, lithology_kwargs_from_args, print_lithology_catalog
from petrology.norm_velocity import norm_velocity_from_bulk_wt

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
WL_KW = dict(R3_DELTA_VP_WL_KW)


def _step2_bounds(
    v_lc: float,
    *,
    f_lower: float,
    f_solid: float,
    p_fc_mpa: float,
) -> tuple[float, float, float]:
    melt = load_kinzler1997_morb_primary()["oxides_wt_percent"]
    bulk = float(
        norm_velocity_from_bulk_wt(
            melt,
            p_pa=FIG5_P_EVAL_MPA * 1e6,
            t_k=FIG5_T_EVAL_C + 273.15,
            mineral_backend=WL_KW["mineral_backend"],
        )["vp_km_s"]
    )
    bnd = bulk_vp_bounds_from_fractionation(
        v_lc,
        f_lower=f_lower,
        f_solid=f_solid,
        p_fc_mpa=p_fc_mpa,
        bulk_vp_guess_km_s=bulk,
        delta_vp_engine="wl1990",
        melt_oxides_wt=melt,
        delta_vp_wl_kw=WL_KW,
    )
    return float(bnd.v_bulk_lower_km_s), float(bnd.v_bulk_upper_km_s), float(bnd.delta_vp_fc_km_s)


def _paper_band_points(result: HvpScanResult, *, h_tol: float) -> list:
    return [
        p
        for p in result.feasible
        if p.chi >= 8.0 and 1250.0 <= p.tp_c <= 1450.0 and abs(p.h_match_km) <= h_tol
    ]


def run(
    *,
    v_lc_obs_km_s: float = 7.0,
    h_obs_km: float = 30.0,
    b_km: float = 0.0,
    pyroxenite_frac: float = 0.0,
    f_lower: float = 0.5,
    f_solid: float = 0.75,
    p_fc_mpa: float = 400.0,
    tp_range_c: tuple[float, float] = (1250.0, 1450.0),
    tp_step_c: float = 10.0,
    chi_values: tuple[float, ...] = (4.0, 6.0, 8.0, 10.0, 12.0, 16.0),
    h_tolerance_km: float = 3.0,
    melting_engine: str = "kinzler_linear",
    require_bulk_in_bounds: bool = False,
    refine_norm_vp: int = 0,
    vp_bias_km_s: float = -0.10,
    n_isentropic_steps: int = 28,
    save_figure: Path | None = None,
    export_csv: Path | None = None,
    show: bool = False,
    lithology_kwargs: dict | None = None,
) -> dict:
    lith_kw = lithology_kwargs or {}
    result = scan_hvp_lip(
        v_lc_obs_km_s=v_lc_obs_km_s,
        h_obs_km=h_obs_km,
        b_km=b_km,
        pyroxenite_frac=pyroxenite_frac,
        f_lower=f_lower,
        f_solid=f_solid,
        p_fc_mpa=p_fc_mpa,
        tp_range_c=tp_range_c,
        tp_step_c=tp_step_c,
        chi_values=list(chi_values),
        h_tolerance_km=h_tolerance_km,
        melting_engine=melting_engine,
        delta_vp_engine="wl1990",
        delta_vp_wl_kw=WL_KW,
        require_bulk_in_bounds=require_bulk_in_bounds,
        refine_norm_vp=refine_norm_vp,
        vp_bias_km_s=vp_bias_km_s,
        n_isentropic_steps=n_isentropic_steps,
        verbose=True,
        **lith_kw,
    )

    v_lo, v_hi, d_vp = _step2_bounds(
        v_lc_obs_km_s, f_lower=f_lower, f_solid=f_solid, p_fc_mpa=p_fc_mpa
    )
    paper = _paper_band_points(result, h_tol=h_tolerance_km)

    summary = {
        "v_lc_obs_km_s": v_lc_obs_km_s,
        "h_obs_km": h_obs_km,
        "n_grid": len(result.points),
        "n_feasible": result.n_feasible,
        "n_paper_band": len(paper),
        "step2_v_bulk_lower_km_s": v_lo,
        "step2_v_bulk_upper_km_s": v_hi,
        "delta_vp_fc_km_s": d_vp,
        "melting_engine": melting_engine,
        "tp_chi_ranges": result.tp_chi_ranges(),
    }

    print(f"\nFig.15c anchor: H={h_obs_km:.0f} km, V_LC={v_lc_obs_km_s:.2f} km/s")
    print(f"Step-2 bounds: V_bulk ∈ [{v_lo:.3f}, {v_hi:.3f}] km/s  (ΔVp={d_vp:.3f})")
    print(f"Grid {summary['n_grid']} pts — feasible {summary['n_feasible']}, paper band (χ≥8) {summary['n_paper_band']}")
    if paper:
        best = min(paper, key=lambda p: abs(p.h_match_km))
        print(
            f"Best H-match: Tp={best.tp_c:.0f}°C χ={best.chi:g} "
            f"H={best.h_km:.1f} km Vp={best.vp_bulk_km_s:.3f} in_bounds={best.bulk_in_bounds}"
        )
    rng = summary["tp_chi_ranges"]
    if rng["tp_c"]:
        print(f"Feasible Tp: {rng['tp_c'][0]:.0f}–{rng['tp_c'][1]:.0f}°C  χ: {rng['chi'][0]:.1f}–{rng['chi'][1]:.1f}")

    if export_csv:
        import csv

        export_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "tp_c", "chi", "h_km", "vp_bulk_km_s", "v_lc_theory_km_s",
            "bulk_in_bounds", "h_match_km", "feasible",
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

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))

    ax = axes[0]
    cmap = plt.cm.plasma
    chi_uniq = sorted({p.chi for p in result.points})
    for i, chi in enumerate(chi_uniq):
        line = sorted([p for p in result.points if p.chi == chi], key=lambda x: x.tp_c)
        if len(line) < 2:
            continue
        ax.plot(
            [p.h_km for p in line],
            [p.vp_bulk_km_s for p in line],
            lw=1.2,
            color=cmap(i / max(len(chi_uniq) - 1, 1)),
            label=f"χ={chi:g}",
        )

    ax.axhspan(v_lo, v_hi, color="#3498db", alpha=0.12, label=f"Step-2 [{v_lo:.2f},{v_hi:.2f}]")
    ax.axhline(v_lc_obs_km_s, color="0.5", ls=":", lw=0.9)
    ax.axvline(h_obs_km, color="0.5", ls="--", lw=0.9)
    if result.feasible:
        ax.scatter(
            [p.h_km for p in result.feasible],
            [p.vp_bulk_km_s for p in result.feasible],
            s=32,
            c="crimson",
            zorder=5,
            label=f"feasible ({result.n_feasible})",
        )
    if paper:
        ax.scatter(
            [p.h_km for p in paper],
            [p.vp_bulk_km_s for p in paper],
            s=48,
            facecolors="none",
            edgecolors="k",
            linewidths=1.2,
            zorder=6,
            label=f"χ≥8 ({len(paper)})",
        )
    ax.scatter([h_obs_km], [v_lc_obs_km_s], s=70, c="k", marker="*", zorder=7, label="obs")
    ax.set_xlabel("Igneous thickness H (km)")
    ax.set_ylabel("Bulk Vp (km/s)")
    ax.set_title("(a) H–Vp vs Step-2 bounds")
    ax.legend(fontsize=7, loc="best")

    ax2 = axes[1]
    if result.points:
        tp_u = sorted({p.tp_c for p in result.points})
        chi_u = sorted({p.chi for p in result.points})
        grid = np.full((len(chi_u), len(tp_u)), np.nan)
        for p in result.points:
            val = 1.0 if p.feasible else 0.0
            if not p.feasible and abs(p.h_match_km) <= h_tolerance_km:
                val = 0.5
            grid[chi_u.index(p.chi), tp_u.index(p.tp_c)] = val
        im = ax2.imshow(
            grid,
            aspect="auto",
            origin="lower",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            extent=[tp_u[0] - 5, tp_u[-1] + 5, chi_u[0] - 0.5, chi_u[-1] + 0.5],
        )
        fig.colorbar(im, ax=ax2, label="feasible / H-match")
    ax2.axhline(8.0, color="k", ls="--", lw=0.8, alpha=0.6)
    ax2.text(tp_range_c[0] + 5, 8.3, "χ=8 (paper)", fontsize=8)
    ax2.set_xlabel("Tp (°C)")
    ax2.set_ylabel("χ")
    ax2.set_title(f"(b) Tp–χ feasibility (n={result.n_feasible})")

    fig.suptitle(
        f"KKHS02 Fig.15c — Greenland anchor (engine={melting_engine}, wl1990 ΔVp)",
        fontsize=11,
    )
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
    parser = argparse.ArgumentParser(description="Reproduce KKHS02 Fig.15c (Greenland anchor)")
    parser.add_argument("--v-lc", type=float, default=7.0)
    parser.add_argument("--h-km", type=float, default=30.0)
    parser.add_argument("--b-km", type=float, default=0.0)
    parser.add_argument("--pyroxenite-frac", type=float, default=0.0)
    parser.add_argument("--f-lower", type=float, default=0.5)
    parser.add_argument("--h-tol", type=float, default=3.0)
    parser.add_argument("--tp-min", type=float, default=1250.0)
    parser.add_argument("--tp-max", type=float, default=1450.0)
    parser.add_argument("--tp-step", type=float, default=10.0)
    parser.add_argument("--chi", type=str, default="4,6,8,10,12,16")
    parser.add_argument(
        "--melting-engine",
        choices=("kinzler_linear", "reebox"),
        default="kinzler_linear",
    )
    parser.add_argument(
        "--require-bulk-bounds",
        action="store_true",
        help="Also require V_bulk in Step-2 interval (often empty with eq.1 Vp)",
    )
    parser.add_argument(
        "--bulk-bounds",
        action="store_true",
        help="REEBOX + norm Vp refine track (greenland_kg1, H+Vp joint closure)",
    )
    parser.add_argument("--refine-norm", type=int, default=0)
    parser.add_argument("--vp-bias", type=float, default=-0.10)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=FIG_DIR / "fig15c_greenland.png",
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    add_lithology_cli(parser)
    args = parser.parse_args()
    if args.list_lithologies:
        print_lithology_catalog()
        return
    chi_vals = tuple(float(x) for x in args.chi.split(",") if x.strip())
    lith_kw = lithology_kwargs_from_args(args)
    if args.bulk_bounds:
        from petrology.melting.pymelt_lithology_adapter import resolve_lithology_col_kwargs

        lith_kw = resolve_lithology_col_kwargs(lithology_preset="greenland_kg1")
        run(
            v_lc_obs_km_s=args.v_lc,
            h_obs_km=args.h_km,
            b_km=args.b_km,
            pyroxenite_frac=0.10 if args.pyroxenite_frac == 0.0 else args.pyroxenite_frac,
            f_lower=args.f_lower,
            tp_range_c=(args.tp_min, args.tp_max),
            tp_step_c=max(args.tp_step, 25.0),
            chi_values=chi_vals or (8.0, 10.0, 12.0, 16.0),
            h_tolerance_km=args.h_tol,
            melting_engine="reebox",
            require_bulk_in_bounds=True,
            refine_norm_vp=max(args.refine_norm, 12),
            vp_bias_km_s=0.0,
            save_figure=args.output.with_name("fig15c_greenland_bulk_bounds.png"),
            export_csv=args.csv,
            show=args.show,
            lithology_kwargs=lith_kw,
        )
        return
    refine = args.refine_norm if args.melting_engine == "reebox" else 0
    run(
        v_lc_obs_km_s=args.v_lc,
        h_obs_km=args.h_km,
        b_km=args.b_km,
        pyroxenite_frac=args.pyroxenite_frac,
        f_lower=args.f_lower,
        tp_range_c=(args.tp_min, args.tp_max),
        tp_step_c=args.tp_step,
        chi_values=chi_vals,
        h_tolerance_km=args.h_tol,
        melting_engine=args.melting_engine,
        require_bulk_in_bounds=args.require_bulk_bounds,
        refine_norm_vp=refine,
        vp_bias_km_s=args.vp_bias,
        save_figure=args.output,
        export_csv=args.csv,
        show=args.show,
        lithology_kwargs=lith_kw,
    )


if __name__ == "__main__":
    main()
