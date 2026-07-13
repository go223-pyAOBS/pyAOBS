"""
Sweep pyroxenite fraction Φ for H + V_LC closure (Greenland-like anchor).

Example:
  python petrology/validation/sweep_phi_lip.py --v-lc 7.0 --h-km 30 --melting-engine reebox
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.melting.lip_scan import scan_hvp_lip_phi
from petrology.melting.pymelt_lithology_adapter import add_lithology_cli, lithology_kwargs_from_args, print_lithology_catalog


def run(
    *,
    v_lc_obs_km_s: float,
    h_obs_km: float,
    b_km: float = 0.0,
    phi_values: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25),
    tp_range_c: tuple[float, float] = (1250.0, 1450.0),
    tp_step_c: float | None = None,
    chi_values: tuple[float, ...] = (4.0, 6.0, 8.0, 10.0, 12.0, 16.0),
    h_tolerance_km: float = 3.0,
    melting_engine: str = "reebox",
    use_norm_vp: bool = False,
    vp_bias_km_s: float = -0.10,
    n_isentropic_steps: int = 32,
    verbose: bool = True,
    refine_norm_vp: int = 12,
    save_figure: Path | None = None,
    show: bool = False,
    lithology_backend: str = "native",
    peridotite_lith: str = "katz_lherzolite",
    pyroxenite_lith: str = "pertermann_g2",
    peridotite_h2o_wt: float = 0.0,
    pyroxenite_h2o_wt: float = 0.0,
    lithology_preset: str | None = None,
) -> dict:
    if tp_step_c is None:
        tp_step_c = 25.0 if melting_engine == "reebox" else 10.0
    scan = scan_hvp_lip_phi(
        v_lc_obs_km_s=v_lc_obs_km_s,
        h_obs_km=h_obs_km,
        b_km=b_km,
        phi_values=list(phi_values),
        tp_range_c=tp_range_c,
        tp_step_c=tp_step_c,
        chi_values=list(chi_values),
        h_tolerance_km=h_tolerance_km,
        melting_engine=melting_engine,
        use_norm_vp=use_norm_vp,
        vp_bias_km_s=vp_bias_km_s,
        n_isentropic_steps=n_isentropic_steps,
        verbose=verbose,
        refine_norm_vp=refine_norm_vp if melting_engine == "reebox" and not use_norm_vp else 0,
        lithology_backend=lithology_backend,
        peridotite_lith=peridotite_lith,
        pyroxenite_lith=pyroxenite_lith,
        peridotite_h2o_wt=peridotite_h2o_wt,
        pyroxenite_h2o_wt=pyroxenite_h2o_wt,
        lithology_preset=lithology_preset,
    )

    vp_note = "norm" if use_norm_vp else "melt_proxy+refine"
    print(
        f"Observation: H={h_obs_km:.1f} km, V_LC={v_lc_obs_km_s:.3f} km/s, "
        f"engine={melting_engine}, vp={vp_note}",
        flush=True,
    )
    print(f"{'Phi':>6} {'n_feas':>7} {'best Tp':>8} {'chi':>6} {'H':>6} {'Vp':>7} {'excess':>7} {'dH':>5} {'':>4}")
    summary_slices = []
    for sl in scan.slices:
        n = sl.result.n_feasible
        pt = sl.best_pareto
        if pt is None:
            print(f"{sl.pyroxenite_frac:6.2f} {n:7d}   —")
            continue
        excess = max(0.0, pt.vp_bulk_km_s - v_lc_obs_km_s)
        dh = pt.h_match_km
        in_b = "B" if pt.bulk_in_bounds else " "
        mark = "Y" if pt.feasible else in_b
        print(
            f"{sl.pyroxenite_frac:6.2f} {n:7d} {pt.tp_c:8.0f} {pt.chi:6.1f} "
            f"{pt.h_km:6.1f} {pt.vp_bulk_km_s:7.3f} {excess:+7.3f} {dh:+5.1f} {mark:>4}"
        )
        summary_slices.append(
            {
                "phi": sl.pyroxenite_frac,
                "n_feasible": n,
                "best_tp": pt.tp_c,
                "best_chi": pt.chi,
                "best_h": pt.h_km,
                "best_vp": pt.vp_bulk_km_s,
                "vp_excess": excess,
                "feasible": pt.feasible,
            }
        )

    print(f"\nTotal feasible across Phi: {scan.n_feasible_total}")
    best = scan.best_overall()
    if best:
        sl, pt = best
        print(
            f"Pareto best: Phi={sl.pyroxenite_frac:.2f} Tp={pt.tp_c:.0f} chi={pt.chi:.1f} "
            f"H={pt.h_km:.1f} Vp={pt.vp_bulk_km_s:.3f} feasible={pt.feasible}"
        )
    else:
        print("No grid points produced.")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable ({exc})")
        return {"slices": summary_slices, "n_feasible_total": scan.n_feasible_total}

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))

    ax = axes[0]
    phis = [s["phi"] for s in summary_slices]
    excess = [s["vp_excess"] for s in summary_slices]
    ax.plot(phis, excess, "o-", lw=1.5)
    ax.axhline(0, color="0.5", ls="--", lw=0.8)
    ax.set_xlabel("Pyroxenite fraction Φ")
    ax.set_ylabel("Vp excess above V_LC (km/s)")
    ax.set_title(f"(a) Min Vp excess vs Φ ({melting_engine})")

    ax2 = axes[1]
    for sl in scan.slices:
        if not sl.result.points:
            continue
        near_h = [p for p in sl.result.points if abs(p.h_km - h_obs_km) <= h_tolerance_km * 2]
        if not near_h:
            near_h = sl.result.points
        best_h = min(near_h, key=lambda p: abs(p.h_km - h_obs_km))
        ax2.scatter(
            sl.pyroxenite_frac,
            best_h.vp_bulk_km_s,
            s=40,
            c="crimson" if best_h.bulk_in_bounds else "0.6",
        )
    ax2.axhline(v_lc_obs_km_s, color="k", ls=":", lw=0.9, label=f"V_LC={v_lc_obs_km_s:.2f}")
    ax2.set_xlabel("Pyroxenite fraction Φ")
    ax2.set_ylabel("Vp_bulk at H≈obs (km/s)")
    ax2.set_title("(b) Vp near target H")
    ax2.legend(fontsize=8)

    fig.suptitle(f"Φ sweep: H={h_obs_km:.0f} km, V_LC={v_lc_obs_km_s:.2f} km/s", fontsize=11)
    fig.tight_layout()

    if save_figure:
        save_figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_figure, dpi=150)
        print(f"Saved {save_figure}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"slices": summary_slices, "n_feasible_total": scan.n_feasible_total}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep pyroxenite Φ for H + V_LC")
    parser.add_argument("--v-lc", type=float, default=7.0)
    parser.add_argument("--h-km", type=float, default=30.0)
    parser.add_argument("--b-km", type=float, default=0.0)
    parser.add_argument("--phi", type=str, default="0,0.05,0.1,0.15,0.2,0.25")
    parser.add_argument("--tp-min", type=float, default=1250.0)
    parser.add_argument("--tp-max", type=float, default=1450.0)
    parser.add_argument("--tp-step", type=float, default=None)
    parser.add_argument("--chi", type=str, default="4,6,8,10,12,16")
    parser.add_argument("--h-tol", type=float, default=3.0)
    parser.add_argument("--melting-engine", choices=("kinzler_linear", "reebox"), default="reebox")
    parser.add_argument("--norm-vp", action="store_true", help="Full-grid BurnMan (very slow)")
    parser.add_argument("--no-refine", action="store_true", help="Skip top-N norm Vp refine")
    parser.add_argument("--isentropic-steps", type=int, default=32)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--vp-bias", type=float, default=-0.10)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures" / "sweep_phi_lip.png",
    )
    parser.add_argument("--show", action="store_true")
    add_lithology_cli(parser)
    args = parser.parse_args()
    if args.list_lithologies:
        print_lithology_catalog()
        return
    phi_vals = tuple(float(x) for x in args.phi.split(",") if x.strip())
    chi_vals = tuple(float(x) for x in args.chi.split(",") if x.strip())
    norm_vp: bool = True if args.norm_vp else False
    refine = 0 if args.no_refine or args.norm_vp else 12
    lith_kw = lithology_kwargs_from_args(args)
    run(
        v_lc_obs_km_s=args.v_lc,
        h_obs_km=args.h_km,
        b_km=args.b_km,
        phi_values=phi_vals,
        tp_range_c=(args.tp_min, args.tp_max),
        tp_step_c=args.tp_step,
        chi_values=chi_vals,
        h_tolerance_km=args.h_tol,
        melting_engine=args.melting_engine,
        use_norm_vp=norm_vp,
        vp_bias_km_s=args.vp_bias,
        n_isentropic_steps=args.isentropic_steps,
        verbose=not args.quiet,
        refine_norm_vp=refine,
        save_figure=args.output,
        show=args.show,
        **lith_kw,
    )


if __name__ == "__main__":
    main()
