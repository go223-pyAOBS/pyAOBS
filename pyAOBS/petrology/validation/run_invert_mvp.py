"""Run Step-2 bounding inversion MVP and optional feasible (P,F) plot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.fc.delta_vp import FIG5_P_EVAL_MPA, FIG5_T_EVAL_C, R3_DELTA_VP_WL_KW
from petrology.fc.wl1990 import load_kinzler1997_morb_primary
from petrology.invert import (
    bulk_vp_bounds_from_fractionation,
    feasible_pf_region,
)
from petrology.norm_velocity import norm_velocity_from_bulk_wt


def _resolve_melt_and_bulk(
    *,
    use_kinzler_melt: bool,
    bulk_vp_guess_km_s: float | None,
    mineral_backend: str,
) -> tuple[dict | None, float]:
    melt: dict | None = None
    bulk = bulk_vp_guess_km_s
    if use_kinzler_melt:
        melt = load_kinzler1997_morb_primary()["oxides_wt_percent"]
        bulk = float(
            norm_velocity_from_bulk_wt(
                melt,
                p_pa=FIG5_P_EVAL_MPA * 1e6,
                t_k=FIG5_T_EVAL_C + 273.15,
                mineral_backend=mineral_backend,
            )["vp_km_s"]
        )
    elif bulk is None:
        bulk = 7.2
    return melt, float(bulk)


def run(
    *,
    v_lc_obs_km_s: float,
    f_lower: float,
    f_solid: float,
    p_fc_mpa: float,
    bulk_vp_guess_km_s: float | None,
    delta_vp_engine: str,
    use_kinzler_melt: bool,
    kd_engine: str,
    mineral_backend: str,
    vp_bias_km_s: float,
    vp_tolerance_km_s: float,
    save_figure: Path | None,
    show: bool,
) -> dict:
    melt, bulk_guess = _resolve_melt_and_bulk(
        use_kinzler_melt=use_kinzler_melt,
        bulk_vp_guess_km_s=bulk_vp_guess_km_s,
        mineral_backend=mineral_backend,
    )
    wl_kw = {"kd_engine": kd_engine, "mineral_backend": mineral_backend}
    bounds = bulk_vp_bounds_from_fractionation(
        v_lc_obs_km_s,
        f_lower=f_lower,
        f_solid=f_solid,
        p_fc_mpa=p_fc_mpa,
        bulk_vp_guess_km_s=bulk_guess,
        delta_vp_engine=delta_vp_engine,
        melt_oxides_wt=melt,
        delta_vp_wl_kw=wl_kw,
    )
    reg = feasible_pf_region(
        bounds,
        vp_bias_km_s=vp_bias_km_s,
        vp_tolerance_km_s=vp_tolerance_km_s,
    )

    eng_label = delta_vp_engine
    print(f"V_LC_obs: {v_lc_obs_km_s:.3f} km/s")
    print(f"ΔVp_fc ({eng_label}): {bounds.delta_vp_fc_km_s:.3f} km/s")
    print(f"f_lower: {bounds.f_lower:.2f}")
    print(
        f"V_bulk bounds: {bounds.v_bulk_lower_km_s:.3f} – "
        f"{bounds.v_bulk_upper_km_s:.3f} km/s"
    )
    print(
        f"eq.(1) scan Vp range (with bias): "
        f"{reg['vp_model_range_km_s'][0]:.3f} – {reg['vp_model_range_km_s'][1]:.3f} km/s"
    )
    if vp_bias_km_s != 0 or vp_tolerance_km_s != 0:
        print(f"Applied bias/tolerance: {vp_bias_km_s:+.3f} / ±{vp_tolerance_km_s:.3f} km/s")
    print(f"Feasible (P,F) points: {reg['n_feasible']}")
    if reg["n_feasible"] > 0:
        print(
            f"P range: {reg['p_range_gpa'][0]:.2f} – {reg['p_range_gpa'][1]:.2f} GPa, "
            f"F range: {reg['f_range'][0]:.3f} – {reg['f_range'][1]:.3f}"
        )
        print(f"Mean P,F: {reg['p_mean_gpa']:.2f} GPa, {reg['f_mean']:.3f}")

    if reg["n_feasible"] == 0:
        if "nearest_point" in reg:
            p, f, vp = reg["nearest_point"]
            print(
                "Nearest model point: "
                f"P={p:.2f} GPa, F={f:.3f}, Vp={vp:.3f} km/s "
                f"(distance {reg['nearest_distance_km_s']:.3f})"
            )
            print(
                "Hint: try --vp-bias -0.10 or --vp-tol 0.10, "
                "or increase V_LC / decrease f_lower."
            )
        return {"bounds": bounds, "region": reg}

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable ({exc}) — skipping figure")
        return {"bounds": bounds, "region": reg}

    pts = reg["points"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=10, cmap="viridis")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Vp_bulk from eq.(1) (km/s)")
    ax.set_xlabel("Pressure of melting P (GPa)")
    ax.set_ylabel("Melt fraction F")
    ax.set_title(f"Step-2 feasible (P,F) region ({eng_label})")
    fig.tight_layout()

    if save_figure:
        save_figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_figure, dpi=150)
        print(f"Saved {save_figure}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return {"bounds": bounds, "region": reg}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step-2 invert MVP")
    parser.add_argument("--v-lc", type=float, default=7.00, help="Observed lower-crust Vp (km/s)")
    parser.add_argument("--f-lower", type=float, default=0.50, help="Lower-crust volume fraction")
    parser.add_argument("--f-solid", type=float, default=0.75, help="FC solid fraction for ΔVp envelope")
    parser.add_argument("--p-fc-mpa", type=float, default=400.0, help="FC pressure (MPa)")
    parser.add_argument(
        "--delta-vp-engine",
        choices=("param", "wl1990", "auto"),
        default="wl1990",
        help="ΔVp source (default wl1990 + Kinzler melt)",
    )
    parser.add_argument(
        "--no-kinzler-melt",
        action="store_true",
        help="Skip Kinzler primary melt for wl1990 bounds (uses --bulk-vp-guess)",
    )
    parser.add_argument(
        "--bulk-vp-guess",
        type=float,
        default=None,
        help="Bulk Vp guess when not using Kinzler melt (km/s)",
    )
    parser.add_argument(
        "--kd-engine",
        default=R3_DELTA_VP_WL_KW["kd_engine"],
        help="Kd engine for wl1990 ΔVp",
    )
    parser.add_argument(
        "--mineral-backend",
        default=R3_DELTA_VP_WL_KW["mineral_backend"],
        help="Mineral backend for wl1990 ΔVp / norm Vp",
    )
    parser.add_argument(
        "--vp-bias",
        type=float,
        default=-0.10,
        help="Additive bias to eq.(1) Vp (km/s), default -0.10",
    )
    parser.add_argument(
        "--vp-tol",
        type=float,
        default=0.05,
        help="Tolerance half-width for bounds matching (km/s), default 0.05",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures" / "invert_mvp_feasible_pf.png",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    use_kinzler = not args.no_kinzler_melt and args.delta_vp_engine in ("wl1990", "auto")
    run(
        v_lc_obs_km_s=args.v_lc,
        f_lower=args.f_lower,
        f_solid=args.f_solid,
        p_fc_mpa=args.p_fc_mpa,
        bulk_vp_guess_km_s=args.bulk_vp_guess,
        delta_vp_engine=args.delta_vp_engine,
        use_kinzler_melt=use_kinzler,
        kd_engine=args.kd_engine,
        mineral_backend=args.mineral_backend,
        vp_bias_km_s=args.vp_bias,
        vp_tolerance_km_s=args.vp_tol,
        save_figure=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
