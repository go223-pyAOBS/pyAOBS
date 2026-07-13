"""Smoke test: Modern melting column + V_LC bounding (with diagnostics)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.melting import forward_lip_column, forward_melting_column


def _print_lip(lip, label: str) -> None:
    c = lip.column
    print(f"\n=== {label} ===")
    print(f"Tp={c.tp_c:.0f}C  chi={c.chi}  phi_pyr={c.pyroxenite_frac}  b={c.b_km:.0f} km")
    print(f"P0={c.p0_gpa:.2f} Pf={c.pf_gpa:.2f} GPa  Fbar={c.fbar:.3f}  H={c.h_km:.1f} km")
    print(f"Vp bulk={lip.vp_bulk_used_km_s:.3f}  eq1={c.vp_bulk_eq1_km_s:.3f} km/s")
    m = c.pooled_melt_wt
    print(f"Melt SiO2={m['SiO2']:.1f} MgO={m['MgO']:.1f} CaO={m['CaO']:.1f} wt%")
    print(f"V_LC obs={lip.v_lc_obs_km_s:.3f}  bulk bounds=[{lip.v_bulk_lower_km_s:.3f}, {lip.v_bulk_upper_km_s:.3f}]")
    print(f"ΔVp(param)={lip.delta_vp_fc_km_s:.3f}  V_LC theory=bulk+ΔVp={lip.v_lc_theory_km_s:.3f}")
    print(f"bulk in bounds? {lip.bulk_in_bounds}  (V_LC−V_bulk={lip.vlc_match_km_s:+.3f} km/s)")
    print(f"eq.(1) grid points in bounds: {lip.n_feasible_pf}")


def main() -> None:
    print("--- Demo A: OJP-like (high Tp, moderate chi) ---")
    col = forward_melting_column(
        tp_c=1350.0,
        b_km=20.0,
        chi=4.0,
        pyroxenite_frac=0.10,
        compute_norm_vp=True,
        mineral_backend="auto",
    )
    print("=== forward_melting_column ===")
    print(f"Tp={col.tp_c:.0f}C  chi={col.chi}  phi_pyr={col.pyroxenite_frac}")
    print(f"P0={col.p0_gpa:.2f} Pf={col.pf_gpa:.2f} GPa  Fbar={col.fbar:.3f}  H={col.h_km:.1f} km")
    print(f"Vp eq1={col.vp_bulk_eq1_km_s:.3f}  Vp norm={col.vp_bulk_norm_km_s:.3f} km/s")
    m = col.pooled_melt_wt
    print(f"Melt SiO2={m['SiO2']:.1f} MgO={m['MgO']:.1f} CaO={m['CaO']:.1f} wt%")

    lip_a = forward_lip_column(
        v_lc_obs_km_s=7.0,
        tp_c=1350.0,
        b_km=20.0,
        chi=4.0,
        pyroxenite_frac=0.10,
        vp_bias_km_s=-0.10,
    )
    _print_lip(lip_a, "forward_lip_column Demo A (V_LC=7.0)")

    print("\n--- Demo B: thinner lid (b=0), stronger active upwelling (chi=8) ---")
    lip_b = forward_lip_column(
        v_lc_obs_km_s=7.0,
        tp_c=1300.0,
        b_km=0.0,
        chi=8.0,
        pyroxenite_frac=0.10,
        vp_bias_km_s=-0.10,
    )
    _print_lip(lip_b, "forward_lip_column Demo B")

    print("\n--- Interpretation ---")
    print(
        "KKHS02 bounding: V_bulk must be <= V_LC (upper bound). "
        "If Vp_bulk > V_LC, in_bounds=False — reject (Tp, chi) or raise V_LC."
    )
    print(
        "V_LC theory = V_bulk + ΔVp is the predicted lower-crust velocity after FC; "
        "compare to V_LC obs for end-to-end fit."
    )
    if not lip_a.bulk_in_bounds:
        print(
            f"Demo A: Vp_bulk exceeds V_LC by ~{lip_a.vp_bulk_used_km_s - lip_a.v_lc_obs_km_s:.2f} km/s "
            f"(primary melt too fast for V_LC=7.0)."
        )

    print("\n--- H-Vp scan smoke (small grid) ---")
    from petrology.melting.hvp_scan import scan_hvp_lip

    scan = scan_hvp_lip(
        v_lc_obs_km_s=7.0,
        h_obs_km=30.0,
        b_km=0.0,
        tp_range_c=(1300, 1400),
        tp_step_c=50.0,
        chi_values=[8.0, 12.0, 16.0],
    )
    print(f"Scan grid={len(scan.points)} feasible={scan.n_feasible}")


if __name__ == "__main__":
    main()
