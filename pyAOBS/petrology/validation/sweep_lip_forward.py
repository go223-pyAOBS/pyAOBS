"""Scan (Tp, chi, phi) for V_LC bounding feasibility."""

from __future__ import annotations

from petrology.melting import forward_lip_column


def main() -> None:
    v_lc = 7.0
    b_km = 20.0
    print(f"V_LC_obs = {v_lc} km/s, b = {b_km} km")
    print("(* = bulk Vp inside [V_LC - f_lower*ΔVp, V_LC])")
    print(f"{'':1s}{'Tp':>5} {'chi':>5} {'phi':>5} {'Vp_bulk':>8} {'lo':>6} {'hi':>6} {'H':>6} {'OK':>4}")
    hits = []
    for tp in (1280, 1320, 1350, 1380):
        for chi in (1.5, 2.5, 4.0, 6.0, 8.0):
            for phi in (0.0, 0.05, 0.10, 0.15):
                lip = forward_lip_column(
                    v_lc_obs_km_s=v_lc,
                    tp_c=float(tp),
                    b_km=b_km,
                    chi=float(chi),
                    pyroxenite_frac=float(phi),
                    use_norm_vp=True,
                )
                mark = "*" if lip.bulk_in_bounds else " "
                if lip.bulk_in_bounds:
                    hits.append(lip)
                print(
                    f"{mark}{tp:5d} {chi:5.1f} {phi:5.2f} "
                    f"{lip.vp_bulk_used_km_s:8.3f} {lip.v_bulk_lower_km_s:6.3f} {lip.v_bulk_upper_km_s:6.3f} "
                    f"{lip.column.h_km:6.1f} {lip.bulk_in_bounds!s:>4}"
                )
    print(f"\nFeasible combinations: {len(hits)}")
    if hits:
        best = min(hits, key=lambda x: abs(x.vp_bulk_used_km_s - 0.5 * (x.v_bulk_lower_km_s + x.v_bulk_upper_km_s)))
        c = best.column
        print(
            f"Example match: Tp={c.tp_c:.0f} chi={c.chi} phi={c.pyroxenite_frac} "
            f"Vp={best.vp_bulk_used_km_s:.3f} H={c.h_km:.1f} km"
        )


if __name__ == "__main__":
    main()
