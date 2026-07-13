"""Smoke test: REEBOX core (isentropic heterogeneous + RMC) vs legacy Kinzler track."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.melting import forward_heterogeneous_column, forward_melting_column


def _print_col(label: str, col) -> None:
    m = col.pooled_melt_wt
    engine = getattr(col, "melting_engine", "reebox")
    print(f"\n=== {label} ({engine}) ===")
    print(f"Tp={col.tp_c:.0f}C chi={col.chi} phi={col.pyroxenite_frac:.2f} H={col.h_km:.1f} km")
    print(f"P0={col.p0_gpa:.2f} Pf={col.pf_gpa:.2f} Fbar={col.fbar:.3f} Fmax={col.f_max:.3f}")
    print(f"SiO2={m['SiO2']:.1f} MgO={m['MgO']:.1f} CaO={m['CaO']:.1f} TiO2={m.get('TiO2', 0):.2f}")
    print(f"Vp eq1={col.vp_bulk_eq1_km_s:.3f} km/s")


def main() -> None:
    tp, chi, phi, b = 1350.0, 4.0, 0.10, 20.0

    legacy = forward_melting_column(
        tp_c=tp, b_km=b, chi=chi, pyroxenite_frac=phi, melting_engine="kinzler_linear"
    )
    reebox = forward_melting_column(
        tp_c=tp, b_km=b, chi=chi, pyroxenite_frac=phi, melting_engine="reebox"
    )
    het = forward_heterogeneous_column(tp_c=tp, b_km=b, chi=chi, pyroxenite_frac=phi)

    _print_col("Legacy Kinzler + linear mix", legacy)
    _print_col("REEBOX engine via column.py", reebox)
    _print_col("Direct heterogeneous forward", het)

    print("\n--- Melt flux by lithology (REEBOX) ---")
    for name, flux in het.melt_flux_by_lithology.items():
        print(f"  {name}: {flux:.4f}")

    print("\n--- Isentropic path (first/last 3 steps) ---")
    steps = het.isentropic_path.steps
    for st in steps[:3] + steps[-3:]:
        fs = ", ".join(f"{k}={v:.3f}" for k, v in st.f_by_name.items())
        print(f"  P={st.p_gpa:.2f} GPa T={st.t_c:.0f}C  F: {fs}")

    # Pyroxenite should contribute more melt flux at same Tp when phi>0
    pyr_flux = het.melt_flux_by_lithology.get("g2_pyroxenite", 0.0)
    per_flux = het.melt_flux_by_lithology.get("peridotite", 0.0)
    print(f"\nPyroxenite/peridotite flux ratio: {pyr_flux / max(per_flux, 1e-9):.2f} (phi={phi})")


if __name__ == "__main__":
    main()
