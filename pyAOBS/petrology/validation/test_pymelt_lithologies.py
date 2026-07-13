"""
Smoke test: pyMelt lithologies wired into REEBOX-core isentropic forward.

  python petrology/validation/test_pymelt_lithologies.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _forward_summary(label: str, col) -> None:
    names = [L.name for L in col.lithologies]
    print(
        f"  [{label}] lith={names}  P0={col.p0_gpa:.2f} Pf={col.pf_gpa:.2f} "
        f"H={col.h_km:.1f} Fmax={col.f_max:.3f} SiO2={col.pooled_melt_wt['SiO2']:.1f}"
    )


def main() -> None:
    from petrology.melting.lithology import katz2003_peridotite_solidus
    from petrology.melting.pymelt_lithology_adapter import (
        list_lithology_presets,
        list_pymelt_lithology_keys,
        lithology_diagnostics,
        print_lithology_catalog,
        pymelt_lithology,
        resolve_lithology_col_kwargs,
    )
    from petrology.melting.heterogeneous import forward_heterogeneous_column

    print_lithology_catalog()

    native_tsol = katz2003_peridotite_solidus(2.0)
    pm_katz = pymelt_lithology("katz_lherzolite")
    print(f"\nKatz Tsol @2GPa: native={native_tsol:.1f}C  pymelt={pm_katz.solidus_gpa(2.0):.1f}C")

    kw = dict(tp_c=1350.0, b_km=0.0, chi=4.0, pyroxenite_frac=0.10, n_isentropic_steps=48)

    print("\n" + "=" * 60)
    print("REEBOX forward (native vs pymelt katz+g2)")
    print("=" * 60)
    nat = forward_heterogeneous_column(**kw, lithology_backend="native")
    pml = forward_heterogeneous_column(
        **kw,
        **resolve_lithology_col_kwargs(lithology_preset="lip_default"),
    )
    _forward_summary("native", nat)
    _forward_summary("pymelt", pml)

    print("\n" + "=" * 60)
    print("Ball primitive mantle + G2 (preset lip_ball)")
    print("=" * 60)
    ball = forward_heterogeneous_column(
        **kw,
        **resolve_lithology_col_kwargs(lithology_preset="lip_ball"),
    )
    _forward_summary("lip_ball", ball)

    print("\n" + "=" * 60)
    print("Enriched end-member: matthews_kg1")
    print("=" * 60)
    kg1 = forward_heterogeneous_column(
        **kw,
        lithology_backend="pymelt",
        peridotite_lith="katz_lherzolite",
        pyroxenite_lith="matthews_kg1",
    )
    print(
        f"  P0={kg1.p0_gpa:.2f} Pf={kg1.pf_gpa:.2f} H={kg1.h_km:.1f} "
        f"Fmax={kg1.f_max:.3f} TiO2={kg1.pooled_melt_wt.get('TiO2', 0):.2f}"
    )

    print("\n" + "=" * 60)
    print("Hydrous KLB-1 preset (lip_klb1_hydrous)")
    print("=" * 60)
    hyd = forward_heterogeneous_column(
        **kw,
        **resolve_lithology_col_kwargs(lithology_preset="lip_klb1_hydrous"),
    )
    d = lithology_diagnostics("matthews_klb1", p_gpa=2.0, h2o_wt=0.1)
    print(f"  KLB-1+0.1H2O Tsol@2GPa={d['tsol_c']:.1f}C")
    _forward_summary("hydrous", hyd)

    print(f"\nRegistry: {len(list_pymelt_lithology_keys())} keys, {len(list_lithology_presets())} presets")


if __name__ == "__main__":
    main()
