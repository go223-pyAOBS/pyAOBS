"""Bulk F must be Σ φ_i F_i so heterogeneous H stays physical."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.melting.lithology import dry_peridotite_lithology, heterogeneous_source
from petrology.melting.reebox_geometry import build_reebox_column


def test_het_bulk_f_below_one() -> None:
    liths = heterogeneous_source(pyroxenite_frac=0.1)
    col = build_reebox_column(liths, tp_c=1350.0, b_km=0.0, chi=1.0, n_isentropic_steps=40)
    last = col.path.steps[-1]
    f_bulk = col.path.f_bulk_at(last)
    f_sum = float(sum(last.f_by_name.values()))
    assert f_bulk < 1.0, f"bulk F={f_bulk:.3f} must be < 1"
    assert f_bulk < f_sum - 0.2, f"bulk F={f_bulk:.3f} should be << raw sum {f_sum:.3f}"


def test_het_h_near_pure_order() -> None:
    """10% G2 thickens crust vs pure peridotite, but not by ~4× via F-sum bug."""
    pure = [dry_peridotite_lithology(u0=1.0)]
    het = heterogeneous_source(pyroxenite_frac=0.1)
    h_pure = build_reebox_column(pure, tp_c=1350.0, b_km=0.0, chi=1.0, n_isentropic_steps=40).h_km
    h_het = build_reebox_column(het, tp_c=1350.0, b_km=0.0, chi=1.0, n_isentropic_steps=40).h_km
    assert 10.0 < h_pure < 30.0, f"pure H={h_pure:.1f} km unexpected"
    assert h_het > h_pure, f"het H={h_het:.1f} should exceed pure {h_pure:.1f}"
    assert h_het < 2.5 * h_pure, f"het H={h_het:.1f} vs pure {h_pure:.1f}: still inflated?"


def test_high_tp_chi_fbar_physical_and_h_reduced() -> None:
    """After mass-weight + Pf–H: Fbar<1, Pf deep, path truncated (not floor 0.05)."""
    liths = heterogeneous_source(pyroxenite_frac=0.1)
    col = build_reebox_column(liths, tp_c=1600.0, b_km=0.0, chi=16.0, n_isentropic_steps=28)
    assert col.path.fbar() < 1.0
    assert col.path.f_total < 1.0
    assert col.pf_gpa > 1.0
    assert col.path.steps[-1].p_gpa > 1.0


def main() -> None:
    test_het_bulk_f_below_one()
    test_het_h_near_pure_order()
    test_high_tp_chi_fbar_physical_and_h_reduced()
    print("test_bulk_f_mass_weight: OK")


if __name__ == "__main__":
    main()
