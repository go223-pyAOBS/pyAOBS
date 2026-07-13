"""Pf–H self-consistency: P_f ≈ (H + b) / 30 and path ends at Pf."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.active_upwelling import pf_from_geometry
from petrology.melting.lithology import dry_peridotite_lithology, heterogeneous_source
from petrology.melting.reebox_geometry import build_reebox_column


def _check_case(liths, *, tp_c: float, chi: float, b_km: float = 0.0) -> None:
    col = build_reebox_column(
        liths, tp_c=tp_c, b_km=b_km, chi=chi, n_isentropic_steps=40, pf_h_consistent=True
    )
    pf_eq7 = pf_from_geometry(col.h_km, b_km)
    assert abs(col.pf_gpa - pf_eq7) < 0.05, (
        f"Tp={tp_c} χ={chi}: Pf={col.pf_gpa:.3f} vs (H+b)/30={pf_eq7:.3f} (H={col.h_km:.1f})"
    )
    assert abs(col.path.pf_gpa - col.pf_gpa) < 0.05, (
        f"path.pf={col.path.pf_gpa:.3f} != col.pf={col.pf_gpa:.3f}"
    )
    assert col.path.steps[-1].p_gpa >= col.pf_gpa - 0.05
    assert col.pf_gpa > 0.2, f"Pf={col.pf_gpa:.3f} still near free-surface floor"


def test_pure_passive() -> None:
    _check_case([dry_peridotite_lithology(u0=1.0)], tp_c=1350.0, chi=1.0)


def test_het_passive_and_active() -> None:
    liths = heterogeneous_source(pyroxenite_frac=0.1)
    _check_case(liths, tp_c=1350.0, chi=1.0)
    _check_case(liths, tp_c=1350.0, chi=4.0)
    _check_case(liths, tp_c=1500.0, chi=8.0)


def test_extreme_not_floor_pf() -> None:
    liths = heterogeneous_source(pyroxenite_frac=0.1)
    col = build_reebox_column(
        liths, tp_c=1600.0, b_km=0.0, chi=16.0, n_isentropic_steps=28, pf_h_consistent=True
    )
    assert col.pf_gpa > 1.0, f"extreme case Pf={col.pf_gpa:.2f} should be deep"
    assert abs(col.pf_gpa - pf_from_geometry(col.h_km, 0.0)) < 0.08
    # Path must stop at Pf (no free-surface chemistry); H itself may still be large at χ=16.
    assert abs(col.path.pf_gpa - col.pf_gpa) < 0.05
    assert col.path.steps[-1].p_gpa > 1.0


def test_legacy_floor_mode() -> None:
    liths = heterogeneous_source(pyroxenite_frac=0.1)
    col = build_reebox_column(
        liths, tp_c=1350.0, b_km=0.0, chi=1.0, n_isentropic_steps=40, pf_h_consistent=False
    )
    assert col.pf_gpa < 0.15
    assert not col.pf_h_consistent


def main() -> None:
    test_pure_passive()
    test_het_passive_and_active()
    test_extreme_not_floor_pf()
    test_legacy_floor_mode()
    print("test_pf_h_consistent: OK")


if __name__ == "__main__":
    main()
