"""Regression: pyroxenite_frac=0 must not include ghost G2 lithology."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.melting.heterogeneous import forward_heterogeneous_column
from petrology.melting.lithology import heterogeneous_source
from petrology.melting.reebox_geometry import p0_gpa_at_solidus


def test_native_phi0_single_lithology() -> None:
    liths = heterogeneous_source(pyroxenite_frac=0.0, backend="native")
    assert len(liths) == 1
    assert liths[0].name == "peridotite"
    assert liths[0].katz_cpx_mass is not None


def test_native_phi0_p0_peridotite_solidus() -> None:
    liths = heterogeneous_source(pyroxenite_frac=0.0, backend="native")
    p0 = p0_gpa_at_solidus(1450.0, liths)
    assert 2.0 < p0 < 2.8, f"P0={p0:.3f} GPa should be peridotite-only (~2.4)"


def test_native_phi0_no_g2_melt() -> None:
    r = forward_heterogeneous_column(
        tp_c=1450.0,
        b_km=0.0,
        chi=8.0,
        pyroxenite_frac=0.0,
        lithology_backend="native",
        compute_norm_vp=False,
    )
    last = r.isentropic_path.steps[-1]
    assert "g2_pyroxenite" not in last.f_by_name
    f_total = float(sum(last.f_by_name.values()))
    assert f_total < 1.05, f"F={f_total:.3f} should not include ghost pyroxenite"


def test_pymelt_phi0_single_lithology() -> None:
    liths = heterogeneous_source(pyroxenite_frac=0.0, backend="pymelt")
    assert len(liths) == 1


def test_native_katz_matches_pymelt_isobaric() -> None:
    from petrology.melting.katz2003 import KATZ_CPX_MASS, katz2003_melt_fraction_dry
    from petrology.melting.pymelt_lithology_adapter import instantiate_pymelt_lithology

    pm = instantiate_pymelt_lithology("katz_lherzolite")
    p = 2.0
    for t in (1350.0, 1400.0, 1500.0):
        fn = katz2003_melt_fraction_dry(p, t, cpx_mass=KATZ_CPX_MASS)
        fp = float(pm.F(p, t))
        assert abs(fn - fp) < 1e-6, f"P={p} T={t}: native={fn} pymelt={fp}"


def main() -> None:
    test_native_phi0_single_lithology()
    test_native_phi0_p0_peridotite_solidus()
    test_native_phi0_no_g2_melt()
    test_pymelt_phi0_single_lithology()
    test_native_katz_matches_pymelt_isobaric()
    print("test_heterogeneous_phi0: OK")


if __name__ == "__main__":
    main()
