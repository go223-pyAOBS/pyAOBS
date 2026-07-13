"""
Regression: ARHENF dual-track must not be mixed.

Three forms coexist in this repo — consolidating them incorrectly changes Kd
by ~10× and saturation indices Q from O(1) to O(100):

1. **basalt1990** — original BASALT.FOR::

       ARHENF(A,B,T) = 10**(A/T) + B

   Production: ``petrology.fc.basalt1990.kd_calc.arhenf_1990``.

2. **modern** — ``basalt_modern.f90`` / most article exploratory scripts::

       ARHENF(A,B,T) = 10**(A/T + B)   (floor at 1e-6)

   Matches Langmuir Arrhenius-in-exponent at P=0.

3. **langmuir1992** — pressure extension::

       ARHENF(A,B,C,T,P) = 10**(A/T + B + C*P)   (P in kbar)

   Production: ``petrology.fc.wl_kd.arhenf_kd``.

Usage::

    py -3.11 petrology/validation/test_kd_arhenf_forms.py
    py -3.11 -m pytest petrology/validation/test_kd_arhenf_forms.py -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.fc.basalt1990.kd_calc import arhenf_1990, kd_olivine_mgo_1990
from petrology.fc.wl_kd import arhenf_kd, kd_olivine_mgo

# Olivine MgO coefficients shared by 1990 1-atm and modern forms
_OL_MGO_A = 2715.0
_OL_MGO_B = -1.158
# Langmuir 1992 olivine MgO (different A,B; pressure term)
_OL_MGO_92_A = 3740.0
_OL_MGO_92_B = -1.87

# Kinzler 1997 model components (mole-based) — used by check_saturation.py
_KINZLER_CS = (0.0816, 0.0494, 0.2049, 0.0737, 0.0527, 0.0078, 0.5300)

# data_original case 1 (1-based Fortran layout, index 0 unused)
_CS_1BASED = (0.0, 0.15, 0.25, 0.08, 0.12, 0.10, 0.02, 0.28)

_T_K = 1473.16  # 1200 °C
_T_DEFAULT = 1273.16  # BASALT.FOR default


def _load_kd_legacy():
    """Load article kd_legacy without requiring package install of article/."""
    path = _ROOT / "petrology" / "article" / "复现korenaga" / "kd_legacy.py"
    spec = importlib.util.spec_from_file_location("kd_legacy_article", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_basalt1990_formula_identity():
    """Production arhenf_1990 == 10**(A/T) + B."""
    val = arhenf_1990(_OL_MGO_A, _OL_MGO_B, _T_DEFAULT)
    expected = 10.0 ** (_OL_MGO_A / _T_DEFAULT) + _OL_MGO_B
    assert abs(val - expected) < 1e-9
    assert val > 50.0  # must NOT equal modern ~11.7


def test_modern_equals_langmuir_at_p0():
    """modern 10**(A/T+B) == Langmuir arhenf_kd(..., C=0, P=0) for same A,B."""
    modern = 10.0 ** (_OL_MGO_A / _T_K + _OL_MGO_B)
    lang = arhenf_kd(_OL_MGO_A, _OL_MGO_B, 0.0, _T_K, 0.0)
    assert abs(modern - lang) < 1e-12


def test_forms_differ_by_order_of_magnitude():
    """Mixing 1990 and modern changes Ol MgO Kd by ~10×."""
    k90 = arhenf_1990(_OL_MGO_A, _OL_MGO_B, _T_K)
    k_mod = 10.0 ** (_OL_MGO_A / _T_K + _OL_MGO_B)
    ratio = k90 / k_mod
    assert 10.0 < ratio < 20.0, f"expected ~14×, got {ratio}"


def test_kd_legacy_dispatch_matches_production():
    kd = _load_kd_legacy()
    assert abs(kd.arhenf_basalt1990(_OL_MGO_A, _OL_MGO_B, _T_K) - arhenf_1990(_OL_MGO_A, _OL_MGO_B, _T_K)) < 1e-12
    assert abs(kd.arhenf_modern(_OL_MGO_A, _OL_MGO_B, _T_K) - arhenf_kd(_OL_MGO_A, _OL_MGO_B, 0.0, _T_K, 0.0)) < 1e-12
    # Default form is modern
    assert abs(kd.arhenf(_OL_MGO_A, _OL_MGO_B, _T_K) - kd.arhenf_modern(_OL_MGO_A, _OL_MGO_B, _T_K)) < 1e-12
    assert abs(
        kd.arhenf(_OL_MGO_A, _OL_MGO_B, _T_K, form="basalt1990")
        - kd.arhenf_basalt1990(_OL_MGO_A, _OL_MGO_B, _T_K)
    ) < 1e-12


def test_kinzler_q_magnitude_by_form():
    """modern Q is O(1); basalt1990 Q is huge — guards against silent form swap."""
    kd = _load_kd_legacy()
    q_mod, an = kd.q_values(_T_K, _KINZLER_CS, form="modern")
    q_90, an90 = kd.q_values(_T_K, _KINZLER_CS, form="basalt1990")
    assert abs(an - an90) < 1e-12
    assert all(abs(x) < 5.0 for x in q_mod), f"modern Q out of range: {q_mod}"
    assert all(abs(x) > 10.0 for x in q_90), f"1990 Q unexpectedly small: {q_90}"


def test_kdcalc_fortran_form_basalt1990():
    kd = _load_kd_legacy()
    mat = kd.kdcalc_fortran(_T_K, kdmode=3, form="basalt1990")
    assert abs(mat[2][3] - arhenf_1990(_OL_MGO_A, _OL_MGO_B, _T_K)) < 1e-12
    assert abs(mat[3][4] - 0.24 * mat[3][3]) < 1e-12


def test_kdcalc_fortran_form_modern():
    kd = _load_kd_legacy()
    mat = kd.kdcalc_fortran(_T_K, kdmode=3, form="modern")
    expected = kd.arhenf_modern(_OL_MGO_A, _OL_MGO_B, _T_K)
    assert abs(mat[2][3] - expected) < 1e-12


def test_q_at_fa0_fortran_modern_reference():
    """Spot-check vs original debug_liquidus inline 10**(A/T+B) at 2000 K."""
    kd = _load_kd_legacy()
    qp, qo, qc = kd.q_at_fa0_fortran(2000.0, _CS_1BASED, form="modern")
    # Reference values from corrected modern form (O(1), negative = undersaturated)
    assert abs(qp - (-0.6558)) < 0.01
    assert abs(qo - (-0.5116)) < 0.01
    assert abs(qc - (-0.7533)) < 0.01


def test_production_ol_mgo_1990_vs_langmuir92():
    """Production engines remain on distinct tracks (compare_kd_three scope)."""
    k90 = kd_olivine_mgo_1990(_T_DEFAULT)
    k92 = kd_olivine_mgo(_T_DEFAULT, 0.0)
    # 1990 uses A=2715,B=-1.158 with +B outside exponent → large
    # 1992 uses A=3740,B=-1.87 with B inside exponent → ~11.7
    assert k90 > 50.0
    assert 8.0 < k92 < 16.0
    assert abs(k92 - arhenf_kd(_OL_MGO_92_A, _OL_MGO_92_B, 0.0, _T_DEFAULT, 0.0)) < 1e-9


def test_langmuir_increases_with_pressure():
    k0 = kd_olivine_mgo(_T_DEFAULT, 0.0)
    k8 = kd_olivine_mgo(_T_DEFAULT, 8.0)
    assert k8 > k0


def main() -> None:
    tests = [
        test_basalt1990_formula_identity,
        test_modern_equals_langmuir_at_p0,
        test_forms_differ_by_order_of_magnitude,
        test_kd_legacy_dispatch_matches_production,
        test_kinzler_q_magnitude_by_form,
        test_kdcalc_fortran_form_basalt1990,
        test_kdcalc_fortran_form_modern,
        test_q_at_fa0_fortran_modern_reference,
        test_production_ol_mgo_1990_vs_langmuir92,
        test_langmuir_increases_with_pressure,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  OK  {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    if failed:
        raise SystemExit(f"{failed}/{len(tests)} tests failed")
    print(f"All {len(tests)} ARHENF dual-track tests passed.")


if __name__ == "__main__":
    main()
