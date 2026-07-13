"""Eq.(1) b1 print-errata: production uses -0.55; printed +0.55 kept for comparison."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.vp_regression import (
    eq1_with_printed_coefficients,
    load_eq1,
    predict_v_bulk_km_s,
)


def test_production_b1_is_negative():
    c = load_eq1()["equation_1"]["coefficients"]
    assert c["b1"] == -0.55


def test_printed_errata_b1_positive():
    err = load_eq1()["equation_1"]["print_errata"]
    assert err["printed_b1"] == 0.55
    assert err["corrected_b1"] == -0.55
    printed = eq1_with_printed_coefficients()
    assert printed["equation_1"]["coefficients"]["b1"] == 0.55


def test_morb_matches_fig3_with_corrected_b1():
    v = predict_v_bulk_km_s(1.0, 0.1)
    assert abs(v - 7.1) < 0.02
    v_printed = predict_v_bulk_km_s(1.0, 0.1, eq1=eq1_with_printed_coefficients())
    assert v_printed > v + 0.10


if __name__ == "__main__":
    test_production_b1_is_negative()
    test_printed_errata_b1_positive()
    test_morb_matches_fig3_with_corrected_b1()
    print("ok")
