"""
Acceptance for KKHS02 eq.(1) — norm-based V_bulk(P_bar, F_bar).

Production coefficients use corrected b1 = -0.55 (printed table had +0.55).

Usage::

    py -3.11 petrology/validation/check_eq1.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.vp_regression import (
    branch_poly,
    eq1_with_printed_coefficients,
    in_eq1_applicable_range,
    load_eq1,
    predict_v_bulk_km_s,
    window_w_h,
    window_w_l,
    window_weights,
)

# Production (sign-corrected) constants
PROD_A0 = 7.52
PROD_B = (-1.73, -0.55, 7.71, -0.11, 8.87, -146.11)
PROD_C = (-0.35, 0.034, 0.51, 0.0016, -0.04, 0.046)
# Printed table (errata) — kept for comparison only
PRINTED_B1 = 0.55
PAPER_ALPHA = 0.6
PAPER_BETA = 8.4
PAPER_PT = 1.0
PAPER_FT = 0.05
MORB_P = 1.0
MORB_F = 0.1
MORB_V_REF = 7.096  # corrected eq.(1); matches paper Fig.3 ~7.1 km/s


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""

    def line(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        return f"  [{mark}] {self.name}" + (f" — {self.detail}" if self.detail else "")


def run() -> int:
    eq1 = load_eq1()
    c = eq1["equation_1"]["coefficients"]
    w = eq1["equation_1"]["window_functions"]
    errata = eq1["equation_1"]["print_errata"]
    checks: list[Check] = []

    checks.append(Check("a0", c["a0"] == PROD_A0, f"a0={c['a0']}"))
    for i, (exp_b, exp_c) in enumerate(zip(PROD_B, PROD_C)):
        got_b = c[f"b{i}"]
        got_c = c[f"c{i}"]
        checks.append(
            Check(
                f"b{i}, c{i} coefficients (production)",
                abs(got_b - exp_b) < 1e-12 and abs(got_c - exp_c) < 1e-12,
                f"b{i}={got_b}, c{i}={got_c}",
            )
        )

    checks.append(
        Check(
            "print_errata retains printed b1=+0.55",
            abs(float(errata["printed_b1"]) - PRINTED_B1) < 1e-12
            and abs(float(errata["corrected_b1"]) - PROD_B[1]) < 1e-12
            and abs(float(errata["printed_coefficients_for_comparison"]["b1"]) - PRINTED_B1)
            < 1e-12,
            f"printed_b1={errata['printed_b1']}, corrected_b1={errata['corrected_b1']}",
        )
    )

    checks.append(
        Check(
            "window alpha, beta, Pt, Ft",
            w["alpha"] == PAPER_ALPHA
            and w["beta"] == PAPER_BETA
            and w["P_t_GPa"] == PAPER_PT
            and w["F_t"] == PAPER_FT,
        )
    )

    wl, wh = window_weights(MORB_P, MORB_F, eq1=eq1)
    checks.append(
        Check(
            "W_L + W_H at MORB",
            0.0 < wl + wh <= 1.0 + 1e-12,
            f"W_L={wl:.4f}, W_H={wh:.4f}, sum={wl+wh:.4f}",
        )
    )
    checks.append(
        Check(
            "window_w_l / window_w_h API",
            abs(window_w_l(MORB_P, MORB_F) - wl) < 1e-15
            and abs(window_w_h(MORB_P, MORB_F) - wh) < 1e-15,
        )
    )

    v_morb = predict_v_bulk_km_s(MORB_P, MORB_F, eq1=eq1)
    checks.append(
        Check(
            "MORB anchor P=1 GPa, F=0.1 (corrected)",
            abs(v_morb - MORB_V_REF) < 1e-3,
            f"V_bulk={v_morb:.4f} km/s (Fig.3 ~7.1)",
        )
    )

    v_printed = predict_v_bulk_km_s(
        MORB_P, MORB_F, eq1=eq1_with_printed_coefficients(eq1)
    )
    checks.append(
        Check(
            "printed b1=+0.55 overpredicts MORB vs Fig.3",
            v_printed > v_morb + 0.10,
            f"printed={v_printed:.4f}, corrected={v_morb:.4f}",
        )
    )

    # Hand-evaluate at MORB
    pb = branch_poly(MORB_P, MORB_F, c, "b")
    pc = branch_poly(MORB_P, MORB_F, c, "c")
    v_hand = PROD_A0 + wl * pb + wh * pc
    checks.append(
        Check(
            "hand evaluation vs predict_v_bulk",
            abs(v_hand - v_morb) < 1e-12,
            f"d={abs(v_hand - v_morb):.2e}",
        )
    )

    ar = eq1["equation_1"]["applicable_range"]
    checks.append(
        Check(
            "applicable range helper",
            in_eq1_applicable_range(2.0, 0.1)
            and not in_eq1_applicable_range(0.5, 0.1)
            and in_eq1_applicable_range(ar["P_GPa"][0], ar["F"][0]),
            f"P in {ar['P_GPa']}, F in {ar['F']}",
        )
    )

    print("=== KKHS02 eq.(1) V_bulk(P_bar, F_bar) acceptance ===")
    print("  production: b1=-0.55 (print errata: table listed +0.55)")
    print("  ref: 600 MPa, 400 C | W_L low-P/low-F, W_H high-P/high-F")
    for chk in checks:
        print(chk.line())

    failed = [x for x in checks if not x.ok]
    if failed:
        print(f"\neq.(1): {len(failed)} check(s) failed.")
        return 1
    print(f"\neq.(1): all {len(checks)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
