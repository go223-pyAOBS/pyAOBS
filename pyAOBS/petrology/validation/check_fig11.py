"""
Acceptance for KKHS02 §4 / Fig.11 active-upwelling melting model.

Checks:
  - closed-form ΔP root vs ``solve_active_upwelling`` (linear 12%/GPa)
  - qualitative trends from paper §32 (χ and b vs Tp)
  - mass–geometry balance residual
  - V_bulk = eq.(1)(P_bar, F_bar) wired through the column solver

Informational (never FAIL): Fig.11 d/h morphology vs digitized paper — see README
``Fig.11 d/h 已知差异`` and ``reproduce_fig11_dh_compare.py``.

Usage::

    py -3.11 petrology/validation/check_fig11.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.active_upwelling import (
    DPDZ_KM_PER_GPA,
    delta_p_closed_form_linear,
    h_from_eq10,
    pf_closed_form_linear,
    solve_active_upwelling,
    sweep_hvp,
)
from petrology.melting_laws import FIG11_MELTING
from petrology.vp_regression import predict_v_bulk_fig11_km_s, predict_v_bulk_km_s

DFDP = FIG11_MELTING.dfdp_per_gpa  # 12%/GPa — Fig.11 / §4 standard model
_DATA = Path(__file__).resolve().parents[1] / "data"
_DIGITIZED_D = _DATA / "ScreenShot_2026-07-04_124209_264.txt"
_DIGITIZED_H = _DATA / "ScreenShot_2026-07-04_124221_433.txt"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""

    def line(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        return f"  [{mark}] {self.name}" + (f" — {self.detail}" if self.detail else "")


def _report_dh_discrepancy(chi_pts, b_pts) -> None:
    """Informational summary: eq.(1) vs Holbrook vs digitized Fig.11 d/h."""
    print("\n=== Fig.11 d/h known discrepancy (informational — does not affect PASS/FAIL) ===")
    print("  §4 P_bar, F_bar, H validated via panels a-c/e-g; only V_bulk mapping differs from")
    print("  digitized printed curves. See README 'Fig.11 d/h 已知差异' and reproduce_fig11_dh_compare.py")

    tp_ref = 1400.0
    r1 = next(p for p in chi_pts if p.chi == 1.0)
    r8 = next(p for p in chi_pts if p.chi == 8.0)
    rb0 = next(p for p in b_pts if p.b_km == 0.0)
    rb30 = next(p for p in b_pts if p.b_km == 30.0)

    eq1_sp_chi = r8.vp_bulk_km_s - r1.vp_bulk_km_s
    hol1 = predict_v_bulk_fig11_km_s(r1.pbar_gpa, r1.fbar)
    hol8 = predict_v_bulk_fig11_km_s(r8.pbar_gpa, r8.fbar)
    hol_sp_chi = hol8 - hol1
    eq1_sp_b = rb30.vp_bulk_km_s - rb0.vp_bulk_km_s
    hol_sp_b = (
        predict_v_bulk_fig11_km_s(rb30.pbar_gpa, rb30.fbar)
        - predict_v_bulk_fig11_km_s(rb0.pbar_gpa, rb0.fbar)
    )

    print(
        f"  @ Tp={tp_ref:.0f} C, eq.(1): chi spread={eq1_sp_chi:+.3f} km/s (chi8-chi1), "
        f"b spread={eq1_sp_b:+.3f} km/s (b30-b0)"
    )
    print(f"  @ Tp={tp_ref:.0f} C, Holbrook: chi spread={hol_sp_chi:+.3f}, b spread={hol_sp_b:+.3f}")
    print(f"  @ Tp={tp_ref:.0f} C, eq.(1) V: chi=1 {r1.vp_bulk_km_s:.3f}, chi=8 {r8.vp_bulk_km_s:.3f}")
    print(f"  @ Tp={tp_ref:.0f} C, Holbrook V: chi=1 {hol1:.3f}, chi=8 {hol8:.3f}")

    if not (_DIGITIZED_D.is_file() and _DIGITIZED_H.is_file()):
        print(f"  (digitized data not found under {_DATA})")
        return

    try:
        import importlib.util

        analyze_path = Path(__file__).resolve().parent / "analyze_fig11_digitized.py"
        spec = importlib.util.spec_from_file_location("analyze_fig11_digitized", analyze_path)
        if spec is None or spec.loader is None:
            raise ImportError("cannot load analyze_fig11_digitized")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        dig_d = mod.parse_getdata(mod.PANEL_D)
        dig_h = mod.parse_getdata(mod.PANEL_H)
        dig_sp_chi = mod.interp_v(dig_d["X=1"], tp_ref) - mod.interp_v(dig_d["x=8"], tp_ref)
        dig_sp_b = mod.interp_v(dig_h["b=0"], tp_ref) - mod.interp_v(dig_h["b=30"], tp_ref)
        print(
            f"  @ Tp={tp_ref:.0f} C, digitized paper: chi spread={dig_sp_chi:.3f}, "
            f"b spread={dig_sp_b:.3f}"
        )
        for label, chi in [("X=1", 1.0), ("x=8", 8.0)]:
            pts = dig_d[label]
            tp_arr = np.array([p[0] for p in pts])
            v_obs = np.array([p[1] for p in pts])
            v_eq1 = np.empty_like(v_obs)
            v_hol = np.empty_like(v_obs)
            for i, t in enumerate(tp_arr):
                r = solve_active_upwelling(tp_c=float(t), b_km=0.0, chi=chi, dfdp_per_gpa=DFDP)
                v_eq1[i] = predict_v_bulk_km_s(r.pbar_gpa, r.fbar)
                v_hol[i] = predict_v_bulk_fig11_km_s(r.pbar_gpa, r.fbar)
            print(
                f"  digitized panel (d) {label}: RMSE eq.(1)={mod.rmse(v_obs, v_eq1):.3f}, "
                f"Holbrook={mod.rmse(v_obs, v_hol):.3f} km/s"
            )
    except Exception as exc:
        print(f"  (skip digitized RMSE: {exc})")


def _mass_geom_residual(r) -> float:
    dp = r.p0_gpa - r.pf_gpa
    h_mass = h_from_eq10(chi=r.chi, delta_p_gpa=dp, fbar=r.fbar)
    h_geom = DPDZ_KM_PER_GPA * r.pf_gpa - r.b_km
    return float(h_mass - h_geom)


def run() -> int:
    checks: list[Check] = []

    # --- closed form vs numerical ---
    cases = [
        (1300.0, 0.0, 1.0),
        (1400.0, 0.0, 2.0),
        (1500.0, 0.0, 8.0),
        (1400.0, 20.0, 1.0),
        (1600.0, 30.0, 4.0),
    ]
    max_dp_err = 0.0
    for tp, b, chi in cases:
        num = solve_active_upwelling(tp_c=tp, b_km=b, chi=chi, dfdp_per_gpa=DFDP)
        dp_num = num.p0_gpa - num.pf_gpa
        dp_cf = delta_p_closed_form_linear(
            p0_gpa=num.p0_gpa, b_km=b, chi=chi, dfdp_per_gpa=DFDP
        )
        if dp_cf is None:
            checks.append(Check(f"closed-form DeltaP @ Tp={tp:.0f}, chi={chi}, b={b}", False, "no root"))
            continue
        err = abs(dp_num - dp_cf)
        max_dp_err = max(max_dp_err, err)
    checks.append(
        Check(
            "closed-form DeltaP vs numerical (max err)",
            max_dp_err < 1e-4,
            f"{max_dp_err:.2e} GPa",
        )
    )

    # --- balance residual ---
    res = sweep_hvp(
        tp_values_c=np.arange(1250.0, 1551.0, 25.0),
        chi_values=[1.0, 2.0, 4.0, 8.0],
        b_km=0.0,
        dfdp_per_gpa=DFDP,
    )
    bal = max(abs(_mass_geom_residual(r)) for r in res) if res else 999.0
    checks.append(Check("mass-geometry balance |dH|", bal < 1e-3, f"{bal:.2e} km"))

    # --- χ trend @ fixed Tp=1400, b=0 (paper §32) ---
    chi_vals = [1.0, 2.0, 4.0, 8.0]
    chi_pts = [solve_active_upwelling(tp_c=1400.0, b_km=0.0, chi=c, dfdp_per_gpa=DFDP) for c in chi_vals]
    h_inc = all(chi_pts[i].h_km < chi_pts[i + 1].h_km for i in range(len(chi_pts) - 1))
    pbar_inc = all(chi_pts[i].pbar_gpa < chi_pts[i + 1].pbar_gpa for i in range(len(chi_pts) - 1))
    fbar_dec = all(chi_pts[i].fbar > chi_pts[i + 1].fbar for i in range(len(chi_pts) - 1))
    checks.append(Check("higher chi -> thicker H (b=0)", h_inc, f"H={[round(p.h_km, 1) for p in chi_pts]}"))
    checks.append(Check("higher chi -> higher Pbar (b=0)", pbar_inc))
    checks.append(Check("higher chi -> lower Fbar (b=0)", fbar_dec))

    # --- b trend @ χ=1 passive ---
    b_vals = [0.0, 10.0, 20.0, 30.0]
    b_pts = [solve_active_upwelling(tp_c=1400.0, b_km=b, chi=1.0, dfdp_per_gpa=DFDP) for b in b_vals]
    pbar_b_inc = all(b_pts[i].pbar_gpa < b_pts[i + 1].pbar_gpa for i in range(len(b_pts) - 1))
    fbar_b_dec = all(b_pts[i].fbar > b_pts[i + 1].fbar for i in range(len(b_pts) - 1))
    h_b_dec = all(b_pts[i].h_km > b_pts[i + 1].h_km for i in range(len(b_pts) - 1))
    checks.append(Check("thicker b -> higher Pbar (chi=1)", pbar_b_inc))
    checks.append(Check("thicker b -> lower Fbar (chi=1)", fbar_b_dec))
    checks.append(Check("thicker b -> thinner H (chi=1)", h_b_dec))

    vp_b = [round(p.vp_bulk_km_s, 3) for p in b_pts]
    vp_chi = [round(p.vp_bulk_km_s, 3) for p in chi_pts]
    print(
        f"  (info) Fig.11 d/h V_bulk = eq.(1)(P_bar, F_bar) @ Tp=1400: "
        f"chi {vp_chi}, b {vp_b}"
    )

    # --- eq.(1) wired through column result ---
    vp_match = all(
        abs(r.vp_bulk_km_s - predict_v_bulk_km_s(r.pbar_gpa, r.fbar)) < 1e-9
        for r in chi_pts + b_pts
    )
    checks.append(
        Check(
            "V_bulk from eq.(1)(P_bar, F_bar) in column result",
            vp_match,
            f"chi {vp_chi}, b {vp_b}",
        )
    )

    # passive chi=1, b=0: eq.(11) => DeltaP = (sqrt(1 + 2 alpha P0) - 1) / alpha
    r_pass = solve_active_upwelling(tp_c=1400.0, b_km=0.0, chi=1.0, dfdp_per_gpa=DFDP)
    dp_cf = delta_p_closed_form_linear(
        p0_gpa=r_pass.p0_gpa, b_km=0.0, chi=1.0, dfdp_per_gpa=DFDP
    )
    pf_cf = pf_closed_form_linear(
        p0_gpa=r_pass.p0_gpa, b_km=0.0, chi=1.0, dfdp_per_gpa=DFDP
    )
    checks.append(
        Check(
            "passive b=0 closed-form DeltaP (eq.11)",
            dp_cf is not None
            and pf_cf is not None
            and abs((r_pass.p0_gpa - r_pass.pf_gpa) - dp_cf) < 1e-6
            and abs(r_pass.pf_gpa - pf_cf) < 1e-6,
            f"DeltaP={r_pass.p0_gpa - r_pass.pf_gpa:.4f} GPa, Pf={r_pass.pf_gpa:.4f} GPa",
        )
    )

    print("=== KKHS02 Fig.11 / section 4 active upwelling acceptance ===")
    print(f"  eq.(6)–(11): (dF/dP)_S={DFDP*100:.0f}%/GPa, Fbar=0.5*F, Pbar=(P0+Pf)/2, H=30*chi*DeltaP*Fbar")
    print(f"  Fig.11 d/h: V_bulk = eq.(1)(P_bar, F_bar) — no morphological correction")
    for c in checks:
        print(c.line())

    _report_dh_discrepancy(chi_pts, b_pts)

    failed = [c for c in checks if not c.ok]
    if failed:
        print(f"\nFig.11 model: {len(failed)} check(s) failed.")
        return 1
    print(f"\nFig.11 model: all {len(checks)} checks passed.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    raise SystemExit(run())


if __name__ == "__main__":
    main()
