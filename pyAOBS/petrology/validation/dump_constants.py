"""
Dump constants + sanity checks with PASS/WARN/FAIL/SKIP verdicts.

  python petrology/validation/dump_constants.py
  python petrology/validation/dump_constants.py --quick   # skip Greenland grid
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

Verdict = Literal["PASS", "WARN", "FAIL", "SKIP"]


@dataclass
class Check:
    section: str
    name: str
    verdict: Verdict
    detail: str


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, section: str, name: str, verdict: Verdict, detail: str) -> None:
        self.checks.append(Check(section, name, verdict, detail))

    def tag(self, verdict: Verdict) -> str:
        return f"[{verdict}]"

    def print_line(self, verdict: Verdict, text: str) -> None:
        print(f"  {self.tag(verdict)} {text}")

    def summary(self) -> None:
        counts = {v: 0 for v in ("PASS", "WARN", "FAIL", "SKIP")}
        for c in self.checks:
            counts[c.verdict] += 1
        print("\n" + "=" * 72)
        print("SUMMARY")
        print("=" * 72)
        print(
            f"  PASS={counts['PASS']}  WARN={counts['WARN']}  "
            f"FAIL={counts['FAIL']}  SKIP={counts['SKIP']}"
        )
        fails = [c for c in self.checks if c.verdict == "FAIL"]
        warns = [c for c in self.checks if c.verdict == "WARN"]
        if fails:
            print("\n  FAIL:")
            for c in fails:
                print(f"    - [{c.section}] {c.name}: {c.detail}")
        if warns:
            print("\n  WARN:")
            for c in warns:
                print(f"    - [{c.section}] {c.name}: {c.detail}")
        if not fails and not warns:
            print("\n  All automated checks passed or skipped.")


def _greenland_h_scan(
    *,
    h_obs_km: float,
    h_tol_km: float,
    pyroxenite_frac: float,
    tp_min: float,
    tp_max: float,
    tp_step: float,
    chi_values: list[float],
    lith_kw: dict,
    n_isentropic_steps: int = 20,
) -> list[dict]:
    """Coarse forward grid: H vs (Tp, chi) only (fast diagnostic)."""
    import numpy as np

    from petrology.melting.heterogeneous import forward_heterogeneous_column

    rows: list[dict] = []
    for tp in np.arange(tp_min, tp_max + tp_step * 0.5, tp_step):
        for chi in chi_values:
            if chi < 1.0:
                continue
            try:
                col = forward_heterogeneous_column(
                    tp_c=float(tp),
                    b_km=0.0,
                    chi=float(chi),
                    pyroxenite_frac=float(pyroxenite_frac),
                    n_isentropic_steps=n_isentropic_steps,
                    **lith_kw,
                )
            except (ValueError, RuntimeError):
                continue
            dh = float(col.h_km - h_obs_km)
            rows.append(
                {
                    "tp_c": float(tp),
                    "chi": float(chi),
                    "h_km": float(col.h_km),
                    "dh_km": dh,
                    "p0_gpa": float(col.p0_gpa),
                    "vp_eq1": float(col.vp_bulk_eq1_km_s),
                }
            )
    return rows


def _summarize_h_grid(rows: list[dict], *, h_tol_km: float) -> dict:
    if not rows:
        return {"n_grid": 0, "n_h": 0, "best": None, "tp_rng": None, "chi_rng": None}
    h_ok = [r for r in rows if abs(r["dh_km"]) <= h_tol_km]
    best = min(rows, key=lambda r: abs(r["dh_km"]))
    tp_rng = chi_rng = None
    if h_ok:
        tp_rng = (min(r["tp_c"] for r in h_ok), max(r["tp_c"] for r in h_ok))
        chi_rng = (min(r["chi"] for r in h_ok), max(r["chi"] for r in h_ok))
    return {
        "n_grid": len(rows),
        "n_h": len(h_ok),
        "best": best,
        "tp_rng": tp_rng,
        "chi_rng": chi_rng,
        "h_ok": h_ok,
    }


def _run_greenland_section(rep: Report, *, quick: bool) -> None:
    h_obs, h_tol = 30.0, 5.0
    chi_values = [4.0, 6.0, 8.0, 10.0, 12.0, 16.0]
    tp_min, tp_max, tp_step = 1150.0, 1400.0, 25.0

    print("\n" + "=" * 72)
    print("K. Greenland anchor heuristic (H ~ 30 km, V_LC = 7.0 km/s)")
    print("=" * 72)
    print(f"  Target H={h_obs:.0f} km  |ΔH|≤{h_tol:g} km  Tp={tp_min:.0f}–{tp_max:.0f} step={tp_step:g}")

    if quick:
        rep.print_line("SKIP", "Greenland grid skipped (--quick)")
        rep.add("K", "greenland_grid", "SKIP", "skipped via --quick")
        return

    try:
        from petrology.melting.pymelt_lithology_adapter import resolve_lithology_col_kwargs

        has_pymelt = True
    except ImportError:
        has_pymelt = False

    configs: list[tuple[str, dict, float]] = []
    configs.append(("native (Katz+G2 analytic)", {"lithology_backend": "native"}, 0.0))
    if has_pymelt:
        try:
            configs.append(
                (
                    "pymelt greenland_kg1 Φ=0",
                    resolve_lithology_col_kwargs(lithology_preset="greenland_kg1"),
                    0.0,
                )
            )
            configs.append(
                (
                    "pymelt greenland_kg1 Φ=0.10",
                    resolve_lithology_col_kwargs(lithology_preset="greenland_kg1"),
                    0.10,
                )
            )
            configs.append(
                (
                    "pymelt lip_default Φ=0.10",
                    resolve_lithology_col_kwargs(lithology_preset="lip_default"),
                    0.10,
                )
            )
        except ImportError:
            has_pymelt = False

    if not has_pymelt:
        rep.print_line("WARN", "pyMelt unavailable — Greenland hint uses native only")
        rep.add("K", "pymelt", "WARN", "pyMelt not installed")

    best_overall: tuple[str, dict] | None = None
    best_n_h = -1

    for label, lith_kw, phi in configs:
        rows = _greenland_h_scan(
            h_obs_km=h_obs,
            h_tol_km=h_tol,
            pyroxenite_frac=phi,
            tp_min=tp_min,
            tp_max=tp_max,
            tp_step=tp_step,
            chi_values=chi_values,
            lith_kw=lith_kw,
        )
        s = _summarize_h_grid(rows, h_tol_km=h_tol)
        print(f"\n  --- {label} ---")
        print(f"      grid={s['n_grid']}  n_H={s['n_h']}")
        if s["best"]:
            b = s["best"]
            print(
                f"      best |ΔH|: Tp={b['tp_c']:.0f} χ={b['chi']:g} "
                f"H={b['h_km']:.1f} ΔH={b['dh_km']:+.1f} Vp_eq1={b['vp_eq1']:.3f}"
            )
        if s["tp_rng"]:
            print(
                f"      H-match window: Tp {s['tp_rng'][0]:.0f}–{s['tp_rng'][1]:.0f} °C, "
                f"χ {s['chi_rng'][0]:g}–{s['chi_rng'][1]:g}"
            )
        else:
            print("      H-match window: (none in grid)")

        if s["n_h"] > best_n_h:
            best_n_h = s["n_h"]
            best_overall = (label, s)

        v: Verdict = "PASS" if s["n_h"] > 0 else "WARN"
        rep.add("K", label, v, f"n_H={s['n_h']} grid={s['n_grid']}")

    print("\n  --- GUI / scan recommendations ---")
    if best_overall and best_overall[1]["n_h"] > 0:
        lbl, s = best_overall
        b = s["best"]
        tp_lo, tp_hi = s["tp_rng"]
        chi_lo, chi_hi = s["chi_rng"]
        pad_tp = 25.0
        print(f"  Best preset for H anchor: {lbl}")
        print(
            f"  Suggested Tp scan: {max(tp_min, tp_lo - pad_tp):.0f}–"
            f"{min(tp_max, tp_hi + pad_tp):.0f} °C (step 25)"
        )
        print(f"  Suggested χ list: include {chi_lo:g}–{chi_hi:g} (extend to 1–16 if exploring)")
        print(
            f"  Near anchor: Tp≈{b['tp_c']:.0f} °C χ≈{b['chi']:g} "
            f"(H={b['h_km']:.1f} km); strict Vp feasible needs refine≠0"
        )
        rep.add(
            "K",
            "recommendation",
            "PASS",
            f"{lbl} Tp {tp_lo:.0f}-{tp_hi:.0f} chi {chi_lo:g}-{chi_hi:g}",
        )
    else:
        print("  No |ΔH|≤tol points in coarse grid — widen Tp (e.g. 1100–1450) or run Φ scan.")
        print("  For Φ=0.10 fixed: try Φ scan; Tp–χ scans often use Φ=0 in literature.")
        rep.add("K", "recommendation", "WARN", "no H-match in coarse grid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Constants dump + PASS/WARN/FAIL checks")
    parser.add_argument("--quick", action="store_true", help="Skip Greenland H grid (section K)")
    args = parser.parse_args()

    rep = Report()

    from petrology.config import DEFAULT_CONFIG
    from petrology.vp_regression import predict_vp_km_s, load_eq1
    from petrology.invert import bulk_vp_bounds_from_fractionation
    from petrology.fractionation import DEFAULT_FRACTIONATION, delta_vp_km_s
    from petrology.minerals import REF_P_PA, REF_T_K
    from petrology.melting.lithology import (
        CP_J_KG_K,
        ALPHA_K_INV,
        RHO_KG_M3,
        DELTA_S_PERIDOTITE,
        DELTA_S_PYROXENITE,
        G2_BULK_WT,
        katz2003_peridotite_solidus,
        katz2003_peridotite_liquidus,
        g2_pyroxenite_solidus,
    )
    from petrology.melting.kinzler1997_batch import HZ_DEP1_WT
    from petrology.melting.reebox_geometry import RHO_GCM3, G_M_S2, P_FLOOR_GPA, PYMELT_TC_TO_KM
    from petrology.active_upwelling import (
        DPDZ_KM_PER_GPA,
        ADIABAT_GRAD_C_PER_GPA,
        DFDP_DEFAULT_PER_GPA,
        solidus_tk83_c,
    )
    from petrology.melting.heterogeneous import forward_heterogeneous_column

    eq1 = load_eq1()
    v = eq1["validation"]
    vp_morb = predict_vp_km_s(v["morb_P_GPa"], v["morb_F"])

    # A
    print("=" * 72)
    print("A. Reference states")
    print("=" * 72)
    ref_json = eq1["reference_state"]
    ok_ref = (
        ref_json["pressure_MPa"] == 600
        and ref_json["temperature_C"] == 400
        and REF_P_PA == 600e6
        and abs(REF_T_K - 673.15) < 0.1
    )
    print(f"  KKHS02 eq.(1): P={ref_json['pressure_MPa']} MPa, T={ref_json['temperature_C']} C")
    print(f"  minerals.py REF: P={REF_P_PA/1e6:.0f} MPa, T={REF_T_K-273.15:.0f} C")
    print(f"  PetrologyConfig: P={DEFAULT_CONFIG.reference_p_pa/1e6:.0f} MPa, T={DEFAULT_CONFIG.reference_t_k-273.15:.0f} C")
    v_a: Verdict = "PASS" if ok_ref else "FAIL"
    rep.print_line(v_a, "600 MPa / 400 °C consistent across config")
    rep.add("A", "reference_state", v_a, "600 MPa / 400 C")

    # B
    print("\n" + "=" * 72)
    print("B. eq.(1) coefficients")
    print("=" * 72)
    c = eq1["equation_1"]["coefficients"]
    w = eq1["equation_1"]["window_functions"]
    print(f"  a0={c['a0']}  windows: alpha={w['alpha']} beta={w['beta']} Pt={w['P_t_GPa']} GPa Ft={w['F_t']}")
    print(f"  MORB P={v['morb_P_GPa']} GPa F={v['morb_F']}: Vp={vp_morb:.3f} km/s (expect {v['expected_Vp_km_s']})")
    d_vp = abs(vp_morb - v["expected_Vp_km_s"])
    if d_vp < 0.05:
        v_b: Verdict = "PASS"
        msg = f"within 0.05 km/s of JSON anchor (Δ={d_vp:.3f})"
    elif d_vp < 0.20:
        v_b = "WARN"
        msg = f"systematic offset Δ={d_vp:.3f} km/s — check coeffs vs KKHS02 Table"
    else:
        v_b = "FAIL"
        msg = f"large offset Δ={d_vp:.3f} km/s"
    rep.print_line(v_b, f"MORB validation: {msg}")
    rep.add("B", "eq1_morb", v_b, msg)

    rng = v.get("catalog_Vp_range_km_s", [6.8, 7.8])
    in_rng = rng[0] <= vp_morb <= rng[1]
    v_b2: Verdict = "PASS" if in_rng else "WARN"
    rep.print_line(v_b2, f"Vp in catalog range [{rng[0]}, {rng[1]}]: {in_rng}")
    rep.add("B", "eq1_catalog_range", v_b2, f"Vp={vp_morb:.3f}")

    # C
    print("\n" + "=" * 72)
    print("C. Step-2 bounding")
    print("=" * 72)
    bnd = bulk_vp_bounds_from_fractionation(7.0)
    dvp = delta_vp_km_s(0.75, bulk_vp_km_s=7.2, p_fc_mpa=400.0)
    print(f"  f_lower=0.5 f_solid=0.75 p_fc=400 → ΔVp={dvp:.3f} km/s")
    print(f"  V_bulk @ V_LC=7.0: [{bnd.v_bulk_lower_km_s:.3f}, {bnd.v_bulk_upper_km_s:.3f}] km/s")
    ok_bnd = bnd.v_bulk_upper_km_s == 7.0 and bnd.v_bulk_lower_km_s < 7.0
    v_c: Verdict = "PASS" if ok_bnd else "WARN"
    rep.print_line(v_c, "bounding interval sane")
    rep.add("C", "bulk_bounds", v_c, f"[{bnd.v_bulk_lower_km_s:.3f}, {bnd.v_bulk_upper_km_s:.3f}]")

    # D–G (informational)
    print("\n" + "=" * 72)
    print("D–G. Thermodynamics & compositions (informational)")
    print("=" * 72)
    p = DEFAULT_FRACTIONATION
    print(f"  FC MVP: delta_vp_max={p.delta_vp_max} exp={p.delta_vp_exp}")
    print(f"  REEBOX: Cp={CP_J_KG_K} alpha={ALPHA_K_INV} ΔS_per={DELTA_S_PERIDOTITE} ΔS_pyr={DELTA_S_PYROXENITE}")
    print(f"  Katz @2GPa: Tsol={katz2003_peridotite_solidus(2.0):.1f} Tliq={katz2003_peridotite_liquidus(2.0):.1f}")
    print(f"  HZ_DEP1 MgO={HZ_DEP1_WT['MgO']}  G2 MgO={G2_BULK_WT['MgO']}")
    rep.add("D-G", "constants_listed", "PASS", "informational dump")

    # I native forward
    print("\n" + "=" * 72)
    print("I. Native REEBOX spot checks")
    print("=" * 72)
    col = forward_heterogeneous_column(
        tp_c=1350, b_km=0, chi=4, pyroxenite_frac=0.10, n_isentropic_steps=48
    )
    print(
        f"  Tp=1350 χ=4 Φ=0.1: P0={col.p0_gpa:.2f} H={col.h_km:.1f} "
        f"Fbar={col.fbar:.3f} Vp_eq1={col.vp_bulk_eq1_km_s:.3f}"
    )
    ok_i = col.h_km > 5 and 6.5 < col.vp_bulk_eq1_km_s < 8.5
    v_i: Verdict = "PASS" if ok_i else "WARN"
    rep.print_line(v_i, "native forward returns physical H/Vp")
    rep.add("I", "native_forward", v_i, f"H={col.h_km:.1f} Vp={col.vp_bulk_eq1_km_s:.3f}")

    # J pyMelt
    print("\n" + "=" * 72)
    print("J. pyMelt cross-check")
    print("=" * 72)
    try:
        from petrology.melting.pymelt_lithology_adapter import (
            lithology_diagnostics,
            pymelt_lithology,
            resolve_lithology_col_kwargs,
        )

        native_tsol = katz2003_peridotite_solidus(2.0)
        pm = pymelt_lithology("katz_lherzolite")
        pm_tsol = pm.solidus_gpa(2.0)
        dt_katz = pm_tsol - native_tsol
        print(f"  Katz Tsol @2GPa: native={native_tsol:.1f} pymelt={pm_tsol:.1f} ΔT={dt_katz:+.1f}")
        v_j1: Verdict = "PASS" if abs(dt_katz) < 1.0 else ("WARN" if abs(dt_katz) < 5.0 else "FAIL")
        rep.print_line(v_j1, "Katz native vs pyMelt solidus")
        rep.add("J", "katz_tsol", v_j1, f"ΔT={dt_katz:+.1f} C")

        g2_native = g2_pyroxenite_solidus(2.0)
        g2_pm = lithology_diagnostics("pertermann_g2", p_gpa=2.0)["tsol_c"]
        dt_g2 = g2_pm - g2_native
        print(f"  G2 Tsol @2GPa: native≈{g2_native:.1f} pymelt={g2_pm:.1f} ΔT={dt_g2:+.1f}")
        v_j2: Verdict = "PASS" if abs(dt_g2) < 5.0 else "WARN"
        rep.print_line(v_j2, "G2 Tsol (native is Katz-150 approximation)")
        rep.add("J", "g2_tsol", v_j2, f"native={g2_native:.1f} pymelt={g2_pm:.1f}")

        pml = forward_heterogeneous_column(
            tp_c=1350,
            b_km=0,
            chi=4,
            pyroxenite_frac=0.10,
            n_isentropic_steps=48,
            **resolve_lithology_col_kwargs(lithology_preset="lip_default"),
        )
        dh = pml.h_km - col.h_km
        print(f"  lip_default vs native H: {pml.h_km:.1f} vs {col.h_km:.1f} (ΔH={dh:+.1f} km)")
        v_j3: Verdict = "PASS" if abs(dh) < 5.0 else "WARN"
        rep.print_line(v_j3, "pymelt lip_default H within 5 km of native")
        rep.add("J", "lip_default_dH", v_j3, f"ΔH={dh:+.1f} km")

        kg1 = forward_heterogeneous_column(
            tp_c=1350,
            b_km=0,
            chi=4,
            pyroxenite_frac=0.10,
            n_isentropic_steps=48,
            **resolve_lithology_col_kwargs(lithology_preset="greenland_kg1"),
        )
        dh30 = kg1.h_km - 30.0
        print(f"  greenland_kg1 @1350/χ4/Φ0.1: H={kg1.h_km:.1f} (ΔH vs 30={dh30:+.1f})")
        v_j4: Verdict = "PASS" if abs(dh30) < 8.0 else "WARN"
        rep.print_line(v_j4, "greenland_kg1 H near literature anchor at spot point")
        rep.add("J", "greenland_kg1_spot", v_j4, f"H={kg1.h_km:.1f} dH={dh30:+.1f}")

    except ImportError as exc:
        rep.print_line("SKIP", f"pyMelt: {exc}")
        rep.add("J", "pymelt", "SKIP", str(exc))

    _run_greenland_section(rep, quick=args.quick)
    rep.summary()


if __name__ == "__main__":
    main()
