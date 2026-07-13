"""
KKHS02 R4 acceptance — Step-2 invert + Greenland H–Vp anchor (Fig.15c).

Uses R3-closed physical chain: langmuir + sb1994_fig2ol ΔVp.

Usage::

    py -3.11 petrology/validation/check_r4.py
    py -3.11 petrology/validation/check_r4.py --strict
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

from petrology.fc.delta_vp import FIG5_P_EVAL_MPA, FIG5_T_EVAL_C, R3_DELTA_VP_WL_KW
from petrology.fc.wl1990 import load_kinzler1997_morb_primary
from petrology.fractionation import delta_vp_km_s
from petrology.invert import bulk_vp_bounds_from_fractionation, feasible_pf_region
from petrology.melting.hvp_scan import scan_hvp_lip
from petrology.norm_velocity import norm_velocity_from_bulk_wt

KD_ENGINE = R3_DELTA_VP_WL_KW["kd_engine"]
MINERAL_BACKEND = R3_DELTA_VP_WL_KW["mineral_backend"]
WL_KW = dict(R3_DELTA_VP_WL_KW)

V_LC_OBS = 7.0
H_OBS_KM = 30.0
F_LOWER = 0.5
F_SOLID = 0.75
P_FC_MPA = 400.0
VP_BIAS = -0.10
VP_TOL = 0.05


@dataclass
class Check:
    name: str
    value: float
    limit: float
    op: str  # "<" or ">"
    unit: str = ""

    @property
    def ok(self) -> bool:
        if self.op == "<":
            return self.value < self.limit
        if self.op == ">":
            return self.value > self.limit
        raise ValueError(self.op)

    def line(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        return f"  [{mark}] {self.name}: {self.value:.4f}{self.unit} {self.op} {self.limit:.4f}{self.unit}"


def _kinzler_bulk_vp() -> float:
    melt = load_kinzler1997_morb_primary()["oxides_wt_percent"]
    return float(
        norm_velocity_from_bulk_wt(
            melt,
            p_pa=FIG5_P_EVAL_MPA * 1e6,
            t_k=FIG5_T_EVAL_C + 273.15,
            mineral_backend=MINERAL_BACKEND,
        )["vp_km_s"]
    )


def _kinzler_delta_vp() -> float:
    melt = load_kinzler1997_morb_primary()["oxides_wt_percent"]
    bulk = _kinzler_bulk_vp()
    return float(
        delta_vp_km_s(
            F_SOLID,
            bulk_vp_km_s=bulk,
            p_fc_mpa=P_FC_MPA,
            engine="wl1990",
            melt_oxides_wt=melt,
            **WL_KW,
        )
    )


def _step2_pf_feasible() -> tuple[int, float, float]:
    melt = load_kinzler1997_morb_primary()["oxides_wt_percent"]
    bulk = _kinzler_bulk_vp()
    bounds = bulk_vp_bounds_from_fractionation(
        V_LC_OBS,
        f_lower=F_LOWER,
        f_solid=F_SOLID,
        p_fc_mpa=P_FC_MPA,
        bulk_vp_guess_km_s=bulk,
        delta_vp_engine="wl1990",
        melt_oxides_wt=melt,
        delta_vp_wl_kw=WL_KW,
    )
    reg = feasible_pf_region(bounds, vp_bias_km_s=VP_BIAS, vp_tolerance_km_s=VP_TOL)
    return int(reg["n_feasible"]), float(bounds.v_bulk_lower_km_s), float(bounds.delta_vp_fc_km_s)


def _greenland_h_anchor() -> tuple[int, float, float]:
    """Coarse (Tp, chi) scan: H-only feasibility in paper discussion band."""
    result = scan_hvp_lip(
        v_lc_obs_km_s=V_LC_OBS,
        h_obs_km=H_OBS_KM,
        b_km=0.0,
        tp_range_c=(1250.0, 1450.0),
        tp_step_c=10.0,
        chi_values=[6.0, 8.0, 10.0, 12.0, 16.0],
        h_tolerance_km=3.0,
        melting_engine="kinzler_linear",
        delta_vp_engine="wl1990",
        delta_vp_wl_kw=WL_KW,
        vp_bias_km_s=VP_BIAS,
        require_bulk_in_bounds=False,
        verbose=False,
        n_isentropic_steps=24,
    )
    paper_band = [
        p
        for p in result.feasible
        if p.chi >= 8.0 and 1250.0 <= p.tp_c <= 1450.0 and abs(p.h_match_km) <= 3.0
    ]
    if not paper_band:
        return 0, float("nan"), float("nan")
    best = min(paper_band, key=lambda p: abs(p.h_match_km))
    return len(paper_band), float(best.tp_c), float(best.chi)


def _reebox_bulk_bounds_closure() -> tuple[int, int, float, float]:
    """REEBOX coarse grid + norm Vp refine (sb1994_fig2ol) with H+Vp bounds."""
    from petrology.melting.pymelt_lithology_adapter import resolve_lithology_col_kwargs

    lith = resolve_lithology_col_kwargs(lithology_preset="greenland_kg1")
    result = scan_hvp_lip(
        v_lc_obs_km_s=V_LC_OBS,
        h_obs_km=H_OBS_KM,
        b_km=0.0,
        pyroxenite_frac=0.10,
        tp_range_c=(1250.0, 1450.0),
        tp_step_c=25.0,
        chi_values=[8.0, 10.0, 12.0, 16.0],
        h_tolerance_km=3.0,
        melting_engine="reebox",
        delta_vp_engine="wl1990",
        delta_vp_wl_kw=WL_KW,
        require_bulk_in_bounds=True,
        refine_norm_vp=12,
        vp_bias_km_s=0.0,
        verbose=False,
        n_isentropic_steps=24,
        **lith,
    )
    paper = [
        p
        for p in result.feasible
        if p.chi >= 8.0 and 1250.0 <= p.tp_c <= 1450.0 and p.bulk_in_bounds
    ]
    if not paper:
        return int(result.n_feasible), 0, float("nan"), float("nan")
    best = min(paper, key=lambda p: abs(p.h_match_km))
    return int(result.n_feasible), len(paper), float(best.tp_c), float(best.chi)


def run_checks(*, strict: bool = False, bulk: bool = False) -> int:
    checks: list[Check] = []

    d_vp = _kinzler_delta_vp()
    checks.append(Check("kinzler ΔVp @ F=0.75", d_vp, 0.08, ">"))
    checks.append(Check("kinzler ΔVp @ F=0.75", d_vp, 0.15 if strict else 0.18, "<"))

    n_pf, v_lower, d_bounds = _step2_pf_feasible()
    checks.append(Check("Step-2 feasible (P,F) count", float(n_pf), 40.0, ">"))
    checks.append(Check("Step-2 V_bulk lower", v_lower, 6.90, ">"))
    checks.append(Check("Step-2 ΔVp in bounds", d_bounds, 0.08, ">"))

    n_h, tp_best, chi_best = _greenland_h_anchor()
    checks.append(Check("Greenland H anchor (χ≥8)", float(n_h), 1.0, ">"))
    if n_h > 0:
        checks.append(Check("Greenland best Tp", tp_best, 1250.0, ">"))
        checks.append(Check("Greenland best Tp", tp_best, 1450.0, "<"))
        checks.append(Check("Greenland best χ", chi_best, 7.99, ">"))

    n_bb_all = n_bb_paper = 0
    tp_bb = chi_bb = float("nan")

    if bulk:
        n_bb_all, n_bb_paper, tp_bb, chi_bb = _reebox_bulk_bounds_closure()
        checks.append(Check("REEBOX bulk+norm feasible (all)", float(n_bb_all), 3.0, ">"))
        checks.append(Check("REEBOX bulk χ≥8 band", float(n_bb_paper), 1.0, ">"))
        if n_bb_paper > 0 and not np.isnan(tp_bb):
            checks.append(Check("REEBOX bulk best Tp", tp_bb, 1250.0, ">"))
            checks.append(Check("REEBOX bulk best Tp", tp_bb, 1450.0, "<"))

    print("KKHS02 R4 checklist (langmuir + sb1994_fig2ol)")
    print(f"  anchor: H={H_OBS_KM:.0f} km, V_LC={V_LC_OBS:.2f} km/s")
    print(f"  Step-2: f_lower={F_LOWER}, vp_bias={VP_BIAS:+.2f}, vp_tol=±{VP_TOL:.2f}")
    print()

    n_fail = 0
    for c in checks:
        print(c.line())
        if not c.ok:
            n_fail += 1

    print()
    if n_fail == 0:
        if n_h > 0:
            print(f"  Greenland best H-match: Tp≈{tp_best:.0f}°C, χ≈{chi_best:g}")
        if bulk and n_bb_paper > 0 and not np.isnan(tp_bb):
            print(f"  REEBOX bulk closure: Tp≈{tp_bb:.0f}°C, χ≈{chi_bb:g} (n={n_bb_paper})")
        print(f"R4 PASS ({len(checks)}/{len(checks)} checks)")
        return 0
    print(f"R4 FAIL ({len(checks) - n_fail}/{len(checks)} checks passed)")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="KKHS02 R4 acceptance checks")
    parser.add_argument("--strict", action="store_true", help="Tighter ΔVp upper bound (0.15 km/s)")
    parser.add_argument(
        "--bulk",
        action="store_true",
        help="Also run REEBOX norm-Vp refine bulk-bounds closure (slower)",
    )
    args = parser.parse_args()
    raise SystemExit(run_checks(strict=args.strict, bulk=args.bulk))


if __name__ == "__main__":
    main()
