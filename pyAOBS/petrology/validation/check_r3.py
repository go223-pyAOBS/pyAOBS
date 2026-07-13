"""
KKHS02 R3 acceptance — langmuir + sb1994_fig2ol (raw ΔVp; no Fig.5 empirical lift).

Runs Fig.2(a) Vp gaps, Fig.2c phase RMS, Fig.5 ΔVp catalog, and w98 high-MgO guard.

Usage::

    py -3.11 petrology/validation/check_r3.py
    py -3.11 petrology/validation/check_r3.py --strict
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

from petrology.data.load_catalog import load_melt_catalog, oxides_from_record
from petrology.fc.cdat_library import fig2_primary_melt
from petrology.fc.fig2_ab import cumulative_phases_pct
from petrology.fc.figure02a_digitized import load_figure02a_curves
from petrology.fc.wl1990 import load_kinzler1997_morb_primary, simulate_crystallization_path
from petrology.norm_velocity import norm_velocity_from_record
from petrology.fc.delta_vp import R3_DELTA_VP_WL_KW, delta_vp_wl_fc

KD_ENGINE = str(R3_DELTA_VP_WL_KW["kd_engine"])
MINERAL_BACKEND = str(R3_DELTA_VP_WL_KW["mineral_backend"])
WL_KW = dict(R3_DELTA_VP_WL_KW)
P_FC_MPA = 400.0
P_EVAL_MPA = 600.0
T_EVAL_C = 400.0
F_CATALOG = 0.75
F_FIG2_P_MPA = 100.0
T_FIG2_C = 100.0


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


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _fig2_vp_rms(curve_key: str, attr: str, path: str) -> float:
    dig = load_figure02a_curves()[curve_key]  # type: ignore[index]
    f_pts = np.array([p[0] for p in dig], dtype=float)
    ref = np.array([p[1] for p in dig], dtype=float)
    f_sim = np.sort(np.unique(np.concatenate([f_pts, [0.0]])))
    st = simulate_crystallization_path(
        primary_melt_oxides_wt=fig2_primary_melt()["oxides_wt_percent"],
        f_grid=f_sim,
        path=path,  # type: ignore[arg-type]
        kd_engine=KD_ENGINE,
        fig2_ab_calibrate=False,
        p_fc_mpa=F_FIG2_P_MPA,
        p_eval_mpa=F_FIG2_P_MPA,
        t_eval_c=T_FIG2_C,
        mineral_backend=MINERAL_BACKEND,
    )
    by_f = {float(s.f_solid): s for s in st}
    got = []
    for f in f_pts:
        s = by_f.get(float(f)) or by_f[min(by_f.keys(), key=lambda k: abs(k - f))]
        got.append(float(getattr(s, attr)))
    return _rms(np.asarray(got), ref)


def _fig2_phase_rms() -> float:
    melt = fig2_primary_melt()["oxides_wt_percent"]
    f_check = np.array([0.08, 0.22, 0.40, 0.55, 0.70, 0.80])
    st = simulate_crystallization_path(
        primary_melt_oxides_wt=melt,
        f_grid=f_check,
        path="fc_100",
        kd_engine=KD_ENGINE,
        fig2_ab_calibrate=False,
        p_fc_mpa=F_FIG2_P_MPA,
        mineral_backend=MINERAL_BACKEND,
    )
    errs = []
    for s in st:
        ol_p, cpx_p, pl_p = cumulative_phases_pct("fc_100", float(s.f_solid))
        ol, pl, cpx = 100 * s.cum_ol, 100 * s.cum_pl, 100 * s.cum_cpx
        errs.append((ol - ol_p) ** 2 + (pl - pl_p) ** 2 + (cpx - cpx_p) ** 2)
    return float(np.sqrt(np.mean(errs)))


def _catalog_delta_vp_stats() -> tuple[float, float, int, int]:
    rows = [r for r in load_melt_catalog() if r.get("include_in_regression")]
    raw: list[float] = []
    neg = 0
    for rec in rows:
        ox = {k: v for k, v in oxides_from_record(rec).items() if k != "P2O5"}
        bulk = float(
            norm_velocity_from_record(
                rec,
                p_pa=P_EVAL_MPA * 1e6,
                t_k=T_EVAL_C + 273.15,
                mineral_backend=MINERAL_BACKEND,
            )["vp_km_s"]
        )
        dr = delta_vp_wl_fc(
            melt_oxides_wt=ox,
            f_solid=F_CATALOG,
            p_fc_mpa=P_FC_MPA,
            bulk_vp_km_s=bulk,
            p_eval_mpa=P_EVAL_MPA,
            t_eval_c=T_EVAL_C,
            mineral_backend=MINERAL_BACKEND,
            kd_engine=KD_ENGINE,
            **{k: v for k, v in WL_KW.items() if k not in ("kd_engine", "mineral_backend")},
        )
        raw.append(float(dr))
        if dr < -0.005:
            neg += 1
    arr = np.asarray(raw)
    return float(arr.mean()), float(arr.max()), neg, len(rows)


def _kinzler_delta_vp() -> float:
    p = load_kinzler1997_morb_primary()
    from petrology.norm_velocity import norm_velocity_from_bulk_wt

    bulk = float(
        norm_velocity_from_bulk_wt(
            p["oxides_wt_percent"],
            p_pa=P_EVAL_MPA * 1e6,
            t_k=T_EVAL_C + 273.15,
            mineral_backend=MINERAL_BACKEND,
        )["vp_km_s"]
    )
    return delta_vp_wl_fc(
        melt_oxides_wt=p["oxides_wt_percent"],
        f_solid=F_CATALOG,
        p_fc_mpa=P_FC_MPA,
        bulk_vp_km_s=bulk,
        p_eval_mpa=P_EVAL_MPA,
        t_eval_c=T_EVAL_C,
        mineral_backend=MINERAL_BACKEND,
        kd_engine=KD_ENGINE,
        **{k: v for k, v in WL_KW.items() if k not in ("kd_engine", "mineral_backend")},
    )


def _w98_high_mgo_delta_vp() -> tuple[float, float]:
    from petrology.validation.debug_delta_vp_sign import _raw_delta_vp

    rec = next(r for r in load_melt_catalog() if r["id"] == "w98_run70_02")
    ox = {k: float(rec[k]) for k in ("SiO2", "TiO2", "Al2O3", "Cr2O3", "FeO", "MgO", "CaO", "Na2O", "K2O")}
    bulk = float(
        norm_velocity_from_record(rec, p_pa=P_EVAL_MPA * 1e6, t_k=T_EVAL_C + 273.15,
                                  mineral_backend=MINERAL_BACKEND)["vp_km_s"]
    )
    r = _raw_delta_vp(
        ox,
        f_solid=F_CATALOG,
        p_fc_mpa=P_FC_MPA,
        bulk_vp_km_s=bulk,
        p_eval_mpa=P_EVAL_MPA,
        t_eval_c=T_EVAL_C,
        mineral_backend=MINERAL_BACKEND,
        cipw_backend="auto",
    )
    d_cal = delta_vp_wl_fc(
        melt_oxides_wt=ox,
        f_solid=F_CATALOG,
        p_fc_mpa=P_FC_MPA,
        bulk_vp_km_s=bulk,
        p_eval_mpa=P_EVAL_MPA,
        t_eval_c=T_EVAL_C,
        mineral_backend=MINERAL_BACKEND,
        kd_engine=KD_ENGINE,
        **{k: v for k, v in WL_KW.items() if k not in ("kd_engine", "mineral_backend")},
    )
    return float(d_cal), 100 * float(r["cum_ol"])


def run(*, strict: bool = False) -> int:
    checks: list[Check] = []

    checks.append(Check("Fig.2c phase RMS", _fig2_phase_rms(), 22.0 if not strict else 10.0, "<", " wt%"))
    checks.append(Check("cum_sol_1kb RMS", _fig2_vp_rms("cum_sol_1kb", "vp_cumulate_km_s", "fc_100"), 0.08, "<", " km/s"))
    checks.append(Check("inc_sol_1kb RMS", _fig2_vp_rms("inc_sol_1kb", "vp_incremental_km_s", "fc_100"), 0.08, "<", " km/s"))
    checks.append(Check("frac_res_1kb RMS", _fig2_vp_rms("frac_res_1kb", "vp_residual_norm_km_s", "fc_100"), 0.05, "<", " km/s"))

    mean_d, max_d, n_neg, n_cat = _catalog_delta_vp_stats()
    checks.append(Check("catalog ΔVp mean @ F=0.75", mean_d, 0.12 if strict else 0.10, ">", " km/s"))
    checks.append(Check("catalog ΔVp mean @ F=0.75", mean_d, 0.18 if strict else 0.20, "<", " km/s"))
    checks.append(Check("catalog ΔVp max @ F=0.75", max_d, 0.40, "<", " km/s"))
    checks.append(Check("catalog ΔVp negative count", float(n_neg), 0.5, "<", ""))

    kinz_d = _kinzler_delta_vp()
    checks.append(Check("Kinzler primary ΔVp @ F=0.75", kinz_d, 0.08, ">", " km/s"))
    checks.append(Check("Kinzler primary ΔVp @ F=0.75", kinz_d, 0.20, "<", " km/s"))

    w98_d, w98_ol = _w98_high_mgo_delta_vp()
    checks.append(Check("w98_run70_02 ΔVp @ F=0.75", w98_d, 0.0, ">", " km/s"))
    checks.append(Check("w98_run70_02 ΔVp @ F=0.75", w98_d, 0.25, "<", " km/s"))
    checks.append(Check("w98_run70_02 cum Ol @ F=0.75", w98_ol, 15.0, ">", " %"))

    print("=== KKHS02 R3 acceptance (langmuir + sb1994_fig2ol) ===")
    print(f"  catalog n={n_cat}  P_fc={P_FC_MPA:.0f} MPa  report {P_EVAL_MPA:.0f} MPa / {T_EVAL_C:.0f} C")
    for c in checks:
        print(c.line())

    failed = [c for c in checks if not c.ok]
    if failed:
        print(f"\nR3: {len(failed)} check(s) failed.")
        return 1
    print("\nR3: all checks passed.")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="KKHS02 R3 acceptance checks")
    p.add_argument("--strict", action="store_true", help="Tighter phase-RMS and ΔVp mean band")
    args = p.parse_args()
    raise SystemExit(run(strict=args.strict))


if __name__ == "__main__":
    main()
