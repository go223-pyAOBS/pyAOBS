"""
Decompose Fig.2(a) Vp errors for sb1994 vs digitized reference.

Focus:
  A) F≈0 cumulative solid — pure Fo90 olivine endpoint
  B) F≥0.7 residual melt norm Vp — CIPW vs mineral backend vs melt chemistry
  C) Mid-F cumulative — phase schedule vs single-mineral offset

Usage::

    py -3.11 petrology/validation/debug_sb1994_fig2_errors.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.cipw import cipw_norm_mass_fractions
from petrology.fc.assemblage import assemblage_vp_rho
from petrology.fc.cdat_library import fig2_primary_melt
from petrology.fc.fig2_ab import cumulative_phases_pct, incremental_compositions
from petrology.fc.figure02a_digitized import digitized_cum_vp_km_s, digitized_residual_vp_km_s
from petrology.fc.wl1990 import (
    olivine_fo_pct,
    plagioclase_an_pct,
    clinopyroxene_di_pct,
    simulate_crystallization_path,
)
from petrology.minerals import phase_properties
from petrology.norm_velocity import norm_velocity_from_bulk_wt

P_EVAL_PA = 100e6
T_EVAL_K = 373.15


def _state(path: str, f: float, *, kd_engine: str = "langmuir") -> object:
    melt = fig2_primary_melt()["oxides_wt_percent"]
    f_grid = np.array([0.0, float(f)], dtype=float) if f > 0 else np.array([0.0, 0.001], dtype=float)
    st = simulate_crystallization_path(
        primary_melt_oxides_wt=melt,
        f_grid=f_grid,
        path=path,  # type: ignore[arg-type]
        kd_engine=kd_engine,  # type: ignore[arg-type]
        fig2_ab_calibrate=False,
        mineral_backend="sb1994",
        cipw_backend="fallback",
    )
    target = 0.0 if f <= 0 else float(f)
    for s in st:
        if abs(s.f_solid - target) < 1e-4:
            return s
    return st[-1]


def _section_a_f0_olivine() -> None:
    print("\n" + "=" * 72)
    print("A) F≈0 cumulative solid — Fo90 olivine endpoint")
    print("=" * 72)

    melt = fig2_primary_melt()["oxides_wt_percent"]
    fo = olivine_fo_pct(melt, t_k=T_EVAL_K, p_mpa=100.0)
    an = plagioclase_an_pct(melt, t_k=T_EVAL_K, p_mpa=100.0)
    di = clinopyroxene_di_pct(melt, t_k=T_EVAL_K, p_mpa=100.0)
    print(f"Primary melt Fo={fo:.1f}%  An={an:.1f}%  Di={di:.1f}%")

    ref_fc = digitized_cum_vp_km_s("fc_100", 0.0)
    ref_pb = digitized_cum_vp_km_s("polybaric_fc", 0.0)
    print(f"Digitized cum @ F=0: fc_100={ref_fc:.3f}  polybaric={ref_pb:.3f} km/s")
    print(f"fig2_ab sparse anchor @ F=0: 7.57 / 7.55 km/s")

    print("\n--- Pure 100% olivine HS (Fo from primary melt) ---")
    rows = []
    for mb in ("sb1994", "fig2", "burnman", "empirical"):
        vp, rho = assemblage_vp_rho(
            ol_frac=1.0, pl_frac=0.0, cpx_frac=0.0,
            fo_pct=fo, an_pct=an, di_pct=di,
            p_pa=P_EVAL_PA, t_k=T_EVAL_K, mineral_backend=mb,  # type: ignore[arg-type]
        )
        ol_only = phase_properties(
            "olivine", fo=fo / 100.0, p_pa=P_EVAL_PA, t_k=T_EVAL_K, backend=mb,  # type: ignore[arg-type]
        )
        rows.append((mb, vp, rho, ol_only["vp_km_s"], ol_only["k_gpa"], ol_only["g_gpa"]))
    print(f"{'backend':10s} {'HS Vp':>7s} {'rho':>6s} {'Ol Vp':>7s} {'K':>7s} {'G':>7s}  Δ vs ref")
    for mb, vp, rho, ovp, k, g in rows:
        print(f"{mb:10s} {vp:7.3f} {rho:6.3f} {ovp:7.3f} {k:7.1f} {g:7.1f}  {vp - ref_fc:+.3f}")

    s0 = _state("fc_100", 0.0)
    print(f"\nLangmuir fc_100 @ F=0: cum Vp={s0.vp_cumulate_km_s:.3f}  (Δ vs ref {s0.vp_cumulate_km_s - ref_fc:+.3f})")

    ol_p, cpx_p, pl_p = cumulative_phases_pct("fc_100", 0.0)
    fo_p, an_p, di_p = incremental_compositions("fc_100", 0.0)
    for mb in ("sb1994", "fig2"):
        vp, _ = assemblage_vp_rho(
            ol_frac=ol_p / 100, pl_frac=pl_p / 100, cpx_frac=cpx_p / 100,
            fo_pct=fo_p, an_pct=an_p, di_pct=di_p,
            p_pa=P_EVAL_PA, t_k=T_EVAL_K, mineral_backend=mb,  # type: ignore[arg-type]
        )
        print(f"Paper Fig.2c-h phases @ F=0 + {mb:6s}: Vp={vp:.3f}  Δ={vp - ref_fc:+.3f}")

    sb_vp = rows[0][1]
    print("\n--- Attribution @ F=0 (fc_100 cum) ---")
    print(f"  Single-mineral (sb1994 Fo{fo:.0f} Ol):  {sb_vp:.3f} vs ref {ref_fc:.3f}  →  {sb_vp - ref_fc:+.3f} km/s")
    print("  Phase schedule: 100% Ol at F=0 (langmuir ≡ paper) — no phase error")
    print(f"  => F=0 cum error is entirely Ol elastic ({sb_vp - ref_fc:+.3f} km/s); fig2 scale fixes this.")


def _residual_breakdown(path: str, f: float) -> None:
    ref = digitized_residual_vp_km_s(path, f)  # type: ignore[arg-type]
    s = _state(path, f)
    melt = s.melt_oxides_wt
    primary = fig2_primary_melt()["oxides_wt_percent"]

    vp_fig2 = norm_velocity_from_bulk_wt(
        melt, p_pa=P_EVAL_PA, t_k=T_EVAL_K, mineral_backend="fig2", cipw_backend="fallback",
    )["vp_km_s"]
    vp_sb = float(s.vp_residual_norm_km_s)
    s_f2 = simulate_crystallization_path(
        primary_melt_oxides_wt=fig2_primary_melt()["oxides_wt_percent"],
        f_grid=np.array([0.0, f], dtype=float),
        path=path,  # type: ignore[arg-type]
        kd_engine="langmuir",
        fig2_ab_calibrate=False,
        mineral_backend="fig2",
        cipw_backend="fallback",
    )[-1]
    vp_lang_fig2 = float(s_f2.vp_residual_norm_km_s)
    vp_pri_fig2 = norm_velocity_from_bulk_wt(
        primary, p_pa=P_EVAL_PA, t_k=T_EVAL_K, mineral_backend="fig2", cipw_backend="fallback",
    )["vp_km_s"]
    vp_pri_sb = norm_velocity_from_bulk_wt(
        primary, p_pa=P_EVAL_PA, t_k=T_EVAL_K, mineral_backend="sb1994", cipw_backend="fallback",
    )["vp_km_s"]

    norm = cipw_norm_mass_fractions(melt, backend="fallback")
    cipw_pri = cipw_norm_mass_fractions(primary, backend="fallback")

    print(f"\n  F={f:.3f}  ref={ref:.3f}  lang+sb1994={vp_sb:.3f}  lang+fig2={vp_lang_fig2:.3f}")
    print(f"    Δ sb={vp_sb - ref:+.3f}  Δ fig2={vp_lang_fig2 - ref:+.3f}  backend-only={vp_sb - vp_fig2:+.3f}")
    print(f"    melt SiO2={melt['SiO2']:.2f} Al2O3={melt['Al2O3']:.2f} MgO={melt['MgO']:.2f} CaO={melt['CaO']:.2f}")
    print(f"    vs primary SiO2={primary['SiO2']:.2f} Al2O3={primary['Al2O3']:.2f} MgO={primary['MgO']:.2f}")
    print(f"    CIPW mass% Ol={100*norm.get('olivine',0):.1f} Pl={100*norm.get('plagioclase',0):.1f} "
          f"Cpx={100*norm.get('clinopyroxene',0):.1f} Qz={100*norm.get('quartz',0):.1f}")
    print(f"    primary CIPW Ol={100*cipw_pri.get('olivine',0):.1f} Pl={100*cipw_pri.get('plagioclase',0):.1f} "
          f"Cpx={100*cipw_pri.get('clinopyroxene',0):.1f}")

    chem_only_fig2 = vp_fig2 - vp_pri_fig2
    chem_only_sb = vp_sb - vp_pri_sb
    backend_at_same_melt = vp_sb - vp_fig2
    unexplained = (vp_sb - ref) - chem_only_sb - backend_at_same_melt

    print(f"    chemistry shift (sb1994):  primary→melt  {vp_pri_sb:.3f}→{vp_sb:.3f}  Δ={chem_only_sb:+.3f}")
    print(f"    chemistry shift (fig2):    primary→melt  {vp_pri_fig2:.3f}→{vp_fig2:.3f}  Δ={chem_only_fig2:+.3f}")
    print(f"    backend @ same melt:       fig2→sb1994   {vp_fig2:.3f}→{vp_sb:.3f}  Δ={backend_at_same_melt:+.3f}")
    print(f"    residual vs ref budget:    total Δ={vp_sb - ref:+.3f}  "
          f"≈ chem({chem_only_sb:+.3f}) + backend({backend_at_same_melt:+.3f}) + cross({unexplained:+.3f})")


def _section_b_residual() -> None:
    print("\n" + "=" * 72)
    print("B) F≥0.7 residual melt norm Vp — chemistry vs mineral backend")
    print("=" * 72)
    for path in ("fc_100", "polybaric_fc"):
        print(f"\n--- {path} ---")
        for f in (0.70, 0.796, 0.798):
            _residual_breakdown(path, f)


def _section_c_mid_cumulative() -> None:
    print("\n" + "=" * 72)
    print("C) Mid-F cumulative — phase schedule vs mineral backend")
    print("=" * 72)
    print(f"{'F':>5s} {'ref':>6s}  {'lang sb':>7s}  {'lang f2':>7s}  "
          f"{'paper sb':>8s}  {'Δphase sb':>9s}  {'Δmineral':>9s}")
    for f in (0.08, 0.22, 0.40, 0.55, 0.70):
        ref = digitized_cum_vp_km_s("fc_100", f)
        s_sb = simulate_crystallization_path(
            primary_melt_oxides_wt=fig2_primary_melt()["oxides_wt_percent"],
            f_grid=np.array([0.0, f], dtype=float),
            path="fc_100",
            kd_engine="langmuir",
            fig2_ab_calibrate=False,
            mineral_backend="sb1994",
            cipw_backend="fallback",
        )[-1]
        s_f2 = simulate_crystallization_path(
            primary_melt_oxides_wt=fig2_primary_melt()["oxides_wt_percent"],
            f_grid=np.array([0.0, f], dtype=float),
            path="fc_100",
            kd_engine="langmuir",
            fig2_ab_calibrate=False,
            mineral_backend="fig2",
            cipw_backend="fallback",
        )[-1]
        ol_p, cpx_p, pl_p = cumulative_phases_pct("fc_100", f)
        fo_p, an_p, di_p = incremental_compositions("fc_100", f)
        vp_paper_sb, _ = assemblage_vp_rho(
            ol_frac=ol_p / 100, pl_frac=pl_p / 100, cpx_frac=cpx_p / 100,
            fo_pct=fo_p, an_pct=an_p, di_pct=di_p,
            p_pa=P_EVAL_PA, t_k=T_EVAL_K, mineral_backend="sb1994",
        )
        d_phase = s_sb.vp_cumulate_km_s - vp_paper_sb
        d_mineral = vp_paper_sb - ref
        print(
            f"{f:5.2f} {ref:6.3f}  {s_sb.vp_cumulate_km_s:7.3f}  {s_f2.vp_cumulate_km_s:7.3f}  "
            f"{vp_paper_sb:8.3f}  {d_phase:+9.3f}  {d_mineral:+9.3f}"
        )
        if f in (0.22, 0.40):
            print(
                f"       phases langmuir Ol={100*s_sb.cum_ol:.0f} Pl={100*s_sb.cum_pl:.0f} "
                f"Cpx={100*s_sb.cum_cpx:.0f}  |  paper Ol={ol_p:.0f} Pl={pl_p:.0f} Cpx={cpx_p:.0f}"
            )

    print("\n--- Column guide ---")
    print("  d_phase sb = langmuir(sb1994) - paper_phases(sb1994)  -> crystallization path error")
    print("  d_mineral  = paper_phases(sb1994) - digitized ref       -> Ol/Pl/Cpx elastic offset")


def main() -> None:
    _section_a_f0_olivine()
    _section_b_residual()
    _section_c_mid_cumulative()
    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print("  F~0: sb1994 Fo90 Ol ~8.34 km/s vs paper ~7.59 — pure mineral offset; fig2 Ol scale fixes it")
    print("  F>=0.7 residual: BOTH backends overshoot ref (~+0.5 fig2, ~+0.8 sb1994 at F~0.8)")
    print("    - Evolved melt CIPW norm stays ~7.1-7.5 while paper frac.res ~6.7-6.9")
    print("    - sb1994 adds +0.24-0.34 vs fig2 on same melt (Pl/Cpx moduli higher)")
    print("  Mid-F cum: phase error (Ol-rich langmuir) dominates over mineral backend")


if __name__ == "__main__":
    main()
