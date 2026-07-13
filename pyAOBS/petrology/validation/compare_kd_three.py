"""
Compare three Kd / crystallization engines for a user-supplied melt.

Engines
-------
1. **1990** — original BASALT.FOR ARHENF Kd (``10**(A/T)+B``), 1 atm.
2. **langmuir1992** — BASALT+langmuir.FOR STATE (``10**(A/T+B+C*P)``).
3. **heuristic** — wl1990 oxide saturation surrogates (Fig.2 default track).

Usage::

    py -3.11 petrology/validation/compare_kd_three.py
    py -3.11 petrology/validation/compare_kd_three.py --p-mpa 100 --t-start-c 1200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.fc.basalt1990.common import CNAMEJ, PNAMEA
from petrology.fc.basalt1990.kd_calc import (
    kd_olivine_feo_1990,
    kd_olivine_mgo_1990,
    kd_plagioclase_caa_1990,
)
from petrology.fc.basalt1990.solver import Basalt1990System
from petrology.fc.wl_components import csj_to_oxides_wt, oxides_wt_to_csj
from petrology.fc.wl_fc_state import state_fc_path, temprun_fc_to_f
from petrology.fc.wl_kd import (
    kd_olivine_feo,
    kd_olivine_mgo,
    kd_plagioclase_caa,
    wl_an_component,
)
from petrology.fc.wl_state import WLStateSystem, equilibrium_state
from petrology.fc.wl1990 import simulate_crystallization_path

# User melt (Kinzler-style MORB primary, wt%)
USER_WT = {
    "SiO2": 48.2,
    "TiO2": 0.94,
    "Al2O3": 16.4,
    "Cr2O3": 0.12,
    "FeO": 7.96,
    "MgO": 12.5,
    "CaO": 11.4,
    "K2O": 0.07,
    "Na2O": 2.27,
}


def _fmt_phases(fa: np.ndarray) -> str:
    parts = [f"{PNAMEA[i]}={fa[i]:.4f}" for i in range(len(fa)) if fa[i] > 1e-4]
    return ", ".join(parts) if parts else "liquid only"


def _fmt_qa(qa: np.ndarray) -> str:
    return "  ".join(f"{PNAMEA[i]} QA={qa[i]:+.3f}" for i in range(len(qa)))


def print_kd_table(t_k: float, p_kbar: float, csj: np.ndarray) -> None:
    an = wl_an_component(csj_to_oxides_wt(csj[:6]))
    print("\n" + "=" * 72)
    print(f"Kd @ T={t_k - 273.16:.0f}°C, P={p_kbar:.2f} kbar  (An≈{an:.2f})")
    print("=" * 72)
    print(f"{'':16s} {'1990 ARHENF':>14s} {'Langmuir 1992':>14s} {'ratio 92/90':>12s}")
    rows = [
        ("Ol MgO", kd_olivine_mgo_1990(t_k), kd_olivine_mgo(t_k, p_kbar)),
        ("Ol FeO", kd_olivine_feo_1990(t_k), kd_olivine_feo(t_k, p_kbar)),
        ("Pl CAA", kd_plagioclase_caa_1990(an, t_k), kd_plagioclase_caa(an, t_k, p_kbar)),
    ]
    for name, k90, k92 in rows:
        ratio = k92 / k90 if abs(k90) > 1e-12 else float("nan")
        print(f"{name:16s} {k90:14.2f} {k92:14.2f} {ratio:12.3f}")


def print_equilibrium_scan(csj: np.ndarray, temps_c: list[float], p_kbar: float) -> None:
    print("\n" + "=" * 72)
    print("单步平衡（Model 1 思路：每温度独立 STATE，无 FC 成分演化）")
    print("=" * 72)
    for eng, eng_name in [("1990", "1990 ARHENF"), ("langmuir1992", "Langmuir 1992 STATE")]:
        print(f"\n--- {eng_name} ---")
        print(f"{'T(°C)':>6s} {'nl':>3s} {'FL':>6s}  {'PLAG':>6s} {'OL':>6s} {'CPX':>6s}  phases / QA")
        for tc in temps_c:
            t_k = tc + 273.16
            if eng == "1990":
                sys = Basalt1990System(t_k=t_k)
                res = sys.solve(csj, fill_temp_kd=True)
                qa = sys.qa
            else:
                res = equilibrium_state(csj, t_k=t_k, p_kbar=p_kbar)
                qa = res.qa
            err = f" nerr={res.nerr}" if res.nerr else ""
            print(
                f"{tc:6.0f} {res.nl:3d} {res.fl:6.4f}  "
                f"{res.fa[0]:6.4f} {res.fa[1]:6.4f} {res.fa[2]:6.4f}  "
                f"{_fmt_phases(res.fa)}{err}"
            )
            if res.nl == 0:
                print(f"       {_fmt_qa(qa)}")


def print_fc_comparison(
    primary_wt: dict,
    csj: np.ndarray,
    f_grid: np.ndarray,
    *,
    p_mpa: float,
    t_fc_k: float,
) -> None:
    p_kbar = p_mpa / 100.0
    print("\n" + "=" * 72)
    print(f"分离结晶 FC 路径  @ P={p_mpa:.0f} MPa, T_start={t_fc_k - 273.16:.0f}°C")
    print("=" * 72)

    # --- Langmuir 1992 STATE FC ---
    print("\n--- Langmuir 1992 (wl_state STATE + TEMPRUN FC) ---")
    print(f"{'F':>5s} {'T(°C)':>6s} {'PLAG':>6s} {'OL':>6s} {'CPX':>6s}  Fo  An  Di  nerr")
    for f in f_grid:
        if f <= 1e-9:
            res = equilibrium_state(csj, t_k=t_fc_k, p_kbar=p_kbar)
            tc = t_fc_k - 273.16
        else:
            _, res, f_act, t_k = temprun_fc_to_f(
                csj,
                f_target=float(f),
                p_kbar=p_kbar,
                t_start_k=t_fc_k,
                t_end_k=973.16,
                dt_k=-5.0,
            )
            tc = t_k - 273.16
            f = f_act
        err = res.nerr
        print(
            f"{f:5.2f} {tc:6.0f} {res.fa[0]:6.4f} {res.fa[1]:6.4f} {res.fa[2]:6.4f}  "
            f"{res.fo_pct:4.0f} {res.an_pct:4.0f} {res.di_pct:4.0f}  {err}"
        )

    # --- 1990 via basalt1990_fc-style isothermal F stepping at fixed T ---
    print("\n--- 1990 ARHENF (等温 F 子步 FC @ fixed T) ---")
    print(f"{'F':>5s} {'PLAG':>6s} {'OL':>6s} {'CPX':>6s}  Fo  An  Di  nerr  note")
    from petrology.fc.basalt1990_fc import basalt1990_fc_path

    steps_1990 = basalt1990_fc_path(
        primary_wt,
        f_grid,
        path="fc_100",
        kd_mode="1990",
        p_fc_mpa=p_mpa,
        t_fc_k=t_fc_k,
        t_end_fc_k=t_fc_k,
        dt_fc_k=-5.0,
    )
    for st in steps_1990:
        note = "liquid" if float(np.sum(st.fa)) < 1e-4 else ""
        print(
            f"{st.f_solid:5.2f} {st.fa[0]:6.4f} {st.fa[1]:6.4f} {st.fa[2]:6.4f}  "
            f"{st.fo_pct:4.0f} {st.an_pct:4.0f} {st.di_pct:4.0f}  {st.result.nerr}  {note}"
        )

    # --- basalt1990 + langmuir Kd injected ---
    print("\n--- basalt1990 solver + Langmuir Kd 注入 (等温 F 子步) ---")
    steps_inj = basalt1990_fc_path(
        primary_wt,
        f_grid,
        path="fc_100",
        kd_mode="langmuir",
        p_fc_mpa=p_mpa,
        t_fc_k=t_fc_k,
        t_end_fc_k=t_fc_k,
        dt_fc_k=-5.0,
    )
    for st in steps_inj:
        print(
            f"{st.f_solid:5.2f} {st.fa[0]:6.4f} {st.fa[1]:6.4f} {st.fa[2]:6.4f}  "
            f"{st.fo_pct:4.0f} {st.an_pct:4.0f} {st.di_pct:4.0f}  {st.result.nerr}"
        )

    # --- heuristic wl1990 ---
    print("\n--- heuristic (wl1990 oxide saturation, Fig.2 track) ---")
    print(f"{'F':>5s} {'cum_ol':>6s} {'cum_pl':>6s} {'cum_cpx':>6s}  Fo  An  Di  Vp")
    for eng in ("heuristic",):
        st_list = simulate_crystallization_path(
            primary_melt_oxides_wt=primary_wt,
            f_grid=f_grid,
            path="fc_100",
            fig2_ab_calibrate=False,
            kd_engine=eng,
            mineral_backend="fig2",
            cipw_backend="fallback",
            p_fc_mpa=p_mpa,
            t_fc_k=t_fc_k,
        )
        for s in st_list:
            print(
                f"{s.f_solid:5.2f} {s.cum_ol:6.3f} {s.cum_pl:6.3f} {s.cum_cpx:6.3f}  "
                f"{s.cum_fo_pct:4.0f} {s.cum_an_pct:4.0f} {s.cum_di_pct:4.0f}  {s.vp_cumulate_km_s:.3f}"
            )


def print_cooling_driver(csj: np.ndarray, *, ti_c: float, tf_c: float, dt_c: float, p_kbar: float) -> None:
    from petrology.fc.basalt1990.driver import driver_run

    print("\n" + "=" * 72)
    print(f"降温 DRIVER Model 2: {ti_c:.0f}°C → {tf_c:.0f}°C, ΔT={dt_c:.0f}°C")
    print("=" * 72)

    print("\n--- 1990 ARHENF ---")
    steps = driver_run(csj, model=2, ti_k=ti_c, tf_k=tf_c, dt_k=dt_c, temp_offset_k=273.16)
    _print_driver_steps(steps, every=3)

    print("\n--- Langmuir 1992 (WLStateSystem 逐步 FC) ---")
    dt_k = -abs(dt_c)
    t_k = ti_c + 273.16
    t_end = tf_c + 273.16
    fa = np.zeros(3)
    flr = 1.0
    csj_run = csj.copy()
    sys = WLStateSystem(t_k=t_k, p_kbar=p_kbar)
    n = 0
    while True:
        res = sys.solve(csj_run, fa0=fa)
        if n % 3 == 0 or res.nerr != 0 or res.nl > 0:
            tc = t_k - 273.16
            print(
                f" TEMP {tc:6.0f}  FLR {flr:7.4f}  "
                f"PLAG={res.fa[0]:.4f} OL={res.fa[1]:.4f} CPX={res.fa[2]:.4f}  "
                f"nl={res.nl} nerr={res.nerr}  {_fmt_phases(res.fa)}"
            )
        if res.nerr != 0:
            break
        flr *= res.fl
        fa = res.fa.copy()
        s = float(np.sum(res.clj[:6]))
        if s > 1e-12:
            csj_run[:6] = res.clj[:6] / s
        t_k += dt_k
        sys.t_k = t_k
        n += 1
        if (t_end - t_k) / dt_k <= 0.0 or n > 200:
            break


def _print_driver_steps(steps, every: int = 3) -> None:
    for i, st in enumerate(steps):
        if i % every != 0 and st.result.nl == 0:
            continue
        tc = st.temp_k - 273.16
        fa = st.result.fa
        print(
            f" TEMP {tc:6.0f}  FLR {st.flr:7.4f}  "
            f"PLAG={fa[0]:.4f} OL={fa[1]:.4f} CPX={fa[2]:.4f}  "
            f"nl={st.nl} nerr={st.nerr}  {_fmt_phases(fa)}"
        )
        if st.nerr != 0:
            break


def main() -> None:
    p = argparse.ArgumentParser(description="Compare 1990 / Langmuir1992 / heuristic Kd engines")
    p.add_argument("--p-mpa", type=float, default=100.0, help="Pressure (MPa), default 100 = 1 kbar")
    p.add_argument("--t-start-c", type=float, default=1200.0, help="FC / equilibrium start T (°C)")
    args = p.parse_args()

    p_kbar = args.p_mpa / 100.0
    t_fc_k = args.t_start_c + 273.16
    csj = oxides_wt_to_csj(USER_WT)
    f_grid = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    print("=" * 72)
    print("三引擎 Kd 对比 — 用户 MORB 成分")
    print("=" * 72)
    print("输入 wt%:", ", ".join(f"{k}={v}" for k, v in USER_WT.items()))
    print("CSJ:", "  ".join(f"{CNAMEJ[j]}={csj[j]:.4f}" for j in range(6)))

    print_kd_table(t_fc_k, p_kbar, csj)
    print_equilibrium_scan(csj, [1200, 1150, 1100, 1050, 1000, 950, 900], p_kbar)
    print_fc_comparison(USER_WT, csj, f_grid, p_mpa=args.p_mpa, t_fc_k=t_fc_k)
    print_cooling_driver(csj, ti_c=1200, tf_c=900, dt_c=-10, p_kbar=p_kbar)


if __name__ == "__main__":
    main()
