"""
Run original BASALT.FOR (1990) Python port — DRIVER model 2 fractional crystallization.

Example (Fortran default PARMS: Ti=Tf=1000 °C, dT=10, offset=273.16 K):

    py -3.11 petrology/validation/run_basalt1990.py

Uses 1990 ARHENF Kd (10**(A/T)+B). Typical MORB may stay liquid-only at 1273 K
because ol/cpx Kd values are very large in this formulation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.fc.basalt1990 import Basalt1990System, driver_run
from petrology.fc.basalt1990.common import PNAMEA
from petrology.fc.basalt1990.kd_calc import kd_olivine_mgo_1990
from petrology.fc.wl_components import oxides_wt_to_csj

# Magnesian tholeiite — crystallizes with Langmuir-scale Kd; useful for solver check.
_DEMO_WT = {
    "SiO2": 45.0,
    "TiO2": 0.5,
    "Al2O3": 10.0,
    "FeO": 8.0,
    "MgO": 25.0,
    "CaO": 10.0,
    "Na2O": 1.0,
    "K2O": 0.1,
    "Cr2O3": 0.0,
}


def _print_step(temp_k: float, flr: float, res) -> None:
    fa = res.fa
    phases = ", ".join(f"{PNAMEA[i]}={fa[i]:.4f}" for i in range(len(fa)) if fa[i] > 1e-4)
    print(
        f"T={temp_k:7.2f} K  FLR={flr:.4f}  FL={res.fl:.4f}  nl={res.nl}  "
        f"nerr={res.nerr}  [{phases or 'liquid only'}]"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="BASALT.FOR (1990) Python DRIVER demo")
    p.add_argument("--model", type=int, default=2, choices=(1, 2, 3))
    p.add_argument("--ti", type=float, default=1000.0, help="Start T (°C if --celsius)")
    p.add_argument("--tf", type=float, default=900.0, help="End T (°C if --celsius)")
    p.add_argument("--dt", type=float, default=-10.0, help="Temperature step (K or °C)")
    p.add_argument("--celsius", action="store_true", default=True)
    p.add_argument("--kelvin", action="store_true", help="Ti/Tf already in Kelvin")
    p.add_argument("--demo", choices=("morb", "magnesian"), default="magnesian")
    p.add_argument("--kd", choices=("1990", "langmuir"), default="1990")
    args = p.parse_args()

    offset = 273.16 if not args.kelvin else 0.0
    if args.demo == "morb":
        from petrology.fc.wl1990 import load_kinzler1997_morb_primary

        csj = oxides_wt_to_csj(load_kinzler1997_morb_primary()["oxides_wt_percent"])
    else:
        csj = oxides_wt_to_csj(_DEMO_WT)

    print("=== BASALT.FOR (1990) Python port ===")
    print(f"Model {args.model}  Kd={args.kd}  demo={args.demo}")
    print(f"Ol MgO Kd @1273 K (1990 ARHENF): {kd_olivine_mgo_1990(1273.16):.2f}")
    print(f"CSJ (6 comp): {np.round(csj[:6], 4)}")
    print()

    if args.kd == "1990":
        steps = driver_run(
            csj,
            model=args.model,
            ti_k=args.ti,
            tf_k=args.tf,
            dt_k=args.dt,
            temp_offset_k=offset,
            fill_temp_kd=True,
        )
    else:
        from petrology.fc.wl_kd import (
            kd_olivine_feo,
            kd_olivine_mgo,
            kd_plagioclase_caa,
            kd_plagioclase_naal,
            wl_an_component,
        )
        from petrology.fc.wl_components import csj_to_oxides_wt

        steps = []
        temp = args.ti + offset
        t_end = args.tf + offset
        dt = float(args.dt)
        if abs(dt) < 1e-12:
            dt = -10.0 if t_end < temp else 10.0
        dt = -abs(dt) if t_end < temp else abs(dt)
        csj_run = csj.copy()
        fa = np.zeros(3)
        flr = 1.0
        sys = Basalt1990System(t_k=temp)

        for _ in range(500):
            sys.sync_temperature(temp)
            melt = csj_to_oxides_wt(csj_run[:6])
            an = wl_an_component(melt)
            sys.kd.fkda[1, 2] = kd_olivine_mgo(temp, 0.0)
            sys.kd.fkda[1, 3] = kd_olivine_feo(temp, 0.0)
            sys.kd.fkda[0, 0] = kd_plagioclase_caa(an, temp, 0.0)
            sys.kd.fkda[0, 1] = kd_plagioclase_naal(an, temp, 0.0)
            res = sys.solve(csj_run, fa0=fa, kd_init_mode=0, fill_temp_kd=False)
            fa = res.fa.copy()
            if args.model == 2:
                flr *= res.fl
            from petrology.fc.basalt1990.driver import DriverStep1990

            steps.append(
                DriverStep1990(temp_k=temp, flr=flr, model=args.model, nerr=res.nerr, nl=res.nl, result=res)
            )
            if res.nerr != 0:
                break
            temp_next = temp + dt
            if (t_end - temp_next) / dt <= 0.0:
                break
            temp = temp_next
            if args.model == 2:
                s = float(np.sum(res.clj[:6]))
                if s > 1e-12:
                    csj_run[:6] = res.clj[:6] / s

    for st in steps:
        _print_step(st.temp_k, st.flr, st.result)

    flrs = [s.flr for s in steps]
    if len(flrs) >= 2 and args.model == 2:
        print(f"\nFLR: {flrs[0]:.4f} → {flrs[-1]:.4f}")


if __name__ == "__main__":
    main()
