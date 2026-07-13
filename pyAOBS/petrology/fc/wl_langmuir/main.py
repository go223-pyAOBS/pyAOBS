"""
Interactive MAIN — BASALT+langmuir.FOR (Langmuir 1992 pressure extension).

Tasks 1–11 and 99 (task 11: single vs polybaric crystallization).

Run::

    py -3.11 -m petrology.fc.wl_langmuir

Batch::

    py -3.11 -m petrology.fc.wl_langmuir --batch --model 2 --polybaric \\
        --ti 1200 --tf 900 --dt -10 --p-high 8 --p-low 1 --dp 0.5
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import numpy as np

from petrology.fc.basalt1990.common import CNAMEJ, NCOMPT, PNAMEA
from petrology.fc.basalt1990.main import (
    Basalt1990Session,
    load_cdat_file,
    task_enter_data as _task_enter_data_shared,
)
from petrology.fc.cdat_library import get_csj, list_samples, resolve_cdat_path
from petrology.fc.wl_driver import TemprunSnapshot, langmuir_driver_run
from petrology.fc.wl_state_stabilize import DEFAULT_FC_STABILIZE, StateStabilize


@dataclass
class BasaltLangmuirSession:
    """Fortran MAIN + COMMON for BASALT+langmuir (MODES 1–6)."""

    cdat: list[np.ndarray] = field(default_factory=list)
    model: int = 0
    init_phases: bool = True  # MODES(2)
    printer_on: bool = False  # MODES(3)
    full_output: bool = True  # MODES(4): True = summary (Fortran inverted)
    phase_only: bool = False  # MODES(5)
    polybaric: bool = False  # MODES(6)
    temp_offset_k: float = 273.16  # separate from PARMS(5) P_HIGH clash in Fortran
    parms: dict[str, float] = field(
        default_factory=lambda: {
            "ti": 1000.0,
            "tf": 1000.0,
            "dt": 10.0,
            "dp": 0.5,
            "p_high": 8.0,
            "p_low": 0.1,
        }
    )
    p_single_kbar: float = 0.0
    nclo: int = 1
    nchi: int = 1
    no_data: bool = True
    no_model: bool = True
    no_temp: bool = True
    last_snapshots: list[TemprunSnapshot] = field(default_factory=list)
    stabilize: StateStabilize = field(default_factory=lambda: DEFAULT_FC_STABILIZE)

    @property
    def ncdat(self) -> int:
        return len(self.cdat)


def _menu(out: TextIO) -> None:
    out.write(
        "\n"
        " 1: ENTER DATA FROM KEYBD OR FILE\n"
        " 2: SPECIFY DATA TO BE RUN\n"
        " 3: CHOOSE CRYSTALLIZATION MODEL\n"
        " 4: INITIALIZE ON/OFF\n"
        " 5: FULL OR SUMMARY OUTPUT\n"
        " 6: REPORT CHANGES ONLY\n"
        " 7: PRINTER ON/OFF\n"
        " 8: INPUT T & P PARAMETERS\n"
        " 9: RUN THE MODEL\n"
        "10: SELECT TEMP SCALE\n"
        "11: SWITCH SINGLE/POLYBARIC MODE\n"
        "99: EXIT THIS PROGRAM\n"
    )


def _print_snapshot(
    snap: TemprunSnapshot,
    session: BasaltLangmuirSession,
    out: TextIO,
) -> None:
    res = snap.result
    t_c = snap.t_k - session.temp_offset_k
    if session.full_output:
        out.write(
            f" TEMP {t_c:7.2f}  P {snap.p_kbar:6.2f}  FLR {snap.flr:7.4f}  "
            f"{' '.join(f'{PNAMEA[i][:4]}={res.fa[i]:7.4f}' for i in range(len(res.fa)))}  "
            f"{' '.join(f'{CNAMEJ[j][:4]}={res.clj[j]:7.4f}' for j in range(min(6, NCOMPT)))}\n"
        )
    else:
        comps = " ".join(f"{CNAMEJ[j][:4]}={res.clj[j]:7.4f}" for j in range(min(6, NCOMPT)))
        out.write(
            f" {t_c:5.0f} {snap.p_kbar:6.2f} {snap.flr:7.4f} "
            f"{''.join(f'{res.fa[i]:7.4f}' for i in range(len(res.fa)))}   {comps}\n"
        )
    if res.nerr == 1:
        out.write(" MATRIX INVERSION PROBLEM IN STATE\n")
    elif res.nerr == 2:
        out.write(" MAXIMUM ITERATIONS REACHED IN STATE\n")


def langmuir_driver_for_case(
    session: BasaltLangmuirSession,
    csj: np.ndarray,
    *,
    out: TextIO | None = None,
) -> list[TemprunSnapshot]:
    """Execute BASALT+langmuir DRIVER for one CSJ row."""
    out = out or sys.stdout
    if session.model not in (1, 2, 3):
        raise ValueError("Model not set (task 3)")

    ti_k = session.parms["ti"] + session.temp_offset_k
    tf_k = session.parms["tf"] + session.temp_offset_k
    dt_k = float(session.parms["dt"])
    if session.parms["ti"] > session.parms["tf"]:
        dt_k = -abs(dt_k)
    else:
        dt_k = abs(dt_k)

    snaps = langmuir_driver_run(
        csj,
        model=session.model,
        ti_k=ti_k,
        tf_k=tf_k,
        dt_k=dt_k,
        polybaric=session.polybaric and session.model == 2,
        p_single_kbar=session.p_single_kbar,
        p_high_kbar=session.parms["p_high"],
        p_low_kbar=session.parms["p_low"],
        dp_kbar=session.parms["dp"],
        init_phases=session.init_phases,
        stabilize=session.stabilize,
    )
    session.last_snapshots = snaps

    if not session.full_output and snaps:
        out.write("TEMP  P(kbar)  FLR   " + " ".join(f"{PNAMEA[i]:>4}" for i in range(3)) + "\n")

    nchang_prev = -1
    for snap in snaps:
        if session.phase_only:
            nchang = int(sum(1 for fa in snap.result.fa if fa > 1e-6))
            if nchang == nchang_prev:
                continue
            nchang_prev = nchang
        _print_snapshot(snap, session, out)
        if snap.result.nerr != 0:
            break

    return snaps


def task_enter_data(session: BasaltLangmuirSession, inp: TextIO, out: TextIO) -> None:
    """Task 1 — CDAT keyboard/file (shared with basalt1990 MAIN)."""
    wrap = Basalt1990Session()
    wrap.cdat = list(session.cdat)
    wrap.no_data = session.no_data
    wrap.nchi = session.nchi
    _task_enter_data_shared(wrap, inp, out)
    session.cdat = wrap.cdat
    session.no_data = wrap.no_data
    session.nchi = wrap.nchi


def task_run(session: BasaltLangmuirSession, inp: TextIO, out: TextIO) -> None:
    if session.no_data:
        out.write("No data — use task 1 first.\n")
        return
    if session.no_model:
        out.write("No model — use task 3 first.\n")
        return
    if session.no_temp:
        out.write("No T/P parameters — use task 8 first.\n")
        return

    nclo = max(1, session.nclo)
    nchi = min(session.nchi, session.ncdat)
    if nclo > nchi:
        nclo, nchi = nchi, nclo

    mode = "POLYBARIC" if session.polybaric else "SINGLE P"
    out.write(f"\nMODE: {mode}  MODEL: {session.model}\n")

    for k in range(nclo, nchi + 1):
        out.write(f"\n--- CASE {k} ---\n")
        langmuir_driver_for_case(session, session.cdat[k - 1], out=out)


def interactive_loop(
    session: BasaltLangmuirSession | None = None,
    inp: TextIO | None = None,
    out: TextIO | None = None,
) -> None:
    session = session or BasaltLangmuirSession()
    inp = inp or sys.stdin
    out = out or sys.stdout

    while True:
        out.write("\nTASK? <CR> FOR MENU: ")
        out.flush()
        line = inp.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            _menu(out)
            continue
        try:
            task = int(line)
        except ValueError:
            out.write("Invalid task.\n")
            continue

        if task == 99:
            out.write("EXIT\n")
            break
        if task == 1:
            task_enter_data(session, inp, out)
        elif task == 2:
            out.write("RANGE OF DATA TO RUN? (nlo nhi): ")
            out.flush()
            parts = inp.readline().split()
            if len(parts) >= 2:
                session.nclo, session.nchi = int(parts[0]), int(parts[1])
        elif task == 3:
            out.write(
                "MODEL (1=equilibrium, 2=fractional crystallization, 3=fractional melting): "
            )
            out.flush()
            m = int(inp.readline().strip())
            if 1 <= m <= 3:
                session.model = m
                session.no_model = False
            else:
                session.no_model = True
        elif task == 4:
            session.init_phases = not session.init_phases
            out.write(f"INITIALIZATION IS {'ON' if session.init_phases else 'OFF'}\n")
        elif task == 5:
            session.full_output = not session.full_output
            out.write(f"{'SUMMARY OUTPUT ONLY' if session.full_output else 'FULL OUTPUT MODE'}\n")
        elif task == 6:
            session.phase_only = not session.phase_only
            out.write(
                f"OUTPUT FOR {'PHASE APPEARANCES ONLY' if session.phase_only else 'ANY TEMPERATURE'}\n"
            )
        elif task == 7:
            session.printer_on = not session.printer_on
            out.write(f"PRINTER IS {'ON' if session.printer_on else 'OFF'}\n")
        elif task == 8:
            out.write("ENTER Ti, Tf, dT, DP, P_HIGH, P_LOW (6 floats, kbar for P): ")
            out.flush()
            vals = [float(x) for x in inp.readline().split()]
            if len(vals) >= 1 and vals[0] > 0:
                keys = ("ti", "tf", "dt", "dp", "p_high", "p_low")
                for i, k in enumerate(keys):
                    if i < len(vals):
                        session.parms[k] = vals[i]
                session.no_temp = False
                if session.parms["ti"] > session.parms["tf"]:
                    session.parms["dt"] = -abs(session.parms["dt"])
                else:
                    session.parms["dt"] = abs(session.parms["dt"])
            else:
                session.no_temp = True
        elif task == 9:
            task_run(session, inp, out)
        elif task == 10:
            if abs(session.temp_offset_k - 273.16) < 1.0:
                session.temp_offset_k = 0.0
                out.write("KELVIN TEMPERATURE SCALE ASSUMED\n")
            else:
                session.temp_offset_k = 273.16
                out.write("CELSIUS TEMPERATURE SCALE ASSUMED\n")
        elif task == 11:
            session.polybaric = not session.polybaric
            if session.polybaric:
                out.write("POLYBARIC MULTI-PRESSURE MODE (LANGMUIR 1992)\n")
            else:
                out.write("SINGLE PRESSURE MODE (P=0 kbar default)\n")
        else:
            _menu(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="BASALT+langmuir.FOR interactive MAIN")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--data", type=Path, help="CDAT file or catalog sample id")
    parser.add_argument("--sample", type=str, help="Catalog sample id (see --list-samples)")
    parser.add_argument("--list-samples", action="store_true", help="List CDAT catalog and exit")
    parser.add_argument("--model", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--ti", type=float, default=1000.0)
    parser.add_argument("--tf", type=float, default=900.0)
    parser.add_argument("--dt", type=float, default=-10.0)
    parser.add_argument("--dp", type=float, default=0.5, help="Polybaric dP (kbar)")
    parser.add_argument("--p-high", type=float, default=8.0, help="P_HIGH (kbar)")
    parser.add_argument("--p-low", type=float, default=1.0, help="P_LOW (kbar)")
    parser.add_argument("--p-kbar", type=float, default=0.0, help="Single-mode pressure (kbar)")
    parser.add_argument("--polybaric", action="store_true", help="MODES(6)=1")
    parser.add_argument("--kelvin", action="store_true", help="Ti/Tf in Kelvin")
    parser.add_argument("--no-stabilize", action="store_true", help="Disable FC stabilization")
    args = parser.parse_args(argv)

    if args.list_samples:
        for s in list_samples():
            print(f"{s.id:28s}  {s.name}  [{', '.join(s.tags)}]")
        return

    stab = DEFAULT_FC_STABILIZE
    if args.no_stabilize:
        from petrology.fc.wl_state_stabilize import NO_STABILIZE

        stab = NO_STABILIZE

    if args.batch:
        session = BasaltLangmuirSession()
        session.model = args.model
        session.no_model = False
        session.no_temp = False
        session.polybaric = args.polybaric
        session.temp_offset_k = 0.0 if args.kelvin else 273.16
        session.p_single_kbar = args.p_kbar
        session.stabilize = stab
        session.parms.update(
            {
                "ti": args.ti,
                "tf": args.tf,
                "dt": args.dt,
                "dp": args.dp,
                "p_high": args.p_high,
                "p_low": args.p_low,
            }
        )

        if args.sample:
            session.cdat = [get_csj(args.sample)]
        elif args.data:
            session.cdat = load_cdat_file(resolve_cdat_path(str(args.data)))
        else:
            from petrology.fc.wl_components import oxides_wt_to_csj

            demo = {
                "SiO2": 45,
                "TiO2": 0.5,
                "Al2O3": 10,
                "FeO": 8,
                "MgO": 25,
                "CaO": 10,
                "Na2O": 1,
                "K2O": 0.1,
                "Cr2O3": 0,
            }
            session.cdat = [oxides_wt_to_csj(demo)]
        session.no_data = False
        session.nchi = session.ncdat
        task_run(session, sys.stdin, sys.stdout)
        return

    interactive_loop()


if __name__ == "__main__":
    main()
