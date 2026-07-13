"""
Interactive MAIN menu — Weaver & Langmuir (1990) BASALT.FOR (version 2.0).

Mirrors Fortran tasks 1–10 and 99.  Run:

    py -3.11 -m petrology.fc.basalt1990.main

Or non-interactive batch:

    py -3.11 -m petrology.fc.basalt1990.main --batch --model 2 --ti 1000 --tf 900 --dt -10
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import numpy as np

from .common import CNAMEJ, NCOMPT, PNAMEA
from ..cdat_library import load_cdat_file, resolve_cdat_path, save_cdat_file, get_csj, list_samples
from .driver import DriverStep1990, driver_run


@dataclass
class Basalt1990Session:
    """Fortran MAIN + COMMON session state."""

    cdat: list[np.ndarray] = field(default_factory=list)
    model: int = 0  # 0=unset, 1=eq, 2=FC, 3=FM
    init_phases: bool = True  # MODES(2): 1=reset FA each run segment
    printer_on: bool = False  # MODES(3)
    full_output: bool = True  # MODES(4): False=full, True=summary only (Fortran inverted)
    phase_only: bool = False  # MODES(5)
    temp_offset_k: float = 273.16  # PARMS(5): 273.16=°C input, 0=Kelvin
    parms: dict[str, float] = field(
        default_factory=lambda: {
            "ti": 1000.0,
            "tf": 1000.0,
            "dt": 10.0,
            "dp": 10.0,
            "flr": 0.5,
        }
    )
    nclo: int = 1
    nchi: int = 1
    no_data: bool = True
    no_model: bool = True
    no_temp: bool = True
    last_steps: list[DriverStep1990] = field(default_factory=list)

    @property
    def ncdat(self) -> int:
        return len(self.cdat)


def _menu(out: TextIO) -> None:
    out.write(
        "\n"
        " 1: ENTER DATA FROM KEYBD OR FILE\n"
        " 2: SPECIFY DATA TO BE RUN\n"
        " 3: CHOOSE MODEL TO RUN\n"
        " 4: INITIALIZE ON/OFF\n"
        " 5: FULL OR SUMMARY OUTPUT\n"
        " 6: REPORT CHANGES ONLY\n"
        " 7: PRINTER ON/OFF\n"
        " 8: ENTER TEMP, ETC AND RUN THE MODEL\n"
        " 9: RUN THE MODEL\n"
        "10: SELECT TEMP SCALE\n"
        "99: EXIT THIS PROGRAM\n"
    )


def _read_csj_line(prompt: str, inp: TextIO, out: TextIO) -> np.ndarray | None:
    out.write(prompt)
    out.flush()
    line = inp.readline()
    if not line or not line.strip():
        return None
    vals = [float(x) for x in line.split()]
    if len(vals) < NCOMPT:
        vals.extend([0.0] * (NCOMPT - len(vals)))
    arr = np.array(vals[:NCOMPT], dtype=float)
    if float(np.sum(np.abs(arr))) <= 1e-6:
        return None
    s = float(np.sum(arr[:6]))
    if s > 1e-12:
        arr[:6] /= s
    return arr


def _print_step_summary(step: DriverStep1990, session: Basalt1990Session, out: TextIO) -> None:
    res = step.result
    if session.full_output:
        out.write(
            f" TEMP {step.temp_k:7.2f}  FLR {step.flr:7.4f}  "
            f"{' '.join(f'{PNAMEA[i][:4]}={res.fa[i]:7.4f}' for i in range(len(res.fa)))}  "
            f"{' '.join(f'{CNAMEJ[j][:4]}={res.clj[j]:7.4f}' for j in range(min(6, NCOMPT)))}\n"
        )
    else:
        comps = " ".join(f"{CNAMEJ[j][:4]}={res.clj[j]:7.4f}" for j in range(min(6, NCOMPT)))
        out.write(
            f" {step.temp_k:5.0f} {step.flr:7.4f} "
            f"{''.join(f'{res.fa[i]:7.4f}' for i in range(len(res.fa)))}   {comps}\n"
        )
    if res.nerr == 1:
        out.write(" MATRIX INVERSION PROBLEM IN STATE\n")
    elif res.nerr == 2:
        out.write(" MAXIMUM ITERATIONS REACHED IN STATE\n")


def run_driver_for_case(
    session: Basalt1990Session,
    csj: np.ndarray,
    *,
    out: TextIO | None = None,
) -> list[DriverStep1990]:
    """Execute Fortran DRIVER for one CSJ row."""
    out = out or sys.stdout
    if session.model not in (1, 2, 3):
        raise ValueError("Model not set (task 3)")

    steps = driver_run(
        csj,
        model=session.model,
        ti_k=session.parms["ti"],
        tf_k=session.parms["tf"],
        dt_k=session.parms["dt"],
        dp_k=session.parms["dp"],
        temp_offset_k=session.temp_offset_k,
        init_phases=session.init_phases,
        sync_sparm_temp=True,
        fill_temp_kd=True,
    )
    session.last_steps = steps

    if not session.full_output and steps:
        out.write("TEMP   FLR   " + " ".join(f"{PNAMEA[i]:>4}" for i in range(3)) + "\n")

    nchang_prev = -1
    for step in steps:
        if session.phase_only:
            nchang = int(sum(1 for i, fa in enumerate(step.result.fa) if fa > 1e-6))
            if nchang == nchang_prev:
                continue
            nchang_prev = nchang
        _print_step_summary(step, session, out)
        if step.nerr != 0:
            break

    return steps


def task_enter_data(session: Basalt1990Session, inp: TextIO, out: TextIO) -> None:
    out.write("DATA FILE PATH? <CR> FOR KEYBOARD: ")
    out.flush()
    path_s = inp.readline().strip()
    if path_s:
        path = Path(path_s)
        if not path.is_file():
            out.write(f"ERROR: file not found: {path}\n")
            return
        session.cdat = load_cdat_file(path)
        session.no_data = len(session.cdat) == 0
        session.nchi = max(1, session.ncdat)
        out.write(f"Loaded {session.ncdat} case(s) from {path}\n")
        return

    out.write("COMPONENTS: " + " ".join(CNAMEJ[:NCOMPT]) + "\n")
    out.write("ENTER DATA (7 floats, blank line to end):\n")
    session.cdat = []
    for icase in range(1, 51):
        row = _read_csj_line(f" CASE {icase:3d}? ", inp, out)
        if row is None:
            break
        session.cdat.append(row)
    session.no_data = len(session.cdat) == 0
    session.nchi = max(1, session.ncdat)
    if session.cdat:
        out.write(f"Stored {session.ncdat} case(s).\n")


def task_run(session: Basalt1990Session, inp: TextIO, out: TextIO) -> None:
    if session.no_data:
        out.write("No data — use task 1 first.\n")
        return
    if session.no_model:
        out.write("No model — use task 3 first.\n")
        return
    if session.no_temp:
        out.write("No temperature params — use task 8 first.\n")
        return

    nclo = max(1, session.nclo)
    nchi = min(session.nchi, session.ncdat)
    if nclo > nchi:
        nclo, nchi = nchi, nclo

    for k in range(nclo, nchi + 1):
        out.write(f"\n--- CASE {k} ---\n")
        run_driver_for_case(session, session.cdat[k - 1], out=out)


def interactive_loop(session: Basalt1990Session | None = None, inp: TextIO | None = None, out: TextIO | None = None) -> None:
    session = session or Basalt1990Session()
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
            out.write("MODEL (1=equilibrium, 2=fractional crystallization, 3=fractional melting): ")
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
            out.write("ENTER TI, TF, DT, DP, FLR (5 floats): ")
            out.flush()
            vals = [float(x) for x in inp.readline().split()]
            if len(vals) >= 1 and vals[0] > 0:
                keys = ("ti", "tf", "dt", "dp", "flr")
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
        else:
            _menu(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="BASALT.FOR (1990) interactive MAIN")
    parser.add_argument("--batch", action="store_true", help="Non-interactive single run")
    parser.add_argument("--data", type=Path, help="CDAT file path or sample id under data/cdat/")
    parser.add_argument(
        "--sample",
        type=str,
        help="Catalog sample id (e.g. kinzler1997_morb_primary, magnesian_tholeiite)",
    )
    parser.add_argument("--list-samples", action="store_true", help="List CDAT catalog and exit")
    parser.add_argument("--model", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--ti", type=float, default=1000.0)
    parser.add_argument("--tf", type=float, default=900.0)
    parser.add_argument("--dt", type=float, default=-10.0)
    parser.add_argument("--dp", type=float, default=10.0)
    parser.add_argument("--kelvin", action="store_true", help="Ti/Tf in Kelvin (no +273.16)")
    args = parser.parse_args(argv)

    if args.list_samples:
        for s in list_samples():
            print(f"{s.id:28s}  {s.name}  [{', '.join(s.tags)}]")
        return

    if args.batch:
        session = Basalt1990Session()
        session.model = args.model
        session.no_model = False
        session.no_temp = False
        session.temp_offset_k = 0.0 if args.kelvin else 273.16
        session.parms.update({"ti": args.ti, "tf": args.tf, "dt": args.dt, "dp": args.dp})

        if args.sample:
            session.cdat = [get_csj(args.sample)]
        elif args.data:
            session.cdat = load_cdat_file(resolve_cdat_path(str(args.data)))
        else:
            from petrology.fc.wl_components import oxides_wt_to_csj

            demo = {
                "SiO2": 45, "TiO2": 0.5, "Al2O3": 10, "FeO": 8,
                "MgO": 25, "CaO": 10, "Na2O": 1, "K2O": 0.1, "Cr2O3": 0,
            }
            session.cdat = [oxides_wt_to_csj(demo)]
        session.no_data = False
        session.nchi = session.ncdat
        task_run(session, sys.stdin, sys.stdout)
        return

    interactive_loop()


if __name__ == "__main__":
    main()
