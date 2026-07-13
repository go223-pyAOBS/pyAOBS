"""
Research diagnostics: BASALT.FOR (1990) vs BASALT+langmuir.FOR vs Python ports.

Prints KDCALC call semantics, QA saturation scan, and TEMPRUN/STATE divergence.

    py -3.11 petrology/validation/research_basalt_fortran.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.fc.basalt1990.common import CNAMEJ, PNAMEA
from petrology.fc.basalt1990.kd_calc import kd_olivine_mgo_1990
from petrology.fc.basalt1990.solver import Basalt1990System
from petrology.fc.wl_components import oxides_wt_to_csj
from petrology.fc.wl_driver import temprun_model2
from petrology.fc.wl_kd import kd_olivine_mgo, mpa_to_kbar
from petrology.fc.wl_state import WLStateSystem, equilibrium_state

# Kinzler MORB primary + user MORB from prior session
MELTS = {
    "kinzler_morb": None,  # filled below
    "user_morb": {
        "SiO2": 48.2,
        "TiO2": 0.94,
        "Al2O3": 16.4,
        "Cr2O3": 0.12,
        "FeO": 7.96,
        "MgO": 12.5,
        "CaO": 11.4,
        "K2O": 0.07,
        "Na2O": 2.27,
    },
    "magnesian_demo": {
        "SiO2": 45.0,
        "TiO2": 0.5,
        "Al2O3": 10.0,
        "FeO": 8.0,
        "MgO": 25.0,
        "CaO": 10.0,
        "Na2O": 1.0,
        "K2O": 0.1,
        "Cr2O3": 0.0,
    },
}


def _qa_line(sys, label: str) -> None:
    parts = [f"{PNAMEA[i]} QA={sys.qa[i]:+.3f}" for i in range(3)]
    print(f"  {label}: " + "  ".join(parts))


def scan_qa_literal_vs_fixed(csj: np.ndarray, t_k: float, p_kbar: float) -> None:
    """Compare Fortran-literal STATE entry (KDCALC 1) vs Python wl_state (mode 4)."""
    print(f"\n--- QA scan T={t_k - 273.16:.0f}°C P={p_kbar:.2f} kbar ---")

    # 1990 port: fill_temp_kd=False, kd_init_mode=1 → literal Fortran STATE (Ol/Cpx Kd=0)
    s90_lit = Basalt1990System(t_k=t_k)
    r90_lit = s90_lit.solve(csj, kd_init_mode=1, fill_temp_kd=False)
    _qa_line(s90_lit, "1990 literal STATE (Kd mode 1 only)")
    print(
        f"  → nl={r90_lit.nl} FL={r90_lit.fl:.4f} "
        f"FA=[{r90_lit.fa[0]:.4f},{r90_lit.fa[1]:.4f},{r90_lit.fa[2]:.4f}]"
    )

    # 1990 port: Python default fix
    s90_fix = Basalt1990System(t_k=t_k)
    r90_fix = s90_fix.solve(csj, fill_temp_kd=True)
    _qa_line(s90_fix, "1990 Python default (fill_temp_kd)")
    print(
        f"  → nl={r90_fix.nl} Ol Kd={s90_fix.kd.fkda[1,2]:.1f} "
        f"FA=[{r90_fix.fa[0]:.4f},{r90_fix.fa[1]:.4f},{r90_fix.fa[2]:.4f}]"
    )

    # Langmuir: wl_state (kd_calc 4 at entry — not literal STATE)
    res_lm = equilibrium_state(csj, t_k=t_k, p_kbar=p_kbar)
    _qa_line_from_result(res_lm, "langmuir wl_state (KDCALC 4 entry)")
    print(
        f"  → nl={res_lm.nl} Ol Kd={res_lm.fkda[1,2]:.2f} "
        f"FA=[{res_lm.fa[0]:.4f},{res_lm.fa[1]:.4f},{res_lm.fa[2]:.4f}] nerr={res_lm.nerr}"
    )

    # Demonstrate TEMPRUN+literal STATE conflict: manually zero after mode 4
    sys = WLStateSystem(t_k=t_k, p_kbar=p_kbar)
    sys.csj[:6] = csj[:6]
    sys.kd_calc(4)
    ol_kd_before = sys.fkda[1, 2]
    sys.kd_calc(1)  # what Fortran STATE does first
    print(
        f"  langmuir TEMPRUN→STATE conflict: Ol Kd after mode4={ol_kd_before:.2f} "
        f"→ after STATE KDCALC(1)={sys.fkda[1,2]:.2f}"
    )


def _qa_line_from_result(res, label: str) -> None:
    parts = [f"{PNAMEA[i]} QA={res.qa[i]:+.3f}" for i in range(3)]
    print(f"  {label}: " + "  ".join(parts))


def temprun_trace(csj: np.ndarray, *, p_kbar: float, t_start: float, t_end: float) -> None:
    print(f"\n--- TEMPRUN MODEL 2 (langmuir) P={p_kbar} kbar ---")
    _, _, flr, _, snaps = temprun_model2(
        csj,
        p_kbar=p_kbar,
        t_start_k=t_start,
        t_end_k=t_end,
        dt_k=-10.0,
    )
    for i, s in enumerate(snaps[:8]):
        fa = s.result.fa
        print(
            f"  step {i:2d} T={s.t_k - 273.16:6.0f}°C FLR={s.flr:.4f} "
            f"PL={fa[0]:.4f} OL={fa[1]:.4f} CPX={fa[2]:.4f} nerr={s.result.nerr}"
        )
    if len(snaps) > 8:
        print(f"  ... ({len(snaps)} steps total, final FLR={flr:.4f})")


def print_call_flow_summary() -> None:
    print("=" * 72)
    print("BASALT.FOR / BASALT+langmuir — 调用链摘要")
    print("=" * 72)
    print(
        """
1990 DRIVER:  STATE only; no KDCALC before STATE; SPARM(1) often stale
             STATE: KDCALC(1) → NR loop → KDCALC(3)

langmuir:     TEMPRUN: SPARM(1)=T; KDCALC(4); STATE
             STATE still starts KDCALC(1) → zeros Ol/Cpx from mode 4
             Python wl_state: KDCALC(4) at STATE entry (intentional fix)

PARMS clash:  langmuir reuses PARMS(5,6) as P_HIGH/P_LOW; 1990 uses PARMS(5) as °C offset
"""
    )
    print(f"Ol MgO Kd @1273K: 1990={kd_olivine_mgo_1990(1273.16):.1f}  langmuir@0kbar={kd_olivine_mgo(1273.16,0):.1f}")
    print(f"Doc: petrology/article/复现korenaga/BASALT_FORTRAN_RESEARCH.md")


def main() -> None:
    from petrology.fc.wl1990 import load_kinzler1997_morb_primary

    MELTS["kinzler_morb"] = load_kinzler1997_morb_primary()["oxides_wt_percent"]

    print_call_flow_summary()

    for name, wt in MELTS.items():
        csj = oxides_wt_to_csj(wt)
        print("\n" + "=" * 72)
        print(f"成分: {name}")
        print("CSJ:", "  ".join(f"{CNAMEJ[j]}={csj[j]:.4f}" for j in range(6)))
        scan_qa_literal_vs_fixed(csj, 1273.16, mpa_to_kbar(100))
        scan_qa_literal_vs_fixed(csj, 1473.16, mpa_to_kbar(100))
        if name == "magnesian_demo":
            temprun_trace(csj, p_kbar=1.0, t_start=1473.16, t_end=1173.16)


if __name__ == "__main__":
    main()
