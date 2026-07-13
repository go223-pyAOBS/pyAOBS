"""Trace FC phases for catalog melts (internal)."""
from __future__ import annotations

import numpy as np

from petrology.data.load_catalog import load_melt_catalog, oxides_from_record
from petrology.fc.wl_components import oxides_wt_to_csj
from petrology.fc.wl_high_al_morb import is_high_al_morb
from petrology.fc.wl_kd import DEFAULT_T_K, mpa_to_kbar
from petrology.fc.wl_state import equilibrium_state
from petrology.fc.wl1990 import simulate_crystallization_path


def show(sid: str) -> None:
    rows = [r for r in load_melt_catalog() if r.get("include_in_regression")]
    rec = next(r for r in rows if r.get("id") == sid or r.get("source_id") == sid)
    ox = {k: v for k, v in oxides_from_record(rec).items() if k != "P2O5"}
    print("===", sid, "Al", ox["Al2O3"], "MgO", ox["MgO"], "high_al", is_high_al_morb(ox))
    st = simulate_crystallization_path(
        primary_melt_oxides_wt=ox,
        f_grid=np.array([0.0, 0.25, 0.75]),
        path="fc_100",
        kd_engine="langmuir",
        fig2_ab_calibrate=False,
        p_fc_mpa=400.0,
        mineral_backend="auto",
        sub_steps_per_interval=25,
    )
    for s in st:
        co, cp, cx = 100 * s.cum_ol, 100 * s.cum_pl, 100 * s.cum_cpx
        io, ip, ic = 100 * s.inc_ol, 100 * s.inc_pl, 100 * s.inc_cpx
        mg = s.melt_oxides_wt["MgO"]
        print(f" F={s.f_solid:.2f} cum {co:.0f}/{cp:.0f}/{cx:.0f} inc {io:.0f}/{ip:.0f}/{ic:.0f} MgO={mg:.1f}")
    m = st[-1].melt_oxides_wt
    res = equilibrium_state(oxides_wt_to_csj(m), t_k=DEFAULT_T_K, p_kbar=mpa_to_kbar(400.0))
    print(" STATE nl", res.nl, "fa", list(res.fa) if res.nl > 0 else None)


if __name__ == "__main__":
    for s in ("w98_run70_02", "k97_grid_P3_F0.02", "k97_exp_11_20"):
        show(s)
