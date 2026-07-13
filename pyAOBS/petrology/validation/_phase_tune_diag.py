"""Quick phase-path diagnostic for high-Al tuning."""
import numpy as np
from petrology.fc.cdat_library import fig2_primary_melt
from petrology.fc.fig2_ab import cumulative_phases_pct, incremental_modes_from_cumulative
from petrology.fc.wl_high_al_morb import HighAlFcRegime, _ol_pl_shares_f_guided, mgo_plagioclase_in
from petrology.fc.wl1990 import simulate_crystallization_path

m0 = fig2_primary_melt()["oxides_wt_percent"]
regime = HighAlFcRegime(
    enabled=True,
    high_al_morb=True,
    primary_al2o3=float(m0.get("Al2O3", 0.0)),
)
f = np.linspace(0, 0.45, 46)
st = simulate_crystallization_path(
    primary_melt_oxides_wt=m0, f_grid=f, path="fc_100", kd_engine="langmuir",
    fig2_ab_calibrate=False, mineral_backend="sb1994_fig2ol",
)
print("primary MgO", m0["MgO"], "mgo_pl_in", mgo_plagioclase_in(m0, 100, regime=regime))
for s in st:
    if abs(s.f_solid - round(s.f_solid, 2)) > 0.005 and s.f_solid not in (0.08, 0.22, 0.35, 0.40):
        continue
    m = s.melt_oxides_wt
    mp = mgo_plagioclase_in(m, 100, regime=regime)
    ol_s, pl_s = _ol_pl_shares_f_guided(s.f_solid, m["MgO"], mp)
    ol_p, cpx_p, pl_p = cumulative_phases_pct("fc_100", s.f_solid)
    if s.f_solid > 0:
        oi, pi, ci = incremental_modes_from_cumulative("fc_100", s.f_solid, s.f_solid - 0.01)
    else:
        oi, pi, ci = 1.0, 0.0, 0.0
    print(
        f"F={s.f_solid:.2f} MgO={m['MgO']:.2f} mp={mp:.2f} ol_s={ol_s:.2f} "
        f"cum=({100*s.cum_ol:.0f},{100*s.cum_pl:.0f},{100*s.cum_cpx:.0f}) "
        f"paper=({ol_p:.0f},{pl_p:.0f},{cpx_p:.0f}) "
        f"inc=({100*s.inc_ol:.0f},{100*s.inc_pl:.0f},{100*s.inc_cpx:.0f}) "
        f"paper_inc=({100*oi:.0f},{100*pi:.0f},{100*ci:.0f})"
    )
