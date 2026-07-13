"""Diagnose high-F cumulative Vp vs phase and melt evolution."""
import numpy as np
from petrology.fc.wl1990 import load_kinzler1997_morb_primary, simulate_crystallization_path

p = load_kinzler1997_morb_primary()
f = np.linspace(0, 0.8, 17)
st = simulate_crystallization_path(
    primary_melt_oxides_wt=p["oxides_wt_percent"],
    f_grid=f,
    path="fc_100",
    cipw_backend="pyrolite",
    mineral_backend="fig2",
)
print("F    cumOl cumPl cumCpx cumFo cumAn incAn  MgO  Vp_cum  Vp_res")
for s in st:
    m = s.melt_oxides_wt
    print(
        f"{s.f_solid:.2f} {s.cum_ol:5.2f} {s.cum_pl:5.2f} {s.cum_cpx:5.2f} "
        f"{s.cum_fo_pct:5.1f} {s.cum_an_pct:5.1f} {s.an_pct:5.1f} "
        f"{m['MgO']:4.1f} {s.vp_cumulate_km_s:.3f} {s.vp_residual_norm_km_s:.3f}"
    )
