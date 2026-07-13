"""Debug melt evolution for wl1990 FC engine."""
import numpy as np
from petrology.fc.wl1990 import load_kinzler1997_morb_primary, simulate_crystallization_path

p = load_kinzler1997_morb_primary()
f = np.linspace(0, 0.8, 9)
st = simulate_crystallization_path(
    primary_melt_oxides_wt=p["oxides_wt_percent"],
    f_grid=f,
    path="fc_100",
    mineral_backend="fig2",
)
print("F    MgO   FeO   SiO2  Fo   An   Di   Vp_cum  Vp_res")
for s in st:
    m = s.melt_oxides_wt
    print(
        f"{s.f_solid:.2f} {m['MgO']:.2f} {m['FeO']:.2f} {m['SiO2']:.2f} "
        f"{s.fo_pct:.1f} {s.an_pct:.1f} {s.di_pct:.1f} "
        f"{s.vp_cumulate_km_s:.3f} {s.vp_residual_norm_km_s:.3f}"
    )
