"""Compare wl1990 FC track vs KKHS02 Fig.2 proxy anchors."""
import numpy as np
from petrology.fc.wl1990 import (
    _melt_an_fraction,
    load_kinzler1997_morb_primary,
    simulate_crystallization_path,
)

p = load_kinzler1997_morb_primary()
f = np.array([0, 0.15, 0.25, 0.35, 0.45, 0.5, 0.6, 0.7, 0.8])
st = simulate_crystallization_path(
    primary_melt_oxides_wt=p["oxides_wt_percent"],
    f_grid=f,
    path="fc_100",
    mineral_backend="fig2",
)
PAPER = {
    0.0: (7.57, 90, 80),
    0.15: (None, 88, 77),
    0.25: (7.55, 85, 73),
    0.45: (None, 81, 69),
    0.5: (7.42, None, None),
    0.6: (None, 72, 62),
    0.7: (7.34, 63, 57),
    0.8: (7.31, 58, 54),
}
print("F    Vp      Fo      An   cumPl  CaO  Na2O An_m | paper")
for s in st:
    m = s.melt_oxides_wt
    an_m = 100 * _melt_an_fraction(m)
    pv, pf, pa = PAPER.get(s.f_solid, (None, None, None))
    ps = f"{pv or '-':>5} {pf or '-':>3} {pa or '-':>3}" if s.f_solid in PAPER else ""
    print(
        f"{s.f_solid:.2f} {s.vp_cumulate_km_s:.3f} {s.cum_fo_pct:5.1f} {s.cum_an_pct:5.1f} "
        f"{s.cum_pl:5.2f} {m['CaO']:4.1f} {m['Na2O']:4.2f} {an_m:4.1f} | {ps}"
    )
