"""Quick fig2 backend anchor check."""
from petrology.norm_velocity import norm_velocity_from_bulk_wt
from petrology.fc.wl1990 import load_kinzler1997_morb_primary
from petrology.fc.assemblage import assemblage_vp_rho

m = load_kinzler1997_morb_primary()["oxides_wt_percent"]
r = norm_velocity_from_bulk_wt(m, p_pa=100e6, t_k=373.15, mineral_backend="fig2")
print(f"fig2 norm Vp: {r['vp_km_s']:.3f} (target 7.17)")

vp, _ = assemblage_vp_rho(
    ol_frac=1, pl_frac=0, cpx_frac=0,
    fo_pct=90, an_pct=73, di_pct=71,
    p_pa=100e6, t_k=373.15, mineral_backend="fig2",
)
print(f"fig2 ol Fo90 Vp: {vp:.3f} (target 7.58)")
