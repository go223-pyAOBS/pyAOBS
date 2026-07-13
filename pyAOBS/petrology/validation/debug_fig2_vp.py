"""Vp backend comparison and Fo formula check."""
from petrology.fc.wl1990 import (
    load_kinzler1997_morb_primary,
    _olivine_fo_pct,
    _plagioclase_an_pct,
    normalize_melt_oxides,
)
from petrology.norm_velocity import norm_velocity_from_bulk_wt
from petrology.fc.assemblage import assemblage_vp_rho

p = load_kinzler1997_morb_primary()
m = p["oxides_wt_percent"]

print("=== norm Vp @ 100 MPa, 100 C ===")
for mb in ("empirical", "burnman", "auto"):
    r = norm_velocity_from_bulk_wt(m, p_pa=100e6, t_k=373.15, mineral_backend=mb)
    print(f"  {mb:10s}  Vp={r['vp_km_s']:.3f}  backend={r.get('mineral_backend')}")

print("\n=== Fo / An from melt (current) ===")
print(f"  Fo={_olivine_fo_pct(m):.1f}  An={_plagioclase_an_pct(m):.1f}")

mg = m["MgO"]
fe = m["FeO"]
fo_molar = 100 / (1 + 0.30 * (fe / 55.85) / (mg / 40.30))
print(f"  Fo molar Kd=0.30: {fo_molar:.1f}")

print("\n=== pure olivine assemblage Vp ===")
for mb in ("empirical", "burnman"):
    for fo in (90, 84):
        vp, rho = assemblage_vp_rho(
            ol_frac=1, pl_frac=0, cpx_frac=0,
            fo_pct=fo, an_pct=75, di_pct=60,
            p_pa=100e6, t_k=373.15, mineral_backend=mb,
        )
        print(f"  {mb} Fo={fo}: Vp={vp:.3f} rho={rho:.3f}")
