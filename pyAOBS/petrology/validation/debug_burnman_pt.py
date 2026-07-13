"""BurnMan olivine Vp vs P,T and HS bounds."""
import numpy as np
from petrology.minerals import phase_properties, _make_burnman_phase
from petrology.mixing import hashin_shtrikman_vp, vp_from_kgrho

fo = 0.90
for p_mpa, t_c in [(100, 100), (100, 400), (600, 400), (100, 25)]:
    p_pa = p_mpa * 1e6
    t_k = t_c + 273.15
    props = phase_properties("olivine", fo=fo, p_pa=p_pa, t_k=t_k, backend="burnman")
    print(f"P={p_mpa} T={t_c}C  vp={props['vp_km_s']:.3f} rho={props['rho_g_cm3']:.3f}")

# HS average vs direct vp for single phase
props = phase_properties("olivine", fo=0.90, p_pa=100e6, t_k=373.15, backend="burnman")
k, g, rho = props["k_gpa"], props["g_gpa"], props["rho_g_cm3"]
mix = hashin_shtrikman_vp(np.array([1.0]), np.array([k]), np.array([g]), np.array([rho]))
direct = props["vp_km_s"]
print(f"direct={direct:.3f} hs_avg={mix['vp_km_s']:.3f} hs_lo={mix['vp_hs_lower_km_s']:.3f}")
