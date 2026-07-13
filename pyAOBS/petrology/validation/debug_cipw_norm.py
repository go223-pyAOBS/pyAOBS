"""CIPW norm breakdown for Kinzler primary."""
from petrology.fc.wl1990 import load_kinzler1997_morb_primary
from petrology.cipw import cipw_norm_mass_fractions
from petrology.minerals import phase_properties
from petrology.mixing import hashin_shtrikman_vp
import numpy as np

m = load_kinzler1997_morb_primary()["oxides_wt_percent"]
norm = cipw_norm_mass_fractions(m, backend="auto")
print("CIPW mass fractions:", {k: round(norm.get(k, 0), 4) for k in ("olivine", "plagioclase", "clinopyroxene", "quartz", "ilmenite")})
print("Fo", norm.get("Fo"), "An", norm.get("An"))

fo, an = norm.get("Fo", 0.9), norm.get("An", 0.5)
cpx_mg = norm.get("CpxMg", fo)
p, t = 100e6, 373.15

vps = []
for ph in ("olivine", "plagioclase", "clinopyroxene", "quartz", "ilmenite"):
    mf = norm.get(ph, 0)
    if mf <= 0:
        continue
    pr = phase_properties(ph, fo=fo, an=an, cpx_mg=cpx_mg, p_pa=p, t_k=t, backend="burnman")
    vol = mf / pr["rho_g_cm3"]
    vps.append((ph, mf, vol, pr["vp_km_s"]))
    print(f"  {ph}: mf={mf:.3f} vp={pr['vp_km_s']:.3f} rho={pr['rho_g_cm3']:.3f}")

vols = np.array([v[2] for v in vps])
vols /= vols.sum()
vp_arith = sum(vols[i] * vps[i][3] for i in range(len(vps)))
print(f"Volume-weighted arithmetic Vp: {vp_arith:.3f}")

mass = np.array([v[1] for v in vps])
ks = np.array([phase_properties(v[0], fo=fo, an=an, cpx_mg=cpx_mg, p_pa=p, t_k=t, backend="burnman")["k_gpa"] for v in vps])
gs = np.array([phase_properties(v[0], fo=fo, an=an, cpx_mg=cpx_mg, p_pa=p, t_k=t, backend="burnman")["g_gpa"] for v in vps])
rhos = np.array([phase_properties(v[0], fo=fo, an=an, cpx_mg=cpx_mg, p_pa=p, t_k=t, backend="burnman")["rho_g_cm3"] for v in vps])
vol = mass / rhos
vol /= vol.sum()
mix = hashin_shtrikman_vp(vol, ks, gs, rhos)
print(f"HS mix Vp: {mix['vp_km_s']:.3f}")
