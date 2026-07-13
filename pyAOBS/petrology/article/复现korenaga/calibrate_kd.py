import math

def q_ol_at_f(fol, kd_mg, kd_fe, cs_mg, cs_fe):
    cl_mg = cs_mg / (1 + fol * (kd_mg - 1))
    cl_fe = cs_fe / (1 + fol * (kd_fe - 1))
    return -2/3 + kd_mg * cl_mg + kd_fe * cl_fe

# Target: find kd_mg such that q_ol=0 at fol=0.1 for MORB-like composition
# Assume kd_fe = 0.24 * kd_mg (simplified from cpx ratio)
cs_mg, cs_fe = 0.08, 0.12

for target_fol in [0.05, 0.10, 0.15, 0.20]:
    lo, hi = 1.0, 1000.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        kd_mg = mid
        kd_fe = 0.24 * kd_mg
        q = q_ol_at_f(target_fol, kd_mg, kd_fe, cs_mg, cs_fe)
        # we want q=0; q>0 means undersaturated (need lower kd); q<0 means supersaturated (need higher kd)
        if q > 0:
            hi = mid
        else:
            lo = mid
    kd_mg = 0.5 * (lo + hi)
    print(f"Target fol={target_fol:.2f}: Kd_mg={kd_mg:.2f}, Kd_fe={0.24*kd_mg:.2f}")

# For high MgO/FeO composition (MgO=0.20, FeO=0.15)
print("\nHigh MgO/FeO composition:")
cs_mg, cs_fe = 0.20, 0.15
for target_fol in [0.10, 0.20, 0.30, 0.40]:
    lo, hi = 1.0, 1000.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        kd_mg = mid
        kd_fe = 0.24 * kd_mg
        q = q_ol_at_f(target_fol, kd_mg, kd_fe, cs_mg, cs_fe)
        if q > 0:
            lo = mid
        else:
            hi = mid
    kd_mg = 0.5 * (lo + hi)
    print(f"Target fol={target_fol:.2f}: Kd_mg={kd_mg:.2f}, Kd_fe={0.24*kd_mg:.2f}")

# Compare with current 1atm params at various temperatures
print("\nCurrent 1atm Kd values:")
for tc in [1300, 1250, 1200, 1150, 1100, 1050, 1000]:
    tk = tc + 273.16
    kd_mg = 10**(2715/tk - 1.158)
    kd_fe = 10**(4230/tk - 2.741)
    print(f"  T={tc}C: Kd_mg={kd_mg:.2f}, Kd_fe={kd_fe:.2f}")

print("\nCurrent high-pressure Kd values (P=0):")
for tc in [1300, 1250, 1200, 1150, 1100, 1050, 1000]:
    tk = tc + 273.16
    kd_mg = 10**(3740/tk - 1.87)
    kd_fe = 10**(3911/tk - 2.50)
    print(f"  T={tc}C: Kd_mg={kd_mg:.2f}, Kd_fe={kd_fe:.2f}")
