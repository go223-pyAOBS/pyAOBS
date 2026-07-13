import math

def compute_q(cs, tk, p):
    an = cs[0] / (cs[0] + 1.5 * cs[1])
    kd_plag_ca = 10**(2446/tk - (1.122+0.2562*an) + 0.012*p)
    kd_plag_na = 10**((3195+3283*an)/tk - (2.318+1.885*an) + 0.007*p)
    kd_ol_mg = 10**(3740/tk - 1.87 + 0.0008*p)
    kd_ol_fe = 10**(3911/tk - 2.50 + 0.0006*p)
    kd_cpx_mg = 10**(3798/tk - 2.28 + 0.004*p)
    kd_cpx_fe = 0.24 * kd_cpx_mg
    kd_cpx_ca = 10**(1738/tk - 0.753 + 0.009*p)
    kd_cpx_ti = 10**(1034/tk - 1.27 + 0.005*p)
    kd_cpx_na = 10**(2418/tk - 2.30 + 0.006*p)
    kd_cpx_ca_al = 10**(5087/tk - 4.48 + 0.003*p)

    # q values at fa=0 (full liquid)
    q_plag = -1 + (5/3)*kd_plag_ca*cs[0] + 2.5*kd_plag_na*cs[1]
    q_ol = -2/3 + kd_ol_mg*cs[2] + kd_ol_fe*cs[3]
    q_cpx = -1 + (4/3)*kd_plag_ca*cs[0] + 2.0*kd_plag_na*cs[1] + \
            2.0*kd_ol_mg*cs[2] + 2.0*kd_ol_fe*cs[3] + kd_cpx_ca*cs[4] + kd_cpx_ti*cs[5]
    return q_plag, q_ol, q_cpx, kd_ol_mg, kd_ol_fe, kd_plag_ca, kd_plag_na

cs_highMgFe = [0.15, 0.15, 0.20, 0.15, 0.10, 0.02, 0.23]
cs_original = [0.15, 0.25, 0.08, 0.12, 0.10, 0.02, 0.28]

print("=== High MgO/FeO composition ===")
for p in [0, 4, 8]:
    print(f"\nP={p} kbar")
    for tk in [1873.16, 1823.16, 1773.16, 1723.16, 1673.16]:
        qp, qo, qc, kdomg, kdfe, kdpca, kdpna = compute_q(cs_highMgFe, tk, p)
        print(f"  T={tk-273.16:5.1f}C  q_plag={qp:8.4f}  q_ol={qo:8.4f}  q_cpx={qc:8.4f}  "
              f"Kd_ol(Mg/Fe)={kdomg:6.3f}/{kdfe:6.3f}")

print("\n=== Original MORB-like composition ===")
for p in [0, 4, 8]:
    print(f"\nP={p} kbar")
    for tk in [1673.16, 1623.16, 1573.16, 1523.16, 1473.16]:
        qp, qo, qc, kdomg, kdfe, kdpca, kdpna = compute_q(cs_original, tk, p)
        print(f"  T={tk-273.16:5.1f}C  q_plag={qp:8.4f}  q_ol={qo:8.4f}  q_cpx={qc:8.4f}  "
              f"Kd_ol(Mg/Fe)={kdomg:6.3f}/{kdfe:6.3f}")
