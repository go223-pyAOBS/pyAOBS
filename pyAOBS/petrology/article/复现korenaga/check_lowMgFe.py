from kd_legacy import q_saturation_pressure

cs_low = [0.15, 0.25, 0.05, 0.08, 0.10, 0.02, 0.35]
for p in [0, 8]:
    print(f"P={p} kbar")
    for tc in [1300, 1250, 1200, 1150, 1100, 1050, 1000]:
        tk = tc + 273.16
        qp, qo, qc = q_saturation_pressure(cs_low, tk, p)
        print(f"  T={tc}C  q_plag={qp:8.4f} q_ol={qo:8.4f} q_cpx={qc:8.4f}")
