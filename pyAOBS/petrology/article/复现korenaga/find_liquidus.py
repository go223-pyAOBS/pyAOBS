from kd_legacy import find_liquidus_pressure, q_saturation_pressure

cs_highMgFe = [0.15, 0.15, 0.20, 0.15, 0.10, 0.02, 0.23]
cs_original = [0.15, 0.25, 0.08, 0.12, 0.10, 0.02, 0.28]

print("Liquidus estimates (temperature in Celsius where first phase saturates):")
print()
print("High MgO/FeO composition:")
for p in [0, 2, 4, 6, 8]:
    tl = find_liquidus_pressure(cs_highMgFe, p)
    if tl:
        qp, qo, qc = q_saturation_pressure(cs_highMgFe, tl, p)
        print(f"  P={p} kbar: T_liq = {tl - 273.16:6.1f}C  q_plag={qp:7.4f} q_ol={qo:7.4f} q_cpx={qc:7.4f}")
    else:
        print(f"  P={p} kbar: above search range")

print()
print("Original MORB-like composition:")
for p in [0, 2, 4, 6, 8]:
    tl = find_liquidus_pressure(cs_original, p)
    if tl:
        qp, qo, qc = q_saturation_pressure(cs_original, tl, p)
        print(f"  P={p} kbar: T_liq = {tl - 273.16:6.1f}C  q_plag={qp:7.4f} q_ol={qo:7.4f} q_cpx={qc:7.4f}")
    else:
        print(f"  P={p} kbar: above search range")
