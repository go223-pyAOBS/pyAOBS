from kd_legacy import q_at_fa0_fortran

# 系统成分 (data_original case 1)
cs = [0.0, 0.15, 0.25, 0.08, 0.12, 0.10, 0.02, 0.28]

print("T(K)    T(C)    q_plag    q_ol      q_cpx")
for tk in range(2000, 1200, -25):
    qp, qo, qc = q_at_fa0_fortran(float(tk), cs)
    print(f"{tk}  {tk - 273.16:5.1f}  {qp:8.4f}  {qo:8.4f}  {qc:8.4f}")
