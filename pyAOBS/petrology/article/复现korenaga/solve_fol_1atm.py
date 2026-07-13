import math

def q_ol(fol, cs, tk):
    kd_mg = 10**(2715/tk - 1.158)
    kd_fe = 10**(4230/tk - 2.741)
    cl_mg = cs[2]/(1+fol*(kd_mg-1))
    cl_fe = cs[3]/(1+fol*(kd_fe-1))
    return -2/3 + kd_mg*cl_mg + kd_fe*cl_fe

cs = [0.15, 0.25, 0.08, 0.12, 0.10, 0.02, 0.28]
for tk in [1473.16, 1423.16, 1373.16]:
    print(f'T={tk-273.16:.0f}C')
    for fol in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
        print(f'  fol={fol:.2f} q={q_ol(fol, cs, tk):.6f}')
    # bisection
    lo, hi = 0.0, 0.9999
    qlo, qhi = q_ol(lo, cs, tk), q_ol(hi, cs, tk)
    if qlo*qhi > 0:
        print(f'  no root in [0,0.9999], q_lo={qlo:.4f} q_hi={qhi:.4f}')
    else:
        for _ in range(60):
            mid = 0.5*(lo+hi)
            qm = q_ol(mid, cs, tk)
            if qlo*qm <= 0:
                hi, qhi = mid, qm
            else:
                lo, qlo = mid, qm
        print(f'  root fol={0.5*(lo+hi):.4f}')
    print()
