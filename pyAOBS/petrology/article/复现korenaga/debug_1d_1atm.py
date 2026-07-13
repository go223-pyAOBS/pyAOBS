import math

def q_ol(fol, cs, tk):
    # 1 atm 参数 (Weaver & Langmuir 1990)
    kd_mg = 10**(2715/tk - 1.158)
    kd_fe = 10**(4230/tk - 2.741)
    cl_mg = cs[2]/(1+fol*(kd_mg-1))
    cl_fe = cs[3]/(1+fol*(kd_fe-1))
    return -2/3 + kd_mg*cl_mg + kd_fe*cl_fe

cs_bulk = [0.15, 0.15, 0.20, 0.15, 0.10, 0.02, 0.23]
for tk in [1873.16, 1868.16, 1863.16, 1858.16, 1853.16]:
    print(f'T={tk}')
    for fol in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
        print(f'  fol={fol:.2f} q={q_ol(fol, cs_bulk, tk):.6f}')
    print()
