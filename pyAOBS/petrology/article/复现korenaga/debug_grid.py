import math
cs = [0.15, 0.15, 0.20, 0.15, 0.10, 0.02, 0.23]
tk = 1873.16
p = 0.0
# Kd for plag at fa=0 (an = cs[0]/(cs[0]+1.5*cs[1]) = 0.15/(0.15+0.225) = 0.4)
an = cs[0]/(cs[0]+1.5*cs[1])
kd_plag_ca = 10**(2446/tk - (1.122+0.2562*an) + 0.012*p)
kd_plag_na = 10**((3195+3283*an)/tk - (2.318+1.885*an) + 0.007*p)
print('an', an, 'Kd_plag_Ca', kd_plag_ca, 'Kd_plag_Na', kd_plag_na)

kd_ol_mg = 10**(3740/tk - 1.87 + 0.0008*p)
kd_ol_fe = 10**(3911/tk - 2.50 + 0.0006*p)

q_ol = -2/3 + kd_ol_mg*cs[2] + kd_ol_fe*cs[3]
q_plag = -1 + (5/3)*kd_plag_ca*cs[0] + 2.5*kd_plag_na*cs[1]
print('q_ol at f=0:', q_ol)
print('q_plag at f=0:', q_plag)

# 2D grid search
best = None
ngrid = 40
for i1 in range(ngrid+1):
    fol = 0.99 * i1 / ngrid
    for i2 in range(ngrid+1-i1):
        fplag = 0.99 * i2 / ngrid
        fl = max(1e-12, 1 - fol - fplag)
        cl = [0.0]*7
        for j in [0,1]:
            denom = fl + fplag*(kd_plag_ca if j==0 else kd_plag_na)
            cl[j] = cs[j] / denom
        cl[2] = cs[2] / (fl + fol*(kd_ol_mg-1))
        cl[3] = cs[3] / (fl + fol*(kd_ol_fe-1))
        for j in [4,5,6]:
            cl[j] = cs[j] / fl
        qol = -2/3 + kd_ol_mg*cl[2] + kd_ol_fe*cl[3]
        qplag = -1 + (5/3)*kd_plag_ca*cl[0] + 2.5*kd_plag_na*cl[1]
        norm = qol*qol + qplag*qplag
        if best is None or norm < best[0]:
            best = (norm, fol, fplag, qol, qplag)
print('best 2D:', best)
