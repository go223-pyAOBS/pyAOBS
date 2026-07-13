cs = [0.15,0.15,0.20,0.15,0.10,0.02,0.23]
# Kds at 1873 K, P=8
Kd = {
 'PLAG_Ca': 10**(2446/1873.16-(1.122+0.2562*0.4)+0.012*8),
 'PLAG_Na': 10**((3195+3283*0.4)/1873.16-(2.318+1.885*0.4)+0.007*8),
 'OL_Mg': 10**(3740/1873.16-1.87+0.0008*8),
 'OL_Fe': 10**(3911/1873.16-2.50+0.0006*8),
}
for k,v in Kd.items(): print(k, v)

def F_ol(fol, fplag):
    fl = max(1e-12, 1-fol-fplag)
    clmg = cs[2]/(fl + fol*Kd['OL_Mg'])
    clfe = cs[3]/(fl + fol*Kd['OL_Fe'])
    return -2/3 + Kd['OL_Mg']*clmg + Kd['OL_Fe']*clfe

def F_plag(fol, fplag):
    fl = max(1e-12, 1-fol-fplag)
    clca = cs[0]/(fl + fplag*Kd['PLAG_Ca'])
    clna = cs[1]/(fl + fplag*Kd['PLAG_Na'])
    return -1 + (5/3)*Kd['PLAG_Ca']*clca + 2.5*Kd['PLAG_Na']*clna

best=None
for i in range(1001):
    fol=i/1000.0
    for j in range(1001-int(i)):
        fplag=j/1000.0
        r1=F_ol(fol,fplag); r2=F_plag(fol,fplag)
        norm=r1*r1+r2*r2
        if best is None or norm<best[0]:
            best=(norm,fol,fplag,r1,r2)
print('best',best)
