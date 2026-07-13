#!/usr/bin/env python3
"""Check q values for a candidate demo composition.

Uses modern ARHENF ``10**(A/T + B)``.  Column ``plag(1)`` is the raw
stoichiometry check ``-TA[0] + 1.67*CAA + 2.5*NAAL`` (no Kd), matching the
original script — not ``q_plag_f``.
"""

from convert_oxides import convert
from kd_legacy import TA, UAJ, kdcalc

wt = {'SiO2': 40, 'TiO2': 2, 'Al2O3': 20, 'Cr2O3': 0, 'FeO': 8, 'MgO': 10, 'CaO': 15, 'K2O': 1, 'Na2O': 3}
comps = convert(wt)
cs = [comps[k] for k in ['CAA', 'NAAL', 'MGO', 'FEO', 'CAWO', 'TIO2', 'SIO2']]
print('CAA=%.4f NAAL=%.4f 1.67*CAA+2.5*NAAL=%.4f' % (cs[0], cs[1], 1.67 * cs[0] + 2.5 * cs[1]))
print('T(C)  AN      q_plag  q_ol    q_cpx   plag(1)')
for tc in [1500, 1400, 1300, 1200, 1100, 1000]:
    tk = tc + 273.16
    kd, an = kdcalc(tk, cs, form="modern")
    qs = []
    for phase in range(3):
        q = -TA[phase]
        for j in range(7):
            if UAJ[phase][j] == 0:
                continue
            q += UAJ[phase][j] * kd[phase][j] * cs[j]
        qs.append(q)
    # Original: raw stoichiometry without Kd (diagnostic, not a saturation index)
    q1 = -TA[0] + 1.67 * cs[0] + 2.5 * cs[1]
    print(f'{tc:4d} {an:.4f} {qs[0]:8.4f} {qs[1]:8.4f} {qs[2]:8.4f} {q1:8.4f}')
