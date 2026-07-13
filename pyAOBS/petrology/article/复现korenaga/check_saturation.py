#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check saturation indices for the Kinzler 1997 composition vs temperature.

Uses ``basalt_modern.f90`` ARHENF: ``10**(A/T + B)``.
"""

from kd_legacy import q_values

# Kinzler 1997 model components (mole-based, normalized)
csj = {
    'CAA': 0.0816,
    'NAAL': 0.0494,
    'MGO': 0.2049,
    'FEO': 0.0737,
    'CAWO': 0.0527,
    'TIO2': 0.0078,
    'SIO2': 0.5300,
}

cs = [csj[k] for k in ['CAA', 'NAAL', 'MGO', 'FEO', 'CAWO', 'TIO2', 'SIO2']]

print('Kinzler 1997 saturation indices (1 atm, modern ARHENF 10**(A/T+B)):')
print('  T(C)   AN      q_plag   q_ol     q_cpx')
for tc in range(1500, 999, -50):
    tk = tc + 273.16
    q, an = q_values(tk, cs, form="modern")
    print(f'  {tc:4d}  {an:6.4f}  {q[0]:8.4f} {q[1]:8.4f} {q[2]:8.4f}')
