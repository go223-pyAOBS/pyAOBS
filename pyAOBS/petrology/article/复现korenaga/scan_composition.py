#!/usr/bin/env python3
"""Scan oxide composition space to find a demo composition with reasonable saturation."""

import numpy as np
from convert_oxides import convert
from kd_legacy import q_values

# Original used BASALT.FOR ARHENF: 10**(A/T)+B (not modern 10**(A/T+B)).
_FORM = "basalt1990"


def check(wt, name):
    try:
        comps = convert(wt)
    except ValueError as e:
        print(f'{name}: conversion error: {e}')
        return
    cs = [comps[k] for k in ['CAA', 'NAAL', 'MGO', 'FEO', 'CAWO', 'TIO2', 'SIO2']]
    print(f'\n{name} wt%: {wt}')
    print('  model comps:', ' '.join(f'{k}={v:.4f}' for k, v in comps.items()))
    print('  T(C)   AN      q_plag   q_ol     q_cpx')
    for tc in [1500, 1400, 1300, 1200, 1100, 1000]:
        tk = tc + 273.16
        q, an = q_values(tk, cs, form=_FORM)
        print(f'  {tc:4d}  {an:6.4f}  {q[0]:8.4f} {q[1]:8.4f} {q[2]:8.4f}')


# Base demo: low Al2O3, moderate MgO/FeO, high CaO, high SiO2
demo1 = {
    'SiO2': 50.0,
    'TiO2': 1.0,
    'Al2O3': 12.0,
    'Cr2O3': 0.0,
    'FeO': 8.0,
    'MgO': 8.0,
    'CaO': 15.0,
    'K2O': 0.1,
    'Na2O': 2.0,
}
check(demo1, 'DEMO-1 low-Al high-Ca')

# Even lower MgO/FeO, more SiO2
demo2 = {
    'SiO2': 52.0,
    'TiO2': 1.0,
    'Al2O3': 12.0,
    'Cr2O3': 0.0,
    'FeO': 6.0,
    'MgO': 6.0,
    'CaO': 15.0,
    'K2O': 0.1,
    'Na2O': 2.0,
}
check(demo2, 'DEMO-2 lower-MgFe')

# Higher CaO, very low Al
demo3 = {
    'SiO2': 48.0,
    'TiO2': 1.0,
    'Al2O3': 10.0,
    'Cr2O3': 0.0,
    'FeO': 8.0,
    'MgO': 8.0,
    'CaO': 18.0,
    'K2O': 0.1,
    'Na2O': 2.0,
}
check(demo3, 'DEMO-3 very-low-Al')

# High Al, low Si (anorthite-rich)
demo4 = {
    'SiO2': 45.0,
    'TiO2': 0.5,
    'Al2O3': 18.0,
    'Cr2O3': 0.0,
    'FeO': 5.0,
    'MgO': 5.0,
    'CaO': 14.0,
    'K2O': 0.1,
    'Na2O': 1.5,
}
check(demo4, 'DEMO-4 high-Al')

# Very low Na (low NAAL), high Ca
demo5 = {
    'SiO2': 50.0,
    'TiO2': 1.0,
    'Al2O3': 14.0,
    'Cr2O3': 0.0,
    'FeO': 8.0,
    'MgO': 8.0,
    'CaO': 13.0,
    'K2O': 0.0,
    'Na2O': 0.5,
}
check(demo5, 'DEMO-5 low-Na')

# Very low Al, very low Na, high Si
demo6 = {
    'SiO2': 55.0,
    'TiO2': 1.0,
    'Al2O3': 8.0,
    'Cr2O3': 0.0,
    'FeO': 6.0,
    'MgO': 5.0,
    'CaO': 12.0,
    'K2O': 0.1,
    'Na2O': 1.5,
}
check(demo6, 'DEMO-6 low-Al high-Si')
