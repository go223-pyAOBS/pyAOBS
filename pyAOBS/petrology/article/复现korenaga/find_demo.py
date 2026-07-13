#!/usr/bin/env python3
"""Search for a demo oxide composition whose 1-atm liquidus is near 1250 C."""

import numpy as np
from convert_oxides import convert
from kd_legacy import cpx_stoich, q_values

# Original find_demo used modern ARHENF: 10**(A/T + B)
_FORM = "modern"


def score(wt):
    try:
        comps = convert(wt)
    except ValueError:
        return None
    cs = [comps[k] for k in ['CAA', 'NAAL', 'MGO', 'FEO', 'CAWO', 'TIO2', 'SIO2']]
    q1300, _ = q_values(1300 + 273.16, cs, form=_FORM)
    q1250, _ = q_values(1250 + 273.16, cs, form=_FORM)
    q1200, _ = q_values(1200 + 273.16, cs, form=_FORM)
    q1150, _ = q_values(1150 + 273.16, cs, form=_FORM)
    q1100, _ = q_values(1100 + 273.16, cs, form=_FORM)
    t_cpx_1250 = cpx_stoich(1250 + 273.16, cs, form=_FORM)
    # Require CPX to be stoichiometrically viable near the liquidus so it can be included.
    penalties = 0.0
    penalties += hinge(t_cpx_1250 - 1.0, +1.0, 0.0) * 20.0
    penalties += hinge(q1300[0], +1.0, 0.05) * 10.0   # plag unsaturated at 1300
    penalties += hinge(q1250[0], -1.0, 0.05) * 5.0    # plag saturated at 1250
    penalties += hinge(q1250[1], +1.0, 0.05) * 4.0    # ol unsaturated at 1250
    penalties += hinge(q1200[1], -1.0, 0.05) * 4.0    # ol saturated at 1200
    penalties += hinge(q1200[2], +1.0, 0.05) * 3.0    # cpx unsaturated at 1200
    penalties += hinge(q1150[2], -1.0, 0.05) * 3.0    # cpx saturated at 1150
    # Encourage all phases saturated by 1100 C
    penalties += hinge(q1100[0], -1.0, 0.05) * 2.0
    penalties += hinge(q1100[1], -1.0, 0.05) * 2.0
    penalties += hinge(q1100[2], -1.0, 0.05) * 2.0
    return penalties, cs, (q1300, q1250, q1200, q1150, q1100), t_cpx_1250


def hinge(x, sign, margin):
    """Return positive penalty if sign*x is not >= margin."""
    return max(0.0, margin - sign * x)


best = None
best_score = 1e9

# Coarse grid search over key oxides.  Widen ranges to allow high-Mg/Fe/Ca
# compositions needed for CPX stoichiometric viability.
for sio2 in np.arange(32.0, 50.1, 2.0):
    for al2o3 in np.arange(8.0, 26.1, 1.0):
        for mgo in np.arange(6.0, 30.1, 2.0):
            for feo in np.arange(5.0, 25.1, 2.0):
                for cao in np.arange(8.0, 30.1, 2.0):
                    for na2o in np.arange(0.5, 8.1, 0.5):
                        total = sio2 + al2o3 + mgo + feo + cao + na2o + 1.0 + 0.1
                        if total > 100.5 or total < 99.5:
                            continue
                        wt = {
                            'SiO2': sio2,
                            'TiO2': 1.0,
                            'Al2O3': al2o3,
                            'Cr2O3': 0.0,
                            'FeO': feo,
                            'MgO': mgo,
                            'CaO': cao,
                            'K2O': 0.1,
                            'Na2O': na2o,
                        }
                        res = score(wt)
                        if res is None:
                            continue
                        sc, cs, qs, t_cpx = res
                        if sc < best_score:
                            best_score = sc
                            best = (wt, cs, qs, t_cpx)

if best is None:
    print('No suitable composition found.')
    raise SystemExit

wt, cs, qs, t_cpx = best
print(f'Best score: {best_score:.4f}')
print(f'Oxide wt%: {wt}')
print(f'Model comps: CAA={cs[0]:.4f} NAAL={cs[1]:.4f} MGO={cs[2]:.4f} FEO={cs[3]:.4f} CAWO={cs[4]:.4f} TIO2={cs[5]:.4f} SIO2={cs[6]:.4f}')
print(f'CPX stoich t at 1250 C: {t_cpx:.4f}')
print('T(C)   q_plag   q_ol     q_cpx')
for tc, q in zip([1300, 1250, 1200, 1150, 1100], qs):
    print(f'{tc:4d}  {q[0]:8.4f} {q[1]:8.4f} {q[2]:8.4f}')
