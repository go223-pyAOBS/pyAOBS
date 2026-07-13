#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert wt% oxides to BASALT model pseudo-components.

Model components (from BASALT.FOR BLOCK DATA comments):
  CAA  ~ CaAl2O4  (anorthite-like component)
  NAAL ~ NaAlO2   (albite-like component)
  MGO  ~ MgO
  FEO  ~ FeO
  CAWO ~ CaSiO3   (wollastonite-like Ca component for pyroxene)
  TIO2 ~ TiO2
  SIO2 ~ SiO2     (bulk silica; implicit in minerals, explicit in mass balance)

Conversion rules:
  - All Al2O3 is distributed between CAA and NAAL.
  - Na2O (+ K2O) determines NAAL via: 1 Na2O -> 2 NaAlO2.
  - Remaining Al2O3 after NAAL goes to CAA.
  - CaO is first used to satisfy CAA; leftover CaO becomes CAWO.
  - Cr2O3 is added to FeO (Cr behaves like Fe).
  - Result is normalized to sum = 1.
"""

import sys

# Molar masses (g/mol)
M = {
    'SiO2': 60.084,
    'TiO2': 79.866,
    'Al2O3': 101.961,
    'Cr2O3': 151.990,
    'FeO': 71.844,
    'MgO': 40.304,
    'CaO': 56.077,
    'Na2O': 61.979,
    'K2O': 94.196,
}


def convert(wt):
    """Convert oxide wt% dict to model component mole fractions."""
    # Moles of each oxide
    mol = {k: wt.get(k, 0.0) / M[k] for k in M}

    # Alkali oxide moles (Na + K treated together for NAAL)
    mol_na2o_eq = mol['Na2O'] + mol['K2O']

    # NAAL = 2 * (Na2O + K2O) because Na2O provides 2 Na cations
    naal = 2.0 * mol_na2o_eq

    # Al2O3 used by NAAL: 0.5 NAAL moles (since 2 NAAL per Al2O3)
    al_used_by_naal = 0.5 * naal

    # CAA uses remaining Al2O3 (1 Al2O3 per CAA)
    caa = mol['Al2O3'] - al_used_by_naal
    if caa < 0:
        raise ValueError(
            f"Not enough Al2O3 to form NAAL: Al2O3={mol['Al2O3']:.4f}, "
            f"needed for NAAL={al_used_by_naal:.4f}"
        )

    # CAWO uses leftover CaO after CAA
    cawo = mol['CaO'] - caa
    if cawo < 0:
        # Not enough CaO for anorthite component; re-balance by reducing CAA
        # This can happen for Ca-poor, Al-rich compositions
        caa = mol['CaO']
        cawo = 0.0

    # Other components
    mgo = mol['MgO']
    feo = mol['FeO'] + mol['Cr2O3']  # Cr2O3 counted as FeO
    tio2 = mol['TiO2']
    sio2 = mol['SiO2']

    comps = {
        'CAA': caa,
        'NAAL': naal,
        'MGO': mgo,
        'FEO': feo,
        'CAWO': cawo,
        'TIO2': tio2,
        'SIO2': sio2,
    }

    total = sum(comps.values())
    if total <= 0:
        raise ValueError('Total component moles <= 0')

    return {k: v / total for k, v in comps.items()}


def main():
    # Default: Kinzler (1997) composition
    wt = {
        'SiO2': 48.2,
        'TiO2': 0.94,
        'Al2O3': 16.4,
        'Cr2O3': 0.12,
        'FeO': 7.96,
        'MgO': 12.5,
        'CaO': 11.4,
        'K2O': 0.07,
        'Na2O': 2.27,
    }

    if len(sys.argv) > 1 and sys.argv[1] not in ('-', ''):
        # Accept a JSON-like dict, e.g.
        # python convert_oxides.py "{'SiO2':50,'TiO2':1,'Al2O3':12,'FeO':8,'MgO':8,'CaO':15,'K2O':0.1,'Na2O':2.0}"
        import ast
        try:
            user = ast.literal_eval(sys.argv[1])
            if isinstance(user, dict):
                wt.update(user)
            else:
                raise ValueError('Argument must be a dict of oxide wt%')
        except Exception as e:
            print(f'Error parsing argument: {e}', file=sys.stderr)
            print("Usage: python convert_oxides.py \"{\\'SiO2\\':50, ...}\"", file=sys.stderr)
            sys.exit(1)

    comps = convert(wt)
    oxides_str = ' '.join(f'{k}={v}' for k, v in wt.items())
    print(f'! Oxide composition converted to BASALT model components')
    print(f'! Oxides (wt%): {oxides_str}')
    print('! Mapping: CAA=CaAl2O4, NAAL=NaAlO2, MGO=MgO, FEO=FeO+Cr2O3, CAWO=CaSiO3, TIO2=TiO2, SIO2=SiO2')
    print('! 列顺序: CAA   NAAL  MGO   FEO   CAWO  TIO2  SIO2')
    print()
    vals = [comps[k] for k in ['CAA', 'NAAL', 'MGO', 'FEO', 'CAWO', 'TIO2', 'SIO2']]
    print(' '.join(f'{v:7.4f}' for v in vals))


if __name__ == '__main__':
    main()
