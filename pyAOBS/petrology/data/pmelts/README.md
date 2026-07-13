# pMELTS KLB-1 melt chemistry (Modern track)

Melt major elements for peridotite lithologies come from pyMelt's bundled grid:

- **Source file**: `pyMelt/phaseDiagrams/build/klb1_pmelts_grid.csv` (requires `pip install pyMelt`)
- **Lookup**: `melting/melt_chemistry.py` → `pmelts_melt_oxides(P, F)`
- **Backend key**: `chemistry_backend="pmelts_klb1"`

Reproduction track (KKHS02 Fig.3) continues to use `kinzler1997`.

## Compare backends

```bash
py -3.11 petrology/validation/compare_melt_chemistry_pmelts.py
```

## Modern column usage

```python
from petrology.melting import forward_heterogeneous_column

het = forward_heterogeneous_column(
    tp_c=1450,
    chi=8.0,
    lithology_backend="pymelt",
    peridotite_lith="katz_lherzolite",
    compute_norm_vp=True,
)
```

Peridotite chemistry defaults to pMELTS when `lithology_backend="pymelt"`.
