# Katz (2003) Fig. 9 — dry melting parameterization comparison

Isobaric **F(T)** at **1 GPa (blue)** and **3 GPa (green)** for models with
implemented parameters:

| Key | Source |
|-----|--------|
| `katz2003` | `katz2003.py` Table 2 |
| `mckenzie1988` | pyMelt `mckenzie.lherzolite` |
| `pmelts2002` | Interpolate `pmelts_klb1_p1_p3.csv` |

Langmuir (1992) and Iwamori (1995) are **not** plotted by default (deferred).

**Axes:** T **1200–1800 °C**, F **0–0.5**.

**Line styles:** Katz — blue/green solid; McKenzie & Bickle — blue/green long dash;
pMELTS — pink long dash.

## Files

| File | Purpose |
|------|---------|
| `pmelts_klb1_p1_p3.csv` | pMELTS KLB-1 grid subset (P = 1, 3 GPa) |

## Reproduce

```bash
py -3.11 petrology/validation/reproduce_katz2003_fig9.py --show
```
