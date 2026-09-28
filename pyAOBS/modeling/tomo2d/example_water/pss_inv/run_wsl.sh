#!/bin/bash
set -euo pipefail
cd /mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/pss_inv
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1
export TOMO2D_INV_OMP=1
export OMP_NUM_THREADS=8

python3 make_pss_inv_case.py

python3 - <<'PY'
from pathlib import Path
import make_pss_inv_case as m
m.write_vs_from_smesh(Path("true_vp.smesh"), Path("start_vs.smesh"), m.KAPPA_START)
print(f"start_vs.smesh <- true_vp / {m.KAPPA_START:g}")
PY

echo "== true PSS  -M true_vp -U true_vs =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_inv.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_true.dat > syn_inv.dat
echo "== start PSS  -M true_vp -U start_vs =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Ustart_vs.smesh -Ggeom_inv.dat \
  -Xconv.refl -Bseafloor.refl "$N" > syn_start.dat
echo "== inverse PSS  冻真 Vp  -k2.0  面下+台侧盖层 Vs =="
"$BIN/tt_inverse" -Mtrue_vp.smesh -Ustart_vs.smesh -Gsyn_inv.dat \
  -Bconv.refl -Yseafloor.refl -w -k2.0 "$N" \
  -I5 -SV200 -TV5 -CVvcorr.dat -Oout -l -Linv.log

echo "== recovered PSS  -M true_vp -U out.smesh =="
VS=$(python3 -c "from pathlib import Path; c=sorted(Path('.').glob('out.smesh.*.*'));
print(max(c, key=lambda p: (int(p.name.split('.')[-2]), int(p.name.split('.')[-1]))))")
echo "rec Vs mesh $VS"
"$BIN/tt_forward" -Mtrue_vp.smesh -U"$VS" -Ggeom_inv.dat -Xconv.refl \
  -Bseafloor.refl "$N" -Rrays_rec.dat > syn_rec.dat

echo "== done (Windows: python check_pss_inv.py --no-show) =="
