#!/bin/bash
set -euo pipefail
cd /mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/converse_inv
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1
export TOMO2D_INV_OMP=1
export OMP_NUM_THREADS=8

python3 make_converse_inv_case.py
echo "== true forward 6 =="
"$BIN/tt_forward" -Mtrue.smesh -Ggeom_inv.dat -Xconv.refl "$N" > syn_inv.dat
echo "== inverse 6 =="
"$BIN/tt_inverse" -Mstart.smesh -Gsyn_inv.dat -Bconv.refl -Yseafloor.refl -w "$N" \
  -I5 -SV200 -TV20 -CVvcorr.dat -Oout -l -Linv.log
echo "== start / recovered forward =="
"$BIN/tt_forward" -Mstart.smesh -Ggeom_inv.dat -Xconv.refl "$N" > syn_start.dat
"$BIN/tt_forward" -Mout.smesh.5.1 -Ggeom_inv.dat -Xconv.refl "$N" \
  -Rrays_rec.dat > syn_rec.dat
python3 check_converse_inv.py --no-show
echo "== done =="
