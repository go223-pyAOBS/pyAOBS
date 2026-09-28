#!/bin/bash
set -euo pipefail
cd /mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/water_inv3
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N4/4/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1
export TOMO2D_INV_OMP=1
export OMP_NUM_THREADS=10

echo "== inverse 2+3 Lv=0.35 SV=500 =="
"$BIN/tt_inverse" -Mstart.smesh -Gsyn_inv.dat -Yseafloor.refl -y "$N" \
  -I5 -SV500 -CVvcorr_lv035.dat -Oout23s -l -Linv23s.log

echo "== recovered forward =="
"$BIN/tt_forward" -Mout23s.smesh.5.1 -Ggeom_inv.dat -Fseafloor.refl "$N" \
  -Rrays_rec23s.dat > syn_rec23s.dat
echo "== done =="
