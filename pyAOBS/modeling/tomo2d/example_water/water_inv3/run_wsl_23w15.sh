#!/bin/bash
set -euo pipefail
cd /mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/water_inv3
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N4/4/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1
export TOMO2D_INV_OMP=1
export OMP_NUM_THREADS=10

echo "== inverse 2+3  dt3=1.5x dt2  Lv=1 SV=200 =="
"$BIN/tt_inverse" -Mstart.smesh -Gsyn_inv_w15.dat -Yseafloor.refl -y "$N" \
  -I5 -SV200 -CVvcorr.dat -Oout23w15 -l -Linv23w15.log

echo "== recovered forward =="
"$BIN/tt_forward" -Mout23w15.smesh.5.1 -Ggeom_inv.dat -Fseafloor.refl "$N" \
  -Rrays_rec23w15.dat > syn_rec23w15.dat
echo "== done =="
