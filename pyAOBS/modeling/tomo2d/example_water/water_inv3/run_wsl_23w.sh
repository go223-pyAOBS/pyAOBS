#!/bin/bash
set -euo pipefail
cd /mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/water_inv3
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N4/4/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1
export TOMO2D_INV_OMP=1
export OMP_NUM_THREADS=10

echo "== inverse 2+3  dt3=3x dt2  Lv=1 SV=200 =="
"$BIN/tt_inverse" -Mstart.smesh -Gsyn_inv_w3.dat -Yseafloor.refl -y "$N" \
  -I5 -SV200 -CVvcorr.dat -Oout23w -l -Linv23w.log

echo "== recovered forward (unweighted geom, for ttimes compare) =="
"$BIN/tt_forward" -Mout23w.smesh.5.1 -Ggeom_inv.dat -Fseafloor.refl "$N" \
  -Rrays_rec23w.dat > syn_rec23w.dat
echo "== done =="
