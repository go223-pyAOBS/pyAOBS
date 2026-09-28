#!/bin/bash
set -euo pipefail
cd /mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/water_checkboard
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N4/4/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1
export TOMO2D_INV_OMP=1
export OMP_NUM_THREADS=8

echo "== true forward 2+3 =="
"$BIN/tt_forward" -Mtrue.smesh -Ggeom_inv.dat -Fseafloor.refl "$N" > syn_inv.dat
echo "== true forward only2 =="
"$BIN/tt_forward" -Mtrue.smesh -Ggeom_inv_c2.dat -Fseafloor.refl "$N" > syn_inv_c2.dat

echo "== inverse 2+3 =="
"$BIN/tt_inverse" -Mstart.smesh -Gsyn_inv.dat -Yseafloor.refl -y "$N" \
  -I5 -SV200 -CVvcorr.dat -Oout23 -l -Linv23.log
echo "== inverse only2 =="
"$BIN/tt_inverse" -Mstart.smesh -Gsyn_inv_c2.dat -Yseafloor.refl -y "$N" \
  -I5 -SV200 -CVvcorr.dat -Oout2 -l -Linv2.log

echo "== start / recovered forward =="
"$BIN/tt_forward" -Mstart.smesh -Ggeom_inv.dat -Fseafloor.refl "$N" > syn_start.dat
"$BIN/tt_forward" -Mstart.smesh -Ggeom_inv_c2.dat -Fseafloor.refl "$N" > syn_start_c2.dat
"$BIN/tt_forward" -Mout23.smesh.5.1 -Ggeom_inv.dat -Fseafloor.refl "$N" \
  -Rrays_rec23.dat > syn_rec23.dat
"$BIN/tt_forward" -Mout2.smesh.5.1 -Ggeom_inv_c2.dat -Fseafloor.refl "$N" \
  -Rrays_rec2.dat > syn_rec2.dat
"$BIN/tt_forward" -Mout2.smesh.5.1 -Ggeom_inv.dat -Fseafloor.refl "$N" \
  > syn_rec2_on23.dat
echo "== done =="
