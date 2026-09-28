#!/bin/bash
set -euo pipefail
cd /mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/water_inv2
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N4/4/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1
export TOMO2D_INV_OMP=1
export OMP_NUM_THREADS=8

echo "== inverse 2+3  SV200 TV1 =="
"$BIN/tt_inverse" -Mstart.smesh -Gsyn_inv.dat -Yseafloor.refl -y "$N" \
  -I5 -SV200 -TV1 -CVvcorr.dat -Oout23d -l -Linv23d.log

echo "== inverse only2  SV200 TV1 =="
"$BIN/tt_inverse" -Mstart.smesh -Gsyn_inv_c2.dat -Yseafloor.refl -y "$N" \
  -I5 -SV200 -TV1 -CVvcorr.dat -Oout2d -l -Linv2d.log

echo "== recovered forward =="
"$BIN/tt_forward" -Mout23d.smesh.5.1 -Ggeom_inv.dat -Fseafloor.refl "$N" \
  -Rrays_rec23d.dat > syn_rec23d.dat
"$BIN/tt_forward" -Mout2d.smesh.5.1 -Ggeom_inv_c2.dat -Fseafloor.refl "$N" \
  -Rrays_rec2d.dat > syn_rec2d.dat
"$BIN/tt_forward" -Mout2d.smesh.5.1 -Ggeom_inv.dat -Fseafloor.refl "$N" \
  > syn_rec2d_on23.dat
echo "== done =="
