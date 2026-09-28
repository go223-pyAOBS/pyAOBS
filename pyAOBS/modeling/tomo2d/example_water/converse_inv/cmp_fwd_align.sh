#!/bin/bash
set -eu
cd /mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/converse_inv
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1
export OMP_NUM_THREADS=8

echo "== converse true =="
"$BIN/tt_forward" -Mtrue.smesh -Ggeom_inv.dat -Xconv.refl $N \
  -Rrays_cv_true.dat > syn_inv.dat
echo "== dual true =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_inv.dat \
  -Xconv.refl -Bseafloor.refl $N -Rrays_psx_true.dat > syn_psx.dat
echo "== converse start =="
"$BIN/tt_forward" -Mstart.smesh -Ggeom_inv.dat -Xconv.refl $N > syn_start.dat
echo "== dual start =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Ustart_vs.smesh -Ggeom_inv.dat \
  -Xconv.refl -Bseafloor.refl $N > syn_psx_start.dat
python3 cmp_fwd_align.py
echo "== done =="
