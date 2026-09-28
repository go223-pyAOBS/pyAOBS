#!/bin/bash
set -euo pipefail
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1
export OMP_NUM_THREADS=8
cd "$ROOT/inv_graph6k_hot"
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_psp6.dat \
  -Xconv.refl -Bseafloor.refl "$N" > /tmp/dual_true_psp.dat
cd "$ROOT/inv_graph6_hot"
"$BIN/tt_forward" -Mtrue_mixed.smesh -Ggeom_psp6.dat -Xconv.refl -Bseafloor.refl \
  "$N" > /tmp/mixed_true_psp.dat
echo done
