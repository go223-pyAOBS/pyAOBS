#!/bin/bash
set -euo pipefail
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
DST=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d/wave_fwd/moho15
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1 OMP_NUM_THREADS=8
cd "$DST"
echo "== tt_forward OBS50  Moho~15km =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_obs50.dat \
  -Xconv.refl -Bseafloor.refl -Fmoho_true.refl -A "$N" -Rrays_obs50.dat \
  > syn_obs50.dat 2>fwd_obs50.log
echo "== done moho15 rays =="
