#!/bin/bash
set -euo pipefail
# Current tt_forward: all reflection-related raytypes, dual, -A.
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
WORK=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d/inv_612
OUT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d/wave_fwd
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=0
unset OMP_NUM_THREADS

cd "$WORK"
echo "== current tt_forward OBS50 all reflection codes =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -G"$OUT/geom_obs50_refl.dat" \
  -Xconv.refl -Bseafloor.refl -Fmoho_true.refl -A "$N" \
  -R"$OUT/rays_obs50_refl.dat" \
  > "$OUT/syn_obs50_refl.dat" 2>"$OUT/fwd_obs50_refl.log"
echo "== done rays_obs50_refl =="
