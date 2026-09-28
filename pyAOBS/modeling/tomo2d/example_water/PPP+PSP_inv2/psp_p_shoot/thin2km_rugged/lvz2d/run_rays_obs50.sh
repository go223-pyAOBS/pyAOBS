#!/bin/bash
set -euo pipefail
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
WORK=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d/inv_612
OUT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d/wave_fwd
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1 OMP_NUM_THREADS=8
cd "$WORK"
echo "== tt_forward OBS50 rays =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -G"$OUT/geom_obs50.dat" \
  -Xconv.refl -Bseafloor.refl -Fmoho_true.refl -A "$N" -R"$OUT/rays_obs50.dat" \
  > "$OUT/syn_obs50.dat"
echo "== done rays_obs50 =="
