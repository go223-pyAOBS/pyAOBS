#!/bin/bash
set -euo pipefail
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
WORK=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d/inv_612
OUT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d/wave_fwd
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1 OMP_NUM_THREADS=8

cd "$WORK"
echo "== current tt_forward OBS50 all reflection codes, NO -A =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -G"$OUT/geom_obs50_refl.dat" \
  -Xconv.refl -Bseafloor.refl -Fmoho_true.refl "$N" \
  -R"$OUT/rays_obs50_refl_noA.dat" \
  > "$OUT/syn_obs50_refl_noA.dat" 2>"$OUT/fwd_obs50_refl_noA.log"
echo "== done rays_obs50_refl_noA =="
