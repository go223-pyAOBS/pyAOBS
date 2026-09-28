#!/bin/bash
set -euo pipefail
# Dual-field type 1 ±A. Current tt_forward (original has no -U).
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
WORK=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d/inv_612
OUT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d/wave_fwd
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=0
unset OMP_NUM_THREADS

cd "$WORK"
echo "== dual PmP WITH -A =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -G"$OUT/geom_obs50_pmp.dat" \
  -Xconv.refl -Bseafloor.refl -Fmoho_true.refl -A "$N" -R"$OUT/rays_pmp_dual_A.dat" \
  > "$OUT/syn_pmp_dual_A.dat" 2>"$OUT/fwd_pmp_dual_A.log"
echo "== dual PmP WITHOUT -A =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -G"$OUT/geom_obs50_pmp.dat" \
  -Xconv.refl -Bseafloor.refl -Fmoho_true.refl "$N" -R"$OUT/rays_pmp_dual_noA.dat" \
  > "$OUT/syn_pmp_dual_noA.dat" 2>"$OUT/fwd_pmp_dual_noA.log"
echo "== done dual PmP =="
