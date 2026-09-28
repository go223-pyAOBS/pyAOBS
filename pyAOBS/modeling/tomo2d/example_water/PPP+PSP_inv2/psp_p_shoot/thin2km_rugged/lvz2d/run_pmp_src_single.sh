#!/bin/bash
set -euo pipefail
# Original single-field tomo2d (src - 副本): type 1 only, ±A. Does not touch current build-tomo2d.
SRC_ORIG="/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src - 副本"
BIN="$SRC_ORIG/build-orig"
WORK=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d/inv_612
OUT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d/wave_fwd
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=0
unset OMP_NUM_THREADS

echo "== cmake original tt_forward =="
cmake -S "$SRC_ORIG" -B "$BIN" >/dev/null
cmake --build "$BIN" -j8 --target tt_forward

cd "$WORK"
echo "== original single-field PmP WITH -A =="
"$BIN/tt_forward" -Mtrue_vp.smesh -G"$OUT/geom_obs50_pmp.dat" \
  -Fmoho_true.refl -A "$N" -R"$OUT/rays_pmp_src_A.dat" \
  > "$OUT/syn_pmp_src_A.dat" 2>"$OUT/fwd_pmp_src_A.log"
echo "== original single-field PmP WITHOUT -A =="
"$BIN/tt_forward" -Mtrue_vp.smesh -G"$OUT/geom_obs50_pmp.dat" \
  -Fmoho_true.refl "$N" -R"$OUT/rays_pmp_src_noA.dat" \
  > "$OUT/syn_pmp_src_noA.dat" 2>"$OUT/fwd_pmp_src_noA.log"
echo "== done original single-field PmP =="
