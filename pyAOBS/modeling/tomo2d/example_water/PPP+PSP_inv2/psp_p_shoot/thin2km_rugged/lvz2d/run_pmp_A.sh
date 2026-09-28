#!/bin/bash
set -euo pipefail
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
WORK=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d/inv_612
OUT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d/wave_fwd
N=-N8/8/0.8/8/1e-4/1e-5
PY=/home/go223/miniconda3/envs/SeisTomo/bin/python
export TOMO2D_FWD_OMP=1 OMP_NUM_THREADS=8

cd "$WORK"
echo "== PmP with -A =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -G"$OUT/geom_obs50_pmp.dat" \
  -Xconv.refl -Bseafloor.refl -Fmoho_true.refl -A "$N" -R"$OUT/rays_pmp_A.dat" \
  > "$OUT/syn_pmp_A.dat" 2>"$OUT/fwd_pmp_A.log"
echo "== PmP without -A =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -G"$OUT/geom_obs50_pmp.dat" \
  -Xconv.refl -Bseafloor.refl -Fmoho_true.refl "$N" -R"$OUT/rays_pmp_noA.dat" \
  > "$OUT/syn_pmp_noA.dat" 2>"$OUT/fwd_pmp_noA.log"
echo "== PmP -A, no -U (single Vp field) =="
"$BIN/tt_forward" -Mtrue_vp.smesh -G"$OUT/geom_obs50_pmp.dat" \
  -Xconv.refl -Bseafloor.refl -Fmoho_true.refl -A "$N" -R"$OUT/rays_pmp_A_noudual.dat" \
  > "$OUT/syn_pmp_A_noudual.dat" 2>"$OUT/fwd_pmp_A_noudual.log"
echo "== PmP no -A, no -U =="
"$BIN/tt_forward" -Mtrue_vp.smesh -G"$OUT/geom_obs50_pmp.dat" \
  -Xconv.refl -Bseafloor.refl -Fmoho_true.refl "$N" -R"$OUT/rays_pmp_noA_noudual.dat" \
  > "$OUT/syn_pmp_noA_noudual.dat" 2>"$OUT/fwd_pmp_noA_noudual.log"
echo "== done pmp A / noA =="
