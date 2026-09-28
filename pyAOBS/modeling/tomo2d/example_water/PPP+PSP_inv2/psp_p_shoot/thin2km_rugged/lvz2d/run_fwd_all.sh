#!/bin/bash
set -euo pipefail
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d"
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
HERE=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d
WORK="$HERE/fwd_all"
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1 OMP_NUM_THREADS=8

python3 "$HERE/make_fwd_all.py"

echo "== fwd all phases 0/1/6-15 =="
"$BIN/tt_forward" -M"$WORK/true_vp.smesh" -U"$WORK/true_vs.smesh" \
  -G"$WORK/geom_all_ph.dat" -X"$WORK/conv.refl" -B"$WORK/seafloor.refl" \
  -F"$WORK/moho_true.refl" "$N" -R"$WORK/rays_true.dat" > "$WORK/syn_true.dat"

echo "== done fwd_all =="
