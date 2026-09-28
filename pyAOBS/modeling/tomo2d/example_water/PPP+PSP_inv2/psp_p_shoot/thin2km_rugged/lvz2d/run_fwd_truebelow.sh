#!/bin/bash
set -euo pipefail
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d"
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1 OMP_NUM_THREADS=8
export PYTHONPATH="$ROOT/inv_2d:${PYTHONPATH:-}"

python3 make_vs_truebelow.py

echo "== rec_vp + lid-rec / below-true =="
"$BIN/tt_forward" -Mrec_vp.smesh -Uvs_lid_truebelow.smesh -Ggeom_all.dat \
  -Xconv.refl -Bseafloor.refl "$N" > syn_truebelow_all.dat
echo wrote syn_truebelow_all.dat

echo "== rec_vp + true_vs (lid+below) =="
"$BIN/tt_forward" -Mrec_vp.smesh -Utrue_vs.smesh -Ggeom_all.dat \
  -Xconv.refl -Bseafloor.refl "$N" > syn_truevs_all.dat
echo wrote syn_truevs_all.dat
