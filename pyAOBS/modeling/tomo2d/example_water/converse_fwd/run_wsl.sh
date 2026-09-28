#!/bin/bash
set -euo pipefail
cd /mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/converse_fwd
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1
export OMP_NUM_THREADS=8

python3 make_converse_fwd_case.py
echo "== tt_forward 0+6 =="
"$BIN/tt_forward" -Mconverse.smesh -Ggeom_conv.dat -Xconv.refl "$N" \
  -Rrays_conv.dat > syn_conv.dat
python3 check_converse_fwd.py --no-show
echo "== done =="
