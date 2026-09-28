#!/bin/bash
set -euo pipefail
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d"
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1 OMP_NUM_THREADS=8
if [ ! -f rec_vp.smesh ] || [ ! -f path_a/rec_vs_lid.smesh ]; then
  echo "missing rec_vp or path_a/rec_vs_lid"
  exit 1
fi
echo "== PPP Vp + PPS/PPP rec_vs_lid  geom_all =="
"$BIN/tt_forward" -Mrec_vp.smesh -Upath_a/rec_vs_lid.smesh -Ggeom_all.dat \
  -Xconv.refl -Bseafloor.refl "$N" > syn_lidvs_all.dat
echo wrote syn_lidvs_all.dat
