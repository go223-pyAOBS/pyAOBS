#!/bin/bash
set -euo pipefail
cd /mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/ps_fwd
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1
export OMP_NUM_THREADS=8

python3 make_ps_fwd_case.py
echo "== tt_forward PPP first-arrival on true Vp (no -U) =="
"$BIN/tt_forward" -Mvp.smesh -Ggeom_ppp.dat "$N" \
  -Rrays_ppp.dat > syn_ppp.dat
echo "== tt_forward PPP reflection on conv (-F, true Vp) =="
"$BIN/tt_forward" -Mvp.smesh -Ggeom_ppr.dat -Fconv.refl "$N" \
  -Rrays_ppr.dat > syn_ppr.dat
echo "== tt_forward PSP on mixed.smesh (same as converse) =="
"$BIN/tt_forward" -Mmixed.smesh -Ggeom_psp.dat -Xconv.refl "$N" \
  -Rrays_psp.dat > syn_psp.dat
echo "== tt_forward 7/8 on vp_psx+vs dual =="
"$BIN/tt_forward" -Mvp_psx.smesh -Uvs.smesh -Ggeom_psx.dat -Xconv.refl -Bseafloor.refl \
  "$N" -Rrays_psx.dat > syn_psx.dat
python3 make_ps_fwd_case.py --merge
echo "== done =="
