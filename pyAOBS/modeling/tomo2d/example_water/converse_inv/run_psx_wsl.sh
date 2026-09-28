#!/bin/bash
# ps_inv 算法 + converse 题目：盖层真 Vp，面下只错 Vs0，-TV20
set -euo pipefail
cd /mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/converse_inv
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1
export TOMO2D_INV_OMP=1
export OMP_NUM_THREADS=8

python3 make_converse_inv_case.py
echo "== true PSP  -M true_vp -U true_vs =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_inv.dat \
  -Xconv.refl -Bseafloor.refl "$N" > syn_psx.dat
echo "== start PSP  -M true_vp -U start_vs =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Ustart_vs.smesh -Ggeom_inv.dat \
  -Xconv.refl -Bseafloor.refl "$N" > syn_psx_start.dat
echo "== inverse PSP below-S  -M true_vp -U start_vs -k1.73 -TV20 =="
"$BIN/tt_inverse" -Mtrue_vp.smesh -Ustart_vs.smesh -Gsyn_psx.dat \
  -Bconv.refl -Yseafloor.refl -w -k1.73 "$N" \
  -I5 -SV200 -TV20 -CVvcorr.dat -Oout_psx -l -Linv_psx.log
echo "== recovered PSP  -M out_psx.vp -U out_psx.smesh =="
VS=$(python3 -c "from pathlib import Path; c=sorted(Path('.').glob('out_psx.smesh.*.*'));
print(max(c, key=lambda p: (int(p.name.split('.')[-2]), int(p.name.split('.')[-1]))))")
echo "rec Vs mesh $VS"
"$BIN/tt_forward" -Mout_psx.vp.smesh -U"$VS" -Ggeom_inv.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_psx_rec.dat > syn_psx_rec.dat
echo "== done =="
