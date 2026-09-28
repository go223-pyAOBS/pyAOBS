#!/bin/bash
set -euo pipefail
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/inv_2d_hot"
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1 TOMO2D_INV_OMP=1 OMP_NUM_THREADS=8
export TOMO2D_INV_LSQR_MAXITER=8000
export PYTHONPATH="/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/inv_2d:${PYTHONPATH:-}"

rm -f out_hot.smesh.* inv_hot.log
echo "== hot start type0 forward =="
"$BIN/tt_forward" -Mstart_hot.smesh -Ggeom_psp0.dat -Bseafloor.refl \
  "$N" > syn_hot_start.dat
echo "== inverse PSP-as-0 from hot start =="
"$BIN/tt_inverse" -Mstart_hot.smesh -Gsyn_inv.dat \
  -Yseafloor.refl -w \
  -DV1 -DQdamp_lid.dat "$N" \
  -I8 -SV200 -CVvcorr.dat \
  -Oout_hot -l -Linv_hot.log
python3 - <<'PY'
import shutil
import inv_grid as g
rec = g.latest("out_hot.smesh*.*")
shutil.copyfile(rec, "rec_hot.smesh")
print(f"rec_hot.smesh <- {rec.name}")
PY
echo "== recovered forward =="
"$BIN/tt_forward" -Mrec_hot.smesh -Ggeom_psp0.dat -Bseafloor.refl \
  "$N" -Rrays_hot.dat > syn_hot_rec.dat
echo "== done hot-start test =="
