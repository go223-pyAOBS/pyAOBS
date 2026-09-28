#!/bin/bash
set -euo pipefail
# PPP 分域两段（-B 只作结点掩膜，不改 type 0 图论）：
# ① FREEZE_LID 冻盖层，只反面下 Vp；② FREEZE_BELOW 冻面下，只反盖层 Vp。
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/inv_ppp_lid"
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1 TOMO2D_INV_OMP=1 OMP_NUM_THREADS=8
export TOMO2D_INV_LSQR_MAXITER=8000
export PYTHONPATH="$ROOT/inv_2d:${PYTHONPATH:-}"
unset TOMO2D_INV_FREEZE_BELOW
unset TOMO2D_INV_FREEZE_LID

python3 "$ROOT/make_joint.py" .

rm -f out_below.smesh.* out_lid.smesh.* inv_below.log inv_lid.log

echo "== PPP obs: true_vp =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Ggeom_ppp.dat -Bseafloor.refl \
  "$N" > syn_inv.dat
cp -f syn_inv.dat syn_true.dat

echo "== PPP start: start_vp =="
"$BIN/tt_forward" -Mstart_vp.smesh -Ggeom_ppp.dat -Bseafloor.refl \
  "$N" -Rrays_start.dat > syn_start.dat

echo "== stage1: freeze lid  invert below  -B mask only =="
export TOMO2D_INV_FREEZE_LID=1
"$BIN/tt_inverse" -Mstart_vp.smesh -Gsyn_inv.dat \
  -Yseafloor.refl -w -Bconv.refl \
  "$N" -I5 -SV200 -TV5 -CVvcorr.dat \
  -Oout_below -l -Linv_below.log
unset TOMO2D_INV_FREEZE_LID

python3 - <<'PY'
from pathlib import Path
import shutil
import inv_grid as g
vp = g.latest("out_below.smesh*.*")
shutil.copyfile(vp, "ppp_vp.smesh")
print(f"ppp_vp.smesh <- {vp.name}  (stage-1 below-only)")
PY

echo "== stage1 recovered =="
"$BIN/tt_forward" -Mppp_vp.smesh -Ggeom_ppp.dat -Bseafloor.refl \
  "$N" > syn_stage1.dat

echo "== stage2: freeze below  invert lid =="
export TOMO2D_INV_FREEZE_BELOW=1
"$BIN/tt_inverse" -Mppp_vp.smesh -Gsyn_inv.dat \
  -Yseafloor.refl -w -Bconv.refl \
  "$N" -I8 -SV200 -TV5 -CVvcorr.dat \
  -Oout_lid -l -Linv_lid.log
unset TOMO2D_INV_FREEZE_BELOW

python3 - <<'PY'
from pathlib import Path
import shutil
import inv_grid as g
vp = g.latest("out_lid.smesh*.*")
shutil.copyfile(vp, "rec_vp.smesh")
shutil.copyfile("true_vs.smesh", "start_vs.smesh")
shutil.copyfile("true_vs.smesh", "rec_vs.smesh")
print(f"rec_vp.smesh <- {vp.name}  (stage-2 lid-only)")
PY

echo "== stage2 recovered =="
"$BIN/tt_forward" -Mrec_vp.smesh -Ggeom_ppp.dat -Bseafloor.refl \
  "$N" -Rrays_rec.dat > syn_rec.dat

echo "== done ppp-lid  plot: python check_joint.py --ppplid =="
