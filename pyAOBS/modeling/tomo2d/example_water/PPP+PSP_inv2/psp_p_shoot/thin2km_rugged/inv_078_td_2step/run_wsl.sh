#!/bin/bash
set -euo pipefail
# 两步（无 PSP）：① 0+7 -td 收台侧盖层 Vs；② 冻盖层，PSS 只反面下。
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/inv_078_td_2step"
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
NINV=-N8/8/0.8/8/1e-3/1e-4
export TOMO2D_FWD_OMP=1 TOMO2D_INV_OMP=1 OMP_NUM_THREADS=8
export TOMO2D_INV_LSQR_MAXITER=8000
export PYTHONPATH="$ROOT/inv_2d:${PYTHONPATH:-}"

python3 "$ROOT/make_joint.py" .
python3 "$ROOT/apply_hot_vs.py" . --lid

rm -f out_lid.smesh.* out_pss.smesh.* out_lid.vp.smesh* out_pss.vp.smesh* inv_lid.log inv_pss.log

echo "== 2step obs: true  geom 0+7+8 =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_078.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_true.dat > syn_inv.dat
cp -f syn_inv.dat syn_true.dat

echo "== 2step start: rec_vp + lid/below Vs+0.50 =="
"$BIN/tt_forward" -Mrec_vp.smesh -Ustart_vs.smesh -Ggeom_078.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_start.dat > syn_start.dat

echo "== step1: PPS-PPP  lid Vs  geom 0+7 -td =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_07.dat \
  -Xconv.refl -Bseafloor.refl "$N" > syn_07.dat
"$BIN/tt_inverse" -Mrec_vp.smesh -Ustart_vs.smesh -Gsyn_07.dat \
  -Bconv.refl -Yseafloor.refl -w -k1.73 -td \
  "$NINV" -I8 -SV200 -TV5 -CVvcorr.dat \
  -Oout_lid -l -Linv_lid.log

python3 - <<'PY'
from pathlib import Path
import shutil
import filecmp
import inv_grid as g
vs = g.latest("out_lid.smesh*.*")
shutil.copyfile(vs, "rec_vs_lid.smesh")
print(f"rec_vs_lid.smesh <- {vs.name}")
if not filecmp.cmp("ppp_vp.smesh", "rec_vp.smesh", shallow=False):
    raise SystemExit("rec_vp was overwritten in step1")
PY

echo "== step2: freeze lid  PSS below only  geom 8 =="
export TOMO2D_INV_FREEZE_LID=1
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_pss8.dat \
  -Xconv.refl -Bseafloor.refl "$N" > syn_08.dat
"$BIN/tt_inverse" -Mrec_vp.smesh -Urec_vs_lid.smesh -Gsyn_08.dat \
  -Bconv.refl -Yseafloor.refl -w -k1.73 \
  "$NINV" -I8 -SV200 -TV5 -CVvcorr.dat \
  -Oout_pss -l -Linv_pss.log
unset TOMO2D_INV_FREEZE_LID

python3 - <<'PY'
from pathlib import Path
import shutil
import filecmp
import inv_grid as g
vs = g.latest("out_pss.smesh*.*")
shutil.copyfile(vs, "rec_vs.smesh")
g.write_psp_speed(Path("rec_vp.smesh"), Path("rec_vs.smesh"), Path("rec_mixed.smesh"))
print(f"rec_vs.smesh <- {vs.name}")
if not filecmp.cmp("ppp_vp.smesh", "rec_vp.smesh", shallow=False):
    raise SystemExit("rec_vp was overwritten in step2")
PY

echo "== 2step recovered (0+7+8) =="
"$BIN/tt_forward" -Mrec_vp.smesh -Urec_vs.smesh -Ggeom_078.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_rec.dat > syn_rec.dat

echo "== holdout geom_all (PSP not inverted) =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_all.dat \
  -Xconv.refl -Bseafloor.refl "$N" > syn_holdout_true.dat
"$BIN/tt_forward" -Mrec_vp.smesh -Urec_vs.smesh -Ggeom_all.dat \
  -Xconv.refl -Bseafloor.refl "$N" > syn_holdout_rec.dat

echo "== PPP check on frozen rec_vp =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Ggeom_ppp.dat -Bseafloor.refl "$N" > syn_ppp.dat
"$BIN/tt_forward" -Mrec_vp.smesh -Ggeom_ppp.dat -Bseafloor.refl "$N" > syn_ppp_rec.dat

echo "== done 2step  plot: python check_joint.py --td2step =="
