#!/bin/bash
set -euo pipefail
# 二维非均匀（崎岖转换面 + 壳/幔低速区）
# 1) PPP 反 Vp（不分域）
# A) 在 PPP 上：0+7 -td（PPS 绝对 + PPS−PPP）收台侧盖层 Vs；再冻盖层，PSS 反面下 Vs
# B) 与 A 同一 PPS 盖层 Vs：真 PSP=6 只反面下
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d"
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
NINV=-N8/8/0.8/8/1e-3/1e-4
export TOMO2D_FWD_OMP=1 TOMO2D_INV_OMP=1 OMP_NUM_THREADS=8
export TOMO2D_INV_LSQR_MAXITER=8000
export PYTHONPATH="$ROOT/inv_2d:${PYTHONPATH:-}"
unset TOMO2D_INV_FREEZE_LID
unset TOMO2D_INV_FREEZE_BELOW

python3 make_lvz.py

rm -f out_ppp.smesh.* inv_ppp.log
echo "== 1. PPP obs / start =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Ggeom_ppp.dat -Bseafloor.refl \
  "$N" > syn_ppp.dat
"$BIN/tt_forward" -Mstart_vp.smesh -Ggeom_ppp.dat -Bseafloor.refl \
  "$N" -Rrays_ppp_start.dat > syn_ppp_start.dat

echo "== 1. inverse PPP  (no -B conv) =="
"$BIN/tt_inverse" -Mstart_vp.smesh -Gsyn_ppp.dat -Yseafloor.refl -w \
  "$N" -I5 -SV200 -TV5 -CVvcorr.dat -Oout_ppp -l -Linv_ppp.log

python3 - <<'PY'
from pathlib import Path
import shutil
import inv_grid as g
from make_lvz import seed_child
rec = g.latest("out_ppp.smesh*.*")
shutil.copyfile(rec, "rec_vp.smesh")
shutil.copyfile(rec, "ppp_vp.smesh")
print(f"rec_vp.smesh <- {rec.name}")
seed_child(Path("path_a"))
seed_child(Path("path_b"))
PY

echo "== 1. recovered PPP =="
"$BIN/tt_forward" -Mrec_vp.smesh -Ggeom_ppp.dat -Bseafloor.refl \
  "$N" -Rrays_ppp_rec.dat > syn_ppp_rec.dat

# ----- A: PPS + PPS-PPP lid Vs, then PSS below Vs -----
cd path_a
python3 "$ROOT/apply_hot_vs.py" . --lid
rm -f out_lid.smesh.* out_pss.smesh.* out_lid.vp.smesh* out_pss.vp.smesh* inv_lid.log inv_pss.log

echo "== A obs 0+7+8 =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_078.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_true.dat > syn_inv.dat
cp -f syn_inv.dat syn_true.dat
"$BIN/tt_forward" -Mrec_vp.smesh -Ustart_vs.smesh -Ggeom_078.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_start.dat > syn_start.dat

echo "== A step1: PPS abs + PPS-PPP  lid Vs =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_07.dat \
  -Xconv.refl -Bseafloor.refl "$N" > syn_07.dat
"$BIN/tt_inverse" -Mrec_vp.smesh -Ustart_vs.smesh -Gsyn_07.dat \
  -Bconv.refl -Yseafloor.refl -w -k1.73 -td \
  "$NINV" -I8 -SV200 -TV5 -CVvcorr.dat \
  -Oout_lid -l -Linv_lid.log

python3 - <<'PY'
from pathlib import Path
import shutil, filecmp
import inv_grid as g
vs = g.latest("out_lid.smesh*.*")
shutil.copyfile(vs, "rec_vs_lid.smesh")
print(f"rec_vs_lid.smesh <- {vs.name}")
if not filecmp.cmp("ppp_vp.smesh", "rec_vp.smesh", shallow=False):
    raise SystemExit("A step1 overwrote rec_vp")
PY

echo "== A step2: freeze lid  PSS below =="
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
import shutil, filecmp
import inv_grid as g
vs = g.latest("out_pss.smesh*.*")
shutil.copyfile(vs, "rec_vs.smesh")
g.write_psp_speed(Path("rec_vp.smesh"), Path("rec_vs.smesh"), Path("rec_mixed.smesh"))
print(f"rec_vs.smesh <- {vs.name}")
if not filecmp.cmp("ppp_vp.smesh", "rec_vp.smesh", shallow=False):
    raise SystemExit("A step2 overwrote rec_vp")
PY

echo "== A recovered 0+7+8 + holdout all =="
"$BIN/tt_forward" -Mrec_vp.smesh -Urec_vs.smesh -Ggeom_078.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_rec.dat > syn_rec.dat
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_all.dat \
  -Xconv.refl -Bseafloor.refl "$N" > syn_holdout_true.dat
"$BIN/tt_forward" -Mrec_vp.smesh -Urec_vs.smesh -Ggeom_all.dat \
  -Xconv.refl -Bseafloor.refl "$N" > syn_holdout_rec.dat

# ----- B: 与 A 同一套 PPS 盖层 Vs，真 PSP 只反面下 -----
cd ../path_b
cp -f ../path_a/rec_vs_lid.smesh rec_vs_lid.smesh
rm -f out_psp.smesh.* out_psp.vp.smesh* inv_psp.log

echo "== B obs PSP=6 =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_psp6.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_true.dat > syn_inv.dat
cp -f syn_inv.dat syn_true.dat
"$BIN/tt_forward" -Mrec_vp.smesh -Urec_vs_lid.smesh -Ggeom_psp6.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_start.dat > syn_start.dat

echo "== B inverse PSP=6  freeze same lid as A =="
export TOMO2D_INV_FREEZE_LID=1
"$BIN/tt_inverse" -Mrec_vp.smesh -Urec_vs_lid.smesh -Gsyn_inv.dat \
  -Bconv.refl -Yseafloor.refl -w -k1.73 \
  "$NINV" -I8 -SV200 -TV5 -CVvcorr.dat \
  -Oout_psp -l -Linv_psp.log
unset TOMO2D_INV_FREEZE_LID

python3 - <<'PY'
from pathlib import Path
import shutil, filecmp
import inv_grid as g
vs = g.latest("out_psp.smesh*.*")
shutil.copyfile(vs, "rec_vs.smesh")
g.write_psp_speed(Path("rec_vp.smesh"), Path("rec_vs.smesh"), Path("rec_mixed.smesh"))
print(f"rec_vs.smesh <- {vs.name}")
if not filecmp.cmp("ppp_vp.smesh", "rec_vp.smesh", shallow=False):
    raise SystemExit("B overwrote rec_vp")
PY

echo "== B recovered PSP + holdout all =="
"$BIN/tt_forward" -Mrec_vp.smesh -Urec_vs.smesh -Ggeom_psp6.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_rec.dat > syn_rec.dat
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_all.dat \
  -Xconv.refl -Bseafloor.refl "$N" > syn_holdout_true.dat
"$BIN/tt_forward" -Mrec_vp.smesh -Urec_vs.smesh -Ggeom_all.dat \
  -Xconv.refl -Bseafloor.refl "$N" > syn_holdout_rec.dat

cd ..
echo "== done lvz2d  plot: python check_lvz.py =="
