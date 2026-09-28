#!/bin/bash
set -euo pipefail
# 冻 rec_vp；现场流程：PSS 绝对走时进核，无 PSP；-td 拟合 PPS−PPP（台侧盖层 Vs）。
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/inv_078_td_hot"
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
NINV=-N8/8/0.8/8/1e-3/1e-4
export TOMO2D_FWD_OMP=1 TOMO2D_INV_OMP=1 OMP_NUM_THREADS=8
export TOMO2D_INV_LSQR_MAXITER=8000
export PYTHONPATH="$ROOT/inv_2d:${PYTHONPATH:-}"

python3 "$ROOT/make_joint.py" .
python3 "$ROOT/apply_hot_vs.py" . --lid

rm -f out_td.smesh.* out_td.vp.smesh* inv_td.log

echo "== 078-td obs: true_vp + true_vs  geom 0+7+8 (no PSP) =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_078.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_true.dat > syn_inv.dat
cp -f syn_inv.dat syn_true.dat

echo "== 078-td start: rec_vp + lid/below Vs+0.50 =="
"$BIN/tt_forward" -Mrec_vp.smesh -Ustart_vs.smesh -Ggeom_078.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_start.dat > syn_start.dat

echo "== inverse Vs-only -td  freeze Vp  abs PSS on  no PSP =="
"$BIN/tt_inverse" -Mrec_vp.smesh -Ustart_vs.smesh -Gsyn_inv.dat \
  -Bconv.refl -Yseafloor.refl -w -k1.73 -td \
  "$NINV" -I8 -SV200 -TV5 -CVvcorr.dat \
  -Oout_td -l -Linv_td.log

python3 - <<'PY'
from pathlib import Path
import shutil
import filecmp
import inv_grid as g
vs = g.latest("out_td.smesh*.*")
shutil.copyfile(vs, "rec_vs.smesh")
g.write_psp_speed(Path("rec_vp.smesh"), Path("rec_vs.smesh"), Path("rec_mixed.smesh"))
print(f"rec_vs.smesh <- {vs.name}")
same = filecmp.cmp("ppp_vp.smesh", "rec_vp.smesh", shallow=False)
print(f"rec_vp == ppp_vp: {same}")
if not same:
    raise SystemExit("rec_vp was overwritten — Vp must stay frozen")
PY

echo "== 078-td recovered (0+7+8) =="
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

echo "== done 078-td  plot: python check_joint.py --td078 =="
