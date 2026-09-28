#!/bin/bash
set -euo pipefail
# P1 对照：四震相联合，Vs 初值=rec_vp/κ，不加 +0.50。
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/inv_joint_kappa"
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
# 弯曲容差略松：κ 初值第 2 轮 type 0 在 (s,r)=3,12 会 bend 超迭代（不改 0/1 算法）
N=-N8/8/0.8/8/1e-3/1e-4
export TOMO2D_FWD_OMP=1 TOMO2D_INV_OMP=1 OMP_NUM_THREADS=8
export TOMO2D_INV_LSQR_MAXITER=8000
export PYTHONPATH="$ROOT/inv_2d:${PYTHONPATH:-}"

python3 "$ROOT/make_joint.py" .
python3 "$ROOT/apply_hot_vs.py" . --kappa

rm -f out_joint.smesh.* out_joint.vp.smesh* inv_joint.log

echo "== kappa joint obs: true_vp + true_vs  geom 0+6+7+8 =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_all.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_true.dat > syn_inv.dat
cp -f syn_inv.dat syn_true.dat

echo "== kappa joint start: ppp_vp + rec_vp/k =="
"$BIN/tt_forward" -Mppp_vp.smesh -Ustart_vs.smesh -Ggeom_all.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_start.dat > syn_start.dat

echo "== inverse JOINT  0+6+7+8  -k1.73  -U kappa Vs =="
"$BIN/tt_inverse" -Mppp_vp.smesh -Ustart_vs.smesh -Gsyn_inv.dat \
  -Bconv.refl -Yseafloor.refl -w -k1.73 \
  "$N" -I8 -SV200 -Ss200 -TV5 -CVvcorr.dat \
  -Oout_joint -l -Linv_joint.log

python3 - <<'PY'
from pathlib import Path
import shutil
import inv_grid as g
vp = g.latest("out_joint.vp.smesh*.*")
vs = g.latest("out_joint.smesh*.*")
shutil.copyfile(vp, "rec_vp.smesh")
shutil.copyfile(vs, "rec_vs.smesh")
g.write_psp_speed(Path("rec_vp.smesh"), Path("rec_vs.smesh"), Path("rec_mixed.smesh"))
print(f"rec_vp.smesh <- {vp.name}")
print(f"rec_vs.smesh <- {vs.name}")
PY

echo "== kappa joint recovered 0+6+7+8 =="
"$BIN/tt_forward" -Mrec_vp.smesh -Urec_vs.smesh -Ggeom_all.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_rec.dat > syn_rec.dat

echo "== done kappa joint  plot: python check_joint.py --kappa =="
