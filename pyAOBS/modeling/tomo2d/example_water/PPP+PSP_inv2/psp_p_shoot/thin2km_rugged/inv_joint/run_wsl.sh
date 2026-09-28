#!/bin/bash
set -euo pipefail
# P1 四震相绝对走时联合：0+6+7+8，一次 tt_inverse。
# Vp 初值=PPP 收回；Vs=rec_vp/κ 且盖层+面下 +0.50。观测各自正演。
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/inv_joint"
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1 TOMO2D_INV_OMP=1 OMP_NUM_THREADS=8
export TOMO2D_INV_LSQR_MAXITER=8000
export PYTHONPATH="$ROOT/inv_2d:${PYTHONPATH:-}"

python3 "$ROOT/make_joint.py" .
python3 "$ROOT/apply_hot_vs.py" . --lid

rm -f out_joint.smesh.* out_joint.vp.smesh* inv_joint.log

echo "== joint obs: true_vp + true_vs  geom 0+6+7+8 =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_all.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_true.dat > syn_inv.dat
cp -f syn_inv.dat syn_true.dat

echo "== joint start: ppp_vp + lid/below Vs+0.50 =="
"$BIN/tt_forward" -Mppp_vp.smesh -Ustart_vs.smesh -Ggeom_all.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_start.dat > syn_start.dat

echo "== inverse JOINT  0+6+7+8  -k1.73  -U hot Vs =="
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

echo "== joint recovered 0+6+7+8 =="
"$BIN/tt_forward" -Mrec_vp.smesh -Urec_vs.smesh -Ggeom_all.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_rec.dat > syn_rec.dat

echo "== done joint  plot: python check_joint.py =="
