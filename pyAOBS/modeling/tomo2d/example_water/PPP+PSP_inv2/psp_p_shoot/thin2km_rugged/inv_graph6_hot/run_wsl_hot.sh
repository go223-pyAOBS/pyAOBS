#!/bin/bash
set -euo pipefail
# 单场公平两步：-M 为 Vp，-k 生成 Vs。观测/初值各自正演，不拷走时。
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/inv_graph6_hot"
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1
export TOMO2D_INV_OMP=1
export OMP_NUM_THREADS=8
export TOMO2D_INV_LSQR_MAXITER=8000
export PYTHONPATH="/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/inv_2d:${PYTHONPATH:-}"

python3 make_hot.py

echo "== single PPP obs: true_vp =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Ggeom_ppp.dat -Bseafloor.refl \
  "$N" > syn_ppp.dat
echo "== single PPP start: start_vp =="
"$BIN/tt_forward" -Mstart_vp.smesh -Ggeom_ppp.dat -Bseafloor.refl \
  "$N" > syn_ppp_start.dat

rm -f out_ppp.smesh.* out_psp.smesh.* inv_ppp.log inv_psp.log

echo "== single inverse PPP =="
"$BIN/tt_inverse" -Mstart_vp.smesh -Gsyn_ppp.dat -Yseafloor.refl -w \
  "$N" -I5 -SV200 -TV5 -CVvcorr.dat -Oout_ppp -l -Linv_ppp.log

python3 - <<'PY'
from pathlib import Path
import shutil
import inv_grid as g
rec = g.latest("out_ppp.smesh*.*")
shutil.copyfile(rec, "rec_vp.smesh")
g.write_vs_from_vp(Path("rec_vp.smesh"), Path("start_vs.smesh"), g.KAPPA)
g.write_psp_speed(Path("rec_vp.smesh"), Path("start_vs.smesh"), Path("start_mixed.smesh"))
print(f"rec_vp.smesh <- {rec.name}")
print("start_vs.smesh <- rec_vp / 1.73")
PY

echo "== single recovered PPP =="
"$BIN/tt_forward" -Mrec_vp.smesh -Ggeom_ppp.dat -Bseafloor.refl \
  "$N" -Rrays_ppp_rec.dat > syn_ppp_rec.dat

echo "== single PSP obs: true_vp -k1.73 =="
"$BIN/tt_forward" -Mtrue_vp.smesh -k1.73 -Ggeom_psp6.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_true.dat > syn_inv.dat
cp -f syn_inv.dat syn_true.dat

echo "== single PSP start: rec_vp -k1.73 =="
"$BIN/tt_forward" -Mrec_vp.smesh -k1.73 -Ggeom_psp6.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_start.dat > syn_start.dat

echo "== single inverse PSP=6  -M rec_vp -k1.73 =="
"$BIN/tt_inverse" -Mrec_vp.smesh -k1.73 -Gsyn_inv.dat \
  -Bconv.refl -Yseafloor.refl -w \
  -DV1 -DQdamp_lid.dat "$N" \
  -I8 -SV200 -CVvcorr.dat \
  -Oout_psp -l -Linv_psp.log

python3 - <<'PY'
from pathlib import Path
import shutil
import inv_grid as g
rec = g.latest("out_psp.smesh*.*")
shutil.copyfile(rec, "rec_vs.smesh")
g.write_psp_speed(Path("rec_vp.smesh"), Path("rec_vs.smesh"), Path("rec_mixed.smesh"))
print(f"rec_vs.smesh <- {rec.name}")
PY

echo "== single recovered PSP=6 =="
"$BIN/tt_forward" -Mrec_vp.smesh -Urec_vs.smesh -Ggeom_psp6.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_rec.dat > syn_rec.dat

echo "== done single fair two-step =="
