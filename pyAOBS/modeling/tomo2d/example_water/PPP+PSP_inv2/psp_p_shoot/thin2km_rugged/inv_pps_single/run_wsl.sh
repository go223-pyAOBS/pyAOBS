#!/bin/bash
set -euo pipefail
# 单场 PPS：观测 -M true_vp -k；初值/反演 -U 热盖层 Vs。
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/inv_pps_single"
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1 TOMO2D_INV_OMP=1 OMP_NUM_THREADS=8
export TOMO2D_INV_LSQR_MAXITER=8000
export PYTHONPATH="$ROOT/inv_2d:${PYTHONPATH:-}"

python3 "$ROOT/make_pps_hot.py" .
python3 "$ROOT/apply_hot_vs.py" . --lid-only

rm -f out_pps.smesh.* out_pps.vp.smesh* inv_pps.log

echo "== single PPS obs: true_vp -k1.73 =="
"$BIN/tt_forward" -Mtrue_vp.smesh -k1.73 -Ggeom_pps7.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_true.dat > syn_inv.dat
cp -f syn_inv.dat syn_true.dat

echo "== single PPS start: rec_vp + lid Vs+0.50 =="
"$BIN/tt_forward" -Mrec_vp.smesh -Ustart_vs.smesh -Ggeom_pps7.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_start.dat > syn_start.dat

echo "== single inverse PPS=7  -M rec_vp -U hot_lid -k1.73 =="
"$BIN/tt_inverse" -Mrec_vp.smesh -Ustart_vs.smesh -k1.73 -Gsyn_inv.dat \
  -Bconv.refl -Yseafloor.refl -w \
  "$N" -I8 -SV200 -TV5 -CVvcorr.dat \
  -Oout_pps -l -Linv_pps.log

python3 - <<'PY'
from pathlib import Path
import shutil
import inv_grid as g
rec = g.latest("out_pps.smesh*.*")
shutil.copyfile(rec, "rec_vs.smesh")
g.write_psp_speed(Path("rec_vp.smesh"), Path("rec_vs.smesh"), Path("rec_mixed.smesh"))
print(f"rec_vs.smesh <- {rec.name}")
PY

echo "== single recovered PPS=7 =="
"$BIN/tt_forward" -Mrec_vp.smesh -Urec_vs.smesh -Ggeom_pps7.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_rec.dat > syn_rec.dat

echo "== done single hot-lid PPS =="
