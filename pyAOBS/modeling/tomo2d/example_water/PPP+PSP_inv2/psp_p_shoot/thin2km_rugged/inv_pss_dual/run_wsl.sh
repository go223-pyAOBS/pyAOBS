#!/bin/bash
set -euo pipefail
# 双场 PSS：冻 rec_vp；盖层+面下 Vs 初值 = rec_vp/κ+0.50。观测=真双场。
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/inv_pss_dual"
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1 TOMO2D_INV_OMP=1 OMP_NUM_THREADS=8
export TOMO2D_INV_LSQR_MAXITER=8000
export PYTHONPATH="$ROOT/inv_2d:${PYTHONPATH:-}"

python3 "$ROOT/make_pss_hot.py" .
python3 "$ROOT/apply_hot_vs.py" . --lid

rm -f out_pss.smesh.* out_pss.vp.smesh* inv_pss.log

echo "== dual PSS obs: true_vp + true_vs =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_pss8.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_true.dat > syn_inv.dat
cp -f syn_inv.dat syn_true.dat

echo "== dual PSS start: rec_vp + rec_vp/k+0.50 lid+below =="
"$BIN/tt_forward" -Mrec_vp.smesh -Ustart_vs.smesh -Ggeom_pss8.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_start.dat > syn_start.dat

echo "== dual inverse PSS=8  -k1.73  hot Vs =="
"$BIN/tt_inverse" -Mrec_vp.smesh -Ustart_vs.smesh -Gsyn_inv.dat \
  -Bconv.refl -Yseafloor.refl -w -k1.73 \
  "$N" -I8 -SV200 -TV5 -CVvcorr.dat \
  -Oout_pss -l -Linv_pss.log

python3 - <<'PY'
from pathlib import Path
import shutil
import inv_grid as g
rec = g.latest("out_pss.smesh*.*")
shutil.copyfile(rec, "rec_vs.smesh")
g.write_psp_speed(Path("rec_vp.smesh"), Path("rec_vs.smesh"), Path("rec_mixed.smesh"))
print(f"rec_vs.smesh <- {rec.name}")
PY

echo "== dual recovered PSS=8 =="
"$BIN/tt_forward" -Mrec_vp.smesh -Urec_vs.smesh -Ggeom_pss8.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_rec.dat > syn_rec.dat

echo "== done dual hot-Vs PSS =="
