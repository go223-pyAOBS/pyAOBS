#!/bin/bash
set -euo pipefail
# B：冻 PPP 的 rec_vp，geom 6+7+8 只反 Vs。热初值盖层+面下 +0.50。
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/inv_678_hot"
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1 TOMO2D_INV_OMP=1 OMP_NUM_THREADS=8
export TOMO2D_INV_LSQR_MAXITER=8000
export PYTHONPATH="$ROOT/inv_2d:${PYTHONPATH:-}"

python3 "$ROOT/make_joint.py" .
python3 "$ROOT/apply_hot_vs.py" . --lid

rm -f out_678.smesh.* out_678.vp.smesh* inv_678.log

echo "== 678-hot obs: true_vp + true_vs =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_678.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_true.dat > syn_inv.dat
cp -f syn_inv.dat syn_true.dat

echo "== 678-hot start: rec_vp + lid/below Vs+0.50 =="
"$BIN/tt_forward" -Mrec_vp.smesh -Ustart_vs.smesh -Ggeom_678.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_start.dat > syn_start.dat

echo "== inverse Vs-only  6+7+8  freeze rec_vp =="
"$BIN/tt_inverse" -Mrec_vp.smesh -Ustart_vs.smesh -Gsyn_inv.dat \
  -Bconv.refl -Yseafloor.refl -w -k1.73 \
  "$N" -I8 -SV200 -TV5 -CVvcorr.dat \
  -Oout_678 -l -Linv_678.log

python3 - <<'PY'
from pathlib import Path
import shutil
import filecmp
import inv_grid as g
vs = g.latest("out_678.smesh*.*")
shutil.copyfile(vs, "rec_vs.smesh")
g.write_psp_speed(Path("rec_vp.smesh"), Path("rec_vs.smesh"), Path("rec_mixed.smesh"))
print(f"rec_vs.smesh <- {vs.name}")
same = filecmp.cmp("ppp_vp.smesh", "rec_vp.smesh", shallow=False)
print(f"rec_vp == ppp_vp: {same}")
if not same:
    raise SystemExit("rec_vp was overwritten — Vp must stay frozen")
PY

echo "== 678-hot recovered =="
"$BIN/tt_forward" -Mrec_vp.smesh -Urec_vs.smesh -Ggeom_678.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_rec.dat > syn_rec.dat

echo "== PPP check on frozen rec_vp =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Ggeom_ppp.dat -Bseafloor.refl "$N" > syn_ppp.dat
"$BIN/tt_forward" -Mrec_vp.smesh -Ggeom_ppp.dat -Bseafloor.refl "$N" > syn_ppp_rec.dat

echo "== done 678-hot =="
