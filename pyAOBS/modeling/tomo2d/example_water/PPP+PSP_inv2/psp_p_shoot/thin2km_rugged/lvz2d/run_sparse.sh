#!/bin/bash
set -euo pipefail
# 复用 PPP Vp + path A 盖层 Vs；冻盖层，PSS + 15% 真 PSP 反面下。
# 不重跑 make_lvz / path A / path B。
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d"
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
NINV=-N8/8/0.8/8/1e-3/1e-4
export TOMO2D_FWD_OMP=1 TOMO2D_INV_OMP=1 OMP_NUM_THREADS=8
export TOMO2D_INV_LSQR_MAXITER=8000
export PYTHONPATH="$ROOT/inv_2d:${PYTHONPATH:-}"
unset TOMO2D_INV_FREEZE_BELOW
unset TOMO2D_INV_PSS_BELOW

python3 make_sparse_holdout.py

cd path_s
rm -f out_sparse.smesh.* out_sparse.vp.smesh* inv_sparse.log

echo "== S obs  PSS + sparse PSP  =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_inv.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_true.dat > syn_inv.dat
python3 ../make_sparse_holdout.py --rewrite-syn syn_inv.dat
cp -f syn_inv.dat syn_true.dat

"$BIN/tt_forward" -Mrec_vp.smesh -Urec_vs_lid.smesh -Ggeom_inv.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_start.dat > syn_start.dat

echo "== S inverse  freeze lid  PSS + sparse PSP =="
export TOMO2D_INV_FREEZE_LID=1
"$BIN/tt_inverse" -Mrec_vp.smesh -Urec_vs_lid.smesh -Gsyn_inv.dat \
  -Bconv.refl -Yseafloor.refl -w -k1.73 \
  "$NINV" -I8 -SV200 -TV5 -CVvcorr.dat \
  -Oout_sparse -l -Linv_sparse.log
unset TOMO2D_INV_FREEZE_LID

python3 - <<'PY'
from pathlib import Path
import shutil, filecmp
import inv_grid as g
vs = g.latest("out_sparse.smesh*.*")
shutil.copyfile(vs, "rec_vs.smesh")
g.write_psp_speed(Path("rec_vp.smesh"), Path("rec_vs.smesh"), Path("rec_mixed.smesh"))
print(f"rec_vs.smesh <- {vs.name}")
if not filecmp.cmp("ppp_vp.smesh", "rec_vp.smesh", shallow=False):
    raise SystemExit("S overwrote rec_vp")
PY

echo "== S recovered + holdout all =="
"$BIN/tt_forward" -Mrec_vp.smesh -Urec_vs.smesh -Ggeom_inv.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_rec.dat > syn_rec.dat
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_all.dat \
  -Xconv.refl -Bseafloor.refl "$N" > syn_holdout_true.dat
"$BIN/tt_forward" -Mrec_vp.smesh -Urec_vs.smesh -Ggeom_all.dat \
  -Xconv.refl -Bseafloor.refl "$N" > syn_holdout_rec.dat

cd ..
echo "== done path_s  plot: python check_sparse.py =="
