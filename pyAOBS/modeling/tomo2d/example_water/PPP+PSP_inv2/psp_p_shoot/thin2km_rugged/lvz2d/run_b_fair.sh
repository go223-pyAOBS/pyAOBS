#!/bin/bash
set -euo pipefail
# 公平 B：与 A/C 同一套 PPP Vp + PPS 盖层 Vs，再真 PSP 只反面下。
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d"
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
NINV=-N8/8/0.8/8/1e-3/1e-4
export TOMO2D_FWD_OMP=1 TOMO2D_INV_OMP=1 OMP_NUM_THREADS=8
export TOMO2D_INV_LSQR_MAXITER=8000
export PYTHONPATH="$ROOT/inv_2d:${PYTHONPATH:-}"
unset TOMO2D_INV_FREEZE_BELOW

python3 - <<'PY'
from pathlib import Path
import shutil
from make_lvz import seed_child
here = Path(".").resolve()
dest = here / "path_bf"
lid = here / "path_a" / "rec_vs_lid.smesh"
if not lid.is_file():
    raise SystemExit("missing path_a/rec_vs_lid.smesh")
seed_child(dest)
shutil.copyfile(lid, dest / "rec_vs_lid.smesh")
print(f"path_bf seeded  rec_vs_lid <- {lid}")
PY

cd path_bf
rm -f out_psp.smesh.* out_psp.vp.smesh* inv_psp.log

echo "== B-fair obs true PSP =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_psp6.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_true.dat > syn_inv.dat
cp -f syn_inv.dat syn_true.dat
"$BIN/tt_forward" -Mrec_vp.smesh -Urec_vs_lid.smesh -Ggeom_psp6.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_start.dat > syn_start.dat

echo "== B-fair inverse  freeze lid  true PSP =="
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
    raise SystemExit("B-fair overwrote rec_vp")
PY

echo "== B-fair recovered + holdout =="
"$BIN/tt_forward" -Mrec_vp.smesh -Urec_vs.smesh -Ggeom_psp6.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_rec.dat > syn_rec.dat
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_all.dat \
  -Xconv.refl -Bseafloor.refl "$N" > syn_holdout_true.dat
"$BIN/tt_forward" -Mrec_vp.smesh -Urec_vs.smesh -Ggeom_all.dat \
  -Xconv.refl -Bseafloor.refl "$N" > syn_holdout_rec.dat

cd ..
echo "== done path_bf  plot: python check_corr_vs.py =="
