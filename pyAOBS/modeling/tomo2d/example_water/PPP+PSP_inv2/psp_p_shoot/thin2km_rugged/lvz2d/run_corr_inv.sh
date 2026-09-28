#!/bin/bash
set -euo pipefail
# 冻 PPP Vp + PPS 盖层 Vs；面下分别用 校正PSP(全体) / 校正PSP(>20km)
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

python3 make_corr_psp_obs.py

run_one() {
  local d="$1"
  cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d/$d"
  rm -f out_corr.smesh.* out_corr.vp.smesh* inv_corr.log
  echo "== $d  start fwd =="
  "$BIN/tt_forward" -Mrec_vp.smesh -Urec_vs_lid.smesh -Gsyn_inv.dat \
    -Xconv.refl -Bseafloor.refl "$N" -Rrays_start.dat > syn_start.dat
  echo "== $d  inverse freeze lid  corr PSP =="
  export TOMO2D_INV_FREEZE_LID=1
  "$BIN/tt_inverse" -Mrec_vp.smesh -Urec_vs_lid.smesh -Gsyn_inv.dat \
    -Bconv.refl -Yseafloor.refl -w -k1.73 \
    "$NINV" -I8 -SV200 -TV5 -CVvcorr.dat \
    -Oout_corr -l -Linv_corr.log
  unset TOMO2D_INV_FREEZE_LID
  python3 - <<'PY'
from pathlib import Path
import shutil, filecmp
import inv_grid as g
vs = g.latest("out_corr.smesh*.*")
shutil.copyfile(vs, "rec_vs.smesh")
g.write_psp_speed(Path("rec_vp.smesh"), Path("rec_vs.smesh"), Path("rec_mixed.smesh"))
print(f"rec_vs.smesh <- {vs.name}")
if not filecmp.cmp("ppp_vp.smesh", "rec_vp.smesh", shallow=False):
    raise SystemExit("corr inv overwrote rec_vp")
PY
  echo "== $d  recovered + holdout =="
  "$BIN/tt_forward" -Mrec_vp.smesh -Urec_vs.smesh -Gsyn_inv.dat \
    -Xconv.refl -Bseafloor.refl "$N" -Rrays_rec.dat > syn_rec.dat
  "$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_all.dat \
    -Xconv.refl -Bseafloor.refl "$N" > syn_holdout_true.dat
  "$BIN/tt_forward" -Mrec_vp.smesh -Urec_vs.smesh -Ggeom_all.dat \
    -Xconv.refl -Bseafloor.refl "$N" > syn_holdout_rec.dat
}

run_one path_c
run_one path_cf
cd /mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d
echo "== done  plot: python check_corr_vs.py =="
