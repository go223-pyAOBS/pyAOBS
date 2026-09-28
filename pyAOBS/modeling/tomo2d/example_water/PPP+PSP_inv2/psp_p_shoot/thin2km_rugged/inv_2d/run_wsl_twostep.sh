#!/bin/bash
set -euo pipefail
# 观测 = 二维积分转折枝；反演 = 图论初至两步（PPP Vp → 折合 0）。不改 graph.cc。
# syn_*.dat 由 Windows: python make_rugged_2d_inv.py 生成（WSL 无 matplotlib）。
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/inv_2d"
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1
export TOMO2D_INV_OMP=1
export OMP_NUM_THREADS=8
export TOMO2D_INV_LSQR_MAXITER=8000

if [ ! -f syn_inv.dat ] || [ ! -f syn_ppp.dat ]; then
  echo "missing syn_*.dat — run: python make_rugged_2d_inv.py"
  exit 1
fi
rm -f out_ppp.smesh.* out_psp.smesh.* inv_ppp.log inv_psp.log

echo "== 1. start PPP forward =="
"$BIN/tt_forward" -Mstart_vp.smesh -Ggeom_ppp.dat -Bseafloor.refl \
  "$N" > syn_ppp_start.dat
echo "== 1. inverse PPP =="
"$BIN/tt_inverse" -Mstart_vp.smesh -Gsyn_ppp.dat -Yseafloor.refl -w \
  "$N" -I5 -SV200 -TV5 -CVvcorr.dat -Oout_ppp -l -Linv_ppp.log

python3 - <<'PY'
from pathlib import Path
import shutil
import inv_grid as g

rec = g.latest("out_ppp.smesh*.*")
shutil.copyfile(rec, "rec_vp.smesh")
g.write_mixed(Path("rec_vp.smesh"), Path("rec_vp.smesh"), Path("start_mixed.smesh"), g.KAPPA)
print(f"rec_vp.smesh <- {rec.name}")
print("start_mixed.smesh <- 盖层 rec_vp，面下 rec_vp/1.73")
PY

echo "== 1. recovered PPP forward =="
"$BIN/tt_forward" -Mrec_vp.smesh -Ggeom_ppp.dat -Bseafloor.refl \
  "$N" -Rrays_ppp_rec.dat > syn_ppp_rec.dat

echo "== 2. start type0 forward =="
"$BIN/tt_forward" -Mstart_mixed.smesh -Ggeom_psp0.dat -Bseafloor.refl \
  "$N" > syn_start.dat
echo "== 2. inverse PSP-as-0  -DQ 盖层钉死 =="
"$BIN/tt_inverse" -Mstart_mixed.smesh -Gsyn_inv.dat \
  -Yseafloor.refl -w \
  -DV1 -DQdamp_lid.dat "$N" \
  -I8 -SV200 -CVvcorr.dat \
  -Oout_psp -l -Linv_psp.log

python3 - <<'PY'
import shutil
import inv_grid as g
rec = g.latest("out_psp.smesh*.*")
shutil.copyfile(rec, "rec_vs.smesh")
print(f"rec_vs.smesh <- {rec.name}")
PY

echo "== 2. recovered type0 forward =="
"$BIN/tt_forward" -Mrec_vs.smesh -Ggeom_psp0.dat -Bseafloor.refl \
  "$N" -Rrays_rec.dat > syn_rec.dat

echo "== done rugged 2d-obs two-step (plot: python plot_rugged_inv.py) =="
