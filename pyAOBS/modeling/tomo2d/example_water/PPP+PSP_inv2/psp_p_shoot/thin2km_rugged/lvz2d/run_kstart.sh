#!/bin/bash
set -euo pipefail
# 错误初值 k=1.50 / 2.20：PPP rec_vp 上正演 PPS，供 corr_psp_kstart.py 拟合台侧 k。
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d"
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1 OMP_NUM_THREADS=8
export PYTHONPATH="$ROOT/inv_2d:${PYTHONPATH:-}"

if [ ! -f rec_vp.smesh ] || [ ! -f syn_true_all.dat ]; then
  echo "missing rec_vp/syn_true_all -- run run_wsl.sh and run_corr_psp.sh first"
  exit 1
fi

for k in 1.50 2.20; do
  tag=$(python3 -c "print('${k}'.replace('.',''))")
  echo "== Vs = rec_vp / ${k}  tag=${tag} =="
  python3 - <<PY
from pathlib import Path
import inv_grid as g
g.write_vs_from_vp(Path("rec_vp.smesh"), Path("vs_k${tag}.smesh"), ${k})
print("vs_k${tag}.smesh <- rec_vp / ${k}")
PY
  "$BIN/tt_forward" -Mrec_vp.smesh -Uvs_k${tag}.smesh -Ggeom_all.dat \
    -Xconv.refl -Bseafloor.refl "$N" -Rrays_k${tag}.dat > syn_k${tag}.dat
done
echo "== done  plot: python corr_psp_kstart.py =="
