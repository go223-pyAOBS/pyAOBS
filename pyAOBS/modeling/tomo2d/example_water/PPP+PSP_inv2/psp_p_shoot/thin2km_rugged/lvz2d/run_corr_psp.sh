#!/bin/bash
set -euo pipefail
# 在 PPP 收回 Vp 上正演 PPP/PPS 路径，供 corr_psp.py 把 PSS 校成 PSP。
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d"
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1 OMP_NUM_THREADS=8
export PYTHONPATH="$ROOT/inv_2d:${PYTHONPATH:-}"

python3 - <<'PY'
from pathlib import Path
import inv_grid as g
if not Path("rec_vp.smesh").is_file():
    raise SystemExit("missing rec_vp.smesh -- run run_wsl.sh first")
g.write_vs_from_vp(Path("rec_vp.smesh"), Path("vs_ppp.smesh"), g.KAPPA)
print("vs_ppp.smesh <- rec_vp / 1.73")
PY

echo "== true all phases (obs) =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_all.dat \
  -Xconv.refl -Bseafloor.refl "$N" > syn_true_all.dat

echo "== PPP Vp + Vp/k  all phases + rays =="
"$BIN/tt_forward" -Mrec_vp.smesh -Uvs_ppp.smesh -Ggeom_all.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_pppvp.dat > syn_pppvp.dat

echo "== done  plot: python corr_psp.py =="
