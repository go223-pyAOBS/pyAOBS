#!/bin/bash
set -euo pipefail
# 公平：单场 -k，只用 0/1/7/8（本工区 syn_obs 无 type 1，实为 0+7+8，无 PSP 拾取）
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d"
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
HERE=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d
WORK="$HERE/cmp_078"
N=-N8/8/0.8/8/1e-4/1e-5
NINV=-N8/8/0.8/8/1e-3/1e-4
export TOMO2D_FWD_OMP=1 TOMO2D_INV_OMP=1 OMP_NUM_THREADS=8
export TOMO2D_INV_LSQR_MAXITER=8000
export PYTHONPATH="$ROOT/inv_2d:${PYTHONPATH:-}"
unset TOMO2D_INV_FREEZE_BELOW TOMO2D_INV_FREEZE_LID TOMO2D_INV_PSS_BELOW TOMO2D_INV_TTDIFF

mkdir -p "$WORK/joint" "$WORK/strat"
for f in true_vp.smesh true_vs.smesh start_vp.smesh seafloor.refl conv.refl vcorr.dat syn_obs.dat; do
  cp -f "$HERE/$f" "$WORK/$f"
  cp -f "$HERE/$f" "$WORK/joint/$f"
  cp -f "$HERE/$f" "$WORK/strat/$f"
done

python3 "$HERE/strategy_flow.py" filter "$WORK/syn_obs.dat" "$WORK/syn_inv.dat" 0,1,7,8
python3 - <<'PY'
from pathlib import Path
import inv_grid as g
here = Path("cmp_078")
g.write_vs_from_vp(here / "start_vp.smesh", here / "start_vs.smesh", g.KAPPA)
for sub in ("joint", "strat"):
    g.write_vs_from_vp(here / sub / "start_vp.smesh", here / sub / "start_vs.smesh", g.KAPPA)
print("filtered 0,1,7,8  (no type 1 in this syn_obs)")
PY

collect_fwd() {
  local d="$1"
  python3 "$HERE/cmp_sj/_collect.py" "$d"
  "$BIN/tt_forward" -M"$d/start_vp.smesh" -U"$d/start_vs.smesh" -G"$WORK/syn_inv.dat" \
    -X"$d/conv.refl" -B"$d/seafloor.refl" "$N" > "$d/syn_start.dat"
  "$BIN/tt_forward" -M"$d/rec_vp.smesh" -U"$d/rec_vs.smesh" -G"$WORK/syn_inv.dat" \
    -X"$d/conv.refl" -B"$d/seafloor.refl" "$N" > "$d/syn_rec.dat"
  cp -f "$WORK/syn_inv.dat" "$d/syn_inv.dat"
  cp -f "$WORK/syn_obs.dat" "$d/syn_holdout_true.dat"
  "$BIN/tt_forward" -M"$d/rec_vp.smesh" -U"$d/rec_vs.smesh" -G"$WORK/syn_obs.dat" \
    -X"$d/conv.refl" -B"$d/seafloor.refl" "$N" > "$d/syn_holdout_rec.dat"
}

echo "== joint  0+7+8  STRATEGY=0 =="
export TOMO2D_INV_STRATEGY=0
cd "$WORK/joint"
rm -f out.smesh.* out.vp.smesh* inv.log
"$BIN/tt_inverse" -Mstart_vp.smesh -G"$WORK/syn_inv.dat" \
  -Bconv.refl -Yseafloor.refl -w -k1.73 \
  "$NINV" -I8 -SV200 -TV5 -CVvcorr.dat -Oout -l -Linv.log
unset TOMO2D_INV_STRATEGY
collect_fwd "$WORK/joint"

echo "== strat  0+7+8  new strategy =="
cd "$WORK/strat"
rm -f out.smesh.* out.vp.smesh* inv.log
"$BIN/tt_inverse" -Mstart_vp.smesh -G"$WORK/syn_inv.dat" \
  -Bconv.refl -Yseafloor.refl -w -k1.73 \
  "$NINV" -I8 -SV200 -TV5 -CVvcorr.dat -Oout -l -Linv.log
collect_fwd "$WORK/strat"

echo "== done cmp_078 =="
