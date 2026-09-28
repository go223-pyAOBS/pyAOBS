#!/bin/bash
set -euo pipefail
# 自动策略：PPP→Vp；PPS+PPS-PPP→盖层 Vs；正演校正远偏移 PSS→PSP；
# 拾取 PSP 优先；冻盖层只反面下。最终 lvz2d/rec_vp.smesh  rec_vs.smesh
#
# 观测：已有 syn_obs.dat 则用之，否则真模型正演 geom_all。
# STRATEGY_FORCE=1  重跑 PPP/PPS（默认有 rec_vp / rec_vs_lid 则复用）
# STRATEGY_PSP_FRAC=1  保留观测 PSP 比例（现场一般 1；合成可改 0.15）
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d"
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
HERE=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d
WORK="$HERE/strategy"
N=-N8/8/0.8/8/1e-4/1e-5
NINV=-N8/8/0.8/8/1e-3/1e-4
export TOMO2D_FWD_OMP=1 TOMO2D_INV_OMP=1 OMP_NUM_THREADS=8
export TOMO2D_INV_LSQR_MAXITER=8000
export PYTHONPATH="$ROOT/inv_2d:${PYTHONPATH:-}"
unset TOMO2D_INV_FREEZE_BELOW
unset TOMO2D_INV_PSS_BELOW
FORCE="${STRATEGY_FORCE:-0}"

mkdir -p "$WORK"
for f in true_vp.smesh true_vs.smesh start_vp.smesh seafloor.refl conv.refl vcorr.dat \
         geom_all.dat geom_ppp.dat geom_07.dat; do
  if [ ! -f "$HERE/$f" ]; then
    echo "missing $HERE/$f  — 先准备网格/几何（make_lvz.py）"
    exit 1
  fi
  cp -f "$HERE/$f" "$WORK/$f"
done

if [ ! -f "$HERE/syn_obs.dat" ]; then
  echo "== obs: true model -> syn_obs.dat =="
  "$BIN/tt_forward" -M"$HERE/true_vp.smesh" -U"$HERE/true_vs.smesh" -G"$HERE/geom_all.dat" \
    -X"$HERE/conv.refl" -B"$HERE/seafloor.refl" "$N" > "$HERE/syn_obs.dat"
fi
cp -f "$HERE/syn_obs.dat" "$WORK/syn_obs.dat"

need_ppp=0
need_lid=0
if [ "$FORCE" = 1 ] || [ ! -f "$HERE/rec_vp.smesh" ]; then need_ppp=1; fi
if [ "$FORCE" = 1 ] || [ ! -f "$HERE/path_a/rec_vs_lid.smesh" ]; then need_lid=1; fi

cd "$WORK"
if [ "$need_ppp" = 1 ]; then
  echo "== 1. PPP -> Vp =="
  python3 "$HERE/strategy_flow.py" filter syn_obs.dat syn_ppp.dat 0
  rm -f out_ppp.smesh.* inv_ppp.log
  "$BIN/tt_inverse" -Mstart_vp.smesh -Gsyn_ppp.dat -Yseafloor.refl -w \
    "$N" -I5 -SV200 -TV5 -CVvcorr.dat -Oout_ppp -l -Linv_ppp.log
  python3 - <<'PY'
from pathlib import Path
import shutil
import inv_grid as g
rec = g.latest("out_ppp.smesh*.*")
shutil.copyfile(rec, "rec_vp.smesh")
shutil.copyfile(rec, "ppp_vp.smesh")
print(f"rec_vp.smesh <- {rec.name}")
PY
  cp -f rec_vp.smesh "$HERE/rec_vp.smesh"
  cp -f rec_vp.smesh "$HERE/ppp_vp.smesh"
else
  echo "== 1. reuse rec_vp.smesh =="
  cp -f "$HERE/rec_vp.smesh" rec_vp.smesh
  cp -f "$HERE/ppp_vp.smesh" ppp_vp.smesh 2>/dev/null || cp -f rec_vp.smesh ppp_vp.smesh
fi

if [ "$need_lid" = 1 ]; then
  echo "== 2. PPS + PPS-PPP -> lid Vs =="
  python3 "$ROOT/apply_hot_vs.py" . --lid
  python3 "$HERE/strategy_flow.py" filter syn_obs.dat syn_07.dat 0,7
  rm -f out_lid.smesh.* out_lid.vp.smesh* inv_lid.log
  "$BIN/tt_inverse" -Mrec_vp.smesh -Ustart_vs.smesh -Gsyn_07.dat \
    -Bconv.refl -Yseafloor.refl -w -k1.73 -td \
    "$NINV" -I8 -SV200 -TV5 -CVvcorr.dat -Oout_lid -l -Linv_lid.log
  python3 - <<'PY'
from pathlib import Path
import shutil, filecmp
import inv_grid as g
vs = g.latest("out_lid.smesh*.*")
shutil.copyfile(vs, "rec_vs_lid.smesh")
print(f"rec_vs_lid.smesh <- {vs.name}")
if not filecmp.cmp("ppp_vp.smesh", "rec_vp.smesh", shallow=False):
    raise SystemExit("lid step overwrote rec_vp")
PY
  mkdir -p "$HERE/path_a"
  cp -f rec_vs_lid.smesh "$HERE/path_a/rec_vs_lid.smesh"
else
  echo "== 2. reuse path_a/rec_vs_lid.smesh =="
  cp -f "$HERE/path_a/rec_vs_lid.smesh" rec_vs_lid.smesh
fi

echo "== 3. fwd PSS/PSP on rec_vp + rec_vs_lid =="
"$BIN/tt_forward" -Mrec_vp.smesh -Urec_vs_lid.smesh -Gsyn_obs.dat \
  -Xconv.refl -Bseafloor.refl "$N" > syn_lidfwd.dat

echo "== 4. build below PSP (pick wins, corr if far) =="
python3 "$HERE/strategy_flow.py" below syn_obs.dat syn_lidfwd.dat syn_below.dat strategy_below.txt

echo "== 5. freeze lid, invert below Vs =="
rm -f out_below.smesh.* out_below.vp.smesh* inv_below.log
export TOMO2D_INV_FREEZE_LID=1
"$BIN/tt_inverse" -Mrec_vp.smesh -Urec_vs_lid.smesh -Gsyn_below.dat \
  -Bconv.refl -Yseafloor.refl -w -k1.73 \
  "$NINV" -I8 -SV200 -TV5 -CVvcorr.dat -Oout_below -l -Linv_below.log
unset TOMO2D_INV_FREEZE_LID

python3 - <<'PY'
from pathlib import Path
import shutil, filecmp
import inv_grid as g
vs = g.latest("out_below.smesh*.*")
shutil.copyfile(vs, "rec_vs.smesh")
g.write_psp_speed(Path("rec_vp.smesh"), Path("rec_vs.smesh"), Path("rec_mixed.smesh"))
print(f"rec_vs.smesh <- {vs.name}")
if not filecmp.cmp("ppp_vp.smesh", "rec_vp.smesh", shallow=False):
    raise SystemExit("below step overwrote rec_vp")
PY

cp -f rec_vp.smesh "$HERE/rec_vp.smesh"
cp -f rec_vs.smesh "$HERE/rec_vs.smesh"
cp -f rec_mixed.smesh "$HERE/rec_mixed.smesh"
cp -f rec_vs_lid.smesh "$HERE/rec_vs_lid.smesh"
cp -f strategy_below.txt "$HERE/strategy_below.txt"

echo "== done  Vp=$HERE/rec_vp.smesh  Vs=$HERE/rec_vs.smesh =="
cat strategy_below.txt
