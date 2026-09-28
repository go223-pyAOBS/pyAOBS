#!/bin/bash
set -euo pipefail
# 公平对比：单场 -M start_vp -k1.73（无 -U），同一 syn_obs（0+6+7+8），
# 同一 -B -Y -w -N -I8 -SV -TV -CV。
#   joint : TOMO2D_INV_STRATEGY=0   前 min(8,I) 收 Vp，再 I 次 6/7/8 收 Vs
#   strat : 新策略（默认）          PPP / 盖层 Vs / 校正 / 面下 Vs
# -I 相同；策略总迭代更多（8+8+1+8），这是算法本身，不另加热 Vs、不换数据。
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d"
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
HERE=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d
WORK="$HERE/cmp_sj"
N=-N8/8/0.8/8/1e-4/1e-5
NINV=-N8/8/0.8/8/1e-3/1e-4
export TOMO2D_FWD_OMP=1 TOMO2D_INV_OMP=1 OMP_NUM_THREADS=8
export TOMO2D_INV_LSQR_MAXITER=8000
export PYTHONPATH="$ROOT/inv_2d:${PYTHONPATH:-}"
unset TOMO2D_INV_FREEZE_BELOW TOMO2D_INV_FREEZE_LID TOMO2D_INV_PSS_BELOW TOMO2D_INV_TTDIFF

mkdir -p "$WORK/joint" "$WORK/strat"
for f in true_vp.smesh true_vs.smesh start_vp.smesh seafloor.refl conv.refl vcorr.dat \
         geom_all.dat syn_obs.dat; do
  if [ ! -f "$HERE/$f" ]; then
    echo "missing $HERE/$f"
    exit 1
  fi
  cp -f "$HERE/$f" "$WORK/$f"
  cp -f "$HERE/$f" "$WORK/joint/$f"
  cp -f "$HERE/$f" "$WORK/strat/$f"
done

python3 - <<'PY'
from pathlib import Path
import inv_grid as g
here = Path("cmp_sj")
g.write_vs_from_vp(here / "start_vp.smesh", here / "start_vs.smesh", g.KAPPA)
for sub in ("joint", "strat"):
    g.write_vs_from_vp(here / sub / "start_vp.smesh", here / sub / "start_vs.smesh", g.KAPPA)
print("start_vs = start_vp/1.73  (forward only; inversion is single-field -k)")
PY

collect() {
  local d="$1"
  python3 - <<PY
from pathlib import Path
import shutil
import inv_grid as g
d = Path("$d")
vs = g.latest("out.smesh*.*", d)
vp = g.latest("out.vp.smesh.*.*", d)
shutil.copyfile(vs, d / "rec_vs.smesh")
shutil.copyfile(vp, d / "rec_vp.smesh")
print(f"{d.name}: rec_vs <- {vs.name}  rec_vp <- {vp.name}")
PY
}

fwd_pack() {
  local d="$1"
  "$BIN/tt_forward" -M"$d/start_vp.smesh" -U"$d/start_vs.smesh" -G"$WORK/syn_obs.dat" \
    -X"$d/conv.refl" -B"$d/seafloor.refl" "$N" > "$d/syn_start.dat"
  "$BIN/tt_forward" -M"$d/rec_vp.smesh" -U"$d/rec_vs.smesh" -G"$WORK/syn_obs.dat" \
    -X"$d/conv.refl" -B"$d/seafloor.refl" "$N" > "$d/syn_rec.dat"
  cp -f "$WORK/syn_obs.dat" "$d/syn_inv.dat"
}

echo "== joint  STRATEGY=0  single-field -k =="
export TOMO2D_INV_STRATEGY=0
cd "$WORK/joint"
rm -f out.smesh.* out.vp.smesh* inv.log
"$BIN/tt_inverse" -Mstart_vp.smesh -G"$WORK/syn_obs.dat" \
  -Bconv.refl -Yseafloor.refl -w -k1.73 \
  "$NINV" -I8 -SV200 -TV5 -CVvcorr.dat -Oout -l -Linv.log
unset TOMO2D_INV_STRATEGY
collect "$WORK/joint"
fwd_pack "$WORK/joint"

echo "== strat  new strategy  single-field -k =="
cd "$WORK/strat"
rm -f out.smesh.* out.vp.smesh* inv.log
"$BIN/tt_inverse" -Mstart_vp.smesh -G"$WORK/syn_obs.dat" \
  -Bconv.refl -Yseafloor.refl -w -k1.73 \
  "$NINV" -I8 -SV200 -TV5 -CVvcorr.dat -Oout -l -Linv.log
collect "$WORK/strat"
fwd_pack "$WORK/strat"

echo "== done  $WORK/joint  $WORK/strat =="
grep -E "strategy|joint staged|joint iters" "$WORK/joint/inv.log" "$WORK/strat/inv.log" || true
