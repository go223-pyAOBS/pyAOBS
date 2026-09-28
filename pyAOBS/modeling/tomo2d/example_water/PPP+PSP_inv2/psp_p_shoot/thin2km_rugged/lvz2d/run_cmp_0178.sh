#!/bin/bash
set -euo pipefail
# 公平：单场 -k，0+1+7+8。传统 PPP+PmP：-F 可动 Moho。
# -SD20 与速度平滑同量级；-TD1 限制平均深度步长（上次 -TD5/-TD2 未触发，左侧被拉浅）。
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d"
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
HERE=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d
WORK="$HERE/cmp_0178"
N=-N8/8/0.8/8/1e-4/1e-5
NINV=-N8/8/0.8/8/1e-3/1e-4
export TOMO2D_FWD_OMP=1 TOMO2D_INV_OMP=1 OMP_NUM_THREADS=8
export TOMO2D_INV_LSQR_MAXITER=8000
export PYTHONPATH="$ROOT/inv_2d:${PYTHONPATH:-}"
unset TOMO2D_INV_FREEZE_BELOW TOMO2D_INV_FREEZE_LID TOMO2D_INV_PSS_BELOW TOMO2D_INV_TTDIFF

python3 "$HERE/make_pmp.py"

echo "== obs: true Vp/Vs + true Moho -> syn (0+1+6+7+8) =="
"$BIN/tt_forward" -M"$WORK/true_vp.smesh" -U"$WORK/true_vs.smesh" \
  -G"$WORK/geom_holdout.dat" -X"$WORK/conv.refl" -B"$WORK/seafloor.refl" \
  -F"$WORK/moho_true.refl" "$N" -R"$WORK/rays_true.dat" > "$WORK/syn_obs.dat"
python3 "$HERE/strategy_flow.py" filter "$WORK/syn_obs.dat" "$WORK/syn_inv.dat" 0,1,7,8

collect_fwd() {
  local d="$1"
  python3 "$HERE/cmp_sj/_collect.py" "$d"
  local frec="$d/rec_moho.refl"
  if [[ ! -f "$frec" ]]; then
    echo "missing recovered Moho $frec" >&2
    exit 1
  fi
  "$BIN/tt_forward" -M"$d/start_vp.smesh" -U"$d/start_vs.smesh" -G"$WORK/syn_inv.dat" \
    -X"$d/conv.refl" -B"$d/seafloor.refl" -F"$d/moho.refl" "$N" \
    -R"$d/rays_start.dat" > "$d/syn_start.dat"
  "$BIN/tt_forward" -M"$d/rec_vp.smesh" -U"$d/rec_vs.smesh" -G"$WORK/syn_inv.dat" \
    -X"$d/conv.refl" -B"$d/seafloor.refl" -F"$frec" "$N" \
    -R"$d/rays_rec.dat" > "$d/syn_rec.dat"
  cp -f "$WORK/syn_inv.dat" "$d/syn_inv.dat"
  cp -f "$WORK/syn_obs.dat" "$d/syn_holdout_true.dat"
  "$BIN/tt_forward" -M"$d/rec_vp.smesh" -U"$d/rec_vs.smesh" -G"$WORK/syn_obs.dat" \
    -X"$d/conv.refl" -B"$d/seafloor.refl" -F"$frec" "$N" \
    -R"$d/rays_holdout.dat" > "$d/syn_holdout_rec.dat"
}

echo "== joint  0+1+7+8  STRATEGY=0  movable Moho =="
export TOMO2D_INV_STRATEGY=0
cd "$WORK/joint"
rm -f out.smesh.* out.vp.smesh* out.refl.* inv.log
"$BIN/tt_inverse" -Mstart_vp.smesh -G"$WORK/syn_inv.dat" \
  -Fmoho.refl -Bconv.refl -Yseafloor.refl -w -k1.73 \
  "$NINV" -I8 -SV200 -SD20 -TV5 -TD1 -CVvcorr.dat -Oout -l -Linv.log
unset TOMO2D_INV_STRATEGY
collect_fwd "$WORK/joint"

echo "== strat  0+1+7+8  new strategy  movable Moho =="
cd "$WORK/strat"
rm -f out.smesh.* out.vp.smesh* out.refl.* inv.log
"$BIN/tt_inverse" -Mstart_vp.smesh -G"$WORK/syn_inv.dat" \
  -Fmoho.refl -Bconv.refl -Yseafloor.refl -w -k1.73 \
  "$NINV" -I8 -SV200 -SD20 -TV5 -TD1 -CVvcorr.dat -Oout -l -Linv.log
collect_fwd "$WORK/strat"

echo "== done cmp_0178 =="
