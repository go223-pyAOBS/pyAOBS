#!/bin/bash
set -euo pipefail
# 真模型多震相射线走时 → wave_fwd/syn_phases.dat。只读 inv_612，不改其中文件。
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d"
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
WORK=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d/inv_612
OUT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d/wave_fwd
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1 OMP_NUM_THREADS=8
mkdir -p "$OUT"
echo "== tt_forward  0/1/6/7/8/12/13  true model  -> wave_fwd =="
cd "$WORK"
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -G"$OUT/geom_phases.dat" \
  -Xconv.refl -Bseafloor.refl -Fmoho_true.refl -A "$N" > "$OUT/syn_phases.dat"
if [ -f "$OUT/geom_obs50.dat" ]; then
  echo "== tt_forward OBS50  -R rays_obs50 =="
  "$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -G"$OUT/geom_obs50.dat" \
    -Xconv.refl -Bseafloor.refl -Fmoho_true.refl "$N" -R"$OUT/rays_obs50.dat" \
    > "$OUT/syn_obs50.dat"
fi
echo "== done syn_phases =="
