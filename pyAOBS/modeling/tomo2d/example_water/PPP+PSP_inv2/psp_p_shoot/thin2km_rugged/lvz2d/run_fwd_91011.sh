#!/bin/bash
set -euo pipefail
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d"
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
HERE=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d
WORK="$HERE/fwd_91011"
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1 OMP_NUM_THREADS=8

python3 "$HERE/make_fwd_91011.py"

echo "== fwd 6/7/8 vs lid-SS 10/11 and water peg 9/14/15 =="
"$BIN/tt_forward" -M"$WORK/true_vp.smesh" -U"$WORK/true_vs.smesh" \
  -G"$WORK/geom_91011.dat" -X"$WORK/conv.refl" -B"$WORK/seafloor.refl" \
  -F"$WORK/moho_true.refl" "$N" -R"$WORK/rays_true.dat" > "$WORK/syn_true.dat"

if python3 -c "import matplotlib" >/dev/null 2>&1; then
  python3 "$HERE/check_fwd_91011.py"
else
  echo "skip plot (WSL python has no matplotlib); run check_fwd_91011.py on Windows"
fi
echo "== done fwd_91011 =="
