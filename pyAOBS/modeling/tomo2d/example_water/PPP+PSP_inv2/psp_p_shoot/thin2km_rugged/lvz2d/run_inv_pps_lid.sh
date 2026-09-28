#!/bin/bash
set -euo pipefail
# 盖层 Vs：冻真 Vp、冻面下。对照 PPS(7)、PPS 盖层 SS 多次(10)、7+10。
# 不改 cmp_0178 / ../PPP+PSP_inv / 0/1 图。
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d"
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
HERE=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d
WORK="$HERE/inv_pps_lid"
N=-N8/8/0.8/8/1e-4/1e-5
NINV=-N8/8/0.8/8/1e-3/1e-4
export TOMO2D_FWD_OMP=1 TOMO2D_INV_OMP=1 OMP_NUM_THREADS=8
export TOMO2D_INV_LSQR_MAXITER=8000
export TOMO2D_INV_FREEZE_BELOW=1
export PYTHONPATH="$ROOT/inv_2d:${PYTHONPATH:-}"
unset TOMO2D_INV_STRATEGY TOMO2D_INV_FREEZE_LID TOMO2D_INV_TTDIFF TOMO2D_INV_PSS_BELOW

python3 "$HERE/make_inv_pps_lid.py"
cd "$WORK"

run_one() {
  local tag="$1" geom="$2"
  echo "== obs ${tag}  true Vp/Vs =="
  "$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -G"$geom" \
    -Xconv.refl -Bseafloor.refl "$N" -R"rays_true_${tag}.dat" > "syn_inv_${tag}.dat"

  echo "== start ${tag}  true Vp + START_1D lid Vs =="
  "$BIN/tt_forward" -Mtrue_vp.smesh -Ustart_vs.smesh -G"$geom" \
    -Xconv.refl -Bseafloor.refl "$N" -R"rays_start_${tag}.dat" > "syn_start_${tag}.dat"

  echo "== inverse ${tag}  freeze true Vp, freeze below, invert lid Vs =="
  rm -f "out_${tag}.smesh."* "out_${tag}.vp.smesh"* "inv_${tag}.log"
  "$BIN/tt_inverse" -Mtrue_vp.smesh -Ustart_vs.smesh -G"syn_inv_${tag}.dat" \
    -Bconv.refl -Yseafloor.refl -w -k1.73 \
    "$NINV" -I8 -SV200 -TV5 -CVvcorr.dat \
    -O"out_${tag}" -l -L"inv_${tag}.log" -V0

  python3 - "$tag" <<'PY'
import sys
from pathlib import Path
import shutil
import inv_grid as g
import make_ppp_psp_inv_case as m2
tag = sys.argv[1]
d = Path(".")
vs = g.latest(f"out_{tag}.smesh*.*", d)
vp = g.latest(f"out_{tag}.vp.smesh*.*", d)
shutil.copyfile(vs, d / f"rec_vs_{tag}.smesh")
shutil.copyfile(vp, d / f"rec_vp_{tag}.smesh")
print(f"rec_vs_{tag} <- {vs.name}  rec_vp_{tag} <- {vp.name}")
_, _, vt = m2.parse_smesh(d / "true_vp.smesh")
_, _, vr = m2.parse_smesh(d / f"rec_vp_{tag}.smesh")
mx = max(abs(a - b) for col_t, col_r in zip(vt, vr) for a, b in zip(col_t, col_r))
print(f"  rec_vp vs true_vp max|d|={mx:.4f}")
if mx > 0.05:
    raise SystemExit(f"{tag} Vs-stage overwrote Vp")
PY

  echo "== recovered ${tag} =="
  "$BIN/tt_forward" -Mtrue_vp.smesh -U"rec_vs_${tag}.smesh" -G"$geom" \
    -Xconv.refl -Bseafloor.refl "$N" -R"rays_rec_${tag}.dat" > "syn_rec_${tag}.dat"
}

run_one 7 geom_7.dat
run_one 10 geom_10.dat
run_one 710 geom_710.dat
echo "== done inv_pps_lid =="
