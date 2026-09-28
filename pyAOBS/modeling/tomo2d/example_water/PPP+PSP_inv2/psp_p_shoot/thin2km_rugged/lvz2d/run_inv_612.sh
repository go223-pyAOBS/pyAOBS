#!/bin/bash
set -euo pipefail
# Vs 段：冻真 Vp、冻盖层 Vs，6+12 反面下 Vs + 可动莫霍。不改 cmp_0178。
# OBS 30/50/70/90/110；模型 0–150 km；转换面最深 4 km。
# 真：面上Vp=3.5 面下Vs=3.5 壳底7.4。初值：START_1D 整体偏快（无 LVZ）。
# -SV200 -SD20 -DV30 -DD200：加大莫霍阻尼，让 12 的残差多进 Vs。
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d"
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
HERE=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d
WORK="$HERE/inv_612"
N=-N8/8/0.8/8/1e-4/1e-5
NINV=-N8/8/0.8/8/1e-3/1e-4
export TOMO2D_FWD_OMP=1 TOMO2D_INV_OMP=1 OMP_NUM_THREADS=8
export TOMO2D_INV_LSQR_MAXITER=8000
export TOMO2D_INV_FREEZE_LID=1
export PYTHONPATH="$ROOT/inv_2d:${PYTHONPATH:-}"
unset TOMO2D_INV_STRATEGY TOMO2D_INV_FREEZE_BELOW TOMO2D_INV_TTDIFF TOMO2D_INV_PSS_BELOW

python3 "$HERE/make_inv_612.py"
cd "$WORK"
rm -f out.smesh.* out.vp.smesh* out.refl.* inv.log

echo "== obs 6+12  true Vp/Vs + true Moho =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_612.dat \
  -Xconv.refl -Bseafloor.refl -Fmoho_true.refl "$N" -Rrays_true.dat > syn_inv.dat

echo "== start  true Vp + start Vs + start Moho =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Ustart_vs.smesh -Ggeom_612.dat \
  -Xconv.refl -Bseafloor.refl -Fmoho.refl "$N" -Rrays_start.dat > syn_start.dat

echo "== inverse 6+12  freeze true Vp, lid frozen, Moho free =="
"$BIN/tt_inverse" -Mtrue_vp.smesh -Ustart_vs.smesh -Gsyn_inv.dat \
  -Bconv.refl -Yseafloor.refl -Fmoho.refl -w -k1.73 \
  "$NINV" -I8 -SV200 -SD20 -DV30 -DD200 -CVvcorr.dat \
  -Oout -l -Linv.log -V0

python3 - <<'PY'
from pathlib import Path
import shutil
import inv_grid as g
import make_ppp_psp_inv_case as m2
d = Path(".")
vs = g.latest("out.smesh*.*", d)
vp = g.latest("out.vp.smesh*.*", d)
shutil.copyfile(vs, d / "rec_vs.smesh")
shutil.copyfile(vp, d / "rec_vp.smesh")
rf = g.latest("out.refl.*.*", d)
shutil.copyfile(rf, d / "rec_moho.refl")
print(f"rec_vs <- {vs.name}  rec_vp <- {vp.name}  rec_moho <- {rf.name}")
_, _, vt = m2.parse_smesh(d / "true_vp.smesh")
_, _, vr = m2.parse_smesh(d / "rec_vp.smesh")
mx = max(abs(a - b) for col_t, col_r in zip(vt, vr) for a, b in zip(col_t, col_r))
print(f"  rec_vp vs true_vp max|d|={mx:.4f}")
if mx > 0.05:
    raise SystemExit("6+12 Vs-stage overwrote Vp")
PY

echo "== recovered 6+12 =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Urec_vs.smesh -Ggeom_612.dat \
  -Xconv.refl -Bseafloor.refl -Frec_moho.refl "$N" -Rrays_rec.dat > syn_rec.dat

echo "== done inv_612 =="
