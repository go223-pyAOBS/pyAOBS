#!/bin/bash
set -euo pipefail
# 加大深度更新 + 收紧面下平滑：不改 inv_612 / inv_dd50。
# -SV400 -SD10 -DV30 -DD20；SENS / LINESEARCH / C2F / LM 关。
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d"
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
HERE=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
N=-N8/8/0.8/8/1e-4/1e-5
NINV=-N8/8/0.8/8/1e-3/1e-4
export TOMO2D_FWD_OMP=1 TOMO2D_INV_OMP=1 OMP_NUM_THREADS=8
export TOMO2D_INV_LSQR_MAXITER=8000
export PYTHONPATH="$HERE:$ROOT:$ROOT/inv_2d:$ROOT/../..:${PYTHONPATH:-}"
unset TOMO2D_INV_STRATEGY TOMO2D_INV_TTDIFF TOMO2D_INV_PSS_BELOW
unset TOMO2D_INV_LSQR_PRECOND
unset TOMO2D_INV_SENS_WEIGHT TOMO2D_INV_LINESEARCH TOMO2D_INV_LM TOMO2D_INV_COARSE2FINE
unset TOMO2D_INV_FREEZE_BELOW
export TOMO2D_INV_FREEZE_LID=1

SRC="$HERE/inv_612"
DST="$HERE/inv_dd20_sv"
rm -rf "$DST"
mkdir -p "$DST"
cp -f "$SRC"/true_vp.smesh "$SRC"/true_vs.smesh "$SRC"/start_vs.smesh \
  "$SRC"/syn_inv.dat "$SRC"/geom_612.dat "$SRC"/vcorr.dat \
  "$SRC"/conv.refl "$SRC"/seafloor.refl "$SRC"/moho.refl "$SRC"/moho_true.refl \
  "$DST/"
cd "$DST"
echo "== inverse 612  -SV400 -SD10 -DD20  freeze lid, Moho free =="
"$BIN/tt_inverse" -Mtrue_vp.smesh -Ustart_vs.smesh -Gsyn_inv.dat \
  -Bconv.refl -Yseafloor.refl -Fmoho.refl -w -k1.73 \
  "$NINV" -I8 -SV400 -SD10 -DV30 -DD20 -CVvcorr.dat \
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
    raise SystemExit("dd20_sv overwrote Vp")
PY

echo "== recovered 612 -SV400 -DD20 =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Urec_vs.smesh -Ggeom_612.dat \
  -Xconv.refl -Bseafloor.refl -Frec_moho.refl "$N" -Rrays_rec.dat > syn_rec.dat
echo "== done inv_dd20_sv =="
