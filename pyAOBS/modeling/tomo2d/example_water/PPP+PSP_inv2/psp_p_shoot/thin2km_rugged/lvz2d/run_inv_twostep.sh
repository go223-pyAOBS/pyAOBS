#!/bin/bash
set -euo pipefail
# 两步：不改 inv_612 / inv_dd20。
# 步1 = 已有 inv_dd20（-SV200 -DD20）。
# 步2 = 冻步1 莫霍（-u），-U 用步1 rec_vs，-SV80 只反面下 Vs。
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

S1="$HERE/inv_dd20"
DST="$HERE/inv_twostep"
if [ ! -f "$S1/rec_vs.smesh" ] || [ ! -f "$S1/rec_moho.refl" ]; then
  echo "missing step1 $S1/rec_vs.smesh or rec_moho.refl" >&2
  exit 1
fi
rm -rf "$DST"
mkdir -p "$DST"
cp -f "$S1"/true_vp.smesh "$S1"/true_vs.smesh \
  "$S1"/syn_inv.dat "$S1"/geom_612.dat "$S1"/vcorr.dat \
  "$S1"/conv.refl "$S1"/seafloor.refl "$S1"/moho_true.refl \
  "$DST/"
cp -f "$S1/rec_vs.smesh" "$DST/start_vs.smesh"
cp -f "$S1/rec_moho.refl" "$DST/moho.refl"
cd "$DST"
echo "== inverse 612 step2  freeze Moho -u  -U=step1 rec  -SV80 =="
"$BIN/tt_inverse" -Mtrue_vp.smesh -Ustart_vs.smesh -Gsyn_inv.dat \
  -Bconv.refl -Yseafloor.refl -Fmoho.refl -w -u -k1.73 \
  "$NINV" -I8 -SV80 -DV30 -CVvcorr.dat \
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
if list(d.glob("out.refl.*.*")):
    rf = g.latest("out.refl.*.*", d)
    shutil.copyfile(rf, d / "rec_moho.refl")
    print(f"rec_vs <- {vs.name}  rec_vp <- {vp.name}  rec_moho <- {rf.name}")
else:
    shutil.copyfile(d / "moho.refl", d / "rec_moho.refl")
    print(f"rec_vs <- {vs.name}  rec_vp <- {vp.name}  rec_moho <- frozen moho.refl")
_, _, vt = m2.parse_smesh(d / "true_vp.smesh")
_, _, vr = m2.parse_smesh(d / "rec_vp.smesh")
mx = max(abs(a - b) for col_t, col_r in zip(vt, vr) for a, b in zip(col_t, col_r))
print(f"  rec_vp vs true_vp max|d|={mx:.4f}")
if mx > 0.05:
    raise SystemExit("twostep overwrote Vp")
PY

echo "== recovered 612 twostep =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Urec_vs.smesh -Ggeom_612.dat \
  -Xconv.refl -Bseafloor.refl -Frec_moho.refl "$N" -Rrays_rec.dat > syn_rec.dat
echo "== done inv_twostep =="
