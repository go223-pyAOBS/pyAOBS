#!/bin/bash
set -euo pipefail
# LM 信赖域对照：不改 inv_612 / inv_pps_lid。SENS / LINESEARCH / C2F 关。
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d"
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
HERE=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged/lvz2d
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
N=-N8/8/0.8/8/1e-4/1e-5
NINV=-N8/8/0.8/8/1e-3/1e-4
export TOMO2D_FWD_OMP=1 TOMO2D_INV_OMP=1 OMP_NUM_THREADS=8
export TOMO2D_INV_LSQR_MAXITER=8000
export TOMO2D_INV_LM=1
export PYTHONPATH="$HERE:$ROOT:$ROOT/inv_2d:$ROOT/../..:${PYTHONPATH:-}"
unset TOMO2D_INV_STRATEGY TOMO2D_INV_TTDIFF TOMO2D_INV_PSS_BELOW
unset TOMO2D_INV_LSQR_PRECOND
unset TOMO2D_INV_SENS_WEIGHT
unset TOMO2D_INV_LINESEARCH
unset TOMO2D_INV_COARSE2FINE

finish_vs() {
  local tag="$1"
  python3 - "$tag" <<'PY'
import sys
from pathlib import Path
import shutil
import inv_grid as g
import make_ppp_psp_inv_case as m2
tag = sys.argv[1]
d = Path(".")
vs = g.latest("out.smesh*.*", d)
vp = g.latest("out.vp.smesh*.*", d)
shutil.copyfile(vs, d / "rec_vs.smesh")
shutil.copyfile(vp, d / "rec_vp.smesh")
print(f"{tag}: rec_vs <- {vs.name}  rec_vp <- {vp.name}")
if list(d.glob("out.refl.*.*")):
    rf = g.latest("out.refl.*.*", d)
    shutil.copyfile(rf, d / "rec_moho.refl")
    print(f"{tag}: rec_moho <- {rf.name}")
_, _, vt = m2.parse_smesh(d / "true_vp.smesh")
_, _, vr = m2.parse_smesh(d / "rec_vp.smesh")
mx = max(abs(a - b) for col_t, col_r in zip(vt, vr) for a, b in zip(col_t, col_r))
print(f"{tag}: rec_vp vs true_vp max|d|={mx:.4f}")
if mx > 0.05:
    raise SystemExit(f"{tag} overwrote Vp")
PY
}

SRC="$HERE/inv_612"
DST="$HERE/inv_lm/612"
rm -rf "$DST"
mkdir -p "$DST"
cp -f "$SRC"/true_vp.smesh "$SRC"/true_vs.smesh "$SRC"/start_vs.smesh \
  "$SRC"/syn_inv.dat "$SRC"/geom_612.dat "$SRC"/vcorr.dat \
  "$SRC"/conv.refl "$SRC"/seafloor.refl "$SRC"/moho.refl "$SRC"/moho_true.refl \
  "$DST/"
cd "$DST"
unset TOMO2D_INV_FREEZE_BELOW
export TOMO2D_INV_FREEZE_LID=1
echo "== inverse 612  LM=1  freeze lid, Moho free =="
"$BIN/tt_inverse" -Mtrue_vp.smesh -Ustart_vs.smesh -Gsyn_inv.dat \
  -Bconv.refl -Yseafloor.refl -Fmoho.refl -w -k1.73 \
  "$NINV" -I8 -SV200 -SD20 -DV30 -DD200 -CVvcorr.dat \
  -Oout -l -Linv.log -V0
finish_vs 612
echo "== recovered 612 =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Urec_vs.smesh -Ggeom_612.dat \
  -Xconv.refl -Bseafloor.refl -Frec_moho.refl "$N" -Rrays_rec.dat > syn_rec.dat

SRC="$HERE/inv_pps_lid"
DST="$HERE/inv_lm/lid7"
rm -rf "$DST"
mkdir -p "$DST"
cp -f "$SRC"/true_vp.smesh "$SRC"/true_vs.smesh "$SRC"/start_vs.smesh \
  "$SRC"/syn_inv_7.dat "$SRC"/geom_7.dat "$SRC"/vcorr.dat \
  "$SRC"/conv.refl "$SRC"/seafloor.refl \
  "$DST/"
cd "$DST"
unset TOMO2D_INV_FREEZE_LID
export TOMO2D_INV_FREEZE_BELOW=1
echo "== inverse lid7  LM=1  freeze below, invert lid Vs =="
"$BIN/tt_inverse" -Mtrue_vp.smesh -Ustart_vs.smesh -Gsyn_inv_7.dat \
  -Bconv.refl -Yseafloor.refl -w -k1.73 \
  "$NINV" -I8 -SV200 -TV5 -CVvcorr.dat \
  -Oout -l -Linv.log -V0
finish_vs lid7
echo "== recovered lid7 =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Urec_vs.smesh -Ggeom_7.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_rec.dat > syn_rec.dat

echo "== done inv_lm =="
