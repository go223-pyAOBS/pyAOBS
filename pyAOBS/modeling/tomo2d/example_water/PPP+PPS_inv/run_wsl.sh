#!/bin/bash
set -euo pipefail
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PPS_inv"
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1
export TOMO2D_INV_OMP=1
export OMP_NUM_THREADS=8

python3 make_ppp_pps_inv_case.py

echo "== true PPP =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Ggeom_ppp.dat -Bseafloor.refl \
  "$N" -Rrays_ppp_true.dat > syn_ppp.dat
echo "== true PPS =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_inv.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_true.dat > syn_inv.dat
echo "== true joint 0+7 =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_joint.dat \
  -Xconv.refl -Bseafloor.refl "$N" > syn_joint.dat

echo "== start PPP / PPS / joint =="
"$BIN/tt_forward" -Mstart_vp.smesh -Ggeom_ppp.dat -Bseafloor.refl \
  "$N" > syn_ppp_start.dat
"$BIN/tt_forward" -Mstart_vp.smesh -Ustart_vs.smesh -Ggeom_inv.dat \
  -Xconv.refl -Bseafloor.refl "$N" > syn_start.dat
"$BIN/tt_forward" -Mstart_vp.smesh -Ustart_vs.smesh -Ggeom_joint.dat \
  -Xconv.refl -Bseafloor.refl "$N" > syn_joint_start.dat

echo "== inverse JOINT  PPP+PPS  同一次 Vp+Vs  -k2.0 =="
"$BIN/tt_inverse" -Mstart_vp.smesh -Ustart_vs.smesh -Gsyn_joint.dat \
  -Bconv.refl -Yseafloor.refl -w -k2.0 "$N" \
  -I8 -SV50 -Ss200 -TV5 -CVvcorr.dat -Oout -l -Linv.log

python3 - <<'PY'
from pathlib import Path
import shutil
cands = sorted(Path(".").glob("out.vp.smesh.*.*"))
if not cands:
    p = Path("out.vp.smesh")
    if not p.is_file():
        raise SystemExit("no out.vp.smesh")
    shutil.copyfile(p, "rec_vp.smesh")
    print("rec_vp.smesh <- out.vp.smesh")
else:
    rec = max(cands, key=lambda p: (int(p.name.split(".")[-2]), int(p.name.split(".")[-1])))
    shutil.copyfile(rec, "rec_vp.smesh")
    print(f"rec_vp.smesh <- {rec.name}")
PY

VS=$(python3 -c "from pathlib import Path; c=sorted(Path('.').glob('out.smesh.*.*'));
print(max(c, key=lambda p: (int(p.name.split('.')[-2]), int(p.name.split('.')[-1]))))")
echo "rec Vs mesh $VS"

echo "== recovered PPP / PPS / joint =="
"$BIN/tt_forward" -Mrec_vp.smesh -Ggeom_ppp.dat -Bseafloor.refl \
  "$N" -Rrays_ppp_rec.dat > syn_ppp_rec.dat
"$BIN/tt_forward" -Mrec_vp.smesh -U"$VS" -Ggeom_inv.dat -Xconv.refl \
  -Bseafloor.refl "$N" -Rrays_rec.dat > syn_rec.dat
"$BIN/tt_forward" -Mrec_vp.smesh -U"$VS" -Ggeom_joint.dat -Xconv.refl \
  -Bseafloor.refl "$N" > syn_joint_rec.dat

echo "== done (Windows: python check_ppp_pps_inv.py --no-show) =="
