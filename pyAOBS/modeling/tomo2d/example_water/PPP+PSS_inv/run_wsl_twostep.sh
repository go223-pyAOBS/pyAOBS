#!/bin/bash
set -euo pipefail
# 旧两步：先 PPP 反 Vp，再冻收回 Vp 只反 PSS。对照用，不是同一次联合。
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSS_inv"
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1
export TOMO2D_INV_OMP=1
export OMP_NUM_THREADS=8

python3 make_ppp_pss_inv_case.py

echo "== 1. true PPP (raytype 0) =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Ggeom_ppp.dat -Bseafloor.refl \
  "$N" -Rrays_ppp_true.dat > syn_ppp.dat
echo "== 1. start PPP =="
"$BIN/tt_forward" -Mstart_vp.smesh -Ggeom_ppp.dat -Bseafloor.refl \
  "$N" > syn_ppp_start.dat
echo "== 1. inverse PPP  -M start_vp 冻水 =="
"$BIN/tt_inverse" -Mstart_vp.smesh -Gsyn_ppp.dat -Yseafloor.refl -w \
  "$N" -I5 -SV50 -CVvcorr.dat -Oout_vp -l -Linv_vp.log

python3 - <<'PY'
from pathlib import Path
import shutil
import make_ppp_pss_inv_case as m
cands = sorted(Path(".").glob("out_vp.smesh.*.*"))
if not cands:
    raise SystemExit("no out_vp.smesh.*")
rec = max(cands, key=lambda p: (int(p.name.split(".")[-2]), int(p.name.split(".")[-1])))
shutil.copyfile(rec, "rec_vp.smesh")
m.write_vs_from_smesh(Path("rec_vp.smesh"), Path("start_vs.smesh"), m.KAPPA_START)
print(f"rec_vp.smesh <- {rec.name}")
print(f"start_vs.smesh <- rec_vp / {m.KAPPA_START:g}")
PY

echo "== 1. recovered PPP forward =="
"$BIN/tt_forward" -Mrec_vp.smesh -Ggeom_ppp.dat -Bseafloor.refl \
  "$N" -Rrays_ppp_rec.dat > syn_ppp_rec.dat

echo "== 2. true PSS  -M true_vp -U true_vs =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_inv.dat \
  -Xconv.refl -Bseafloor.refl "$N" -Rrays_true.dat > syn_inv.dat
echo "== 2. start PSS  -M rec_vp -U start_vs =="
"$BIN/tt_forward" -Mrec_vp.smesh -Ustart_vs.smesh -Ggeom_inv.dat \
  -Xconv.refl -Bseafloor.refl "$N" > syn_start.dat
echo "== 2. inverse PSS  冻收回 Vp  -k2.0 =="
"$BIN/tt_inverse" -Mrec_vp.smesh -Ustart_vs.smesh -Gsyn_inv.dat \
  -Bconv.refl -Yseafloor.refl -w -k2.0 "$N" \
  -I5 -SV200 -TV5 -CVvcorr.dat -Oout -l -Linv.log

echo "== 2. recovered PSS  -M out.vp -U out.smesh =="
VS=$(python3 -c "from pathlib import Path; c=sorted(Path('.').glob('out.smesh.*.*'));
print(max(c, key=lambda p: (int(p.name.split('.')[-2]), int(p.name.split('.')[-1]))))")
echo "rec Vs mesh $VS"
"$BIN/tt_forward" -Mout.vp.smesh -U"$VS" -Ggeom_inv.dat -Xconv.refl \
  -Bseafloor.refl "$N" -Rrays_rec.dat > syn_rec.dat

echo "== done two-step (Windows: python check_ppp_pss_inv.py --no-show) =="
