#!/bin/bash
set -euo pipefail
# 传统两步：PPP 反 Vp → 面下 Vp/1.73 → 混合网格上把 PSP 当 raytype 0 反（不用 -X/6）。
# 对照用，不是同一次联合。不改 ../PPP+PSP_inv。
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2"
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1
export TOMO2D_INV_OMP=1
export OMP_NUM_THREADS=8
export TOMO2D_INV_LSQR_MAXITER=8000

python3 make_ppp_psp_inv_case.py
rm -f out_ppp.smesh.* out_psp.smesh.* inv_ppp.log inv_psp.log

echo "== 1. true PPP (raytype 0) =="
"$BIN/tt_forward" -Mtrue_vp.smesh -Ggeom_ppp.dat -Bseafloor.refl \
  "$N" -Rrays_ppp_true.dat > syn_ppp.dat
echo "== 1. start PPP =="
"$BIN/tt_forward" -Mstart_vp.smesh -Ggeom_ppp.dat -Bseafloor.refl \
  "$N" > syn_ppp_start.dat
echo "== 1. inverse PPP  -M start_vp 冻水（无 -B，避免 solve_conv） =="
"$BIN/tt_inverse" -Mstart_vp.smesh -Gsyn_ppp.dat -Yseafloor.refl -w \
  "$N" -I5 -SV200 -TV5 -CVvcorr.dat -Oout_ppp -l -Linv_ppp.log

python3 - <<'PY'
from pathlib import Path
import shutil
import make_ppp_psp_inv_case as m

cands = sorted(Path(".").glob("out_ppp*.smesh*.*"))
if not cands:
    raise SystemExit("no out_ppp*.smesh")
rec = max(cands, key=lambda p: (int(p.name.split(".")[-2]), int(p.name.split(".")[-1])))
shutil.copyfile(rec, "rec_vp.smesh")
# 折合：盖层保持收回 Vp，面下 rec_vp/1.73（对齐 edit_smesh -Cb）。
m.write_mixed_smesh(
    Path("true_vp.smesh"), Path("true_vp.smesh"), Path("true_mixed.smesh"),
    below_kappa=m.KAPPA_TRUE,
)
m.write_mixed_smesh(
    Path("rec_vp.smesh"), Path("rec_vp.smesh"), Path("start_mixed.smesh"),
    below_kappa=m.KAPPA_TRUE,
)
m.write_lid_damp(Path("damp_lid.dat"))
print(f"rec_vp.smesh <- {rec.name}")
print(f"true_mixed.smesh <- 盖层 true_vp，面下 true_vp/{m.KAPPA_TRUE:g}")
print(f"start_mixed.smesh <- 盖层 rec_vp，面下 rec_vp/{m.KAPPA_TRUE:g}")
print(f"damp_lid.dat <- 盖层 w={m.DAMP_LID:.3g}  面下 w={m.DAMP_BELOW:g}")
PY

echo "== 1. recovered PPP forward =="
"$BIN/tt_forward" -Mrec_vp.smesh -Ggeom_ppp.dat -Bseafloor.refl \
  "$N" -Rrays_ppp_rec.dat > syn_ppp_rec.dat

echo "== 2. true 折合 type0  -M true_mixed（观测） =="
"$BIN/tt_forward" -Mtrue_mixed.smesh -Ggeom_psp0.dat -Bseafloor.refl \
  "$N" -Rrays_true.dat > syn_inv.dat
echo "== 2. true PSP type6  同网格对照射线 =="
"$BIN/tt_forward" -Mtrue_mixed.smesh -Ggeom_inv.dat -Xconv.refl -Bseafloor.refl \
  "$N" -Rrays_true6.dat > syn_psp6.dat
echo "== 2. start type0  -M start_mixed =="
"$BIN/tt_forward" -Mstart_mixed.smesh -Ggeom_psp0.dat -Bseafloor.refl \
  "$N" > syn_start.dat
echo "== 2. inverse PSP-as-0  无 -B，-DQ 面上1000/面下30 =="
"$BIN/tt_inverse" -Mstart_mixed.smesh -Gsyn_inv.dat \
  -Yseafloor.refl -w \
  -DV1 -DQdamp_lid.dat "$N" \
  -I8 -SV200 -CVvcorr.dat \
  -Oout_psp -l -Linv_psp.log

echo "== 2. recovered PSP-as-0 =="
VS=$(python3 -c "from pathlib import Path; c=sorted(Path('.').glob('out_psp.smesh.*.*'));
print(max(c, key=lambda p: (int(p.name.split('.')[-2]), int(p.name.split('.')[-1]))))")
echo "rec mixed mesh $VS"
python3 -c "from pathlib import Path; import shutil; shutil.copyfile('$VS', 'rec_vs.smesh'); print('rec_vs.smesh <-', '$VS')"
"$BIN/tt_forward" -M"$VS" -Ggeom_psp0.dat -Bseafloor.refl \
  "$N" -Rrays_rec.dat > syn_rec.dat

echo "== done two-step converse (Windows: python check_ppp_psp_inv.py --smesh rec_vs.smesh --no-show) =="
