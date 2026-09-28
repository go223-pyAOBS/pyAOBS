#!/bin/bash
set -euo pipefail
# 已有 rec_vp：折合 type0 观测+反演；无 -B，-DQ 1000/30。须先用新模型跑完 PPP。
cd "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2"
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1
export TOMO2D_INV_OMP=1
export OMP_NUM_THREADS=8
export TOMO2D_INV_LSQR_MAXITER=8000

python3 - <<'PY'
from pathlib import Path
import make_ppp_psp_inv_case as m
m.write_mixed_smesh(
    Path("true_vp.smesh"), Path("true_vp.smesh"), Path("true_mixed.smesh"),
    below_kappa=m.KAPPA_TRUE,
)
m.write_mixed_smesh(
    Path("rec_vp.smesh"), Path("rec_vp.smesh"), Path("start_mixed.smesh"),
    below_kappa=m.KAPPA_TRUE,
)
m.write_lid_damp(Path("damp_lid.dat"))
m.write_geom(Path("geom_psp0.dat"), codes=m.CODES_PSP0)
m.write_geom(Path("geom_inv.dat"), codes=m.CODES)
print(f"start_mixed.smesh <- 盖层 rec_vp，面下 rec_vp/{m.KAPPA_TRUE:g}")
PY

echo "== 2. true 折合 type0 =="
"$BIN/tt_forward" -Mtrue_mixed.smesh -Ggeom_psp0.dat -Bseafloor.refl \
  "$N" -Rrays_true.dat > syn_inv.dat
echo "== 2. true PSP type6 对照 =="
"$BIN/tt_forward" -Mtrue_mixed.smesh -Ggeom_inv.dat -Xconv.refl -Bseafloor.refl \
  "$N" -Rrays_true6.dat > syn_psp6.dat
echo "== 2. start type0 =="
"$BIN/tt_forward" -Mstart_mixed.smesh -Ggeom_psp0.dat -Bseafloor.refl \
  "$N" -Rrays_start.dat > syn_start.dat

echo "== 2. inverse type0  无 -B，-DQ 1000/30 =="
"$BIN/tt_inverse" -Mstart_mixed.smesh -Gsyn_inv.dat \
  -Yseafloor.refl -w \
  -DV1 -DQdamp_lid.dat "$N" \
  -I8 -SV200 -CVvcorr.dat \
  -Oout_psp -l -Linv_psp.log

VS=$(python3 -c "from pathlib import Path; c=sorted(Path('.').glob('out_psp.smesh.*.*'));
print(max(c, key=lambda p: (int(p.name.split('.')[-2]), int(p.name.split('.')[-1]))))")
echo "rec mixed mesh $VS"
python3 -c "import shutil; shutil.copyfile('$VS', 'rec_vs.smesh'); print('rec_vs.smesh <-', '$VS')"
"$BIN/tt_forward" -M"$VS" -Ggeom_psp0.dat -Bseafloor.refl \
  "$N" -Rrays_rec.dat > syn_rec.dat
echo "== psp-only done =="
