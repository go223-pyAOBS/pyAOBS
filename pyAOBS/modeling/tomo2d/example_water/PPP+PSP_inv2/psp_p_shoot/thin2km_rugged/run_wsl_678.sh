#!/bin/bash
set -euo pipefail
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
bash "$ROOT/inv_678_hot/run_wsl.sh"
bash "$ROOT/inv_678_kappa/run_wsl.sh"
echo "== B 6+7+8 done  plot: python check_joint.py --678 hot && python check_joint.py --678 kappa =="
