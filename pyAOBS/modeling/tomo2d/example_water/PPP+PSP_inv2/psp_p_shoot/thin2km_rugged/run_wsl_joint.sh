#!/bin/bash
set -euo pipefail
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
bash "$ROOT/inv_joint/run_wsl.sh"
echo "== P1 joint done  plot: python check_joint.py =="
