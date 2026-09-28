#!/bin/bash
set -euo pipefail
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
bash "$ROOT/inv_pps_dual/run_wsl.sh"
bash "$ROOT/inv_pps_single/run_wsl.sh"
echo "== PPS hot-lid done  plot: python compare_single_dual.py --pps =="
