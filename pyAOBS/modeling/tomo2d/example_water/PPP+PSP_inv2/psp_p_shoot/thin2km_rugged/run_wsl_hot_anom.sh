#!/bin/bash
set -euo pipefail
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
bash "$ROOT/inv_graph6k_hot/run_wsl_hot_anom.sh"
bash "$ROOT/inv_graph6_hot/run_wsl_hot_anom.sh"
echo "== hot-Vs PSP done  plot: python compare_single_dual.py =="
