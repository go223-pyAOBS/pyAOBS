#!/bin/bash
set -euo pipefail
# 单场 / 双场各自两步，走时互不拷贝。
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
bash "$ROOT/inv_graph6k_hot/run_wsl_hot.sh"
bash "$ROOT/inv_graph6_hot/run_wsl_hot.sh"
echo "== fair two-step done  plot: python compare_single_dual.py =="
