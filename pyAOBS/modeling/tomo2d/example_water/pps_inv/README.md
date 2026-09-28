# PPS 反 Vs — 只有台侧盖层 S

真模型与 `../ps_inv` 相同。数据只用 raytype **7**。冻真 Vp，初 Vs = 真 Vp / 2。

PPS：炮侧盖层 P、面下 P、台侧盖层 S。Vs 核只有台侧盖层；面下走冻住的 Vp，不应改面下 Vs。光滑/阻尼在转换面断开（与 PSS 相同）。

```bash
python make_pps_inv_case.py
# WSL: bash run_wsl.sh
python check_pps_inv.py --no-show
```
