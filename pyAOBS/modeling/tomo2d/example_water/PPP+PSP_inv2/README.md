# PPP + PSP（盖层底 4.0，面下顶 7.2）

传统两步：面下 Vs=7.20/1.73≈**4.16 > 4.00**，混合网格初至能进面下。

1. PPP 反 Vp（`-Y -w` 冻水，不要 `-B`，否则会走 `solve_conv`）
2. 盖层保持收回 Vp，面下 `Vp/1.73`
3. 观测=混合网格 raytype **0**（不用把 6 标成 0）。无 `-B`，`-DQ` 面上 1000、面下 30

正演射线：`check_fwd_psp_rays.png`（左 PSP/6，右折合初至/0）。

```bash
bash run_wsl_twostep.sh
python check_ppp_psp_inv.py --smesh rec_vs.smesh --no-show
```
