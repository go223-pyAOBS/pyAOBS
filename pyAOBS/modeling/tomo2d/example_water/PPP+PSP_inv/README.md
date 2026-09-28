# PPP + PSP 同一次联合反演

一次 `tt_inverse -k`，数据里同时有 raytype **0** 和 **6**。LSQR 未知数是 **Vp 结点 + Vs 结点**。

几何与 `../ps_inv` / `../PPP+PSS_inv` 相同。正演 6 真双场 `-M`+`-U`（不必 `-k`）。反演 κ 用 `-k2.0`。联合光滑：**`-SV50` 只管 Vp，`-Ss200` 只管 Vs**。PSP 整层冻盖层，只反面下 Vs。联合时分两段（一次 `tt_inverse`）：前几轮只改 Vp（`-I8` 时前 3 轮）；解锁 Vs 时把面下 Vs 重设为当前 Vp/κ（盖层 Vs 不动），此后冻 Vp 只反 Vs。转换面结点不进核。`check_inv_rays.png` 左收回、右真值。

```bash
python make_ppp_psp_inv_case.py
bash run_wsl.sh
python check_ppp_psp_inv.py --no-show
```
