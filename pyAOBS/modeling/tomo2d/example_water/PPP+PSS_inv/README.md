# PPP + PSS 同一次联合反演

一次 `tt_inverse -k`，数据里同时有 raytype **0** 和 **8**。LSQR 未知数是 **Vp 结点 + Vs 结点**。

| 模式 | 数据 | 行为 |
|------|------|------|
| 无 `-k` | 0/1（可带 converse 6） | 传统只反 Vp，源代码策略不变 |
| `-k` 且只有 6/7/8 | 冻 Vp，只反 Vs | 现有 PSP/PPS/PSS 策略不变 |
| `-k` 且 0/1 **和** 6/7/8 | 同一次联合 | 0/1 写 Vp；6/7/8 只把 S 段写 Vs（P 段不写 Vp） |

几何与 `../ps_inv` 相同。正演 8 真双场 `-M`+`-U`（不必 `-k`）。反演 κ 用 `-k2.0`。联合光滑：**`-SV50` 只管 Vp，`-Ss200` 只管 Vs**（不写 `-Ss` 则两场共用 `-SV`）。传统无 `-k` 和只 6/7/8 的 `-k` 不受 `-Ss` 影响。转换面结点两边都不进核，Vp/Vs 光滑都不跨转换面。6/7/8 的 P 段不进 Vp 核，避免 PSS 残差拧坏盖层 Vp。联合时分两段（一次 `tt_inverse`）：前几轮只改 Vp（`-I8` 时前 3 轮）；解锁 Vs 时把面下 Vs 重设为当前 Vp/κ（盖层 Vs 不动），此后冻 Vp 只反 Vs。转换面结点不进核。不改 LSQR 解。`check_inv_rays.png` 左收回、右真值。

```bash
python make_ppp_pss_inv_case.py
# WSL 同一次联合:
bash run_wsl.sh
# 旧两步对照:
bash run_wsl_twostep.sh
python check_ppp_pss_inv.py --no-show
```

一次性 PPP+PPS+PSS+PSP：把 `geom_joint.dat` 的 codes 改成 `(0, 7, 8, 6)` 即可，反演命令不变。
