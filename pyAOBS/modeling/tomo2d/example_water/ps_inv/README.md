# 两步反演 — PPP 反 Vp，再只用 PSP 面下 S 核反 Vs

真 / 初 Vp 都是两段梯度：转换面以上**沉积**、以下**壳幔**；初值截距和斜率故意偏真值。
路径只按最短走时，不强制 Snell。κ 只作反演 `-k` 冷启动；造数/正演用独立 Vs 网格，不改 P 波 `-M`。

| 步 | 数据 | `-M` | 反什么 | 冻 |
|----|------|------|--------|----|
| ① | `geom_ppp.dat` raytype **0** | `start_vp.smesh` | Vp | `-Y -w` 水 |
| ② | `geom_inv.dat` **6** | 收回的 `rec_vp.smesh` | 面下 Vs | Vp（`-k` 双场）、水、**盖层冻死**（核/平滑/阻尼/dm） |

| 旗标 | 含义 |
|------|------|
| `-k2.0` | 反演冷启动：冻 `-M` 的 Vp，初 Vs=Vp/k（水不缩放） |
| 正演双场 | 造数/初值/收回都是 `-M` Vp + `-U` Vs，**不用 `-k`**。P 段只读 `-M` |
| Vs 步阻尼 | `-SV200 -TV5`，`vcorr` **Lh=8、Lv=2**（加重竖向光滑，压高低速条带） |
| `-B` | 仅第二步：转换面钉点；只 6 时整层冻盖层（只反面下） |
| `-Y -w` | 海底 + 冻水 |
| 收回 `-U` | `-Mout.vp.smesh -Uout.smesh.<iter>`（与造数同一套双场） |

输出：第一步 `out_vp.smesh.*`（Vp）、第二步 `out.smesh.*`（Vs）与 `out.vp.smesh`（冻 Vp）。

```bash
python make_ps_inv_case.py
# WSL: bash run_wsl.sh
python check_ps_inv.py --no-show
```

写出 `check_inv_vp_models.png` / `check_inv_vp_ttimes.png`（PPP），以及 `check_inv_models.png`、`check_inv_rays.png`、`check_inv_ttimes.png`（Vs）。
第二步成败看 `inv.log` 残差和**面下** Vs 均值；盖层冻死，应等于初值（收回 Vp/κ）。冻真 Vp 的 PSP 见 `../psp_inv`；PSS 见 `../pss_inv`。

PSP 在 `-k`/`-U` 下与 7/8 同一套双场：盖层 P、面下 S。无 `-k` 的折合 PSP 仍见 `../converse_inv`。正演 0/1/6/7/8 见 `../ps_fwd`。
