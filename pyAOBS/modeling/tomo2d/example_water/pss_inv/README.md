# PSS 反 Vs — 面下 S + 台侧盖层 S

真模型与 `../ps_inv` 相同。数据只用 raytype **8**。**冻真 Vp**（`-M true_vp`），初 Vs = 真 Vp / 2，双场 `-U` 只反 Vs。不跑 PPP。

路径只按最短走时，不强制 Snell。正演 8 真双场 `-M`+`-U` 不必 `-k`。反演单场从 Vp 生成 Vs 才用 `-k`（`-Q` 在反演里是 LSQR 容差，不要写成 κ）。

| 步 | 数据 | `-M` | 反什么 | 冻 |
|----|------|------|--------|----|
| ① | `geom_inv.dat` **8** | **真 Vp** | **面下 Vs + 台侧盖层 Vs** | 真 Vp、水；盖层不解冻（有 8） |

炮侧盖层是 P，不进 Vs 核。台侧盖层是 S，会改盖层。初值误差只来自 κ（2.0 vs 1.73），不掺 Vp 偏差。

盖层和面下**都反**，但光滑/阻尼在转换面断开（与 PSP 冻盖层时一样），两边核不写界面结点，避免在界面下打出高速薄层、射线贴面滑行。

| 旗标 | 含义 |
|------|------|
| `-k2.0` | 冻 `-M` 的真 Vp；无 `-U` 时初 Vs=Vp/k |
| `-U start_vs` | 初 Vs = 真 Vp / 2 |
| `-SV200 -TV5` | 同 `ps_inv`，`vcorr` **Lh=8、Lv=2** |
| `-B -Y -w` | 转换面钉点、海底、冻水 |

```bash
python make_pss_inv_case.py
# WSL: bash run_wsl.sh
python check_pss_inv.py --no-show
```

输出：`out.smesh.*`（Vs）、`out.vp.smesh`（冻真 Vp）。对照图 `check_inv_models.png` / `check_inv_rays.png` / `check_inv_ttimes.png`。
