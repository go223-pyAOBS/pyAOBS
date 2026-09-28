# 折合 PSP 反演 — tt_inverse `-B`

混合慢度：转换面以上水 `1.5` + 沉积盖层梯度 Vp（约 1.80→3.15）**冻结**；只反界面以下壳幔 Vs。真/初值 Vs 梯度相同（`g=0.12 /km`），Vs0 从 `3.55` 收到 `3.20`。转换点按整条射线最短选。

| 文件 | 含义 |
|------|------|
| `true.smesh` | 真模型：Vs0=3.20 |
| `start.smesh` | 初值：Vs0=3.55；水/盖层与真值相同 |
| `conv.refl` | 平底转换面 z=5（反演 `-B`；正演 `-X`） |
| `seafloor.refl` | 平底海底 z=2（反演 `-Y` 冻水；也作图） |
| `geom_inv.dat` | 5 个 OBS 在海底 z=2（x=30–70）；炮在海面 z=0.01（x=20–80 km、间隔 2 km）；只用 raytype **6** |
| `vcorr.dat` | 速度相关长度（Lh=8 km，Lv=0.5 km），供 `-CV`；配 `-SV200 -TV20` |

`-B` 是转换波界面，不能当海底。加 `-Y -w` 冻水。有 `-B` 时，转换面**以上**结点从数据核、`-CV/-TV` 平滑和 LSQR 更新里全部去掉（界面结点划到面下 Vs）。图论先取整条 `t_P+t_S+t_P` 最短，再弯曲细化。前向星 `8/8`，不要 `-g`。

```bash
python make_converse_inv_case.py

tt_forward -Mtrue.smesh -Ggeom_inv.dat -Xconv.refl \
  -N8/8/0.8/8/1e-4/1e-5 > syn_inv.dat

tt_inverse -Mstart.smesh -Gsyn_inv.dat -Bconv.refl -Yseafloor.refl -w \
  -N8/8/0.8/8/1e-4/1e-5 -I5 -SV200 -TV20 -CVvcorr.dat -Oout -l -Linv.log

tt_forward -Mstart.smesh -Ggeom_inv.dat -Xconv.refl \
  -N8/8/0.8/8/1e-4/1e-5 > syn_start.dat
tt_forward -Mout.smesh.5.1 -Ggeom_inv.dat -Xconv.refl \
  -N8/8/0.8/8/1e-4/1e-5 -Rrays_rec.dat > syn_rec.dat

python check_converse_inv.py
```

写出三张图（无窗口加 `--no-show`）：

| 文件 | 内容 |
|------|------|
| `check_inv_models.png` | 初值 / 反演 / 真值；反演图叠射线（P 蓝 / S 红）；三角=海底 OBS，圆点=海面炮 |
| `check_inv_rays.png` | 收回模型 + PSP 射线：水柱和盖层为 **P（蓝）**，转换面以下为 **S（红）** |
| `check_inv_ttimes.png` | 走时拟合：观测（真模型正演）vs 初值正演 vs 反演正演，及残差 |

照明区浅 S 均值应靠近真值；水应仍为 1.5（`-w`）。盖层可能被界面平滑轻轻拉动。

也可在 WSL 下执行 `bash run_wsl.sh`（需本仓库 `src/build-tomo2d` 里已编好的 `tt_forward` / `tt_inverse`）。

本仓库已跑（弯曲 + 前向星 8，盖层完全冻结）：浅 S（z=5–7.5）均值 **3.343**（真 3.344，初 3.694），走时 RMS **0.488 → 0.010 s**（155 条），水 1.500，盖层均值 2.475（期望 2.48，与真值一致）。
