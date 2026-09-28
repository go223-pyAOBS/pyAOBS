# 水+壳联合反演 — 0/1 + 2/3，无 `-y`/`-w`

`topo=0`。**不保证** 2/3 只改水、0/1 只改壳：核按整条路径累。本工区用几何把照明大致分开——2/3 海面炮，0/1 海底炮——再同时进核。

| 文件 | 含义 |
|------|------|
| `true.smesh` | 真模型：水 1.45，沉积 1.8 |
| `start.smesh` | 初值：水 1.55，沉积 2.0（两边都偏） |
| `seafloor.refl` | 海底 z=2（正演 `-B`，反演 `-Y`） |
| `basement.refl` | 沉积内反射面 z=3.2（`-F`，code 1；反演 `-u` 冻界面） |
| `geom_inv.dat` | OBS 在海底；**0/1 炮在海底**，**2/3 炮在海面** |
| `vcorr.dat` | Lh=8 km，Lv=0.5 km |

海面接收的 code 0 图论初至是水波，所以 0/1 不用海面炮。无 `-y`/`-w` 时平滑可以跨海底。分步做法见 `../water_inv`（`-y`）和 `../crust_inv`（`-w`）。

```bash
python make_joint_inv_case.py

tt_forward -Mtrue.smesh -Ggeom_inv.dat -Bseafloor.refl -Fbasement.refl \
  -N4/4/0.8/8/1e-4/1e-5 > syn_inv.dat

tt_inverse -Mstart.smesh -Gsyn_inv.dat -Yseafloor.refl -Fbasement.refl -u \
  -N4/4/0.8/8/1e-4/1e-5 -I5 -SV20 -CVvcorr.dat -Oout -l -Linv.log

tt_forward -Mstart.smesh -Ggeom_inv.dat -Bseafloor.refl -Fbasement.refl \
  -N4/4/0.8/8/1e-4/1e-5 > syn_start.dat
tt_forward -Mout.smesh.5.1 -Ggeom_inv.dat -Bseafloor.refl -Fbasement.refl \
  -N4/4/0.8/8/1e-4/1e-5 -Rrays_rec.dat > syn_rec.dat

python check_joint_inv.py
```

写出三张图（无窗口加 `--no-show`）：

| 文件 | 内容 |
|------|------|
| `check_inv_models.png` | 初值 / 反演 / 真值；反演图叠射线 |
| `check_inv_rays.png` | 收回模型 + 0/1/2/3 射线 + 台、炮 |
| `check_inv_ttimes.png` | 四类震相走时拟合与残差 |

本仓库已用 `-SV20 -CVvcorr.dat` 跑过：照明区水均值约 **1.450**（初值 1.55，结点约 1.36–1.52），沉积约 **1.805**（初值 2.0，真值 1.8），走时 RMS 约 **0.012 s**（初值正演 0.61 s）。无 `-y`/`-w`，水柱结点范围比只反水更宽，平滑可以跨海底。
