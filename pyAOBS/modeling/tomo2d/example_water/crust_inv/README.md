# 壳核反演 — tt_inverse `-Y` / `-w`

`topo=0` 时水柱仍在速度网格里。本工区验收：**冻水、只反壳**。

| 文件 | 含义 |
|------|------|
| `true.smesh` | 真模型：水 1.45，沉积 1.8 |
| `start.smesh` | 初值：水 1.45（已对），沉积 2.0 |
| `seafloor.refl` | 平底海底 z=2（反演 `-Y`） |
| `geom_inv.dat` | 台和炮都在海底，raytype **0**（避免海面接收时图论初至走水柱） |
| `vcorr.dat` | 速度相关长度（Lh=8 km，Lv=0.5 km），供 `-CV` |

水步见 `../water_inv`（`-Y`/`-y`，只用 2/3）。0/1+2/3 同时进核见 `../joint_inv`。`-y` 与 `-w` 互斥。0/1 默认路径未改。

```bash
python make_crust_inv_case.py

tt_forward -Mtrue.smesh -Ggeom_inv.dat -Fseafloor.refl \
  -N4/4/0.8/8/1e-4/1e-5 > syn_inv.dat

tt_inverse -Mstart.smesh -Gsyn_inv.dat -Yseafloor.refl -w \
  -N4/4/0.8/8/1e-4/1e-5 -I5 -SV20 -CVvcorr.dat -Oout -l -Linv.log

tt_forward -Mstart.smesh -Ggeom_inv.dat -Fseafloor.refl \
  -N4/4/0.8/8/1e-4/1e-5 > syn_start.dat
tt_forward -Mout.smesh.5.1 -Ggeom_inv.dat -Fseafloor.refl \
  -N4/4/0.8/8/1e-4/1e-5 -Rrays_rec.dat > syn_rec.dat

python check_crust_inv.py
```

写出三张图（无窗口加 `--no-show`）：

| 文件 | 内容 |
|------|------|
| `check_inv_models.png` | 初值 / 反演 / 真值；反演图叠射线；三角=OBS 台，圆点=炮 |
| `check_inv_rays.png` | 收回模型 + code 0 射线 + 台、炮 |
| `check_inv_ttimes.png` | 走时拟合：观测（真模型正演）vs 初值正演 vs 反演正演，及残差 |

海底以上水速应仍为 1.45；沉积应从 2.0 靠近 1.8。海底结点划到壳侧，可以动。

本仓库已用 `-SV20 -CVvcorr.dat` 跑过：海底以上水保持 **1.45**，沉积均值约 **1.801**（初值 2.0，真值 1.8），走时 RMS 约 **0.001 s**（初值正演 0.67 s）。
