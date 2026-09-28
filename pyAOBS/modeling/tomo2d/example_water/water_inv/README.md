# 水核反演 — tt_inverse `-Y` / `-y`

`topo=0` 时水柱已经在速度网格里。本工区验收：**冻壳、只反水**。

| 文件 | 含义 |
|------|------|
| `true.smesh` | 真模型：水 1.45，沉积 1.8 |
| `start.smesh` | 初值：水 1.55，沉积 1.8 |
| `seafloor.refl` | 平底海底 z=2（反演 `-Y`；正演可用 `-B` 或 `-F`） |
| `geom_inv.dat` | 5 个 OBS 在海底（x=30,40,50,60,70）；炮仍在海面 30–70 km、间隔 0.2 km；raytype **2** 与 **3** |
| `vcorr.dat` | 速度相关长度（Lh=8 km，Lv=2 km，约水柱厚），供 `-CV` |

反演 `-B` 仍是转换波界面，不能当海底。`-y` 时海底以下结点不进速度核。做壳冻水用 `-w`（见 `../crust_inv`）。0/1+2/3 同时进核见 `../joint_inv`。0/1 默认路径未改。

```bash
python make_water_inv_case.py

tt_forward -Mtrue.smesh -Ggeom_inv.dat -Fseafloor.refl \
  -N4/4/0.8/8/1e-4/1e-5 > syn_inv.dat

tt_inverse -Mstart.smesh -Gsyn_inv.dat -Yseafloor.refl -y \
  -N4/4/0.8/8/1e-4/1e-5 -I5 -SV200 -CVvcorr.dat -Oout -l -Linv.log

tt_forward -Mstart.smesh -Ggeom_inv.dat -Fseafloor.refl \
  -N4/4/0.8/8/1e-4/1e-5 > syn_start.dat
tt_forward -Mout.smesh.5.1 -Ggeom_inv.dat -Fseafloor.refl \
  -N4/4/0.8/8/1e-4/1e-5 -Rrays_rec.dat > syn_rec.dat

python check_water_inv.py
```

写出三张图（无窗口加 `--no-show`）：

| 文件 | 内容 |
|------|------|
| `check_inv_models.png` | 初值 / 反演 / 真值；反演图叠射线；三角=OBS 台，圆点=炮 |
| `check_inv_rays.png` | 收回模型 + 直达/多次射线 + 台、炮 |
| `check_inv_ttimes.png` | 走时拟合：观测（真模型正演）vs 初值正演 vs 反演正演，及残差 |

照明区水速均值应靠近 1.45，沉积应仍为 1.8。`-y` 时平滑不跨海底。无 `-Y` 时 2/3 仍可把 `-F` 当海底（与 `water_fwd` 兼容）。

本仓库已用 `-SV200 -CVvcorr.dat`（Lv=2 km）、5 台、**200 m 一炮**重跑：照明区水均值约 **1.451**（初值 1.55，真值 1.45，结点约 1.44–1.46），沉积保持 **1.8**，走时 RMS 约 **0.044 s**（初值正演 0.82 s，2010 条 2/3）。均值已贴真值；结点范围比同几何 `-SV50` 明显收窄。
