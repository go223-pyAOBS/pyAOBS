# 水核反演 2 — 收回 `water_fwd` 扰动场

真模型是 `../water_fwd/water.smesh`（背景 1.5 km/s，RMS 5%）；初值是**均匀 1.5**；海底以下沉积 1.8 冻结。**4 台**（x=30, 43.3, 56.7, 70 km）；海面炮 30–70 km、间隔 0.2 km。反演：`-Y -y`、**`-SV200`**、Lh=8 km、Lv=1.0 km。

对照：同一套平滑下，**2/3 同时进核** vs **只用直达 2**。

| 文件 | 含义 |
|------|------|
| `true.smesh` | `water_fwd` 同一套扰动（seed=42） |
| `start.smesh` | 均匀水 1.5，沉积 1.8 |
| `seafloor.refl` | 平底海底 z=2（反演 `-Y`） |
| `geom_inv.dat` | 4 台；炮 30–70 km、间隔 0.2 km；raytype **2** 与 **3** |
| `geom_inv_c2.dat` | 同上，仅 raytype **2** |
| `vcorr.dat` | Lh=8 km，Lv=1.0 km，供 `-CV` |

```bash
python make_water_inv2_case.py

tt_forward -Mtrue.smesh -Ggeom_inv.dat -Fseafloor.refl \
  -N4/4/0.8/8/1e-4/1e-5 > syn_inv.dat
tt_forward -Mtrue.smesh -Ggeom_inv_c2.dat -Fseafloor.refl \
  -N4/4/0.8/8/1e-4/1e-5 > syn_inv_c2.dat

tt_inverse -Mstart.smesh -Gsyn_inv.dat -Yseafloor.refl -y \
  -N4/4/0.8/8/1e-4/1e-5 -I5 -SV200 -CVvcorr.dat -Oout23 -l -Linv23.log
tt_inverse -Mstart.smesh -Gsyn_inv_c2.dat -Yseafloor.refl -y \
  -N4/4/0.8/8/1e-4/1e-5 -I5 -SV200 -CVvcorr.dat -Oout2 -l -Linv2.log

tt_forward -Mstart.smesh -Ggeom_inv.dat -Fseafloor.refl \
  -N4/4/0.8/8/1e-4/1e-5 > syn_start.dat
tt_forward -Mstart.smesh -Ggeom_inv_c2.dat -Fseafloor.refl \
  -N4/4/0.8/8/1e-4/1e-5 > syn_start_c2.dat
tt_forward -Mout23.smesh.5.1 -Ggeom_inv.dat -Fseafloor.refl \
  -N4/4/0.8/8/1e-4/1e-5 -Rrays_rec23.dat > syn_rec23.dat
tt_forward -Mout2.smesh.5.1 -Ggeom_inv_c2.dat -Fseafloor.refl \
  -N4/4/0.8/8/1e-4/1e-5 -Rrays_rec2.dat > syn_rec2.dat
tt_forward -Mout2.smesh.5.1 -Ggeom_inv.dat -Fseafloor.refl \
  -N4/4/0.8/8/1e-4/1e-5 > syn_rec2_on23.dat

python check_water_inv2.py --no-show --smesh out23.smesh.5.1 \
  --syn-rec syn_rec23.dat --rays rays_rec23.dat --tag 23 --rec-label 2/3
python check_water_inv2.py --no-show --smesh out2.smesh.5.1 \
  --geom geom_inv_c2.dat --obs syn_inv_c2.dat --syn-start syn_start_c2.dat \
  --syn-rec syn_rec2.dat --rays rays_rec2.dat --tag c2 --rec-label only2
python check_water_inv2.py --no-show --compare out23.smesh.5.1 out2.smesh.5.1
```

| 文件 | 内容 |
|------|------|
| `check_inv_models_23.png` / `_c2.png` | 各案初值 / 反演 / 真值（不叠射线） |
| `check_inv_compare.png` | 真值、2/3、仅 2，及各自相对真场残差 |
| `check_inv_ttimes_23.png` / `_c2.png` | 各自训练震相的走时拟合 |

`-SV200`、Lh=8、Lv=1.0（初值 RMS vs 真 = 0.081）：

| | 1 台 | | **4 台** | |
|---|---|---|---|---|
| | 2/3 | 仅 2 | 2/3 | 仅 2 |
| 水 RMS vs 真 | 0.074 | 0.082 | **0.026** | 0.032 |
| 相关系数 | 0.51 | 0.44 | **0.95** | 0.93 |
| 走时 RMS（同一套 2+3） | 0.051 s | 0.070 s | **0.031 s** | 0.055 s |

4 台时两边都收回得很好，2/3 仍略优（0.026 vs 0.032），差距接近原先 5 台。1 台时多次才明显补覆盖。沉积两侧都保持 1.8。
