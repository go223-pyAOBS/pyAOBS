# 水核反演 3 — 200 km 测线、水深 3 km、水中高速异常

模型范围 **0–200 km**，`topo=0`，水深 **3 km**。真模型：背景水 1.5 km/s，中部 **30 km 宽 × 1 km 高** 高速异常（x=85–115、z=1–2、v=1.65）；初值均匀水 1.5；海底以下沉积 1.8 冻结（`-y`）。

**10 台**，台距 10 km（x=55, 65, …, 145 km）。海面炮间隔 **200 m**。每台：直达 **2** 偏移 ≤**20 km**，多次 **3** 偏移 ≤**40 km**。反演：`-Y -y`、**`-SV200`**、Lh=8 km、Lv=1.0 km。

对照：同一套平滑下，**2/3 同时进核** vs **只用直达 2**。

| 文件 | 含义 |
|------|------|
| `true.smesh` | 背景 1.5 + 30×1 km 高速异常 1.65 |
| `start.smesh` | 均匀水 1.5，沉积 1.8 |
| `seafloor.refl` | 平底海底 z=3（反演 `-Y`） |
| `geom_inv.dat` | 10 台；2：\|dx\|≤20 km；3：\|dx\|≤40 km |
| `geom_inv_c2.dat` | 同上，仅 raytype **2** |
| `vcorr.dat` | Lh=8 km，Lv=1.0 km，供 `-CV` |

```bash
python make_water_inv3_case.py

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

python check_water_inv3.py --no-show --smesh out23.smesh.5.1 \
  --syn-rec syn_rec23.dat --rays rays_rec23.dat --tag 23 --rec-label 2/3
python check_water_inv3.py --no-show --smesh out2.smesh.5.1 \
  --geom geom_inv_c2.dat --obs syn_inv_c2.dat --syn-start syn_start_c2.dat \
  --syn-rec syn_rec2.dat --rays rays_rec2.dat --tag c2 --rec-label only2
python check_water_inv3.py --no-show --compare out23.smesh.5.1 out2.smesh.5.1
```

| 文件 | 内容 |
|------|------|
| `check_inv_models_23.png` / `_c2.png` | 各案初值 / 反演 / 真值（不叠射线；绿框=异常） |
| `check_inv_compare.png` | 真值、2/3、仅 2，及各自相对真场残差 |
| `check_inv_ttimes_23.png` / `_c2.png` | 各自训练震相的走时拟合 |

反演模型图不叠射线；射线只在 `check_inv_rays_*.png`。

`-SV200`、Lh=8、Lv=1.0（初值照明区水 RMS vs 真 = 0.037）：

| | **2/3** | **仅 2** |
|---|---|---|
| 水 RMS vs 真（全照明区） | 0.038 | 0.032 |
| 相关系数 | 0.42 | 0.56 |
| 异常框内均值（真 1.650） | 1.566 | 1.581 |
| 同 x 框上 0–1 km（真 1.500） | **1.507** | 1.573 |
| 同 x 框下 2–3 km（真 1.500） | 1.616 | 1.583 |
| 训练走时 RMS | 0.081 s（2+3） | 0.023 s（仅 2） |
| 同一套 2+3 走时 RMS | 0.081 s | 0.106 s |

全场 RMS 偏向仅 2，因为背景更干净；但仅 2 在异常柱内几乎没有垂向反差（0–3 km 都是 ~1.58，贯穿柱）。2/3 浅部仍接近 1.50，能看出一块高速体，只是峰值被多次波核拉到框下、靠近海底。沉积两侧都保持 1.8。
