# 水柱棋盘格 — 2/3 vs 仅 2 能收回多少异常

几何同 `water_inv`：**5 台**（x=30, 40, 50, 60, 70 km），海面炮 30–70 km、间隔 0.2 km。真模型只在水里放 **10 km × 1 km** 棋盘（±0.10 km/s，背景 1.5）；初值均匀 1.5；海底以下 1.8 冻结。

相关长度 **小于格子**（Lh=6、Lv=0.4），避免正则先把棋盘抹平。对照：同一套 `-Y -y -SV200` 下 **2/3** vs **仅 2**。

| 文件 | 含义 |
|------|------|
| `true.smesh` | 水柱棋盘 10×1 km，±0.10 |
| `start.smesh` | 均匀水 1.5，沉积 1.8 |
| `seafloor.refl` | 平底海底 z=2（反演 `-Y`） |
| `geom_inv.dat` | 5 台；炮 30–70 km；raytype **2** 与 **3** |
| `geom_inv_c2.dat` | 同上，仅 raytype **2** |
| `vcorr.dat` | Lh=6 km，Lv=0.4 km，供 `-CV` |

```bash
python make_water_checkboard_case.py

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

python check_water_checkboard.py --no-show --smesh out23.smesh.5.1 \
  --syn-rec syn_rec23.dat --rays rays_rec23.dat --tag 23 --rec-label 2/3
python check_water_checkboard.py --no-show --smesh out2.smesh.5.1 \
  --geom geom_inv_c2.dat --obs syn_inv_c2.dat --syn-start syn_start_c2.dat \
  --syn-rec syn_rec2.dat --rays rays_rec2.dat --tag c2 --rec-label only2
python check_water_checkboard.py --no-show --compare out23.smesh.5.1 out2.smesh.5.1
```

反演模型图不叠射线。白虚线是棋盘格子。

`-SV200`、Lh=6、Lv=0.4（初值照明区水 RMS vs 真 = 0.100）：

| | **2/3** | **仅 2** |
|---|---|---|
| 水 RMS vs 真 | 0.079 | 0.079 |
| 相关系数 | **0.65** | 0.62 |
| 极性一致率 | **0.80** | 0.78 |
| 收回 \|Δv\| / 真 \|Δv\| | **0.70** | 0.58 |
| 训练走时 RMS | 0.047 s（2+3） | 0.021 s（仅 2） |
| 同一套 2+3 走时 RMS | **0.047 s** | 0.062 s |

两边都能看出 10 km 水平格子。垂向 1 km 两层都糊，仅 2 更像贯穿柱、幅度更弱；2/3 幅度收回更多（0.70 vs 0.58），x=50 剖面更跟得上 1.6→1.4 的台阶。沉积两侧保持 1.8。
