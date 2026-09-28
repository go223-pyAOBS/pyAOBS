# 台侧多次进核对照 — 壳内高速体

OBS 在海底、炮在海面。真模型是**背景壳梯度 + 一块高速体**，初值是无异常背景（不是整层均匀偏置）。`-w` 冻水，`-u` 冻莫霍。

高速体：x=42–58 km，z=**2.4–4.0 km**（紧挨海底之下），相对背景 **+0.50 km/s**。块内约 4.64–5.20，避开壳底 5.4–6.1 的背景速度带。

| 文件 | 含义 |
|------|------|
| `true.smesh` | 背景 + 高速体 |
| `start.smesh` | 无异常背景 |
| `seafloor.refl` / `moho.refl` | `-Y` / `-F` |
| `geom_inv_01.dat` | 5 台（**38/44/50/56/62**）；**0/1 偏移 ±10…±30 km** |
| `geom_inv_45.dat` | 同上；**4/5 偏移 ±10…±40 km**（距网格边不足 8 km 的炮丢掉） |
| `geom_inv_0145.dat` | 同上两套合在一起 |
| `vcorr.dat` | Lh=6 km，Lv=0.6 km，配 `-SV80`、`-TV20` |

两套对照（同一平滑、同一初值）：

1. **只用 0/1** vs **0/1/4/5**
2. **只用 0/1** vs **只用 4/5**

```bash
python make_recv_peg_inv_case.py

tt_forward -Mtrue.smesh -Ggeom_inv_01.dat -Bseafloor.refl -Fmoho.refl -A \
  -N4/4/0.8/8/1e-4/1e-5 > syn_inv_01.dat
tt_forward -Mtrue.smesh -Ggeom_inv_45.dat -Bseafloor.refl -Fmoho.refl -A \
  -N4/4/0.8/8/1e-4/1e-5 > syn_inv_45.dat
tt_forward -Mtrue.smesh -Ggeom_inv_0145.dat -Bseafloor.refl -Fmoho.refl -A \
  -N4/4/0.8/8/1e-4/1e-5 > syn_inv_0145.dat
python -c "from make_recv_peg_inv_case import set_pick_uncert
for p in ('syn_inv_01.dat','syn_inv_45.dat','syn_inv_0145.dat'): set_pick_uncert(p)"

tt_inverse -Mstart.smesh -Gsyn_inv_01.dat -Yseafloor.refl -Fmoho.refl -w -u -A \
  -N4/4/0.8/8/1e-4/1e-5 -I5 -SV80 -TV20 -CVvcorr.dat -Oout01 -l -Linv01.log
tt_inverse -Mstart.smesh -Gsyn_inv_45.dat -Yseafloor.refl -Fmoho.refl -w -u -A \
  -N4/4/0.8/8/1e-4/1e-5 -I5 -SV80 -TV20 -CVvcorr.dat -Oout45 -l -Linv45.log
tt_inverse -Mstart.smesh -Gsyn_inv_0145.dat -Yseafloor.refl -Fmoho.refl -w -u -A \
  -N4/4/0.8/8/1e-4/1e-5 -I5 -SV80 -TV20 -CVvcorr.dat -Oout0145 -l -Linv0145.log

tt_forward -Mstart.smesh -Ggeom_inv_01.dat -Bseafloor.refl -Fmoho.refl -A \
  -N4/4/0.8/8/1e-4/1e-5 > syn_start_01.dat
tt_forward -Mstart.smesh -Ggeom_inv_45.dat -Bseafloor.refl -Fmoho.refl -A \
  -N4/4/0.8/8/1e-4/1e-5 > syn_start_45.dat
tt_forward -Mstart.smesh -Ggeom_inv_0145.dat -Bseafloor.refl -Fmoho.refl -A \
  -N4/4/0.8/8/1e-4/1e-5 > syn_start_0145.dat
tt_forward -Mout01.smesh.5.1 -Ggeom_inv_01.dat -Bseafloor.refl -Fmoho.refl -A \
  -N4/4/0.8/8/1e-4/1e-5 -Rrays_rec_01.dat > syn_rec_01.dat
tt_forward -Mout45.smesh.5.1 -Ggeom_inv_45.dat -Bseafloor.refl -Fmoho.refl -A \
  -N4/4/0.8/8/1e-4/1e-5 -Rrays_rec_45.dat > syn_rec_45.dat
tt_forward -Mout0145.smesh.5.1 -Ggeom_inv_0145.dat -Bseafloor.refl -Fmoho.refl -A \
  -N4/4/0.8/8/1e-4/1e-5 -Rrays_rec_0145.dat > syn_rec_0145.dat

python check_recv_peg_inv.py --no-show --smesh out01.smesh.5.1 \
  --geom geom_inv_01.dat --obs syn_inv_01.dat --syn-start syn_start_01.dat \
  --syn-rec syn_rec_01.dat --rays rays_rec_01.dat --tag 01 --rec-label 只用0/1
python check_recv_peg_inv.py --no-show --smesh out45.smesh.5.1 \
  --geom geom_inv_45.dat --obs syn_inv_45.dat --syn-start syn_start_45.dat \
  --syn-rec syn_rec_45.dat --rays rays_rec_45.dat --tag 45 --rec-label 只用4/5
python check_recv_peg_inv.py --no-show --smesh out0145.smesh.5.1 \
  --geom geom_inv_0145.dat --obs syn_inv_0145.dat --syn-start syn_start_0145.dat \
  --syn-rec syn_rec_0145.dat --rays rays_rec_0145.dat --tag 0145 --rec-label 0/1/4/5
python check_recv_peg_inv.py --no-show --compare out01.smesh.5.1 out0145.smesh.5.1 \
  --compare-labels 只用0/1 0/1/4/5 --compare-out check_inv_compare_01_vs_0145.png
python check_recv_peg_inv.py --no-show --compare out01.smesh.5.1 out45.smesh.5.1 \
  --compare-labels 只用0/1 只用4/5 --compare-out check_inv_compare_01_vs_45.png
```

绿虚线框是真异常位置。不要只看全壳 RMS：框上 / 框内 / 框下均值更能看出是否被拉成贯穿柱。

台位在 **38–62 km**（夹住高速体 42–58），炮再留 **8 km** 边距。不要用 30/70：其 0/1 ±30 会落到 x=0/100，图方法贴边，边台斜穿还会把框内/框下拧成一次更新。

本仓库已按新几何用 `-SV80 -TV20`、拾取误差 0.05、0/1 ±30、4/5 ±40、Lh=6、**Lv=0.6** 跑过（初值框内 4.420，真 4.920；框上真 4.070，框下真 5.435）。`vcorr.dat` 用 `:g` 写入，不再把 0.35 收成 0.3。

| | **只用 0/1（±30）** | **只用 4/5（±40）** | **0/1+4/5** |
|---|---|---|---|
| 壳 RMS vs 真（全照明区） | 0.201 | **0.136** | 0.187 |
| 相关 | 0.95 | **0.98** | 0.95 |
| 框内均值（真 4.920） | 4.777 | **4.802** | 4.750 |
| 框上（真 4.070） | 4.362 | **4.383** | 4.364 |
| 框下 4.05–8 km（真 5.435） | **5.431** | 5.494 | 5.453 |
| 其中 4–6 km（真 5.095） | 5.363 | 5.377 | **5.329** |
| 训练走时 RMS | 0.120→0.081 s | **0.103→0.038 s** | 0.111→0.063 s |

相对 Lv=0.3：壳场更干净（4/5 0.23→0.14），0145 不再在第 3 步冲高（0.27→0.06）。框内略欠拟合（4.80 vs 先前 4.85）。4–6 km 比 0.3 时更糊（约 +0.25），是加长 Lv 的预期代价。水三套都保持 1.500。图：`check_inv_models_{01,45,0145}.png`、`check_inv_compare_01_vs_45.png`、`check_inv_compare_01_vs_0145.png`。
