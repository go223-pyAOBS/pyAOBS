# 台侧一阶 peg-leg — tt_forward

OBS 在海底、炮在海面（与 `water_fwd` 相同的 s/r 约定）。水柱均匀 1.5 km/s，海底到莫霍壳速递增，莫霍以下地幔 8.0 km/s。

| raytype | 含义 |
|---------|------|
| 0 | 折射 |
| 4 | 折射的**台侧一阶**（OBS 侧海面弹跳后再走壳） |
| 1 | **莫霍反射**（`-F` = 莫霍） |
| 5 | 莫霍反射的台侧一阶（`-B` 海底 + `-F` 莫霍） |

同一偏移上 `t4 − t0` 与 `t5 − t1` 大约 `2H/v ≈ 2.667 s`（镜像台，垂直近似）。

```bash
python make_recv_peg_case.py
tt_forward -Mcrust.smesh -Ggeom_peg.dat -Bseafloor.refl -Fmoho.refl -A \
  -N4/4/0.8/8/1e-4/1e-5 -Rrays_peg.dat > syn_peg.dat
python check_peg.py --no-show
```

写出 `check_peg_ttimes.png`（正演圆点 vs 层状解析虚线：0/1 一次波，4/5 台侧多次）和 `check_peg_rays.png`。

水层 2/3 仍可只给 `-F` 当海底（与 `water_fwd` 兼容）。反演水核见 `example_water/water_inv`（`-Y`/`-y`），壳核见 `example_water/crust_inv`（`-Y`/`-w`）。4/5 进核验收见 `example_water/recv_peg_inv`（`-Y`/`-F`/`-w`/`-u`）。
