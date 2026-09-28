# 直达水波 / 一阶多次 — tt_forward 最小正演

背景为均匀水 `v=1.5 km/s`，平底水深 `H=2 km`，`topo` 全 0（网格从海面挂到 4 km；`z≤2` 为水，以下为 **1.8 km/s** 浅沉积）。水柱叠加空间相关随机扰动（默认 RMS 5%、水平相关 6 km、垂向 0.4 km、seed=42），沉积不扰动。解析解仍按均匀 1.5，只作对照。

解析解（只用于验收，不是正演算法）：

- raytype **2**（直达）: \(t=\sqrt{\Delta x^2+(H-z_\mathrm{shot})^2}/v\)
- raytype **3**（多次）: \(t=\sqrt{\Delta x^2+(3H-z_\mathrm{shot})^2}/v\)
- 零偏移: \(t_3-t_2=2H/v=2.667\,\mathrm{s}\)（与炮深无关）

本仓库已用当前 `tt_forward` 跑通：stderr 为 `wwwwwwmmmmmm`（6 条直达 + 6 条多次）。零偏移与解析解差约 **7 ms**（直达 1.327 s / 多次 3.993 s）。中等偏移的多次可能比 3H 展开差几百 ms，先看 `-R` 射线是否炮→海底→海面→台；零偏移和直达是第一道关。

对照：跑完后在本目录执行

```bash
python check_analytic.py
```

会打印走时表，弹出 **T–X 走时图** 和 **射线剖面**，并写成 `check_ttimes.png` / `check_rays.png`。无窗口时加 `--no-show` 只存图。射线底图与 GUI 绘制 smesh 相同：`water.smesh` 速度场 + 内置 Vp 色标，叠 `seafloor.refl`。

## GUI

工作目录指到本目录。`tt_forward` 页：

| 字段 | 文件 |
|------|------|
| smesh (-M) | `water.smesh` |
| geom (-G) | `geom_water.dat` |
| refl_file (-F) | `seafloor.refl` |
| out_ttime | `syn_water.dat`（stdout 走时，可当反演 -G） |
| out_ray (-R) | `rays_water.dat`（看折返） |

不要勾 `do_full_refl (-A)`。跑完用「预览 ttimes」看 2/3；零偏移附近 \(t_2\approx 1.333\,\mathrm{s}\)、\(t_3\approx 4.000\,\mathrm{s}\)。

对照 `analytic.txt`。射线应贴海底与海面，水相应停在 `z≤2`；界面以下是 1.8 km/s 沉积，反差小，弯曲不容易往下「抄近道」。

## 命令行（WSL）

```bash
tt_forward -Mwater.smesh -Ggeom_water.dat -Fseafloor.refl \
  -N4/4/0.8/8/1e-4/1e-5 -Rrays_water.dat > syn_water.dat
```

重新生成输入：

```bash
python make_water_fwd_case.py
```

水核反演（冻壳只反水）见同级目录 `../water_inv/`。用本目录扰动场当真模型、均匀 1.5 当初值见 `../water_inv2/`。
