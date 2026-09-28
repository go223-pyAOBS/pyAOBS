# 折合 PSP — tt_forward `-X`

混合慢度网格：转换面 **以上是 P**（水 `1.5` + 沉积盖层 `Vp=1.80+0.45(z-2)`，界面处约 3.15），**以下是壳幔 Vs**（`3.40+0.12(z-5)`）。`topo` 全 0。平底水深 `H=2 km`，转换面 `Zc=5 km`。转换点按 **整条射线最短** 选，不是炮/台正下的最近点。连续介质里最短路径的结果常接近 Snell，代码并不强制 \(p_P=p_S\)。

| raytype | 含义 |
|---------|------|
| **0** | Fermat 初至。短偏移走水柱；中长偏移在**混合网格**里会用面下数字（此处是 Vs）。Vs ≳ 盖层底 Vp 时，长偏移 `t0≈t6`，不是「壳内快 P」 |
| **6** | 折合 PSP：炮侧 P↓ → 转换面 → 界面下 S → 再过面 → 盖层+水柱 P↑ 到 OBS。须 `-X`。转换点 B、C 由整条 `t_P+t_S+t_P` 最短选出，不是炮/台正下 |

零偏移参考（盖层用线性梯度积分，不是均匀 Vp）：

\[
t_0 = H/v_\mathrm{w},\quad
t_6 \approx H/v_\mathrm{w} + 2\ln(V_\mathrm{p}(Z_\mathrm{c})/V_\mathrm{p}(H))/g
\]

约 `1.327 s` 对 `3.81 s`。`analytic.txt` 的 t6 按界面 Vs0 水平展开、不含下潜与斜 P；正演会略快。`check_converse_fwd.py` 事后打印转换点的 `p=\sin i/v`：联合最短时连续介质里常有 `pP≈pS`（Snell 是结果，不是选点约束）。

对照：跑完后在本目录执行

```bash
python check_converse_fwd.py
```

打印走时表与事后慢度对照，写出 **T–X** 与 **射线剖面**（`check_ttimes.png` / `check_rays.png`）。无窗口加 `--no-show`。射线底图是 `converse.smesh`：盖层看 Vp、面下看 Vs。6 必须两过 `z=5`；长偏移 S 下潜，短偏移弯曲可能把 S 贴在转换面上（Vs ≳ 盖层底 Vp 时的 Fermat 捷径）。射线按相位上色：**P 蓝**、**S 红**。默认开弯曲、前向星 `8/8`（不要 `-g`）。

## GUI

工作目录指到本目录。`tt_forward` 页：

| 字段 | 文件 |
|------|------|
| smesh (-M) | `converse.smesh` |
| geom (-G) | `geom_conv.dat` |
| conv_file (-X) | `conv.refl` |
| out_ttime | `syn_conv.dat` |
| out_ray (-R) | `rays_conv.dat` |

不要勾 `do_full_refl (-A)`。不要勾 `graph_only (-g)`（要弯曲）。前向星用 `8/8`。跑完用「预览 ttimes」看 0/6。

## 命令行（WSL）

```bash
python make_converse_fwd_case.py

tt_forward -Mconverse.smesh -Ggeom_conv.dat -Xconv.refl \
  -N8/8/0.8/8/1e-4/1e-5 -Rrays_conv.dat > syn_conv.dat

python check_converse_fwd.py --no-show
```

也可在 WSL 下执行 `bash run_wsl.sh`（需本仓库 `src/build-tomo2d` 里已编好的 `tt_forward`）。

只反转换面以下 Vs、冻盖层见同级 `../converse_inv/`。
