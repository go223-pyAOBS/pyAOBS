# 用水波反演水速（策略）

**已确认（2026-08-26）：** `topo=0`，海底作独立界面（正演 `-B` / 反演 `-Y`；反演 `-B` 仍是转换波）；水进 `vgrid`。直达 / 一阶多次走独立水域图论。不用 3H 镜像造路径；水相不 `add_kernel_refl`。

**从 v.in 生成 topo=0 网格：** `gen_smesh -S`（GUI：挂海面 topo=0）。`ilayer` 为海底层；可选 `-G` 写出海底给 `-B`/`-Y`。不勾 `-S` 时 zelt 仍把该层写入 topo（原行为）。

## 隔离（保证现有 0/1 正反演不变）

默认作业（data 里只有 raytype 0/1，不加 `-u`，`tx2tomo2d` 不填水分相）与改之前同一条路：

| 约束 | 做法 |
|------|------|
| `solve` / `solve_refl` / `pickPath*` 函数体 | 不改逻辑 |
| 水相图论 | 独立 `prev_w0/w1/w2`、`ttime_w*`、`B_w*`、`C_w`，不覆盖地壳的 `prev_node*` / `C` |
| 何时调用新图论 | 该炮出现 raytype 2 或 3 才 `solve_water`；该炮有 0 或 1 时仍先 `graph.solve` |
| 仅水相的炮 | **跳过** `solve()`，避免台在水中触发 `unsupported source` |
| `-L` 两槽 rms | 仍只统计 0/1，列格式不变；总 rms 把 2/3 算进去（仅有 0/1 时与原来相同） |
| 冻结 `-F` | **`tt_inverse -u`，默认不加**。无 `-u` 时 `-F` 仍会更新界面 |
| `tx2tomo2d` | `water_phases` / `mult_phases` 默认空 |
| GUI | 转换页水分相默认空；反演页「冻结 -F (-u)」默认不勾 |
| 不改 | `inverse_old.cc` / `inverse_new.cc`、vedit 侧栏 |

用法：data 里 raytype **2**=直达水波、**3**=一阶水柱多次；必须 `-F` 海底；只反水速时加 `-u`。不要把 Pw 放进 `tx.refr_phases`。最小正演工区：`examples/water_fwd/`（`tt_forward` + 解析解对照）。

相关：`src/graph.cc` `solve_water` / `solve_water_mult`、`src/inverse.cc` `add_kernel` vs `add_kernel_refl`。

## 现网为什么反不了水速

- 水速是 smesh **文件头标量**（`gen_smesh -Q`，默认 1.5）。`atWater()` 返回 `p_water`，没有 `setWater()`。
- `add_kernel` 只在 `locateInCell>0` 时打 `vgrid`。水点在 `topo` 之上，核≈0。把 Pw 当 Pg 喂进去，残差会乱灌地壳，水速仍不动。
- **禁止**把直达水波放进 `tx.refr_phases`。那会走 `pickPathThruWater`（水腿+壳内初至）。

## 直达水波怎么定 v

均匀水柱射线是直线。炮 `(x_s, 0)` 到台 `(x_r, H)`：

```
t = p_water · L + τ
```

`L` 为直线距离（海底起伏大、直线会穿进固体时，改成沿 `bathyp` 折线）。**t–L 图：斜率 = 慢度，截距 = 该台钟差。** 近偏移到约 2–3 倍水深就够同时估 `p` 和 `τ`。

## 多次在这里干什么

一阶水柱多次路径约 `L+2H`，相对直达多 `2H/v`。`t3−t1` 对钟差不敏感，专门约束 `H/v`。若 `topo` 有系统偏差，直达和多次会给出两套不一致的 v——这才是多次的价值。

偏移距已经够、水深可信时，多次只做 QC，不必进核。peg-leg（地壳+水弹跳）对反水速不是刚需。

**不要**把二维水当成「关掉 in_water 就行」。必须先铺负 `zpos` 水层，再让核打到这些结点。


## 推荐做法：先水后壳

1. `tx.water_phases` → raytype **2**（与 0/1 分开）。
2. **Python t–L**（先不改 `tt_inverse`）：每台拟合 `v` 与 `τ`。验收：v 约 1.48–1.54 km/s；点在直线上。
3. 有一阶多次则检查 `t3−t1 ≈ 2H/v`。不一致先查相位 / `topo`。
4. 把 `v_water` 写回 smesh 头（`-Q`），再跑现有 Pg/PmP。Pg 水腿用对了的 `p_water`，地壳核不被水波污染。

需要进正演/同一套迭代时：`syngen`/`inverse` 支持 code 2（全水路径，**不用图论**）；LSQR 多 **1** 个未知数 `p_water`（下标 `nnodev+nnoded+1`），`A[i,j_w]=L_water`；`outMesh` 已会写 `1/p_water` 到头，但要补 `setWater`。联合反演（Pg 打网格、Pw 只打 `j_w`）以后再说。

## 水层当速度节点（地形固定）

这是「二维水进 vgrid」：水柱结点与壳层同一套 `at()` / `add_kernel` / LSQR。

**地形本来就不变。** `tt_inverse` 未知数只有 `nnodev` 个慢度（+ 可选 `-F` 界面）。`topo` 从不进 `A`。不要用 `-F` 去反海底。

**现网还做不到**，因为网格挂在海底：`z = topo(i)+zpos(k)`，`zpos` 通常从 0 起。水柱在网格上方。`in_water` 时 `at()` 返回头 `p_water`，`locateInCell` 返回 -1，核为 0。`gen_smesh -W` 是相对海底的等厚盖层，不是 0→海底的真实水柱。GUI `air_water_node_mask` 会冻结 `zpos<0` 和 `v≈v_water`。

做法：

1. `zpos` 增加负数层，使 `topo+zpos` 从约 0 铺到海底；浅水多出的层穿到 `z<0` 当空气冻结。`topo` 不动。
2. 水点：`at()` / `locateInCell` 走 `vgrid`（加开关，旧网格行为保持）。空气仍短路。
3. 平滑在 `zpos=0`（海底）断开，避免 1.5 渗进地壳。水层更长水平相关长度。
4. code 2/3 对水结点求核。远偏移图论初至是 Pg，Pw 仍要显式路径。
5. 验收：冻壳只反水；再确认迭代前后 `topo` 数组不变。

### 多次波（负 zpos 上）

只反水速、数据只有 code 2 时，不跑图论，也不存在「Pg 被收成 Pw」。那句话指的是**事后同一份网格跑地壳**：若 `inWater()` 被整段关掉，炮点会从 `pickPathThruWater` 改走 `pickPath`，图论若含负 `zpos`，近偏移最短路才是水波。插值关掉水短路即可；路径分支仍认 `inWater(炮)`；图论继续只走 `zpos≥0`。

不要用 `solve_refl` 找水多次（源已在海底时用 `bathyp` 当反射面是退化的；多次也不是最短路）。

| code | 路径 | 钉点 | 核 |
|------|------|------|-----|
| 2 | 炮(z≈0)→台(zpos=0)，1 段水 | 台端 `bathyp`；炮端 z=0 | 只打水结点 |
| 3 | 炮→海底→海面→台，3 段水 | `bathyp` + 新建 `Interface2d(z=0)`，复用 `bend.refine` 多界面 | 只打水结点 |
| 0 | ThruWater：水腿 + 壳内图论 | 入水点仍钉 `bathyp` | 水腿打水，壳内打壳 |

海面**不是**某一层 `k`（`zpos=-topo(i)` 随 x 变），必须用 z=0 界面钉住，防止弯进空气。海底仍是 `zpos=0` 格面，钉 `bathyp` 即可。初值：直达用直线（禁止穿 `zpos>0`）；多次用均匀 1.5 的镜像源。

`pickWaterBouncePath` 不调 `graph`。同一炮集 code 0 仍 `solve`（仅壳结点）。平滑在 `zpos=0` 断开。验收：射线折返贴 `bathyp` 与 z=0，不进入 `zpos>0`；`t3-t1≈2H/v`。

peg-leg（壳+水弹跳）先不做。

## 备选：topo 全 0，海底当固定界面

网格改挂在海面：`z = zpos`（绝对深度）。`in_water` 是 `z < topo`，topo=0 时水柱不再短路，浅部 `vgrid` 就是水，核自然能打到。

**海底不要占用 `-F`。** `addRefl` 会设 `nrefl=1`、深度进 LSQR，`add_kernel_refl` 写深度导数；而且全程序只有一个 `reflp`，占了就没有莫霍。要界面不动：做成今天的 `bathyp`（只钉射线）。注意 `bathyp = Interface2d(smesh)` **复制的是 topo**——topo 全 0 后它会变成海面，必须改为从水深文件读入。

代价：现网把网格挂在海底，正是为了让水–固落在格面上。topo=0 后海底斜切格子，1.5 与 3+ 会在同一 cell 里被双线性平均。莫霍用 `-F` 能凑合是因为两边都是岩石；水–固反差大一个量级。平滑也必须沿界面断开，不能按整列 `k`。不要 `-A`（会把界面以下改成水速）。

相对负 `zpos` 方案：概念更干净，对现网入侵更大。固定海底、只反水速时，仍更贴现网的是负 `zpos`（海底保持格面，`topo` 本就不进 `A`）。

## 选定：topo=0，海底 -F 固定；多次接反射图论

不用 3H 镜像、不用 `2H/v` 造路径（那只是平坦近偏移的粗检）。

一阶多次 = **现有 `solve_refl` 再接一次海面下行**：

1. **下行到海底**：图论源用炮 `(z=0)`。结点只收 `z≤z_F`（水）。即 `solve_refl` 的 `B_down`。
2. **海底反射后向上**：从全部 `-F` 结点以 `ttime_down` 为初值搜上行。即现成 `B_up`。
3. **碰到海面再向下**：把 `z=0` 结点上的 `ttime_up` 当种子，在水里再 Dijkstra 到台。结构复制 `B_down`，种子不是单个 src。

回溯：台 → `prev_dn2` → 海面（钉 `bathyp`）→ `prev_up` → 海底（钉 `reflp`）→ `prev_down` → 炮。然后现有 `bend.refine`。

无约束 `solve()` 找不到多次。强制过界面，与 Moho 反射同一思想。域停在 `-F` 以上，避免变成 Pg。

图论源按你说的顺序是炮；从台反转三条腿物理等价，每个台只解一次更省。台在 `-F` 上当 code 1 源才是退化（变成直达）。

水相不 `add_kernel_refl`，不 `reflp->set`。验收看折返贴界面，不看是否等于 `2H/v`。

## 不要先做

无约束 `solve()` 里找多次；用 3H 镜像当路径；水相调用 `add_kernel_refl`。把 Pw 当 Pg 喂现网；用声波 FD/FWI 替换当前走时核。

