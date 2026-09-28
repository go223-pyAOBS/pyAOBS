# TOMO2D 编译与并行运行说明

本文档汇总 `tt_inverse` / `tt_forward`（含 OpenMP 并行）的**编译命令**与**环境变量配置**。

---

## 1) 编译（CMake + OpenMP）

> 推荐使用 `src/CMakeLists.txt`（已接入 `TOMO2D_ENABLE_OPENMP` 选项）。

### 方式 A：在仓库根目录执行

```bash
cmake -S pyAOBS/modeling/tomo2d/src -B build-tomo2d -DCMAKE_BUILD_TYPE=Release -DTOMO2D_ENABLE_OPENMP=ON
cmake --build build-tomo2d -j
```

生成的可执行文件位于：

- `build-tomo2d/tt_inverse`
- `build-tomo2d/tt_forward`
- `build-tomo2d/gen_smesh`
- 以及其它工具程序

### 方式 B：在 `modeling/tomo2d/src` 目录执行

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTOMO2D_ENABLE_OPENMP=ON
cmake --build build -j
```

---

## 2) 运行前环境变量（并行）

最常用配置（示例）：

```bash
export OMP_NUM_THREADS=3
export TOMO2D_INV_OMP=1
```

> **Qt GUI**：顶栏「并行 / 策略环境变量」可设置下列变量，并写入子进程（覆盖同名系统变量）；
> 亦随工区/`env.*` 配置保存。命令行仍可用 `export`。

### 变量含义

- `TOMO2D_INV_OMP=1`  
  启用 `tt_inverse` 的 OpenMP 并行路径（关闭可设为 `0`）。

- `TOMO2D_FWD_OMP=1`  
  启用 `tt_forward` 的 OpenMP source 级并行路径（关闭可设为 `0`）。
  
  说明：
  - `tt_forward` 在 `-A`（full reflection）模式下会自动回退串行；
  - 并行时射线路径输出仍按 source 顺序写入。

- `TOMO2D_GRAPH_FS_ENUM=1`  
  图论 `graph.solve`（含反射/转换）按 forward-star 下标枚举邻居。  
  未设置或 `=0`：回退原策略（扫剩余 C/B 再 `isIn`）。  
  GUI「并行/策略 → 图论FS枚举」勾选写 1，不勾选写 0（可对拍）。

- `TOMO2D_INV_LEGACY_BASELINE=1`  
  启用“回退基线模式”，用于与优化版做 A/B 对比。该模式会同时回退以下三项到旧行为：
  1) 关闭前向复用（不再按模型变化阈值复用 `A/path/res_ttime`）；  
  2) kernel 归并回退为 `vector<pair> + sort + merge`；  
  3) 关闭 LSQR 列预条件（恢复原始求解路径）；
  4) 关闭分阶段反演（coarse-to-fine）。
  
  > 优先级最高：开启后，`TOMO2D_INV_REUSE_*` 与 `TOMO2D_INV_COARSE2FINE` 将被忽略；LSQR 列预条件也被强制关闭。

- `TOMO2D_INV_LSQR_MAXITER=<n>`  
  LSQR 迭代硬上限（与默认 `nnode*100` 取较小值）。≤0 或不设 = 不额外封顶。联合 Vs-only 的 `-TV` 无阻尼探步病态时，用来避免顶满上百万步。

- `TOMO2D_INV_LSQR_ATOL=<tol>`  
  覆盖 LSQR 的 `test2` 阈值。未设则用代码默认 `1e-3`。

- `TOMO2D_INV_LSQR_PRECOND=1`  
  只打开 LSQR 列预条件，不整包加速（hash 核 / 复用 / C2F 仍按各自开关）。  
  默认关闭。`D_j=clip(s_med/‖A列‖, 1/κ, κ)`，空列 `D=0`；`test2` 至少 `MINITER` 步后才许因 ATOL 停下（避免 iter=1 假收敛）。`-TV` 无阻尼探步不加列缩放。

- `TOMO2D_INV_LSQR_PRECOND_MAX`  
  相对列范数中位数的夹逼 `κ`，默认 `10`。`<=0` = 相对不封顶（空列仍为 0）。

- `TOMO2D_INV_LSQR_PRECOND_MINITER`  
  预条件开启后允许 ATOL 停机的最少 LSQR 迭代，默认 `20`。

- `TOMO2D_INV_SENS_WEIGHT=1`  
  灵敏度加权阻尼/平滑（默认关）。按已缩放数据核的列和（DWS）对本块中位数求
  `w_j=clip(s_med/(s_j+ε s_med), 1/κ, κ)`：暗结点 `T` 加大，亮→暗 `R` 耦合改为 `2/(w_i+w_j)`。
  Vp、面上 Vs、面下 Vs、莫霍深度各自一块中位数。用来减弱射线路径拖曳，不针对某一层。
  与 LSQR 列预条件都按照明缩放，减拖曳时优先只开本项。

- `TOMO2D_INV_SENS_KAPPA`  
  相对本块 DWS 中位数的夹逼 `κ`，默认 `10`。

- `TOMO2D_INV_SENS_EPS`  
  DWS 分母稳定项 `ε`，默认 `0.05`。

- `TOMO2D_INV_LINESEARCH=1`  
  Gauss–Newton 步长线搜索（默认关）。LSQR 给出方向后不整步加上去，
  按重追后的真实 χ² 做 Armijo 回退（α=1, ρ, ρ², … 直到 α_min）。
  Vp / Vs / 面上 / 面下 / 莫霍同一套外迭代。扫描多组 `-SV/-SD` 时自动跳过。
  可选：`TOMO2D_INV_LS_C`（Armijo c，默认 `1e-4`）、
  `TOMO2D_INV_LS_RHO`（回退因子，默认 `0.5`）、
  `TOMO2D_INV_LS_AMIN`（最小 α，默认 `0.03125`=1/32）。
  与 `TOMO2D_INV_LM` 同时开时线搜索被忽略。

- `TOMO2D_INV_LM=1`  
  Levenberg–Marquardt 信赖域（默认关）。按重追真实 χ² 与线性预测的比 ρ：
  ρ 差则把阻尼乘 `UP`（默认 4）并重新 LSQR，不是沿原方向缩步长。
  接受且 ρ 好则把 λ 乘 `DOWN`（默认 0.5）。无 `-D/-T` 时自动关。
  可选：`TOMO2D_INV_LM_LAMBDA`、`TOMO2D_INV_LM_UP`、`TOMO2D_INV_LM_DOWN`、
  `TOMO2D_INV_LM_LMAX`（默认 256）、`TOMO2D_INV_LM_RHO_ACCEPT`（默认 0.1）、
  `TOMO2D_INV_LM_RHO_GOOD`（默认 0.5）。

- `TOMO2D_INV_REUSE_FORWARD=1` 与 `TOMO2D_INV_REUSE_THRESH=<阈值>`  
  启用前向复用（默认关闭）。仅在 **未开启** `TOMO2D_INV_LEGACY_BASELINE` 时生效。  
  示例：`export TOMO2D_INV_REUSE_FORWARD=1; export TOMO2D_INV_REUSE_THRESH=1e-3`

- `TOMO2D_INV_COARSE2FINE=1`  
  启用分阶段反演（coarse-to-fine）：前期更强平滑/阻尼，后期逐步放松到目标值。  
  可选参数（均为正数）：
  - `TOMO2D_INV_C2F_SMOOTH_START`（默认 `3.0`）
  - `TOMO2D_INV_C2F_SMOOTH_END`（默认 `1.0`）
  - `TOMO2D_INV_C2F_DAMP_START`（默认 `3.0`）
  - `TOMO2D_INV_C2F_DAMP_END`（默认 `1.0`）

  含义（按迭代从第 1 轮到最后 1 轮）：
  - `*_START`：第一轮的阶段系数；
  - `*_END`：最后一轮的阶段系数；
  - 中间迭代按对数插值平滑过渡（适合跨数量级参数）。
  
  例如默认 `3 -> 1` 表示：前期平滑/阻尼约为原参数 3 倍，后期回落到原参数。
  
  建议起步配置：
  `export TOMO2D_INV_COARSE2FINE=1; export TOMO2D_INV_C2F_SMOOTH_START=3; export TOMO2D_INV_C2F_DAMP_START=3`

- `OMP_NUM_THREADS`  
  OpenMP 线程数。对 source 并行建议不超过 source 数。


---

## 3) 运行示例

```bash
../../../../src/build-tomo2d/tt_inverse \
  -Minputs/mesh.dat -Ginputs/data.dat \
  -N4/4/0.8/8/0.0001/1e-05 \
  -Finputs/refl.dat -W1 \
  -Loutputs/tt_inverse.log -Ooutputs/out -Koutputs/dws.dat \
  -Q0.001 -I5 -J1.0 \
  -SV100 -SD10 -CVinputs/corr_v.dat -CDinputs/corr_d.dat \
  -DV1 -DD20 -DQinputs/damp_v.dat -V-1
```

> `-V-1` 已支持并解析为 `verbose_level=-1`。  
> 若只做性能测试，可不传 `-V`。

---

## 4) 诊断开关（可选）

若需比较串行/并行是否从某一迭代开始分叉，可打开诊断哈希：

```bash
export TOMO2D_INV_DIAG=1
```

会输出每迭代与每个 `iset` 的 hash（`hash_res/hash_A/hash_dmodel/hash_modelv/hash_modeld`），用于定位差异来源。

### 监视通道（status.jsonl）

每完成一轮 `iter×iset` 追加一行 NDJSON（需重新编译含该改动的 `tt_inverse`）：

```bash
export TOMO2D_INV_STATUS_JSONL=outputs/status.jsonl
```

字段示例：`iter` `iset` `rms` `chi2` `pred_chi` `dv_norm` `dd_norm` `rough_v` `rough_d` `is_final` `smesh`。  
GUI 默认经 `env.inv_status_jsonl_path` 写入该变量；监视窗优先用其刷新末态。留空则关闭。

---

## 5) `-DV` 与 `-DQ` 如何共同决定阻尼

- `-DV<wdv>`：固定速度阻尼的全局系数（标量）
- `-DQ<damp_v_file>`：速度阻尼的空间权重场 `w(x,z)`（squeezing）

在源码中的顺序是：

1. 先在 `calc_damping_matrix()` 中把局部速度阻尼核乘以 `w(x,z)`；
2. 再在 `_solve()` 中由 `wdv` 对整个速度阻尼块做全局缩放。

因此可近似理解为局部等效阻尼强度：

`damp_local ~ wdv * w(x,z)`

注意：

- `-DQ` 只作用于速度阻尼，不作用于 `-DD`；
- 仅给 `-DQ` 无法生效，当前程序会直接报错（要求 `-DV>0`）。

---

## 6) 常见问题

### Q1: `libgomp: Invalid value for environment variable OMP_NUM_THREADS`

`OMP_NUM_THREADS` 值非法（空串、非数字等）。请显式设置为整数，例如：

```bash
export OMP_NUM_THREADS=3
```

### Q2: 编译日志看不到 `OpenMP enabled for tomo2d_core`

该提示通常出现在 **cmake 配置阶段**（`cmake -S ... -B ...`），不是 build 阶段。  
可检查 `CMakeCache.txt` 中 `OpenMP_CXX_FOUND` 是否为 `TRUE`。

### Q3: 想用 Makefile，而不是 CMake

`src/Makefile` 已支持 OpenMP 与单目标编译。进入 `modeling/tomo2d/src` 后可直接：

```bash
make -j
```

只编译某个可执行程序：

```bash
make tt_inverse
make tt_forward
make gen_smesh
```

只编译某个对象文件：

```bash
make inverse.o
make tt_inverse.o
```

关闭 OpenMP（临时）：

```bash
make tt_inverse USE_OPENMP=0
```

清理并重编：

```bash
make clean
make -j tt_inverse
```

