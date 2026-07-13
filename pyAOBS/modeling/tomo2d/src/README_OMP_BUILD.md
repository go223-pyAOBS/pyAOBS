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

### 变量含义

- `TOMO2D_INV_OMP=1`  
  启用 `tt_inverse` 的 OpenMP 并行路径（关闭可设为 `0`）。

- `TOMO2D_FWD_OMP=1`  
  启用 `tt_forward` 的 OpenMP source 级并行路径（关闭可设为 `0`）。
  
  说明：
  - `tt_forward` 在 `-A`（full reflection）模式下会自动回退串行；
  - 并行时射线路径输出仍按 source 顺序写入。

- `TOMO2D_INV_LEGACY_BASELINE=1`  
  启用“回退基线模式”，用于与优化版做 A/B 对比。该模式会同时回退以下三项到旧行为：
  1) 关闭前向复用（不再按模型变化阈值复用 `A/path/res_ttime`）；  
  2) kernel 归并回退为 `vector<pair> + sort + merge`；  
  3) 关闭 LSQR 列预条件（恢复原始求解路径）；
  4) 关闭分阶段反演（coarse-to-fine）。
  
  > 优先级最高：开启后，`TOMO2D_INV_REUSE_*` 与 `TOMO2D_INV_COARSE2FINE` 将被忽略。

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

