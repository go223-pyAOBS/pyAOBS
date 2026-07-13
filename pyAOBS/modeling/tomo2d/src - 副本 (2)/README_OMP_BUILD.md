# TOMO2D 编译与并行运行说明

本文档汇总 `tt_inverse`（含 OpenMP 并行）的**编译命令**与**环境变量配置**。

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

## 5) 常见问题

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

