# 地震数据去噪算法分析与实现路径（WSST/SSQ + GCV + 形态学，支持剖面与选道）

## 一、总体目标

在 `pyAOBS/processors/denoise/` 中实现一套可复用的时频域去噪算法，并通过稳定 API 暴露给 `zplotpy`：

- **单道级**：对一条地震道做高质量去噪与时频可视化；
- **剖面级**：对二维剖面（若干道 × 采样点）进行批量去噪；
- **选道机制**：用户在 `zplotpy` 中手动选择“某几道”或“全部道”，仅对选择的道应用去噪，其余保持不变；
- **非破坏性**：默认不覆盖原始数据，而是以新数据层 / 内存副本形式返回。

当前代码中：

- `denoise_trace(...)` 已经完成 **单道 WSST/SSQ + GCV** 去噪的可用实现；
- `denoise_section(...)` 目前还是“参数验证 + pass-through”的占位实现，需要按照本文档给出的步骤扩展为真实剖面去噪。

---

## 二、算法核心流程

单道去噪的目标：在保留有效反射事件和相位结构的前提下，抑制宽频随机噪声和孤立能量。

整体流程分为三步（其中 Step 1–2 已在代码中落实，Step 3 预留）：

1. **时频表示（WSST / 同步压缩 CWT）**
   - 输入：时域信号 \(x(t)\)、采样间隔 `dt`。
   - 输出：高分辨率复数时频表示 \(T(f, t)\)，以及频率轴 `freq`。
   - 在实现上：
     - 优先使用 `ssq_backend.ssq_forward()`（基于 `ssqueezepy.ssq_cwt`）；
     - 若缺少依赖，则回退到 `wsst_backend.wsst_like_sst()` 自研后端。

2. **GCV 自适应收缩（频域统计抑噪）**
   - 思路：在每个时间帧上，对频率向量 \(T(\cdot, t)\) 估计最优收缩强度，使残差在统计意义上接近噪声。
   - 在实现上：
     - 调用 `gcv_threshold.gcv_shrink_per_frame()`，输出与 \(T(f,t)\) 同形状的权重矩阵 `weights`；
     - 得到加权时频图 \(T_{\text{gcv}}(f,t) = T(f,t) \cdot w(f,t)\)。

3. **形态学与连通性筛选（结构约束，规划中）**
   - 在 \(T_{\text{gcv}}(f,t)\) 上构造掩膜：
     - 计算能量图 \(|T_{\text{gcv}}(f,t)|^2\)；
     - 通过自适应阈值、连通域分析、开闭运算等形态学操作过滤孤立小块；
     - 最终得到二值或灰度掩膜 \(M(f,t)\)，与 \(T_{\text{gcv}}\) 相乘得到 \(T_{\text{final}} = M \odot T_{\text{gcv}}\)。
   - 当前代码里尚未完全落地该步，可在后续通过单独的 `morphology_mask.py` 模块补齐。

4. **逆变换重建**
   - 将加权后的时频表示 \(T_{\text{final}}(f,t)\) 逆变换回时域：
     - `ssq_backend.ssq_inverse()`：基于 `issq_cwt`；
     - 或 `wsst_backend.reconstruct_from_weights()`：将权重映射回原始 STFT 后再做 iSTFT；
   - 输出去噪后的时域信号 \(\hat{x}(t)\)。  

---

## 三、与现有代码的对应关系

当前 `pipeline.py` 中的核心设计：

- `denoise_trace(...)`
  - 校验参数：`dt, f_s, f_e, bwconn, strength` 等（`_validate_common`）。
  - 决定后端：
    - 若 `ssq_backend.HAVE_SSQ` 为真：
      - `ssq_res = ssq_backend.ssq_forward(...)` 得到 `tf, freq`；
      - `gcv = gcv_shrink_per_frame(ssq_res.tf, strength=...)`；
      - `tf_weighted = ssq_res.tf * gcv.weights`；
      - `y = ssq_backend.ssq_inverse(tf_weighted, ssq_res.meta)`；
      - 调试输出：`org_tf = ssq_res.tf`，`gcv_tf = tf_weighted`。
    - 否则使用 `wsst_backend.wsst_like_sst(...)` 与 `reconstruct_from_weights(...)`。
  - 返回策略：
    - 默认（`return_debug=False` 且没有 `return_result=True`）：只返回 `np.ndarray`，即去噪后的 1D 波形 `y`；
    - 若 `return_debug=True` 或 `return_result=True`：返回 `DenoiseResult`，其中：
      - `data = y`（去噪波形）；
      - `debug.org_tf / gcv_tf / final_tf / freq` 填好，供 `zplotpy` 绘制。

- `denoise_section(...)`
  - 已做：
    - 参数与数组形状检查；
    - 保证输入为二维 `traces`（`n_traces, n_samples`）；
    - 当前仅返回 `Y = X.astype(float64, copy=True)`，即 **尚未真正去噪**。
  - 预留：
    - `workers` 参数用于未来的多进程/多线程并行；
    - 可以通过 `kwargs["return_result"]` 返回 `DenoiseResult`。

结论：**单道算法已基本实现；剖面与选道逻辑需要在 `denoise_section` 与 `zplotpy` 交互层补齐。**

---

## 四、zplotpy 调用场景与选道需求

从 `zplotpy` 视角，典型剖面去噪使用场景包括：

1. **单道试算**
   - 用户在剖面图上点选某一条道（或多条道）进行试算；
   - 快速查看“原始 vs 去噪”波形以及时频图，微调参数。

2. **选中道批量去噪**
   - 用户框选 / 列表中勾选若干条道（例如 10 条）；
   - 对这些选中的道应用相同参数的去噪，便于对比效果。

3. **全剖面去噪**
   - 对当前剖面中的全部道执行去噪；
   - 通常结合 `workers` 做并行计算，付出较多时间但获得整体更干净的剖面。

UI 侧的参数输入建议：

- `f_s, f_e`：去噪保留频带下限 / 上限（Hz）；
- `strength`：抑噪强度（建议映射为“保守 / 标准 / 激进”三档）；
- `bwconn`：连通域邻域（4/8，GUI 可默认 8）；
- “仅选中道” vs “全部道”：由用户勾选。

---

## 五、剖面去噪与选道的实现步骤

下面给出从当前状态到“可被 `zplotpy` 调用、支持选道/全剖面”的**完整实现路径**，分成四个层次：

### 5.1 层次 A：最小可用实现（利用现有 `denoise_trace` 循环）

这一层不需要改动 `denoise_section` 内部逻辑，只在 `zplotpy` 侧做循环：

1. **道选择**（`zplotpy` 内部完成）
   - `indices_selected`: 用户通过交互选择的一维整型数组/列表，表示需要去噪的道索引；
   - `use_all_traces`: 若用户选择“全部道”，则 `indices_selected = range(n_traces)`。

2. **数据准备**
   - `traces`：形状为 `(n_traces, n_samples)` 的 numpy 数组；
   - `dt`：采样间隔（秒）；
   - 创建输出数组 `out = traces.copy()`，用于存放去噪结果。

3. **循环调用单道去噪**
   - 对于每个 `i` in `indices_selected`：
     - 取出当前道 `trace_i = traces[i, :]`；
     - 调用：
       - 仅要结果波形：`y = denoise_trace(trace_i, dt, f_s=..., f_e=..., strength=..., return_debug=False)`；
       - 需要展示时频：`res = denoise_trace(..., return_debug=True)`，`y = res.data`。
     - 将 `y` 写回 `out[i, :]`。

4. **结果使用**
   - 将 `out` 作为新剖面数据源显示在 `zplotpy` 中；
   - 保留原始 `traces` 作为“原始层”，方便前后对比。

> 这一层即可满足“剖面数据去噪 + 用户手动选道”需求，无需等待 `denoise_section` 内部优化。

### 5.2 层次 B：在 `denoise_section` 中封装选道与并行（推荐下一步）

为减少 `zplotpy` 中的重复逻辑，可在 `processors/denoise/pipeline.py` 中扩展 `denoise_section`：

1. **接口扩展建议**

```python
def denoise_section(
    traces,
    dt,
    *,
    f_s,
    f_e,
    bwconn=8,
    workers=1,
    indices=None,           # 新增：可选，一维索引列表/数组
    return_debug=False,
    **kwargs,
):
    ...
```

- `indices is None`：默认对全部道进行去噪；
- `indices` 为一维索引序列：只对这些道运行去噪，其余道原样拷贝。

2. **内部实现思路**

在现有形状与数值检查之后：

1. 规范化 `indices`：
   - 若 `indices is None`：生成 `np.arange(n_traces)`；
   - 否则将 `indices` 转为一维 `np.ndarray`，并做越界检查。
2. 构造输出数组 `Y = X.astype(np.float64, copy=True)`（保留未选中的道）；
3. 根据 `workers` 决定串行 / 并行：
   - `workers == 1`：简单 for 循环，调用现有 `denoise_trace(...)`；
   - `workers > 1`：后续可选用 `joblib` / `concurrent.futures` 等做并行（P3 阶段实现）。
4. 返回策略：
   - 若不需要 debug：直接返回 `Y`；
   - 若需要 debug 或 `kwargs.get("return_result", False)`：构造 `DenoiseResult`，其中 `data=Y`。

> 这样 `zplotpy` 只需要传入 `indices` 和 `workers`，无需自己循环。

### 5.3 层次 C：形态学约束与更强抑噪（可选后续增强）

在层次 B 基础上，为了进一步抑制孤立噪声而不过度抹平连贯反射，可在单道或剖面级别引入形态学约束：

1. 在 `morphology_mask.py` 中实现：
   - `build_mask_from_tf(tf, freq, params) -> mask`：返回与 `tf` 同形状的掩膜；
   - `params` 可包括最小连通域面积、结构元素大小等。
2. 在 `denoise_trace` 中：
   - 在 GCV 后增加一步 `mask = build_mask_from_tf(tf_gcv, freq, params)`；
   - 得到 `tf_final = tf_gcv * mask`，用于逆变换。
3. 在 `denoise_section` 中：
   - 保持接口一致，只是每道内部多一步形态学处理。

形态学参数可在 `zplotpy` 中提供三档预设，对应不同抑噪强度。

---

## 六、zplotpy 侧典型调用流程（伪代码）

### 6.1 单道试算（当前即可使用）

```python
from pyAOBS.processors.denoise import denoise_trace

trace = section[i_trace, :]   # 选中的单道
res = denoise_trace(
    trace,
    dt,
    f_s=1.0,
    f_e=60.0,
    bwconn=8,
    strength=1.0,
    return_debug=True,        # 需要时频图
)
denoised = res.data
```

### 6.2 剖面选道去噪（层次 A：zplotpy 循环）

```python
import numpy as np
from pyAOBS.processors.denoise import denoise_trace

X = section          # (n_traces, n_samples)
n_traces = X.shape[0]
indices_selected = user_selected_indices()  # 例如 [0, 3, 5] 或 range(n_traces)

Y = X.copy().astype(float)

for i in indices_selected:
    trace_i = X[i, :]
    y_i = denoise_trace(
        trace_i,
        dt,
        f_s=f_s,
        f_e=f_e,
        bwconn=bwconn,
        strength=strength,
        return_debug=False,
    )
    Y[i, :] = y_i

# 在 zplotpy 中使用 Y 作为去噪后的剖面数据
```

### 6.3 剖面选道去噪（层次 B：使用未来的 `denoise_section`）

一旦按 5.2 中的思路扩展了 `denoise_section`，`zplotpy` 侧可以简化为：

```python
from pyAOBS.processors.denoise import denoise_section

indices_selected = user_selected_indices_or_none()

Y = denoise_section(
    section,
    dt,
    f_s=f_s,
    f_e=f_e,
    bwconn=bwconn,
    workers=workers,           # 1 表示串行
    indices=indices_selected,  # None 表示全部道
)
```

---

## 七、落地 checklist（面向实现）

结合当前代码状态，要实现“zplotpy 可调用、支持选道或全部道的剖面去噪”，建议按如下顺序推进：

1. **确认单道接口稳定**
   - [x] `denoise_trace` 参数与返回类型固定（已基本完成）；
   - [x] 在简单示例脚本中验证单道去噪效果。

2. **在 zplotpy 中接入单道试算**
   - [ ] 提供 UI 参数面板与“单道试算”按钮；
   - [ ] 调用 `denoise_trace(..., return_debug=True)` 并绘制时频图。

3. **实现层次 A 的选道剖面去噪（不改 `denoise_section`）**
   - [ ] 在 zplotpy 侧实现“选中道列表 + for 循环调用 `denoise_trace`”；
   - [ ] 支持“仅选中道 / 全部道”开关。

4. **扩展 `denoise_section`（层次 B）**
   - [ ] 在 `pipeline.py` 中实现按 `indices` 去噪的逻辑；
   - [ ] 保持与现有 API 兼容（`workers`、`return_debug` 等）。

5. **（可选）引入形态学增强与并行优化（层次 C）**
   - [ ] 实现 `morphology_mask.py` 并接入 `denoise_trace` / `denoise_section`；
   - [ ] 对典型数据集做性能与效果评估。

只要完成第 1–3 步，`zplotpy` 即可基于当前实现对剖面数据进行去噪，并支持用户手动选择某些道或所有道；第 4–5 步属于后续优化与增强。

