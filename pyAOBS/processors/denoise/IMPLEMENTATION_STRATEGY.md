# 去噪模块落地策略（processors/denoise）

## 1. 目标

在 `pyAOBS/processors/denoise/` 落地可复用的去噪处理链，并通过稳定 API 提供给 `zplotpy` 调用。

核心原则：

- 算法在 `processors`，交互在 `zplotpy`；
- API 稳定优先于内部实现细节；
- 先单道可用，再批处理优化。

## 2. 模块边界

推荐文件布局：

```text
pyAOBS/processors/denoise/
├── __init__.py
├── ALGORITHM_ANALYSIS.md
├── IMPLEMENTATION_STRATEGY.md
├── types.py
├── wsst_backend.py
├── gcv_threshold.py
├── morphology_mask.py
├── pipeline.py
└── metrics.py
```

- `wsst_backend.py`：正/逆 WSST 统一封装。
- `gcv_threshold.py`：GCV 阈值估计与收缩函数。
- `morphology_mask.py`：连通域筛选、形态学掩膜。
- `pipeline.py`：串联流程与主入口。
- `types.py`：参数与结果数据结构。
- `metrics.py`：SNR/RMS/CC 评估函数（可选）。

## 3. API 约定（第一版）

建议入口函数：

```python
denoise_trace(trace, dt, *, f_s, f_e, bwconn=8, return_debug=False, **kwargs)
denoise_section(traces, dt, *, f_s, f_e, bwconn=8, workers=1, **kwargs)
```

建议结果对象（`types.py`）：

```python
@dataclass
class DenoiseDebug:
    org_tf: np.ndarray
    gcv_tf: np.ndarray
    final_tf: np.ndarray
    freq: np.ndarray

@dataclass
class DenoiseResult:
    data: np.ndarray
    debug: Optional[DenoiseDebug] = None
    meta: Dict[str, Any] = field(default_factory=dict)
```

约束：

- `trace` 输入为 1D `np.ndarray`；
- `dt` 统一单位为秒；
- `bwconn` 只接受 `4|8`；
- 对非法参数抛 `ValueError`，错误信息可直接显示在 UI。

## 4. 与 zplotpy 的集成方式

`zplotpy` 侧只做三件事：

1. 收集参数（`f_s/f_e/bwconn/窗口`）；
2. 调用 `pyAOBS.processors.denoise.pipeline`；
3. 显示波形和时频中间结果（当 `return_debug=True`）。

建议集成点：

- 单道试算按钮：快速体验，低延时；
- 批处理入口：对当前记录或全部记录执行去噪；
- 结果回写：作为新数据层或内存副本，不覆盖原始数据。

### 4.1 最小调用示例（zplotpy 侧）

下面示例展示 `zplotpy` 在拿到单道数据 `trace` 与采样间隔 `dt` 后，如何调用去噪并拿到时频中间结果用于显示：

```python
import numpy as np
from pyAOBS.processors.denoise import denoise_trace

# trace: (n_samples,) 的 numpy 数组
# dt: 秒
res = denoise_trace(
    trace,
    dt,
    f_s=1.0,
    f_e=60.0,
    bwconn=8,
    strength=1.0,       # 建议 UI 提供：0.8/1.0/1.3 三档
    return_debug=True,  # 需要画时频图/调试时打开
)

denoised = res.data

# debug：用于 zplotpy 时频显示
if res.debug is not None:
    tf = res.debug.final_tf      # (n_freq, n_frames) 复数 TF
    freq = res.debug.freq        # (n_freq,) Hz
    power = np.abs(tf)           # 或 20*log10(|tf|+eps)
```

建议 UI 策略：

- **保守**：`strength=0.8`（更保形）
- **标准**：`strength=1.0`
- **激进**：`strength=1.3`（更强压噪）

## 5. 分阶段实施（可执行）

### P0（1-2 天）：骨架与接口打通

- 新增 `types.py`、`pipeline.py`、`__init__.py`；
- `denoise_trace()` 返回“占位结果”（先直通）；
- `zplotpy` 成功调用并显示结果。

验收：调用链贯通，参数可传，异常可见。

### P1（2-3 天）：WSST + GCV 最小可用版本

- 完成 `wsst_backend.py`、`gcv_threshold.py`；
- 在 `pipeline.py` 串联：`WSST -> GCV -> iWSST`；
- 支持 `return_debug` 输出 `org_tf/gcv_tf`。

验收：单道可稳定运行，去噪结果优于未处理。

### P2（2-3 天）：形态学增强

- 实现 `morphology_mask.py`；
- 串联：`WSST -> GCV -> morphology -> iWSST`；
- 输出 `final_tf`，并新增关键参数（连通域最小面积等）。

验收：时频孤立噪声显著减少，细节保留优于仅 GCV。

### P3（1-2 天）：批处理与性能优化

- 新增 `denoise_section()`；
- 分块与并行（可选）；
- 增加缓存与降采样策略以支持交互场景。

验收：典型数据规模下耗时可接受，内存稳定。

### P4（1-2 天）：评估与回归

- 实现 `metrics.py`（SNR/RMS/CC）；
- 固定测试集 + 基线方法（带通/硬软阈值）对比；
- 补充最小单元测试和接口测试。

验收：指标输出稳定，可复现。

## 6. 依赖策略

- 采用“懒加载 + 明确报错”：
  - 在算法执行时导入可选库；
  - 缺依赖时提示安装命令；
  - 不影响 `zplotpy` 主程序启动。

## 7. 风险与规避

- 风险：WSST 后端差异导致结果不一致。  
  规避：在 `wsst_backend.py` 做统一归一化与单元测试。

- 风险：逐点 GCV 耗时高。  
  规避：先实现分块/向量化，再加并行。

- 风险：形态学参数敏感。  
  规避：提供“保守/标准/激进”三档预设给 `zplotpy`。

## 8. 立即下一步

1. 先创建 `__init__.py + types.py + pipeline.py` 的空实现；
2. 在 `zplotpy` 接入 `denoise_trace()` 单道入口；
3. 打通后再替换内部实现为 WSST+GCV+形态学真实流程。

