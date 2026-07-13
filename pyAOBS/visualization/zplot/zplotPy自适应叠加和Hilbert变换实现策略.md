# zplotPy 自适应叠加和 Hilbert 变换实现策略

## 一、概述

本文档描述如何在 zplotPy（Python版本的zplot）中实现"自适应叠加和Hilbert变换"功能，这些功能在原始Fortran代码中由 `tcas`、`pstack` 和 `hilbert` 子程序实现。

---

## 二、功能需求分析

### 2.1 核心功能

1. **自适应叠加对齐（Adaptive Stacking Alignment）**
   - 使用迭代方法确定最优道对齐
   - Lp范数优化（默认L3范数）
   - 改进的归一化处理
   - 误差估计（基于叠加功率半宽度）

2. **Hilbert变换（Hilbert Transform）**
   - 计算信号的Hilbert变换
   - 提取瞬时相位
   - 提取信号包络

3. **相位一致性加权叠加（Phase Coherence Weighted Stacking）**
   - 使用Hilbert变换计算相位一致性
   - 基于相位一致性进行加权叠加
   - 提高信噪比，抑制噪声

### 2.2 使用场景

- **震相拾取优化**：自动对齐多道数据，提高拾取精度
- **信号增强**：通过叠加提高弱信号的信噪比
- **质量评估**：误差估计帮助评估拾取质量
- **OBS数据处理**：特别适合处理振幅和相位差异较大的OBS数据

---

## 三、架构设计

### 3.1 模块划分

```
zplotpy/
├── signal_processor.py      # 新增：信号处理模块（Hilbert变换）
├── adaptive_stack.py         # 新增：自适应叠加模块
├── data_processor.py         # 扩展：集成新功能
├── parameters.py            # 扩展：添加新参数
└── zplot_gui.py             # 扩展：添加入口
```

### 3.2 类设计

#### 3.2.1 SignalProcessor（信号处理类）

```python
class SignalProcessor:
    """信号处理类 - 实现Hilbert变换等功能"""
    
    def hilbert_transform(self, x: np.ndarray, output_mode: int = 0) -> np.ndarray:
        """Hilbert变换
        
        Args:
            x: 输入信号
            output_mode: 输出模式
                0 = 标准Hilbert变换
                1 = 输出相位
                2 = 输出包络
        
        Returns:
            处理后的信号
        """
    
    def extract_phase(self, x: np.ndarray) -> np.ndarray:
        """提取瞬时相位"""
    
    def extract_envelope(self, x: np.ndarray) -> np.ndarray:
        """提取信号包络"""
    
    def phase_coherence(self, phases: List[np.ndarray]) -> np.ndarray:
        """计算相位一致性"""
```

#### 3.2.2 AdaptiveStacker（自适应叠加类）

```python
class AdaptiveStacker:
    """自适应叠加类 - 实现tcas和pstack功能"""
    
    def __init__(self, nsi: int = 10, pjgl: int = 3, 
                 stkwb: float = 0.04, stkwl: float = None,
                 dtcw: float = None, ratio: float = 1.0):
        """初始化自适应叠加器
        
        Args:
            nsi: 叠加迭代次数（默认10）
            pjgl: Lp范数指数（默认3）
            stkwb: 叠加窗口起始时间（秒，默认0.04）
            stkwl: 叠加窗口长度（秒）
            dtcw: 差分时间搜索范围（秒）
            ratio: 相位一致性加权比例因子
        """
    
    def align_traces(self, traces: List[np.ndarray], 
                    times: np.ndarray,
                    initial_picks: Optional[List[int]] = None,
                    sampling_rate: float = None) -> Dict:
        """自适应对齐多道数据
        
        Args:
            traces: 道数据列表
            times: 时间数组
            initial_picks: 初始拾取样本号列表（可选）
            sampling_rate: 采样率（Hz）
        
        Returns:
            {
                'time_shifts': List[float],  # 每道的时间偏移
                'errors': List[float],        # 每道的误差估计
                'stacked_trace': np.ndarray,  # 最终叠加结果
                'quality_metric': float       # 叠加质量指标
            }
        """
    
    def stack_traces(self, traces: List[np.ndarray],
                    time_shifts: List[float],
                    times: np.ndarray,
                    window_start: int,
                    window_length: int,
                    use_phase_weighting: bool = True) -> np.ndarray:
        """叠加多道数据（带相位一致性加权）"""
```

---

## 四、实现步骤

### 4.1 第一阶段：基础信号处理（SignalProcessor）

#### 步骤1.1：实现Hilbert变换

**文件：** `signal_processor.py`

**实现方法：**
- 使用 `scipy.signal.hilbert` 作为基础实现
- 或者使用FFT方法实现（更接近原始Fortran代码）

```python
import numpy as np
from scipy import signal
from typing import List, Optional

class SignalProcessor:
    """信号处理类"""
    
    def hilbert_transform(self, x: np.ndarray, output_mode: int = 0) -> np.ndarray:
        """Hilbert变换
        
        使用scipy.signal.hilbert实现，与原始Fortran代码等价
        """
        if len(x) == 0:
            return x
        
        # 计算Hilbert变换
        analytic_signal = signal.hilbert(x)
        
        if output_mode == 0:
            # 标准Hilbert变换（虚部）
            return np.imag(analytic_signal)
        elif output_mode == 1:
            # 输出相位
            return np.angle(analytic_signal)
        elif output_mode == 2:
            # 输出包络
            return np.abs(analytic_signal)
        else:
            return np.imag(analytic_signal)
    
    def extract_phase(self, x: np.ndarray) -> np.ndarray:
        """提取瞬时相位"""
        analytic_signal = signal.hilbert(x)
        return np.angle(analytic_signal)
    
    def extract_envelope(self, x: np.ndarray) -> np.ndarray:
        """提取信号包络"""
        analytic_signal = signal.hilbert(x)
        return np.abs(analytic_signal)
    
    def phase_coherence(self, phases: List[np.ndarray]) -> np.ndarray:
        """计算相位一致性
        
        Args:
            phases: 相位数组列表（每道一个相位数组）
        
        Returns:
            相位一致性数组（长度与相位数组相同）
        """
        if not phases or len(phases) == 0:
            return np.array([])
        
        n_traces = len(phases)
        n_samples = len(phases[0])
        
        # 计算相位余弦和、正弦和
        cos_sum = np.zeros(n_samples)
        sin_sum = np.zeros(n_samples)
        
        for phase in phases:
            cos_sum += np.cos(phase)
            sin_sum += np.sin(phase)
        
        # 计算相位一致性：C = |Σe^(iφ)|/N
        coherence = np.sqrt(cos_sum**2 + sin_sum**2) / n_traces
        
        return coherence
```

**测试：**
- 单元测试验证Hilbert变换的正确性
- 对比scipy实现与原始Fortran代码的结果

#### 步骤1.2：集成到DataProcessor

**文件：** `data_processor.py`

```python
from .signal_processor import SignalProcessor

class DataProcessor:
    def __init__(self):
        self.signal_processor = SignalProcessor()  # 新增
    
    def apply_hilbert_transform(self, data: np.ndarray, 
                                output_mode: int = 0) -> np.ndarray:
        """应用Hilbert变换"""
        return self.signal_processor.hilbert_transform(data, output_mode)
```

---

### 4.2 第二阶段：自适应叠加（AdaptiveStacker）

#### 步骤2.1：实现数据归一化

```python
def normalize_traces(self, traces: List[np.ndarray]) -> List[np.ndarray]:
    """归一化道数据（改进方法）
    
    对每条道独立归一化，使用最大绝对值
    """
    normalized = []
    for trace in traces:
        max_val = np.max(np.abs(trace))
        if max_val > 0:
            normalized.append(trace / max_val)
        else:
            normalized.append(trace.copy())
    return normalized
```

#### 步骤2.2：实现初始叠加（pstack）

```python
def stack_traces(self, traces: List[np.ndarray],
                time_shifts: List[float],
                times: np.ndarray,
                window_start_idx: int,
                window_length: int,
                use_phase_weighting: bool = True,
                ratio: float = 1.0) -> Tuple[np.ndarray, float]:
    """叠加多道数据（带相位一致性加权）
    
    Returns:
        (stacked_trace, quality_metric)
    """
    n_samples = window_length
    stacked = np.zeros(n_samples)
    quadratic_sum = np.zeros(n_samples)
    
    # 相位一致性相关
    phase_cos_sum = np.zeros(n_samples)
    phase_sin_sum = np.zeros(n_samples)
    
    n_valid_traces = 0
    
    for i, trace in enumerate(traces):
        # 计算窗口起始位置（考虑时间偏移）
        shift_samples = int(time_shifts[i] / (times[1] - times[0]))
        start_idx = window_start_idx + shift_samples
        
        if start_idx < 0 or start_idx + n_samples > len(trace):
            continue
        
        window_data = trace[start_idx:start_idx + n_samples]
        
        # 线性叠加和二次叠加
        stacked += window_data
        quadratic_sum += window_data**2
        
        # Hilbert变换提取相位
        if use_phase_weighting:
            phases = self.signal_processor.extract_phase(window_data)
            phase_cos_sum += np.cos(phases)
            phase_sin_sum += np.sin(phases)
        
        n_valid_traces += 1
    
    if n_valid_traces == 0:
        return stacked, 0.0
    
    # 计算相位一致性因子
    if use_phase_weighting:
        phase_coherence = np.sqrt(phase_cos_sum**2 + phase_sin_sum**2) / n_valid_traces
        phase_weight = phase_coherence ** ratio
    else:
        phase_weight = np.ones(n_samples)
    
    # 加权叠加
    stacked = stacked * phase_weight / n_valid_traces
    
    # 归一化
    max_val = np.max(np.abs(stacked))
    if max_val > 0:
        stacked = stacked / max_val
    
    # 计算质量指标（L2测度的道失配）
    quality_metric = np.sum(np.abs(quadratic_sum)) / (n_valid_traces * n_samples)
    
    return stacked, quality_metric
```

#### 步骤2.3：实现迭代对齐优化

```python
def align_traces(self, traces: List[np.ndarray], 
                times: np.ndarray,
                initial_picks: Optional[List[int]] = None,
                sampling_rate: Optional[float] = None) -> Dict:
    """自适应对齐多道数据"""
    
    # 1. 归一化
    normalized_traces = self.normalize_traces(traces)
    
    # 2. 计算采样率
    if sampling_rate is None:
        sampling_rate = 1.0 / (times[1] - times[0])
    
    dt = 1.0 / sampling_rate
    
    # 3. 确定叠加窗口
    window_start_idx = int(self.stkwb / dt)
    window_length = int(self.stkwl / dt) if self.stkwl else len(times) - window_start_idx
    
    # 4. 初始叠加
    initial_shifts = [0.0] * len(traces)
    stacked, _ = self.stack_traces(
        normalized_traces, initial_shifts, times,
        window_start_idx, window_length
    )
    
    # 5. 迭代优化
    time_shifts = [0.0] * len(traces)
    errors = [0.0] * len(traces)
    
    search_range_samples = int(self.dtcw / dt) if self.dtcw else 50
    
    for iteration in range(self.nsi):
        for i, trace in enumerate(normalized_traces):
            if initial_picks and initial_picks[i] <= 0:
                continue
            
            # 确定搜索中心
            if initial_picks:
                center_idx = initial_picks[i] + window_start_idx
            else:
                center_idx = window_start_idx
            
            # 搜索最优对齐
            min_error = float('inf')
            best_shift = 0
            
            for shift_samples in range(-search_range_samples, search_range_samples + 1):
                start_idx = center_idx + shift_samples
                
                if start_idx < 0 or start_idx + window_length > len(trace):
                    continue
                
                window_data = trace[start_idx:start_idx + window_length]
                
                # 计算Lp范数差异
                diff = np.abs(stacked - window_data) ** self.pjgl
                error = np.mean(diff)
                
                if error < min_error:
                    min_error = error
                    best_shift = shift_samples * dt
            
            time_shifts[i] = best_shift
        
        # 使用新对齐重新叠加
        stacked, quality = self.stack_traces(
            normalized_traces, time_shifts, times,
            window_start_idx, window_length
        )
        
        # 最后一次迭代：计算误差估计
        if iteration == self.nsi - 1:
            errors = self.estimate_errors(
                normalized_traces, stacked, times,
                time_shifts, window_start_idx, window_length
            )
    
    return {
        'time_shifts': time_shifts,
        'errors': errors,
        'stacked_trace': stacked,
        'quality_metric': quality
    }
```

#### 步骤2.4：实现误差估计

```python
def estimate_errors(self, traces: List[np.ndarray],
                   stacked: np.ndarray,
                   times: np.ndarray,
                   time_shifts: List[float],
                   window_start_idx: int,
                   window_length: int) -> List[float]:
    """估计每道的误差（基于叠加功率半宽度）"""
    
    errors = []
    dt = times[1] - times[0]
    emin, emax = 0.025, 0.150
    erl = 1.25
    
    for i, trace in enumerate(traces):
        # 计算叠加功率曲线
        shift_samples = int(time_shifts[i] / dt)
        center_idx = window_start_idx + shift_samples
        
        search_range = 50  # 搜索范围（样本数）
        power_curve = []
        
        for shift in range(-search_range, search_range + 1):
            start_idx = center_idx + shift
            if start_idx < 0 or start_idx + window_length > len(trace):
                power_curve.append(float('inf'))
                continue
            
            window_data = trace[start_idx:start_idx + window_length]
            diff = np.abs(stacked - window_data) ** self.pjgl
            power = np.mean(diff)
            power_curve.append(power)
        
        # 找到最小值位置
        power_array = np.array(power_curve)
        min_idx = np.argmin(power_array)
        min_power = power_array[min_idx]
        
        # 计算半宽度（左右交叉点）
        threshold = min_power * erl
        
        # 左侧交叉点
        left_idx = min_idx
        while left_idx > 0 and power_array[left_idx] < threshold:
            left_idx -= 1
        
        # 右侧交叉点
        right_idx = min_idx
        while right_idx < len(power_array) - 1 and power_array[right_idx] < threshold:
            right_idx += 1
        
        # 线性插值计算精确位置
        if left_idx < min_idx:
            if left_idx + 1 < len(power_array):
                denom = power_array[left_idx + 1] - power_array[left_idx]
                if abs(denom) > 1e-5:
                    err_left = (min_idx - left_idx) * dt - \
                              (threshold - power_array[left_idx]) / denom * dt
                else:
                    err_left = (min_idx - left_idx) * dt
            else:
                err_left = (min_idx - left_idx) * dt
        else:
            err_left = 0.0
        
        if right_idx > min_idx:
            if right_idx > 0:
                denom = power_array[right_idx] - power_array[right_idx - 1]
                if abs(denom) > 1e-5:
                    err_right = (right_idx - min_idx - 1) * dt + \
                               (threshold - power_array[right_idx - 1]) / denom * dt
                else:
                    err_right = (right_idx - min_idx) * dt
            else:
                err_right = (right_idx - min_idx) * dt
        else:
            err_right = 0.0
        
        # 平均误差
        if err_left > 0 and err_right > 0:
            err = (err_left + err_right) / 2.0
        elif err_left > 0:
            err = err_left
        elif err_right > 0:
            err = err_right
        else:
            err = emax
        
        # 约束误差范围
        err = max(emin, min(emax, err))
        errors.append(err)
    
    return errors
```

---

### 4.3 第三阶段：参数扩展

#### 步骤3.1：扩展ZPlotParameters

**文件：** `parameters.py`

```python
@dataclass
class ZPlotParameters:
    # ... 现有参数 ...
    
    # 自适应叠加参数（新增）
    nsi: int = 10              # 叠加迭代次数
    pjgl: int = 3              # Lp范数指数（1=L1, 2=L2, 3=L3）
    stkwb: float = 0.04        # 叠加窗口起始时间（秒）
    stkwl: float = 0.5         # 叠加窗口长度（秒）
    dtcw: float = 0.1          # 差分时间搜索范围（秒）
    hilbratio: float = 1.0     # Hilbert变换比例因子（相位一致性加权）
    
    # 自适应叠加开关
    iadaptive: int = 0         # 自适应叠加开关：0=关闭, 1=启用
```

#### 步骤3.2：添加菜单项（可选）

如果需要通过GUI菜单控制，可以在 `top_toolbar.py` 中添加新的菜单项。

---

### 4.4 第四阶段：GUI集成

#### 步骤4.1：在ZPlotGUI中添加入口

**文件：** `zplot_gui.py`

```python
from .adaptive_stack import AdaptiveStacker

class ZPlotGUI:
    def __init__(self):
        # ... 现有初始化 ...
        self.adaptive_stacker = AdaptiveStacker(
            nsi=self.params.nsi,
            pjgl=self.params.pjgl,
            stkwb=self.params.stkwb,
            stkwl=self.params.stkwl,
            dtcw=self.params.dtcw,
            ratio=self.params.hilbratio
        )
    
    def perform_adaptive_stacking(self):
        """执行自适应叠加对齐"""
        if not self.traces or len(self.traces) == 0:
            return
        
        # 获取当前拾取字对应的初始拾取
        initial_picks = []
        for i in range(len(self.traces)):
            if i in self.pick_manager.picks:
                picks = self.pick_manager.picks[i]
                if self.params.apick in picks:
                    pick_time = picks[self.params.apick]
                    # 转换为样本号
                    pick_sample = int((pick_time - self.times[0]) / 
                                     (self.times[1] - self.times[0]))
                    initial_picks.append(pick_sample)
                else:
                    initial_picks.append(-1)
            else:
                initial_picks.append(-1)
        
        # 执行自适应叠加
        result = self.adaptive_stacker.align_traces(
            self.traces,
            self.times,
            initial_picks=initial_picks if any(p >= 0 for p in initial_picks) else None
        )
        
        # 更新拾取（应用时间偏移）
        for i, shift in enumerate(result['time_shifts']):
            if i in self.pick_manager.picks:
                picks = self.pick_manager.picks[i]
                if self.params.apick in picks:
                    picks[self.params.apick] += shift
        
        # 显示结果
        self.show_stacking_result(result)
        
        # 重新绘制
        self.update_plot()
    
    def show_stacking_result(self, result: Dict):
        """显示叠加结果"""
        print(f"Adaptive Stacking Results:")
        print(f"  Quality Metric: {result['quality_metric']:.6f}")
        print(f"  Average Error: {np.mean(result['errors']):.6f} s")
        print(f"  Max Error: {np.max(result['errors']):.6f} s")
        print(f"  Min Error: {np.min(result['errors']):.6f} s")
```

#### 步骤4.2：添加键盘快捷键（可选）

```python
def setup_event_handlers(self):
    # ... 现有事件绑定 ...
    
    # 自适应叠加快捷键（例如：'s'键）
    self.root.bind('<KeyPress-s>', lambda e: self.perform_adaptive_stacking())
```

---

## 五、技术实现细节

### 5.1 依赖库

```python
import numpy as np
from scipy import signal
from typing import List, Optional, Dict, Tuple
```

### 5.2 性能优化

1. **向量化计算**：使用NumPy向量化操作替代循环
2. **内存管理**：避免不必要的数据复制
3. **并行计算**：对于大量道的处理，可以考虑使用多进程

### 5.3 数值稳定性

1. **除零检查**：在所有除法操作前检查分母
2. **边界检查**：确保数组索引在有效范围内
3. **NaN处理**：检查并处理NaN值

---

## 六、测试策略

### 6.1 单元测试

**文件：** `test_signal_processor.py`

```python
def test_hilbert_transform():
    """测试Hilbert变换"""
    processor = SignalProcessor()
    x = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
    
    # 测试标准Hilbert变换
    h = processor.hilbert_transform(x, output_mode=0)
    assert len(h) == len(x)
    
    # 测试相位提取
    phase = processor.extract_phase(x)
    assert len(phase) == len(x)
    assert np.all(phase >= -np.pi) and np.all(phase <= np.pi)
    
    # 测试包络提取
    envelope = processor.extract_envelope(x)
    assert len(envelope) == len(x)
    assert np.all(envelope >= 0)
```

**文件：** `test_adaptive_stack.py`

```python
def test_adaptive_stacking():
    """测试自适应叠加"""
    # 生成测试数据
    n_traces = 10
    n_samples = 1000
    traces = []
    for i in range(n_traces):
        t = np.linspace(0, 1, n_samples)
        # 添加时间偏移
        shift = 0.01 * i
        trace = np.sin(2 * np.pi * 10 * (t + shift))
        traces.append(trace)
    
    times = np.linspace(0, 1, n_samples)
    
    stacker = AdaptiveStacker(nsi=5, pjgl=3)
    result = stacker.align_traces(traces, times)
    
    assert len(result['time_shifts']) == n_traces
    assert len(result['errors']) == n_traces
    assert len(result['stacked_trace']) > 0
```

### 6.2 集成测试

- 测试与现有数据加载和绘图系统的集成
- 测试与拾取管理系统的集成
- 测试GUI交互

### 6.3 对比测试

- 与原始Fortran代码结果对比
- 使用相同输入数据，验证输出一致性

---

## 七、实施优先级

### 7.1 高优先级（核心功能）

1. ✅ **SignalProcessor类**：Hilbert变换基础功能
2. ✅ **AdaptiveStacker类**：自适应叠加核心算法
3. ✅ **参数扩展**：添加必要参数

### 7.2 中优先级（集成功能）

4. ✅ **GUI集成**：添加入口和结果显示
5. ✅ **拾取更新**：自动更新拾取时间

### 7.3 低优先级（优化功能）

6. ⚠️ **性能优化**：并行计算、内存优化
7. ⚠️ **可视化增强**：叠加结果显示、误差可视化
8. ⚠️ **参数调优界面**：GUI参数调整

---

## 八、文件清单

### 8.1 新增文件

1. `signal_processor.py` - 信号处理模块（Hilbert变换）
2. `adaptive_stack.py` - 自适应叠加模块
3. `test_signal_processor.py` - 信号处理测试
4. `test_adaptive_stack.py` - 自适应叠加测试

### 8.2 修改文件

1. `data_processor.py` - 集成SignalProcessor
2. `parameters.py` - 添加新参数
3. `zplot_gui.py` - 添加入口和结果显示
4. `plot_manager.py` - 可选：叠加结果显示

---

## 九、注意事项

### 9.1 兼容性

- 保持与现有代码的兼容性
- 新功能通过参数开关控制，默认关闭

### 9.2 性能考虑

- 自适应叠加是计算密集型操作，注意性能优化
- 对于大量道的数据，考虑进度显示

### 9.3 用户体验

- 提供清晰的进度反馈
- 显示叠加结果和质量指标
- 允许用户撤销操作

---

## 十、后续扩展

### 10.1 功能扩展

1. **批量处理**：支持批量处理多个记录
2. **参数自动优化**：自动寻找最优参数
3. **可视化增强**：叠加前后对比显示

### 10.2 算法改进

1. **多窗口叠加**：支持多个时间窗口
2. **频率域处理**：在频率域进行叠加
3. **机器学习**：使用ML方法优化对齐

---

## 十一、参考文档

- `自适应叠加和Hilbert变换功能分析.md` - 功能分析文档
- `Haibo_Huang_2023修改总结.md` - 原始修改总结
- `代码分析报告.md` - 代码结构分析

---

**文档生成日期：** 2024年  
**版本：** 1.0
