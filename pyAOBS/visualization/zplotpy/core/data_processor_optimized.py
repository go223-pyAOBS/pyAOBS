"""
data_processor_optimized.py - ZPLOT 数据处理模块（优化版本）

展示性能优化后的实现，特别是变增益模式的向量化优化
"""

import numpy as np
from typing import List, Optional
from scipy import signal
from scipy.ndimage import uniform_filter1d

try:
    from .parameters import ZPlotParameters
    from .signal_processor import SignalProcessor
except ImportError:
    from parameters import ZPlotParameters
    from signal_processor import SignalProcessor


class DataProcessorOptimized:
    """优化版数据处理类
    
    主要优化：
    1. 变增益模式使用向量化操作替代循环
    2. 批量处理多条道
    3. 添加结果缓存机制
    """
    
    def __init__(self, enable_cache: bool = True, cache_size: int = 100):
        """初始化数据处理器
        
        Args:
            enable_cache: 是否启用结果缓存
            cache_size: 缓存大小
        """
        self.signal_processor = SignalProcessor()
        self.enable_cache = enable_cache
        self._cache = {}
        self._cache_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _generate_cache_key(self, data: np.ndarray, params: dict) -> str:
        """生成缓存键"""
        import hashlib
        # 使用数据的哈希值和参数生成缓存键
        data_hash = hashlib.md5(data.tobytes()).hexdigest()
        params_str = str(sorted(params.items()))
        return f"{data_hash}_{hash(params_str)}"
    
    def apply_gain_vectorized(self, data: np.ndarray, iscale: int, amp: float,
                             rcor: float, offset: float, times: Optional[np.ndarray] = None,
                             tvg: float = 1.0, pvg: float = 1.0) -> np.ndarray:
        """应用增益控制（向量化优化版本）
        
        优化点：
        - 使用 scipy.ndimage.uniform_filter1d 实现滑动窗口的向量化计算
        - 避免 Python 循环，大幅提升性能
        
        Args:
            data: 输入数据
            iscale: 缩放模式 (0=自动, 1=固定, 2=变增益)
            amp: 振幅缩放因子
            rcor: 距离校正指数
            offset: 炮检距 (km)
            times: 时间数组（用于变增益模式的时间变化增益）
            tvg: 时间变化增益参数
            pvg: 功率变化增益参数
            
        Returns:
            增益处理后的数据
        """
        if iscale == 0:
            # 自动增益：归一化到最大绝对值，然后乘以amp
            max_val = np.max(np.abs(data))
            if max_val > 0:
                normalized = data / max_val
                return normalized * amp
            else:
                return data * amp
        elif iscale == 1:
            # 固定增益：直接乘以amp
            return data * amp
        elif iscale == 2:
            # 变增益：根据距离校正和时间变化增益
            # 基础增益：gain = amp * (offset + 1)^rcor
            if rcor != 0 and offset > 0:
                gain = amp * np.power(offset + 1.0, rcor)
            else:
                gain = amp
            
            # 应用时间变化增益（tvg和pvg）- 向量化版本
            if times is not None and len(times) == len(data) and (tvg != 1.0 or pvg != 1.0):
                # 计算窗口大小（基于tvg，如果tvg>0，使用时间窗口）
                if tvg > 0 and len(times) > 1:
                    dt = times[1] - times[0]  # 采样间隔
                    window_samples = max(1, int(tvg / dt))  # 窗口采样点数
                    window_samples = min(window_samples, len(data) // 2)  # 不超过数据长度的一半
                    
                    if window_samples > 0 and pvg != 0:
                        # 向量化实现：使用 uniform_filter1d 计算滑动窗口的功率和
                        # 这是关键优化：避免 Python 循环
                        abs_data = np.abs(data)
                        power_data = np.power(abs_data, pvg)
                        
                        # 使用 uniform_filter1d 计算滑动窗口的平均值
                        # 注意：uniform_filter1d 计算的是平均值，需要乘以窗口大小得到总和
                        window_mean = uniform_filter1d(
                            power_data, 
                            size=window_samples, 
                            mode='constant', 
                            cval=0.0
                        )
                        window_sum = window_mean * window_samples
                        
                        # 避免除零错误
                        window_sum = np.maximum(window_sum, 1e-10)
                        
                        # 应用时间变化增益
                        # 每个点除以窗口内功率的 pvg 次方根
                        normalization_factor = np.power(window_sum, 1.0 / pvg)
                        processed_data = data / normalization_factor
                        
                        return processed_data * gain
            
            return data * gain
        else:
            # 未知模式，使用固定增益
            return data * amp
    
    def process_traces_batch(self, traces: List[np.ndarray], times: np.ndarray,
                            offsets: np.ndarray, params: ZPlotParameters,
                            sampling_rate: Optional[float] = None) -> List[np.ndarray]:
        """批量处理地震道数据（优化版本）
        
        优化点：
        - 批量应用滤波（如果可能）
        - 并行处理多条道（可选）
        - 使用缓存避免重复计算
        
        Args:
            traces: 道数据列表
            times: 时间数组
            offsets: 炮检距数组
            params: 绘图参数
            sampling_rate: 采样率（Hz），如果为None则从times计算
            
        Returns:
            处理后的道数据列表
        """
        if not traces or len(traces) == 0:
            return traces
        
        # 计算采样率
        if sampling_rate is None:
            if len(times) > 1:
                sampling_rate = 1.0 / (times[1] - times[0])
            else:
                sampling_rate = 1000.0  # 默认采样率
        
        processed_traces = []
        
        # 批量处理：如果启用了带通滤波，可以尝试批量应用
        if params.ibndps != 0:
            # 批量滤波（如果所有道的长度相同）
            trace_lengths = [len(trace) for trace in traces]
            if len(set(trace_lengths)) == 1:
                # 所有道长度相同，可以批量处理
                traces_array = np.array(traces, dtype=np.float64)
                
                # 批量应用带通滤波
                filtered_array = self._apply_bandpass_filter_batch(
                    traces_array, 
                    params.freqlo, 
                    params.freqhi, 
                    params.npoles,
                    params.izerop,
                    sampling_rate
                )
                
                # 转换为列表
                traces = [filtered_array[i] for i in range(len(traces))]
        
        # 逐道处理其他操作（增益、静校正、裁剪）
        for i, trace in enumerate(traces):
            # 检查缓存
            cache_key = None
            if self.enable_cache:
                cache_params = {
                    'iscale': params.iscale,
                    'amp': params.amp,
                    'rcor': params.rcor,
                    'offset': offsets[i] if i < len(offsets) else 0.0,
                    'imute': params.imute,
                    'clip': params.clip,
                    'tvg': params.tvg,
                    'pvg': params.pvg
                }
                cache_key = self._generate_cache_key(trace, cache_params)
                
                if cache_key in self._cache:
                    self._cache_hits += 1
                    processed_traces.append(self._cache[cache_key])
                    continue
            
            self._cache_misses += 1
            
            # 复制数据，避免修改原始数据
            processed_trace = trace.copy().astype(np.float64)
            
            # 应用增益控制（使用优化版本）
            offset = offsets[i] if i < len(offsets) else 0.0
            processed_trace = self.apply_gain_vectorized(
                processed_trace,
                params.iscale,
                params.amp,
                params.rcor,
                offset,
                times=times if len(times) > 0 else None,
                tvg=params.tvg,
                pvg=params.pvg
            )
            
            # 应用静校正（如果启用）
            if params.imute != 0:
                processed_trace = self._apply_mute(
                    processed_trace,
                    times,
                    params.imute,
                    offset,
                    vmute=params.vmute,
                    tmute=params.tmute,
                    vred=params.vred
                )
            
            # 应用数据裁剪
            if params.clip > 0:
                processed_trace = np.clip(processed_trace, -params.clip, params.clip)
            
            # 转换回原始数据类型
            result = processed_trace.astype(trace.dtype)
            processed_traces.append(result)
            
            # 更新缓存
            if self.enable_cache and cache_key:
                if len(self._cache) >= self._cache_size:
                    # 删除最旧的缓存项（简单的 FIFO）
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                self._cache[cache_key] = result
        
        return processed_traces
    
    def _apply_bandpass_filter_batch(self, traces_array: np.ndarray, 
                                    freqlo: float, freqhi: float,
                                    npoles: int, izerop: int, 
                                    sampling_rate: float) -> np.ndarray:
        """批量应用带通滤波器
        
        Args:
            traces_array: 道数据数组，shape (n_traces, n_samples)
            freqlo: 低截止频率
            freqhi: 高截止频率
            npoles: 滤波器阶数
            izerop: 是否使用零相位滤波
            sampling_rate: 采样率
            
        Returns:
            滤波后的道数据数组
        """
        if freqlo <= 0 or freqhi <= 0 or freqlo >= freqhi:
            return traces_array
        
        # 计算Nyquist频率
        nyquist = sampling_rate / 2.0
        
        # 确保截止频率在有效范围内
        freqlo = min(freqlo, nyquist * 0.99)
        freqhi = min(freqhi, nyquist * 0.99)
        
        if freqlo >= freqhi:
            return traces_array
        
        # 归一化频率
        low = freqlo / nyquist
        high = freqhi / nyquist
        
        # 检查数据长度
        n_samples = traces_array.shape[1]
        min_length = max(30, npoles * 3)
        
        if n_samples < min_length:
            return traces_array
        
        # 设计Butterworth带通滤波器
        order = max(1, npoles // 2)
        
        try:
            b, a = signal.butter(order, [low, high], btype='bandpass')
            
            # 批量应用滤波器
            if izerop == 1:
                # 零相位滤波（使用filtfilt）
                filtered = signal.filtfilt(b, a, traces_array, axis=1)
            else:
                # 非零相位滤波（使用lfilter）
                filtered = signal.lfilter(b, a, traces_array, axis=1)
            
            return filtered
        except Exception as e:
            print(f"批量滤波失败: {e}")
            return traces_array
    
    def _apply_mute(self, data: np.ndarray, times: np.ndarray, 
                   mute_type: int, offset: float = 0.0,
                   vmute: float = 0.0, tmute: float = 0.0,
                   vred: float = 0.0) -> np.ndarray:
        """应用静校正（mute）- 从原 DataProcessor 复制"""
        if mute_type == 0:
            return data
        
        muted_data = data.copy()
        
        if vmute > 0:
            if mute_type > 0:
                if vred > 0:
                    ttmute = tmute + abs(offset) * (1.0/vmute - 1.0/vred)
                else:
                    ttmute = tmute + abs(offset) / vmute
                
                mute_mask = times >= ttmute
                muted_data[~mute_mask] = 0.0
            elif mute_type < 0:
                if vred > 0:
                    ttmute = tmute + abs(offset) * (1.0/vmute - 1.0/vred)
                else:
                    ttmute = tmute + abs(offset) / vmute
                
                mute_mask = times <= ttmute
                muted_data[~mute_mask] = 0.0
        else:
            if mute_type > 0:
                mute_index = int(len(times) * 0.05)
                if mute_index > 0:
                    muted_data[:mute_index] = 0.0
            elif mute_type < 0:
                mute_index = int(len(times) * 0.95)
                if mute_index < len(times):
                    muted_data[mute_index:] = 0.0
        
        return muted_data
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_size': len(self._cache),
            'max_cache_size': self._cache_size,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate
        }
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


# 性能对比测试函数
def benchmark_gain_application():
    """对比原始实现和优化实现的性能"""
    import time
    
    # 创建测试数据
    n_samples = 10000
    data = np.random.randn(n_samples).astype(np.float64)
    times = np.linspace(0, 10, n_samples)
    
    # 测试参数
    iscale = 2
    amp = 1.0
    rcor = 0.5
    offset = 10.0
    tvg = 0.5
    pvg = 2.0
    
    # 原始实现（模拟）
    def apply_gain_original(data, iscale, amp, rcor, offset, times, tvg, pvg):
        if iscale == 2:
            gain = amp * np.power(offset + 1.0, rcor)
            if tvg > 0 and len(times) > 1:
                dt = times[1] - times[0]
                window_samples = max(1, int(tvg / dt))
                window_samples = min(window_samples, len(data) // 2)
                
                processed_data = data.copy()
                if window_samples > 0 and pvg != 0:
                    for j in range(len(data)):
                        start_idx = max(0, j - window_samples // 2)
                        end_idx = min(len(data), j + window_samples // 2 + 1)
                        window_data = np.abs(data[start_idx:end_idx])
                        if len(window_data) > 0:
                            sum_power = np.sum(np.power(window_data, pvg))
                            if sum_power > 0:
                                processed_data[j] = data[j] / np.power(sum_power, 1.0 / pvg)
                return processed_data * gain
        return data * amp
    
    # 优化实现
    processor = DataProcessorOptimized()
    
    # 预热
    _ = apply_gain_original(data, iscale, amp, rcor, offset, times, tvg, pvg)
    _ = processor.apply_gain_vectorized(data, iscale, amp, rcor, offset, times, tvg, pvg)
    
    # 测试原始实现
    n_iterations = 10
    start = time.time()
    for _ in range(n_iterations):
        result1 = apply_gain_original(data, iscale, amp, rcor, offset, times, tvg, pvg)
    time_original = time.time() - start
    
    # 测试优化实现
    start = time.time()
    for _ in range(n_iterations):
        result2 = processor.apply_gain_vectorized(data, iscale, amp, rcor, offset, times, tvg, pvg)
    time_optimized = time.time() - start
    
    # 验证结果一致性
    max_diff = np.max(np.abs(result1 - result2))
    
    print(f"性能对比结果（{n_samples} 个采样点，{n_iterations} 次迭代）：")
    print(f"原始实现耗时: {time_original:.4f} 秒")
    print(f"优化实现耗时: {time_optimized:.4f} 秒")
    print(f"速度提升: {time_original / time_optimized:.2f}x")
    print(f"结果最大差异: {max_diff:.2e}")
    
    return {
        'time_original': time_original,
        'time_optimized': time_optimized,
        'speedup': time_original / time_optimized,
        'max_diff': max_diff
    }


if __name__ == '__main__':
    # 运行性能测试
    benchmark_gain_application()
