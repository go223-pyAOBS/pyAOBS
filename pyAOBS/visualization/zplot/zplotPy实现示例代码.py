"""
zplotPy 自适应叠加和Hilbert变换实现示例代码

本文件展示核心功能的简化实现，供参考
"""

import numpy as np
from scipy import signal
from typing import List, Optional, Dict, Tuple


class SignalProcessor:
    """信号处理类 - 实现Hilbert变换等功能"""
    
    def hilbert_transform(self, x: np.ndarray, output_mode: int = 0) -> np.ndarray:
        """Hilbert变换
        
        Args:
            x: 输入信号
            output_mode: 输出模式
                0 = 标准Hilbert变换（虚部）
                1 = 输出相位
                2 = 输出包络
        
        Returns:
            处理后的信号
        """
        if len(x) == 0:
            return x
        
        # 计算解析信号（Hilbert变换）
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


class AdaptiveStacker:
    """自适应叠加类 - 实现tcas和pstack功能"""
    
    def __init__(self, nsi: int = 10, pjgl: int = 3, 
                 stkwb: float = 0.04, stkwl: float = 0.5,
                 dtcw: float = 0.1, ratio: float = 1.0):
        """初始化自适应叠加器
        
        Args:
            nsi: 叠加迭代次数（默认10）
            pjgl: Lp范数指数（默认3）
            stkwb: 叠加窗口起始时间（秒，默认0.04）
            stkwl: 叠加窗口长度（秒，默认0.5）
            dtcw: 差分时间搜索范围（秒，默认0.1）
            ratio: 相位一致性加权比例因子（默认1.0）
        """
        self.nsi = nsi
        self.pjgl = pjgl
        self.stkwb = stkwb
        self.stkwl = stkwl
        self.dtcw = dtcw
        self.ratio = ratio
        self.signal_processor = SignalProcessor()
    
    def normalize_traces(self, traces: List[np.ndarray]) -> List[np.ndarray]:
        """归一化道数据（改进方法）
        
        对每条道独立归一化，使用最大绝对值
        这是Haibo改进的关键点之一
        """
        normalized = []
        for trace in traces:
            max_val = np.max(np.abs(trace))
            if max_val > 0:
                normalized.append(trace / max_val)
            else:
                normalized.append(trace.copy())
        return normalized
    
    def stack_traces(self, traces: List[np.ndarray],
                    time_shifts: List[float],
                    times: np.ndarray,
                    window_start_idx: int,
                    window_length: int,
                    use_phase_weighting: bool = True) -> Tuple[np.ndarray, float]:
        """叠加多道数据（带相位一致性加权）
        
        Args:
            traces: 道数据列表
            time_shifts: 每道的时间偏移（秒）
            times: 时间数组
            window_start_idx: 窗口起始样本索引
            window_length: 窗口长度（样本数）
            use_phase_weighting: 是否使用相位一致性加权
        
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
        dt = times[1] - times[0] if len(times) > 1 else 0.001
        
        for i, trace in enumerate(traces):
            # 计算窗口起始位置（考虑时间偏移）
            shift_samples = int(time_shifts[i] / dt)
            start_idx = window_start_idx + shift_samples
            
            if start_idx < 0 or start_idx + n_samples > len(trace):
                continue
            
            window_data = trace[start_idx:start_idx + n_samples]
            
            # 线性叠加和二次叠加
            stacked += window_data
            quadratic_sum += window_data**2
            
            # Hilbert变换提取相位（Haibo新增）
            if use_phase_weighting:
                phases = self.signal_processor.extract_phase(window_data)
                phase_cos_sum += np.cos(phases)
                phase_sin_sum += np.sin(phases)
            
            n_valid_traces += 1
        
        if n_valid_traces == 0:
            return stacked, 0.0
        
        # 计算相位一致性因子（Haibo新增）
        if use_phase_weighting:
            phase_coherence = np.sqrt(phase_cos_sum**2 + phase_sin_sum**2) / n_valid_traces
            phase_weight = phase_coherence ** self.ratio
        else:
            phase_weight = np.ones(n_samples)
        
        # 加权叠加（Haibo改进）
        stacked = stacked * phase_weight / n_valid_traces
        
        # 归一化
        max_val = np.max(np.abs(stacked))
        if max_val > 0:
            stacked = stacked / max_val
        
        # 计算质量指标（L2测度的道失配）
        quality_metric = np.sum(np.abs(quadratic_sum)) / (n_valid_traces * n_samples)
        
        return stacked, quality_metric
    
    def align_traces(self, traces: List[np.ndarray], 
                    times: np.ndarray,
                    initial_picks: Optional[List[int]] = None,
                    sampling_rate: Optional[float] = None) -> Dict:
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
        # 1. 归一化（Haibo改进）
        normalized_traces = self.normalize_traces(traces)
        
        # 2. 计算采样率
        if sampling_rate is None:
            if len(times) > 1:
                sampling_rate = 1.0 / (times[1] - times[0])
            else:
                sampling_rate = 1000.0  # 默认采样率
        
        dt = 1.0 / sampling_rate
        
        # 3. 确定叠加窗口
        window_start_idx = int(self.stkwb / dt)
        window_length = int(self.stkwl / dt) if self.stkwl else len(times) - window_start_idx
        window_length = min(window_length, len(times) - window_start_idx)
        
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
    
    def estimate_errors(self, traces: List[np.ndarray],
                       stacked: np.ndarray,
                       times: np.ndarray,
                       time_shifts: List[float],
                       window_start_idx: int,
                       window_length: int) -> List[float]:
        """估计每道的误差（基于叠加功率半宽度）
        
        这是Haibo改进的误差估计方法
        """
        errors = []
        dt = times[1] - times[0] if len(times) > 1 else 0.001
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


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """使用示例"""
    
    # 1. 生成测试数据
    n_traces = 10
    n_samples = 1000
    sampling_rate = 1000.0  # Hz
    times = np.linspace(0, 1, n_samples)
    
    traces = []
    for i in range(n_traces):
        # 生成带时间偏移的信号
        shift = 0.01 * i  # 每道偏移10ms
        t_shifted = times + shift
        trace = np.sin(2 * np.pi * 10 * t_shifted)  # 10Hz信号
        trace += 0.1 * np.random.randn(n_samples)  # 添加噪声
        traces.append(trace)
    
    # 2. 创建自适应叠加器
    stacker = AdaptiveStacker(
        nsi=10,          # 迭代10次
        pjgl=3,          # L3范数
        stkwb=0.04,      # 窗口起始0.04秒
        stkwl=0.5,       # 窗口长度0.5秒
        dtcw=0.1,        # 搜索范围0.1秒
        ratio=1.0        # 相位一致性加权比例
    )
    
    # 3. 执行自适应叠加
    result = stacker.align_traces(traces, times, sampling_rate=sampling_rate)
    
    # 4. 显示结果
    print("自适应叠加结果：")
    print(f"  质量指标: {result['quality_metric']:.6f}")
    print(f"  平均误差: {np.mean(result['errors']):.6f} s")
    print(f"  最大误差: {np.max(result['errors']):.6f} s")
    print(f"  最小误差: {np.min(result['errors']):.6f} s")
    print(f"\n各道时间偏移:")
    for i, shift in enumerate(result['time_shifts']):
        print(f"  道 {i}: {shift*1000:.2f} ms, 误差: {result['errors'][i]*1000:.2f} ms")
    
    return result


if __name__ == "__main__":
    result = example_usage()
